"""Review lanes -> review pool: the one-time migration M (PR-3, package C).

The former ``OUROBOROS_REVIEWER_SLOTS`` lanes (triad / scope / advisory / deep review)
and their surface keys become reviewer rows of the subagent catalog
(``OUROBOROS_SUBAGENTS`` rows marked ``review_eligible``). ``review_pool_migration``
carries FROZEN copies of the lane readers so the migration keeps reading old documents
exactly as the release that wrote them did, after the live readers are gone.

Pinned here: the frozen readers resolve Anton's install and the N-1 fixture to the same
effective executions as the base readers (expected values inline, never imported from
old code); the contract's F4-F8 tables; the F6 catalog x lanes matrix (25 cells);
Anton's install and the N-1/N-2 fixtures; idempotency by bytes; seats, engines and
deliveries preserved; the read-seam wiring (``config.normalize_settings_raw``); the
snapshot written once and the owner told once (``server_maintenance``).
"""

from __future__ import annotations

import copy
import json
import pathlib
from collections import Counter

import pytest

import ouroboros.configured_subagents as cs
from ouroboros import config as cfg
from ouroboros import review_pool_migration as m
from ouroboros import server_maintenance
from ouroboros.settings_defaults import (
    OPENROUTER_REVIEW_DEFAULTS,
    RETIRED_COMMA_LIST_SETTING_KEYS,
    RETIRED_SETTING_KEYS,
    RETIRED_SETTING_SUCCESSORS,
    REVIEW_POOL_MIGRATED_SETTING_KEYS,
    SETTINGS_DEFAULTS,
)
from supervisor import message_bus, state

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "nminus1"
N1_DOC = json.loads((FIXTURES / "settings_v6.113.4.json").read_text(encoding="utf-8"))
N2_DOC = json.loads((FIXTURES / "settings_v6.87.5.json").read_text(encoding="utf-8"))

SLOTS, SUBAGENTS = "OUROBOROS_REVIEWER_SLOTS", "OUROBOROS_SUBAGENTS"
LANE_RECOMMENDATION = "Minted from the former review lane"


@pytest.fixture(autouse=True)
def _pool_ceiling(monkeypatch):
    """Package A raises the catalog ceiling to 26 for the pool; until it lands, this tree
    caps at 10. The migration reads the live constant, so the tests set the pool value."""
    monkeypatch.setattr(cs, "MAX_CONFIGURED_SUBAGENTS", 26)
    m._MIGRATIONS_SEEN.clear()
    cfg._RETIREMENT_NOTICE_SEEN.clear()
    yield
    m._MIGRATIONS_SEEN.clear()
    cfg._RETIREMENT_NOTICE_SEEN.clear()


# --- document builders -------------------------------------------------------


def session_row(subagent_id, target, effort="", access="full", **extra):
    row = {"subagent_id": subagent_id, "recommended_use": f"helper {subagent_id}",
           "route": {"kind": "agent_session", "target_id": target, "credential_profile_id": ""}}
    if effort:
        row["effort"] = effort
    row["access"] = access
    row.update(extra)
    return row


def api_row(subagent_id, target, effort="", **extra):
    row = {"subagent_id": subagent_id, "recommended_use": f"helper {subagent_id}",
           "route": {"kind": "api_model", "target_id": target}}
    if effort:
        row["effort"] = effort
    row.update(extra)
    return row


def catalog(*rows, enabled=True):
    return json.dumps({"enabled": enabled, "items": list(rows)})


def direct(slot_id, target, effort="", delivery=None, kind="api_chat", **extra):
    row = {"slot_id": slot_id, "route": {"kind": kind, "target_id": target}}
    if effort:
        row["effort"] = effort
    if delivery is not None:
        row["delivery"] = delivery
    row.update(extra)
    return row


def ref(slot_id, subagent_id, effort=""):
    row = {"slot_id": slot_id, "subagent_id": subagent_id}
    if effort:
        row["effort"] = effort
    return row


def lanes(triad=(), scope=(), advisory=None, deep_review=None):
    payload = {"triad": list(triad), "scope": list(scope)}
    if advisory is not None:
        payload["advisory"] = advisory
    if deep_review is not None:
        payload["deep_review"] = deep_review
    return json.dumps(payload)


# Anton's install, structurally (contract §2 "Установка Антона", 2026-10-07): a 9-row
# catalog, all enabled; three referenced triad seats, one referenced scope seat, an
# advisory reference, a direct deep-review row; surface keys that acted on no seat.
ANTON_CATALOG_ROWS = [
    session_row("subagent_osabav", "claude=claude-fable-5-1", "xhigh"),
    session_row("subagent_k0ofra", "cursor=grok-4.7-xhigh-fast"),
    session_row("subagent_sa5g1l", "codex=gpt-6-astra", "ultra"),
    session_row("subagent_a1", "codex=gpt-5.6-sol", "high"),
    api_row("subagent_a2", "openai/gpt-5.6-terra", "medium"),
    api_row("subagent_a3", "anthropic/claude-opus-5", "high", processing_preference="economy"),
    session_row("subagent_a4", "claude=claude-sonnet-5", "medium", access="workspace_write"),
    api_row("subagent_a5", "google/gemini-3.8-flash"),
    session_row("subagent_a6", "cursor=gpt-5.6-sol-high"),
]
ANTON_LANES = lanes(
    triad=[ref("triad_286lhb", "subagent_osabav", "xhigh"), ref("triad_w45a8z", "subagent_k0ofra"),
           ref("triad_bkydwq", "subagent_sa5g1l", "xhigh")],
    scope=[ref("scope_slot_1", "subagent_sa5g1l", "xhigh")],
    advisory={"enabled": True, "subagent_id": "subagent_k0ofra"},
    deep_review={"route": {"kind": "api_chat", "target_id": "claudexor::codex=gpt-6-astra"},
                 "effort": "xhigh", "processing_preference": "economy"},
)


def anton_document():
    return {
        SUBAGENTS: catalog(*copy.deepcopy(ANTON_CATALOG_ROWS)),
        SLOTS: ANTON_LANES,
        "OUROBOROS_EFFORT_REVIEW": "medium",
        "OUROBOROS_EFFORT_SCOPE_REVIEW": "medium",
        "OUROBOROS_EFFORT_DEEP_SELF_REVIEW": "high",
        "OUROBOROS_MODEL_DEEP_SELF_REVIEW": "openai/gpt-5.6-sol-pro",
        "OUROBOROS_MODEL": "openai/gpt-5.6-sol",
        "OPENROUTER_API_KEY": "present",
        "TOTAL_BUDGET": 10.0,
    }


def _seat(slot_id, kind, target, effort, source, delivery, **extra):
    payload = {"slot_id": slot_id, "subagent_id": extra.pop("subagent_id", ""), "kind": kind,
               "target_id": target, "effort": effort, "effort_source": source, "delivery": delivery,
               "credential_profile_id": "", "processing_preference": extra.pop("processing_preference", "")}
    if kind == "agent_session":
        payload["access"] = extra.pop("access", "full")
    payload.update(extra)
    return payload


# What the base readers (reviewer_slot_config + row_effort) resolved Anton's lanes to,
# written out by hand: the frozen copies must agree, field by field.
ANTON_EXPECTED_EXECUTIONS = {
    "triad": [
        _seat("triad_286lhb", "agent_session", "claude=claude-fable-5-1", "xhigh", "row", "session",
              subagent_id="subagent_osabav"),
        _seat("triad_w45a8z", "agent_session", "cursor=grok-4.7-xhigh-fast", "xhigh", "compound", "session",
              subagent_id="subagent_k0ofra"),
        _seat("triad_bkydwq", "agent_session", "codex=gpt-6-astra", "xhigh", "row", "session",
              subagent_id="subagent_sa5g1l"),
    ],
    "scope": [
        _seat("scope_slot_1", "agent_session", "codex=gpt-6-astra", "xhigh", "row", "session",
              subagent_id="subagent_sa5g1l"),
    ],
    "advisory": {"slot_id": "advisory_slot_1", "subagent_id": "subagent_k0ofra", "enabled": True},
    "deep_review": _seat("deep_review_slot_1", "api_chat", "claudexor::codex=gpt-6-astra", "xhigh", "row", "native",
                         processing_preference="economy"),
}

# The N-1 fixture (6.113.4, no provider keys, lanes "" and catalog ""): the shipped
# OpenRouter panel at the document's "high" — the factory rows themselves (one factory
# source: no scope seat beside them), plus the deep row the legacy key synthesized.
N1_EXPECTED_EXECUTIONS = {
    "triad": [
        _seat("slot_1", "api_chat", "google/gemini-3.8-flash", "high", "document", "native", authored=False),
        _seat("slot_2", "api_chat", "openai/gpt-5.6-terra", "high", "document", "native", authored=False),
        _seat("slot_3", "api_chat", "anthropic/claude-opus-5", "high", "document", "native", authored=False),
    ],
    "scope": [],
    "advisory": _seat("advisory_slot_1", "api_chat", "", "low", "row", "native", authored=False, enabled=True),
    "deep_review": _seat("deep_review_slot_1", "api_chat", "openai/gpt-5.6-sol-pro", "high", "document", "native"),
}


def _executions(document):
    raw = document.get(SLOTS)
    authored = isinstance(raw, str) and bool(raw.strip())
    parsed = m.parse_reviewer_slots(document, raw) if authored else m.factory_lanes(document)
    return m._executions_dict(m.effective_executions(document, parsed, authored=authored))


def _migrated(document):
    outcome = m.migrate_review_lanes(dict(document))
    assert outcome is not None and not outcome.error, outcome
    return outcome, json.loads(outcome.catalog_after)


def _marked(items):
    return [row["subagent_id"] for row in items if row.get("review_eligible")]


# --- 1. the frozen readers --------------------------------------------------------


def test_frozen_readers_resolve_antons_document_like_the_base():
    assert _executions(anton_document()) == ANTON_EXPECTED_EXECUTIONS


def test_frozen_readers_resolve_the_nminus1_fixture_like_the_base():
    assert _executions(N1_DOC) == N1_EXPECTED_EXECUTIONS
    assert list(OPENROUTER_REVIEW_DEFAULTS["triad"]) == [s["target_id"] for s in N1_EXPECTED_EXECUTIONS["triad"]]


def test_frozen_readers_reject_what_the_base_rejected():
    for bad in ('{"triad": [{"model": "x/y"}]}', '{"triad": [], "scope": []}', "{broken",
                lanes(triad=[direct("t", "x/y", delivery="packet")], scope=[direct("s", "x/y", delivery="native")]),
                lanes(triad=[ref("t", "ghost")], scope=[])):
        with pytest.raises(ValueError):
            m.parse_reviewer_slots({}, bad)
    with pytest.raises(ValueError, match="JSON string"):
        m.parse_reviewer_slots({}, {"triad": []})


def test_the_review_seat_recommendation_is_the_presets_sentence():
    presets = pytest.importorskip("ouroboros.subscription_install_presets")
    expected = getattr(presets, "_REVIEW_SEAT_RECOMMENDATION", None)
    if expected is None:
        pytest.skip("package A moved the sentence; the frozen copy stands on its own")
    assert m.REVIEW_SEAT_RECOMMENDATION == expected


# --- 2. Anton's install, N-1, N-2 ---------------------------------------------------


def test_antons_install_migrates_to_eleven_rows_three_marked():
    doc = anton_document()
    outcome, after = _migrated(doc)
    assert (outcome.catalog_state, outcome.slots_state) == ("configured", "mixed")
    assert after["enabled"] is True and len(after["items"]) == 11
    assert _marked(after["items"]) == ["subagent_osabav", "subagent_k0ofra", "review-1"]
    unchanged = {row["subagent_id"]: row for row in ANTON_CATALOG_ROWS}
    for row in after["items"][:9]:
        original = unchanged[row["subagent_id"]]
        assert {k: v for k, v in row.items() if k != "review_eligible"} == original, "rows are never rewritten"
    assert after["items"][2]["effort"] == "ultra" and "review_eligible" not in after["items"][2]
    assert after["items"][9] == {
        "subagent_id": "review-1", "recommended_use": LANE_RECOMMENDATION,
        "route": {"kind": "agent_session", "target_id": "codex=gpt-6-astra", "credential_profile_id": ""},
        "effort": "xhigh", "access": "full", "review_eligible": True, "minted_from": "review_lane"}
    assert after["items"][10] == {
        "subagent_id": "review-2", "recommended_use": LANE_RECOMMENDATION,
        "route": {"kind": "api_model", "target_id": "claudexor::codex=gpt-6-astra", "credential_profile_id": ""},
        "effort": "xhigh", "processing_preference": "economy", "minted_from": "review_lane"}
    assert outcome.snapshot["summary"] == {"seats_before": 4, "rows_marked_after": 3, "distinct_models": 3,
                                           "helper_rows_minted": 1}
    assert outcome.snapshot["not_in_effect"] == [
        "OUROBOROS_EFFORT_REVIEW=medium", "OUROBOROS_EFFORT_SCOPE_REVIEW=medium",
        "OUROBOROS_EFFORT_DEEP_SELF_REVIEW=high", "OUROBOROS_MODEL_DEEP_SELF_REVIEW=openai/gpt-5.6-sol-pro"]
    assert outcome.consumed_keys == REVIEW_POOL_MIGRATED_SETTING_KEYS and outcome.retained_keys == ()
    rows = {entry["subagent_id"]: entry for entry in outcome.snapshot["rows"]}
    assert rows["review-1"]["from_seats"] == ["triad_bkydwq", "scope_slot_1"] and "merged" in rows["review-1"]["note"]
    assert rows["subagent_k0ofra"]["from_seats"] == ["triad_w45a8z", "advisory_slot_1"]
    text = m.owner_message(outcome, "state/review_migrations/x.json")
    assert text.startswith("⚙️ Review settings migrated.")
    assert "Before: 3 triad seats + 1 scope seat (+ advisory reference, deep review row). After: 3 reviewer rows, 3 distinct models." in text
    assert "• subagent_osabav (claude=claude-fable-5-1, xhigh) — marked" in text
    assert "• review-2 (api claudexor::codex=gpt-6-astra, xhigh, economy) — new row without the mark" in text
    assert "Not in effect before, retired: OUROBOROS_EFFORT_REVIEW=medium" in text
    assert text.rstrip().endswith(f"Snapshot: state/review_migrations/x.json. {m.ROLLBACK_SENTENCE} Adjust in Settings → Agents.")


@pytest.mark.parametrize("document", [N1_DOC, {"OPENROUTER_API_KEY": "present"}, None], ids=["n-1", "never-configured", "anton"])
def test_the_owner_message_names_the_snapshot_and_the_rollback_its_before_really_is(document):
    """VD3-07: every migration message names its snapshot path and the rollback — restore
    OUROBOROS_SUBAGENTS and OUROBOROS_REVIEWER_SLOTS from the snapshot's ``before`` — and that
    ``before`` really is the rollback source: written back over the migrated document (null =
    remove the key) it yields the pre-migration inputs, which migrate to the same catalog again."""
    document = anton_document() if document is None else dict(document)
    outcome = m.migrate_review_lanes(document)
    assert outcome is not None and not outcome.error and not outcome.noop
    text = m.owner_message(outcome, "state/review_migrations/20261008T000000Z-slots-to-pool.json")
    assert "Snapshot: state/review_migrations/20261008T000000Z-slots-to-pool.json. " + m.ROLLBACK_SENTENCE in text
    assert "restore OUROBOROS_SUBAGENTS and OUROBOROS_REVIEWER_SLOTS" in text and "`before`" in text
    assert "no rollback source" in m.owner_message(outcome, "")

    migrated = cfg.normalize_settings_raw(dict(document))
    assert SLOTS not in migrated and migrated[SUBAGENTS]
    before = outcome.snapshot["before"]
    rolled_back = {k: v for k, v in migrated.items() if not str(k).startswith("_")}
    for key in (SUBAGENTS, SLOTS, "OUROBOROS_EFFORT_REVIEW", "OUROBOROS_EFFORT_SCOPE_REVIEW",
                "OUROBOROS_EFFORT_DEEP_SELF_REVIEW", "OUROBOROS_MODEL_DEEP_SELF_REVIEW"):
        if key in before and before[key] is not None:
            rolled_back[key] = before[key]
        else:
            rolled_back.pop(key, None)
    again = m.migrate_review_lanes(rolled_back)
    assert again is not None and again.trigger == outcome.trigger and again.catalog_after == outcome.catalog_after
    assert again.snapshot["before"] == before


def test_the_nminus1_fixture_migrates_to_the_contract_catalog():
    outcome, after = _migrated(N1_DOC)
    assert (outcome.catalog_state, outcome.slots_state) == ("absent", "absent")
    factory = m.REVIEW_SEAT_RECOMMENDATION
    assert after == {"enabled": True, "items": [
        {"subagent_id": "review-1", "recommended_use": factory,
         "route": {"kind": "api_model", "target_id": "google/gemini-3.8-flash"},
         "effort": "high", "review_eligible": True, "minted_from": "factory_default"},
        {"subagent_id": "review-2", "recommended_use": factory,
         "route": {"kind": "api_model", "target_id": "openai/gpt-5.6-terra"},
         "effort": "high", "review_eligible": True, "minted_from": "factory_default"},
        {"subagent_id": "review-3", "recommended_use": factory,
         "route": {"kind": "api_model", "target_id": "anthropic/claude-opus-5"},
         "effort": "high", "review_eligible": True, "minted_from": "factory_default"},
        {"subagent_id": "review-4", "recommended_use": LANE_RECOMMENDATION,
         "route": {"kind": "api_model", "target_id": "openai/gpt-5.6-sol-pro"},
         "effort": "high", "minted_from": "review_lane"},
    ]}
    assert outcome.snapshot["summary"] == {"seats_before": 3, "rows_marked_after": 3, "distinct_models": 3,
                                           "helper_rows_minted": 1}
    rows = {entry["subagent_id"]: entry for entry in outcome.snapshot["rows"]}
    assert [rows[f"review-{n}"]["from_seats"] for n in (1, 2, 3)] == [["slot_1"], ["slot_2"], ["slot_3"]]
    assert outcome.snapshot["not_in_effect"] == []
    message = m.owner_message(outcome, "x")
    assert "shipped default review lanes" in message and "Before: the shipped default panel (3 seats)" in message
    # The retired comma keys never entered the panel (ABI-10) and are not read.
    assert set(N1_DOC) & set(RETIRED_COMMA_LIST_SETTING_KEYS)
    stripped = {k: v for k, v in N1_DOC.items() if k not in RETIRED_COMMA_LIST_SETTING_KEYS}
    assert m.migrate_review_lanes(stripped).catalog_after == outcome.catalog_after


def test_the_nminus2_document_direct_equals_sequential():
    """A pre-6.90 document (no lanes key, retired comma keys) migrates straight to the pool
    exactly as it would have through the 6.90+ ``""`` lanes era: one hop == two hops."""
    assert SLOTS not in N2_DOC and SUBAGENTS not in N2_DOC
    assert set(N2_DOC) & set(RETIRED_COMMA_LIST_SETTING_KEYS)
    one_hop = cfg.normalize_settings_raw(dict(N2_DOC))
    via_lanes = {k: v for k, v in N2_DOC.items() if k not in RETIRED_COMMA_LIST_SETTING_KEYS}
    via_lanes[SLOTS] = ""
    two_hops = cfg.normalize_settings_raw(via_lanes)
    assert one_hop[SUBAGENTS] == two_hops[SUBAGENTS]
    items = json.loads(one_hop[SUBAGENTS])["items"]
    assert _marked(items) == ["review-1", "review-2", "review-3"]
    assert items[3]["route"]["target_id"] == "openai/gpt-5.6-sol-pro" and "review_eligible" not in items[3]
    for key in REVIEW_POOL_MIGRATED_SETTING_KEYS + RETIRED_COMMA_LIST_SETTING_KEYS:
        assert key not in one_hop and key not in two_hops
    assert cfg.normalize_settings_raw(dict(one_hop)) == one_hop


@pytest.mark.parametrize("document", [anton_document(), N1_DOC, N2_DOC], ids=["anton", "n-1", "n-2"])
def test_the_migration_is_idempotent_by_bytes(document):
    once = cfg.normalize_settings_raw(dict(document))
    twice = cfg.normalize_settings_raw(dict(once))
    assert json.dumps(once, sort_keys=True) == json.dumps(twice, sort_keys=True)
    assert m.migrate_review_lanes(once) is None, "a pool document carries nothing to migrate"
    assert not m.migration_applies(once)


@pytest.mark.parametrize("document", [anton_document(), N1_DOC, N2_DOC], ids=["anton", "n-1", "n-2"])
def test_seats_engines_and_deliveries_are_preserved(document):
    """Every engine (with its delivery) a lane seat ran is an engine of a marked row after
    M, AS MANY TIMES as triad seats ran it; the only fold is the declared scope merge, so
    the marked rows are the triad engines (a multiset) plus the scope engines not among them."""
    if SLOTS not in document:
        document = {**{k: v for k, v in document.items() if k not in RETIRED_COMMA_LIST_SETTING_KEYS}, SLOTS: ""}
    executions = _executions(document)
    outcome, after = _migrated(document)

    def seat_engine(seat):
        return (seat["kind"], seat["target_id"], seat["credential_profile_id"], seat["effort"],
                seat["processing_preference"], seat.get("access", ""), seat["delivery"])

    def row_engine(row):
        route = row["route"]
        kind = "agent_session" if route["kind"] == "agent_session" else "api_chat"
        delivery = "session" if kind == "agent_session" else (row.get("delivery") or "native")
        effort = row.get("effort") or m.compound_session_effort(m._row_route(row))
        processing = m.resolve_processing_preference("", override=row.get("processing_preference") or None,
                                                    settings=dict(document))
        return (kind, route["target_id"], route.get("credential_profile_id", ""), effort, processing,
                row.get("access", "full") if kind == "agent_session" else "", delivery)

    triad = Counter(seat_engine(s) for s in executions["triad"])
    before = triad + Counter({seat_engine(s) for s in executions["scope"]} - set(triad))
    marked = Counter(row_engine(r) for r in after["items"] if r.get("review_eligible"))
    assert before == marked
    assert outcome.snapshot["summary"]["seats_before"] == len(executions["triad"]) + len(executions["scope"])
    assert outcome.snapshot["summary"]["rows_marked_after"] == sum(marked.values())
    for row in after["items"]:
        if row.get("review_eligible") or row.get("minted_from"):
            assert row.get("effort") or m.compound_session_effort(m._row_route(row)), "no seat row leaves without effort"


# --- 3. F4-F8 -------------------------------------------------------------------------


def test_f4_slot_ids_unique_subagent_ids_repeat_seat_effort_above_row_effort():
    doc = {SUBAGENTS: catalog(api_row("helper", "openai/gpt-5.6-terra", "medium")),
           SLOTS: lanes(triad=[ref("r1", "helper"), ref("r2", "helper", "xhigh")], scope=[ref("s1", "helper")])}
    outcome, after = _migrated(doc)
    assert after["items"] == [
        {**api_row("helper", "openai/gpt-5.6-terra", "medium"), "review_eligible": True},
        {"subagent_id": "review-1", "recommended_use": LANE_RECOMMENDATION,
         "route": {"kind": "api_model", "target_id": "openai/gpt-5.6-terra"}, "effort": "xhigh",
         "review_eligible": True, "minted_from": "review_lane"}]
    rows = {entry["subagent_id"]: entry for entry in outcome.snapshot["rows"]}
    assert rows["helper"]["action"] == "marked" and rows["helper"]["from_seats"] == ["r1", "s1"]
    assert "merged" in rows["helper"]["note"]
    assert rows["review-1"]["from_seats"] == ["r2"] and "helper keeps effort medium" in rows["review-1"]["note"]
    assert outcome.snapshot["summary"] == {"seats_before": 3, "rows_marked_after": 2, "distinct_models": 1,
                                           "helper_rows_minted": 0}


def one_harness_wizard_document(override=""):
    """What the previous wizard wrote for a sole connected harness (Claude): the task actor
    row, the advisory helper it minted, and three triad references to the ONE actor row
    (``_compile_policy_seats`` repeated the harness three times), the scope and deep review
    referencing it too. ``override`` puts the same effort override on every triad seat."""
    rows = [session_row("claude-code", "claude=claude-opus-5", "medium", recommended_use="Claude Code actor"),
            session_row("review-claude", "claude=claude-sonnet-5", "low")]
    return {SUBAGENTS: catalog(*rows), "OUROBOROS_MODEL": "openai/gpt-5.6-sol", "OPENROUTER_API_KEY": "present",
            SLOTS: lanes(triad=[ref(f"slot_{n}", "claude-code", override) for n in (1, 2, 3)],
                         scope=[ref("scope_slot_1", "claude-code")],
                         advisory={"enabled": True, "subagent_id": "review-claude"},
                         deep_review={"subagent_id": "claude-code"})}


def test_m2_three_references_of_the_one_harness_wizard_stay_three_runs():
    from ouroboros import reviewer_slot_config as rs
    from ouroboros.review_model_routes import adaptive_quorum

    doc = one_harness_wizard_document()
    outcome, after = _migrated(doc)
    assert _marked(after["items"]) == ["claude-code", "review-1", "review-2"]
    actor = session_row("claude-code", "claude=claude-opus-5", "medium", recommended_use="Claude Code actor")
    assert after["items"][0] == {**actor, "review_eligible": True}, "the source row is marked once, otherwise untouched"
    assert after["items"][1] == session_row("review-claude", "claude=claude-sonnet-5", "low"), "the helper stays a helper"
    twin = {"subagent_id": "", "recommended_use": LANE_RECOMMENDATION,
            "route": {"kind": "agent_session", "target_id": "claude=claude-opus-5", "credential_profile_id": ""},
            "effort": "medium", "access": "full", "review_eligible": True, "minted_from": "review_lane"}
    assert after["items"][2:] == [{**twin, "subagent_id": "review-1"}, {**twin, "subagent_id": "review-2"}]
    rows = {entry["subagent_id"]: entry for entry in outcome.snapshot["rows"]}
    assert rows["claude-code"]["from_seats"] == ["slot_1", "scope_slot_1"], "the scope seat still merges"
    assert rows["review-1"]["from_seats"] == ["slot_2"] and rows["review-2"]["from_seats"] == ["slot_3"]
    assert "also referenced claude-code" in rows["review-1"]["note"] and "twin" in rows["review-2"]["note"]
    assert outcome.snapshot["summary"] == {"seats_before": 4, "rows_marked_after": 3, "distinct_models": 1,
                                           "helper_rows_minted": 0}
    loaded = cfg.normalize_settings_raw(dict(doc))
    pool = rs.review_pool_rows(loaded)
    assert [row.target_id for row in pool] == ["claude=claude-opus-5"] * 3 and adaptive_quorum(len(pool)) == 2


def test_m2_repeated_identical_effort_overrides_mint_one_row_per_seat():
    doc = one_harness_wizard_document(override="xhigh")
    outcome, after = _migrated(doc)
    assert _marked(after["items"]) == ["claude-code", "review-1", "review-2", "review-3"]
    assert after["items"][0]["effort"] == "medium", "the actor keeps its own medium (marked by the scope seat alone)"
    assert all(r["effort"] == "xhigh" and r["route"]["target_id"] == "claude=claude-opus-5" for r in after["items"][2:])
    rows = {entry["subagent_id"]: entry for entry in outcome.snapshot["rows"]}
    assert [rows[f"review-{n}"]["from_seats"] for n in (1, 2, 3)] == [["slot_1"], ["slot_2"], ["slot_3"]]
    assert rows["claude-code"]["action"] == "marked" and rows["claude-code"]["from_seats"] == ["scope_slot_1"]
    assert outcome.snapshot["summary"]["rows_marked_after"] == 4, "three overridden runs plus the scope seat's own row"


@pytest.mark.parametrize("enabled", [False, True], ids=["disabled-helper", "enabled-helper"])
def test_m3_a_disabled_row_never_takes_the_mark_the_seat_mints_an_enabled_row(enabled):
    """A direct seat whose engine coincides with a helper row: the helper is marked only when
    it is ON. A helper switched off stays as it was and the seat mints its own enabled row —
    the runtime pool (enabled marked rows) is never emptied by the migration."""
    from ouroboros import reviewer_slot_config as rs

    helper = api_row("helper", "openai/gpt-5.6-terra", "high", **({} if enabled else {"enabled": False}))
    doc = {SUBAGENTS: catalog(dict(helper)), "OUROBOROS_MODEL": "openai/gpt-5.6-sol", "OPENROUTER_API_KEY": "present",
           SLOTS: lanes(triad=[direct("t", "openai/gpt-5.6-terra", "high", delivery="native")],
                        scope=[direct("s", "openai/gpt-5.6-terra", "high")])}
    outcome, after = _migrated(doc)
    rows = {entry["subagent_id"]: entry for entry in outcome.snapshot["rows"]}
    if enabled:
        assert after["items"] == [{**helper, "review_eligible": True}] and rows["helper"]["from_seats"] == ["t", "s"]
    else:
        assert after["items"][0] == helper, "the disabled helper is untouched: no mark, still off"
        assert _marked(after["items"]) == ["review-1"] and after["items"][1]["minted_from"] == "review_lane"
        assert "enabled" not in after["items"][1] and rows["review-1"]["from_seats"] == ["t", "s"]
    pool = rs.review_pool_rows(cfg.normalize_settings_raw(dict(doc)))
    assert [(row.target_id, row.effort) for row in pool] == [("openai/gpt-5.6-terra", "high")]
    assert outcome.snapshot["summary"]["rows_marked_after"] == 1


def test_m3_a_disabled_row_of_a_factory_engine_does_not_claim_the_factory_seat():
    from ouroboros import reviewer_slot_config as rs

    off = api_row("helper", OPENROUTER_REVIEW_DEFAULTS["triad"][1], "high", enabled=False)
    doc = {SUBAGENTS: catalog(off), "OUROBOROS_MODEL": "openai/gpt-5.6-sol", "OPENROUTER_API_KEY": "present", SLOTS: ""}
    _outcome, after = _migrated(doc)
    assert after["items"][0] == off and _marked(after["items"]) == ["review-1", "review-2", "review-3"]
    pool = rs.review_pool_rows(cfg.normalize_settings_raw(dict(doc)))
    assert [row.target_id for row in pool] == list(OPENROUTER_REVIEW_DEFAULTS["triad"])


def test_f5a_the_pool_ceiling_is_the_catalog_ceiling():
    rows = [api_row(f"h{i}", f"vendor/model-{i}", "low") for i in range(10)]
    triad = [direct(f"t{i}", f"vendor/triad-{i}", "high", delivery="native") for i in range(10)]
    scope = [direct(f"s{i}", f"vendor/scope-{i}", "high") for i in range(4)]
    doc = {SUBAGENTS: catalog(*rows),
           SLOTS: lanes(triad=triad, scope=scope,
                        advisory={"enabled": True, "route": {"kind": "api_chat", "target_id": "vendor/adv"}, "effort": "low"},
                        deep_review={"route": {"kind": "api_chat", "target_id": "vendor/deep"}, "effort": "high"})}
    outcome, after = _migrated(doc)
    assert len(after["items"]) == 26 and len(_marked(after["items"])) == 14
    assert [r["subagent_id"] for r in after["items"][10:]] == [f"review-{i}" for i in range(1, 17)]
    # A 27th row refuses the WHOLE migration: no partial pool, the lanes stay.
    doc27 = dict(doc, **{SUBAGENTS: catalog(*rows, api_row("h10", "vendor/model-10", "low"))})
    refused = m.migrate_review_lanes(doc27)
    assert refused.error and "27 rows" in refused.error and "26" in refused.error
    assert refused.catalog_after is None and refused.retained_keys == (SLOTS,)
    loaded = dict(doc27)
    m.apply_outcome(loaded, refused)
    assert loaded == doc27


def test_f5b_twins_of_one_engine_become_two_marked_rows():
    doc = {SLOTS: lanes(triad=[direct("a", "openai/gpt-5.6-sol", delivery="native"),
                               direct("b", "openai/gpt-5.6-sol", delivery="native")],
                        scope=[direct("s", "openai/gpt-5.6-sol")])}
    outcome, after = _migrated(doc)
    assert _marked(after["items"]) == ["review-1", "review-2"] and len(after["items"]) == 2
    assert after["items"][0]["route"] == after["items"][1]["route"] == {"kind": "api_model", "target_id": "openai/gpt-5.6-sol"}
    assert all(r["effort"] == "high" and "delivery" not in r for r in after["items"])
    assert outcome.snapshot["summary"]["distinct_models"] == 1
    rows = {entry["subagent_id"]: entry for entry in outcome.snapshot["rows"]}
    assert rows["review-1"]["from_seats"] == ["a", "s"], "the scope twin merges into the first produced row"
    assert rows["review-2"]["from_seats"] == ["b"]


def test_f5c_a_direct_advisory_coinciding_with_a_direct_triad_seat_keeps_its_own_helper_row():
    doc = {SLOTS: lanes(triad=[direct("t", "anthropic/claude-sonnet-5", "low", delivery="native")],
                        scope=[direct("s", "openai/gpt-5.6-terra", "high")],
                        advisory={"enabled": True, "route": {"kind": "api_chat", "target_id": "anthropic/claude-sonnet-5"},
                                  "effort": "low"})}
    outcome, after = _migrated(doc)
    assert _marked(after["items"]) == ["review-1", "review-2"]
    helper = after["items"][2]
    assert helper["subagent_id"] == "review-3" and "review_eligible" not in helper
    assert helper["minted_from"] == "review_lane" and helper["recommended_use"] == LANE_RECOMMENDATION
    assert helper["route"] == after["items"][0]["route"] and helper["effort"] == "low"
    note = {e["subagent_id"]: e for e in outcome.snapshot["rows"]}["review-3"]["note"]
    assert "coincided with triad seat t" in note and "separate helper row review-3 (no mark)" in note
    assert "preflight is now chosen per commit" in note


def _save_judgement(loaded, items):
    """What the Settings save path says about re-posting ``items`` over the migrated document."""
    posted = json.dumps({"enabled": True, "items": items})
    return cs.roster_save_error(posted, loaded, {SUBAGENTS: posted})


@pytest.mark.parametrize("helper_lane", ["advisory", "deep_review"])
def test_m4_the_reviewer_helper_pair_the_migration_minted_survives_a_description_edit(helper_lane):
    """F5(c) end to end: a direct triad seat and a direct advisory (or deep-review) seat of one
    engine become a marked reviewer and an unmarked helper, both minted. The owner editing any
    row's description must save without deleting a row or changing an engine."""
    seat = {"route": {"kind": "api_chat", "target_id": "anthropic/claude-sonnet-5"}, "effort": "low"}
    extra = {"advisory": {"enabled": True, **seat}} if helper_lane == "advisory" else {"deep_review": seat}
    doc = {"OUROBOROS_MODEL": "openai/gpt-5.6-sol", "OPENROUTER_API_KEY": "present",
           SLOTS: lanes(triad=[direct("t", "anthropic/claude-sonnet-5", "low", delivery="native")],
                        scope=[direct("s", "openai/gpt-5.6-terra", "high")], **extra)}
    loaded = cfg.normalize_settings_raw(dict(doc))
    items = json.loads(loaded[SUBAGENTS])["items"]
    assert [(r["subagent_id"], r.get("review_eligible", False), r["minted_from"]) for r in items] == [
        ("review-1", True, "review_lane"), ("review-2", True, "review_lane"), ("review-3", False, "review_lane")]
    assert items[2]["route"] == items[0]["route"] and items[2]["effort"] == items[0]["effort"] == "low"
    assert _save_judgement(loaded, items) == "", "re-posting the migrated catalog unchanged is accepted"
    for index in range(3):
        edited = copy.deepcopy(items)
        edited[index]["recommended_use"] = "Owner's wording"
        assert _save_judgement(loaded, edited) == "", f"a description edit of items[{index}] is an ordinary save"
    # The ordinary refusal stands: two rows the owner authored on one engine.
    owner_twin = {k: v for k, v in items[2].items() if k != "minted_from"}
    owned = [{k: v for k, v in items[0].items() if k not in ("minted_from", "review_eligible")}, items[1], owner_twin]
    assert "runs the same engine" in _save_judgement(loaded, owned)


def test_f7_an_empty_scope_effort_materializes_before_the_key_is_retired():
    doc = {"OUROBOROS_EFFORT_REVIEW": "high", "OUROBOROS_EFFORT_SCOPE_REVIEW": "xhigh",
           SLOTS: lanes(triad=[direct("t", "openai/gpt-5.6-terra")], scope=[direct("s", "openai/gpt-5.6-terra")])}
    outcome, after = _migrated(doc)
    assert after["items"] == [
        {"subagent_id": "review-1", "recommended_use": LANE_RECOMMENDATION,
         "route": {"kind": "api_model", "target_id": "openai/gpt-5.6-terra"}, "effort": "high",
         "review_eligible": True, "delivery": "packet", "minted_from": "review_lane"},
        {"subagent_id": "review-2", "recommended_use": LANE_RECOMMENDATION,
         "route": {"kind": "api_model", "target_id": "openai/gpt-5.6-terra"}, "effort": "xhigh",
         "review_eligible": True, "minted_from": "review_lane"}]
    assert outcome.snapshot["not_in_effect"] == []
    loaded = cfg.normalize_settings_raw(dict(doc))
    assert "OUROBOROS_EFFORT_SCOPE_REVIEW" not in loaded and json.loads(loaded[SUBAGENTS]) == after


def test_f7b_packet_triad_and_native_scope_of_one_engine_stay_two_rows_at_equal_effort():
    doc = {SLOTS: lanes(triad=[direct("t", "openai/gpt-5.6-terra", "high")],
                        scope=[direct("s", "openai/gpt-5.6-terra", "high")])}
    _outcome, after = _migrated(doc)
    assert [(r["subagent_id"], r.get("delivery")) for r in after["items"]] == [("review-1", "packet"), ("review-2", None)]


def test_f8_delivery_is_serialized_only_as_packet_and_only_on_direct_api_triad_rows():
    doc = {SLOTS: lanes(triad=[direct("p", "a/one", "high", delivery="packet"),
                               direct("n", "b/two", "high", delivery="native"),
                               direct("d", "c/three", "high"),
                               direct("s", "codex=gpt-5.6-sol", "high", kind="agent_session")],
                        scope=[direct("sc", "a/one", "high")])}
    _outcome, after = _migrated(doc)
    by_id = {r["subagent_id"]: r for r in after["items"]}
    assert by_id["review-1"]["delivery"] == "packet"
    assert "delivery" not in by_id["review-2"], "native is the default and is not written"
    assert by_id["review-3"]["delivery"] == "packet", "a direct api triad row without delivery ran as a packet"
    assert "delivery" not in by_id["review-4"] and by_id["review-4"]["route"]["kind"] == "agent_session"
    assert by_id["review-4"]["access"] == "full"
    assert "delivery" not in by_id["review-5"], "the scope seat of a/one (reads) is its own row beside the packet twin"
    assert len(after["items"]) == 5
    assert m.parse_reviewer_slots({}, lanes(triad=[direct("t", "a/one")],
                                            scope=[direct("s", "a/one")])).triad[0].delivery == ""
    with pytest.raises(ValueError, match="delivery"):
        m.parse_reviewer_slots({}, lanes(triad=[direct("t", "a/one")], scope=[direct("s", "a/one", delivery="packet")]))


# --- 4. F6: the catalog x lanes matrix ------------------------------------------------


def _catalog_for(state_id):
    rows = [api_row("helper", "openai/gpt-5.6-terra", "medium"), session_row("coder", "codex=gpt-5.6-sol", "high")]
    return {
        "A": None,
        "E": catalog(),
        "I": '{"enabled": true, "items": [{"subagent_id": "helper"}]}',
        "D": catalog(*rows, enabled=False),
        "C": catalog(*rows),
    }[state_id]


def _lanes_for(state_id):
    return {
        "0": "",
        "P": lanes(triad=[direct("t1", "x/one", "high", delivery="native"), direct("t2", "y/two", "high")],
                   scope=[direct("s1", "x/one", "high")],
                   advisory={"enabled": True, "route": {"kind": "api_chat", "target_id": "z/adv"}, "effort": "low"},
                   deep_review={"route": {"kind": "api_chat", "target_id": "w/deep"}, "effort": "high"}),
        "R": lanes(triad=[ref("t1", "helper"), ref("t2", "coder")], scope=[ref("s1", "helper")],
                   advisory={"enabled": True, "subagent_id": "coder"}),
        "M": lanes(triad=[ref("t1", "helper"), direct("t2", "y/two", "high")], scope=[direct("s1", "x/one", "high")]),
        "X": '{"triad": [{"model": "x/y"}]}',
    }[state_id]


F6_CELLS = [f"{c}{s}" for c in "AEIDC" for s in "0PRMX"]


@pytest.mark.parametrize("cell", F6_CELLS)
def test_f6_catalog_by_lanes_matrix(cell):
    catalog_id, lanes_id = cell
    doc = {"OUROBOROS_MODEL": "openai/gpt-5.6-sol", "OPENROUTER_API_KEY": "present", SLOTS: _lanes_for(lanes_id)}
    stored = _catalog_for(catalog_id)
    if stored is not None:
        doc[SUBAGENTS] = stored
    before = copy.deepcopy(doc)
    outcome = m.migrate_review_lanes(doc)
    assert doc == before, "pure: the input is never mutated"
    assert outcome is not None
    expected_catalog = {"A": "absent", "E": "empty", "I": "invalid", "D": "disabled", "C": "configured"}[catalog_id]
    assert outcome.catalog_state == expected_catalog
    loaded = dict(doc)
    m.apply_outcome(loaded, outcome)

    # The refusal cells: an invalid catalog, invalid lanes, or references that cannot
    # resolve (no catalog, an empty one, a disabled one) — no partial migration, the
    # lane keys stay for the owner's catalog save, the catalog is untouched.
    refused = catalog_id == "I" or lanes_id == "X" or (lanes_id in "RM" and catalog_id in "AED")
    if refused:
        assert outcome.error and outcome.catalog_after is None, outcome
        assert outcome.retained_keys == (SLOTS,) and outcome.consumed_keys == ()
        assert loaded == doc, "nothing rewritten, the lanes key stays"
        assert outcome.slots_state == ("invalid" if lanes_id in "XRM" or catalog_id != "I" else outcome.slots_state)
        assert m.owner_message(outcome, "snap.json").startswith("⚙️ Review settings could not be migrated automatically")
        assert "snap.json" in m.owner_message(outcome, "snap.json")
        return

    assert not outcome.error and outcome.consumed_keys == (SLOTS,) and outcome.retained_keys == ()
    after = json.loads(loaded[SUBAGENTS])
    assert SLOTS not in loaded
    expected_slots = {"0": "absent", "P": "direct", "R": "referenced", "M": "mixed"}[lanes_id]
    assert outcome.slots_state == expected_slots
    # The catalog switch is never touched: a disabled catalog stays disabled (review
    # stays on through the pool; delegation stays off) and the report says so.
    assert after["enabled"] is (catalog_id != "D")
    if catalog_id == "D":
        assert any("review stays on; delegation stays off" in note for note in outcome.snapshot["notes"])
    existing = json.loads(stored)["items"] if stored else []
    assert [r["subagent_id"] for r in after["items"][:len(existing)]] == [r["subagent_id"] for r in existing]
    marked = _marked(after["items"])
    if lanes_id == "0":
        # The shipped default panel ran: factory rows, minted as such, three distinct
        # OpenRouter models with the scope seat merged into the terra row.
        minted = [r for r in after["items"] if r.get("minted_from") == "factory_default"]
        assert [r["route"]["target_id"] for r in minted] == list(OPENROUTER_REVIEW_DEFAULTS["triad"])
        assert marked == [r["subagent_id"] for r in minted] and all(r["effort"] == "high" for r in minted)
        assert all(r["recommended_use"] == m.REVIEW_SEAT_RECOMMENDATION for r in minted)
        assert len(after["items"]) == len(existing) + 3, "no advisory row (unauthored), no deep row (key empty)"
        assert outcome.snapshot["summary"]["distinct_models"] == 3
    elif lanes_id == "P":
        minted = after["items"][len(existing):]
        assert [r["subagent_id"] for r in minted] == ["review-1", "review-2", "review-3", "review-4"]
        assert marked == ["review-1", "review-2"], "advisory and deep helpers carry no mark"
        assert minted[1]["delivery"] == "packet" and "delivery" not in minted[0]
        assert minted[2]["route"]["target_id"] == "z/adv" and minted[3]["route"]["target_id"] == "w/deep"
        assert all(r["minted_from"] == "review_lane" for r in minted)
    elif lanes_id == "R":
        assert marked == ["helper", "coder"] and len(after["items"]) == len(existing)
        assert not any(r.get("minted_from") for r in after["items"])
    else:  # M
        assert marked == ["helper", "review-1", "review-2"]
        assert after["items"][2]["route"]["target_id"] == "y/two" and after["items"][3]["route"]["target_id"] == "x/one"
    assert all(r.get("effort") for r in after["items"] if r.get("minted_from")), "no minted row without effort"


def test_f6_c0_is_the_ordinary_upgrade_of_antons_catalog_without_lanes():
    doc = anton_document()
    doc[SLOTS] = ""
    doc["OUROBOROS_MODEL_DEEP_SELF_REVIEW"] = ""
    for key in ("OUROBOROS_EFFORT_REVIEW", "OUROBOROS_EFFORT_SCOPE_REVIEW"):
        doc.pop(key)  # the shipped "high"
    outcome, after = _migrated(doc)
    assert (outcome.catalog_state, outcome.slots_state) == ("configured", "absent")
    assert len(after["items"]) == 12 and _marked(after["items"]) == ["review-1", "review-2", "review-3"]
    assert [r["route"]["target_id"] for r in after["items"][9:]] == list(OPENROUTER_REVIEW_DEFAULTS["triad"])
    rows = {entry["subagent_id"]: entry for entry in outcome.snapshot["rows"]}
    assert rows["review-2"]["from_seats"] == ["slot_2"]  # one factory source: no scope seat beside the rows
    assert "advisory" in " ".join(outcome.snapshot["notes"]) and outcome.snapshot["summary"]["helper_rows_minted"] == 0
    # The panel at the document's "medium" coincides with an existing terra@medium row:
    # that row is marked (at most once) instead of a new one, the other two are minted.
    doc["OUROBOROS_EFFORT_REVIEW"] = doc["OUROBOROS_EFFORT_SCOPE_REVIEW"] = "medium"
    outcome, after = _migrated(doc)
    assert len(after["items"]) == 11 and _marked(after["items"]) == ["subagent_a2", "review-1", "review-2"]
    assert {e["subagent_id"]: e["from_seats"] for e in outcome.snapshot["rows"]}["subagent_a2"] == ["slot_2"]
    assert outcome.snapshot["not_in_effect"] == [], "the shipped scope reader ran the scope effort: retired, not idle"


@pytest.mark.parametrize("document", [
    {"OUROBOROS_MODEL": "x/y", "OPENROUTER_API_KEY": "present"},
    {"OUROBOROS_MODEL": "x/y", "OPENROUTER_API_KEY": "present", SUBAGENTS: ""},
    {"TOTAL_BUDGET": 1.0},
], ids=["provider-key-only", "blank-catalog-string", "no-provider"])
def test_m1_a_never_configured_document_reads_the_factory_rows_at_the_seam(document):
    """Contract §1.5, the both-absent cell: an install with neither the lanes key nor a
    catalog (Docker / Colab / a mounted volume without the wizard) ran the shipped default
    panel; the read seam mints the factory rows with the mark, and the runtime pool reader
    sees them. The same document read twice is one migration, never a second set of rows."""
    from ouroboros import reviewer_slot_config as rs

    assert m.migration_trigger(document) == m.TRIGGER_NEVER_CONFIGURED
    loaded = cfg.normalize_settings_raw(dict(document))
    items = json.loads(loaded[SUBAGENTS])["items"]
    assert _marked(items) == ["review-1", "review-2", "review-3"]
    assert all(row["minted_from"] == "factory_default" and row["effort"] == "high" for row in items)
    assert [row["route"]["target_id"] for row in items] == list(OPENROUTER_REVIEW_DEFAULTS["triad"])
    pool = rs.review_pool_rows(loaded)
    assert [row.slot_id for row in pool] == ["review-1", "review-2", "review-3"]
    assert rs.review_pool_state(loaded[SUBAGENTS])["state"] == "structured"
    (outcome,) = cfg.review_pool_migrations_seen()
    assert outcome.trigger == m.TRIGGER_NEVER_CONFIGURED and outcome.snapshot["before"]["trigger"] == outcome.trigger
    assert not outcome.error and not outcome.noop
    # Idempotent: the migrated document is a pool document — read again, nothing is re-minted.
    assert m.migrate_review_lanes(dict(loaded)) is None
    assert cfg.normalize_settings_raw(dict(loaded)) == loaded
    assert len(cfg.review_pool_migrations_seen()) == 1
    # The owner hears what RUNS, not a migration story about lanes this install never had.
    text = m.owner_message(outcome, "snap.json")
    assert text.startswith("⚙️ Review pool initialized.") and "3 reviewer rows" in text and "Settings → Agents" in text
    assert "review lanes" not in text.split("\n", 1)[1]


@pytest.mark.parametrize("stored", [
    catalog(),
    catalog(api_row("helper", "x/y", "high")),
    catalog(api_row("helper", "x/y", "high", enabled=False)),
], ids=["items-empty", "unmarked-rows", "disabled-row"])
def test_m1_a_structural_catalog_the_owner_saved_empty_stays_empty(stored):
    """Empty is not never-configured: a catalog saved as a structure (``items: []``, or
    rows the owner left unmarked under ``allow_empty_review_pool``) is the owner's
    document; the seam mints nothing and the pool reports ``empty`` loudly."""
    from ouroboros import reviewer_slot_config as rs

    document = {SUBAGENTS: stored, "OPENROUTER_API_KEY": "present"}
    assert m.migration_trigger(document) == "" and m.migrate_review_lanes(dict(document)) is None
    loaded = cfg.normalize_settings_raw(dict(document))
    assert loaded[SUBAGENTS] == stored
    assert rs.review_pool_rows(loaded) == [] and rs.review_pool_state(stored)["state"] == "empty"
    assert cfg.review_pool_migrations_seen() == ()
    pool = {SUBAGENTS: catalog(api_row("r", "x/y", "high", review_eligible=True, minted_from="factory_default"))}
    assert m.migrate_review_lanes(pool) is None, "a pool document without the lanes key is done"


def test_m1_the_no_settings_file_path_reaches_the_factory_pool(tmp_path, monkeypatch):
    """``load_settings_lock_held`` without a document (``SETTINGS_DEFAULTS`` + env) is the
    other never-configured entry: the env-merged defaults go through the same seam, so a
    container started with provider keys in its environment has a review pool."""
    from ouroboros import reviewer_slot_config as rs

    monkeypatch.setattr(cfg, "SETTINGS_PATH", tmp_path / "absent" / "settings.json")
    monkeypatch.setenv("OPENROUTER_API_KEY", "present")
    settings = cfg.load_settings_lock_held(_settings_lock_held=False)
    pool = rs.review_pool_rows(settings)
    assert [row.target_id for row in pool] == list(OPENROUTER_REVIEW_DEFAULTS["triad"])
    assert all(row["minted_from"] == "factory_default" for row in json.loads(settings[SUBAGENTS])["items"])
    (outcome,) = cfg.review_pool_migrations_seen()
    assert outcome.trigger == m.TRIGGER_NEVER_CONFIGURED
    # Read again: the same digest replays the recorded outcome, one set of rows.
    assert cfg.load_settings_lock_held(_settings_lock_held=False)[SUBAGENTS] == settings[SUBAGENTS]
    assert len(cfg.review_pool_migrations_seen()) == 1


DIRECT_PROVIDER_CELLS = [
    ("OPENAI_API_KEY", "openai", "openai::gpt-5.6-terra"),
    ("ANTHROPIC_API_KEY", "anthropic", "anthropic::claude-opus-5"),
]


def _exactly_the_factory_pool(loaded, document, provider, main):
    """FIX5 M1-direct: the marked rows ARE ``factory_review_rows(document)`` — the provider's
    three direct rows around its default Main — and nothing else; every seat routes to the
    provider the document holds a credential for, none to the credential-less OpenRouter ids."""
    from ouroboros import reviewer_slot_config as rs
    from ouroboros.provider_models import model_has_credentials_in_settings, provider_for_model
    from ouroboros.subscription_install_presets import factory_review_rows

    items = json.loads(loaded[SUBAGENTS])["items"]
    assert items == factory_review_rows(document), items
    assert _marked(items) == ["review-1", "review-2", "review-3"]
    assert [row["route"]["target_id"] for row in items] == [main] * 3
    assert all(row["minted_from"] == "factory_default" and row["effort"] == "high" for row in items)
    pool = rs.review_pool_rows(loaded)
    assert [(row.slot_id, row.target_id) for row in pool] == [(f"review-{n}", main) for n in (1, 2, 3)]
    assert {provider_for_model(row.target_id) for row in pool} == {provider}
    assert all(model_has_credentials_in_settings(row.target_id, dict(document)) for row in pool)
    assert not {row.target_id for row in pool} & set(OPENROUTER_REVIEW_DEFAULTS["triad"])
    assert rs.review_pool_state(loaded[SUBAGENTS])["state"] == "structured"


@pytest.mark.parametrize("catalog_cell", ["absent", "blank-string"])
@pytest.mark.parametrize("key, provider, main", DIRECT_PROVIDER_CELLS, ids=["openai-only", "anthropic-only"])
def test_m1_a_never_configured_direct_provider_document_reads_exactly_its_factory_rows(
        catalog_cell, key, provider, main):
    """FIX5 M1-direct (Astra, Coupling): a document holding ONLY a direct provider's key and no
    Main is read before the defaults merge, so the frozen panel sees no Main and derives the
    OpenRouter triad — which it then placed BESIDE the provider's three factory rows: six marked
    seats, three of them on routes the install has no credential for. The never-configured cell
    owns exactly the canonical factory pool; the frozen seats are the authored-lane migration's
    business, not this install's. Repeated reads are one migration; the document is not written."""
    document = {key: "present"}
    if catalog_cell == "blank-string":
        document[SUBAGENTS] = ""
    assert m.migration_trigger(document) == m.TRIGGER_NEVER_CONFIGURED
    loaded = cfg.normalize_settings_raw(dict(document))
    _exactly_the_factory_pool(loaded, document, provider, main)
    (outcome,) = cfg.review_pool_migrations_seen()
    assert outcome.trigger == m.TRIGGER_NEVER_CONFIGURED and not outcome.error and not outcome.noop
    assert outcome.snapshot["summary"]["rows_marked_after"] == 3 and outcome.snapshot["summary"]["distinct_models"] == 1
    assert document == ({key: "present", SUBAGENTS: ""} if catalog_cell == "blank-string" else {key: "present"})
    # Idempotent: the migrated document is a pool document; a second read re-mints nothing.
    assert m.migrate_review_lanes(dict(loaded)) is None
    assert cfg.normalize_settings_raw(dict(loaded)) == loaded
    assert len(cfg.review_pool_migrations_seen()) == 1
    text = m.owner_message(outcome, "snap.json")
    assert text.startswith("⚙️ Review pool initialized.") and "3 reviewer rows, 1 distinct models" in text


@pytest.mark.parametrize("key, provider, main", DIRECT_PROVIDER_CELLS, ids=["openai-only", "anthropic-only"])
def test_m1_the_no_settings_file_path_of_a_direct_provider_reaches_exactly_its_factory_pool(
        tmp_path, monkeypatch, key, provider, main):
    """The other never-configured entry for a direct provider: no settings file, the key in the
    environment (a container). The env-merged defaults carry the shipped OpenRouter Main, which is
    not on the provider — the factory rows take the provider's default Main, nothing else is
    minted, no file is created, and a second read replays the one outcome."""
    from ouroboros import reviewer_slot_config as rs

    path = tmp_path / "absent" / "settings.json"
    monkeypatch.setattr(cfg, "SETTINGS_PATH", path)
    for other in m._SHA_PRESENCE_KEYS:  # the operator's shell must not add a provider to the cell
        monkeypatch.delenv(other, raising=False)
    monkeypatch.setenv(key, "present")
    settings = cfg.load_settings_lock_held(_settings_lock_held=False)
    assert not path.exists() and not path.parent.exists(), "a read never creates the document"
    pool = rs.review_pool_rows(settings)
    assert [(row.slot_id, row.target_id) for row in pool] == [(f"review-{n}", main) for n in (1, 2, 3)]
    items = json.loads(settings[SUBAGENTS])["items"]
    assert _marked(items) == ["review-1", "review-2", "review-3"]
    assert all(row["minted_from"] == "factory_default" for row in items)
    assert not {row.target_id for row in pool} & set(OPENROUTER_REVIEW_DEFAULTS["triad"])
    (outcome,) = cfg.review_pool_migrations_seen()
    assert outcome.trigger == m.TRIGGER_NEVER_CONFIGURED
    assert cfg.load_settings_lock_held(_settings_lock_held=False)[SUBAGENTS] == settings[SUBAGENTS]
    assert len(cfg.review_pool_migrations_seen()) == 1 and not path.exists()


@pytest.mark.parametrize("key", ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"])
def test_m1_a_direct_provider_structural_empty_catalog_stays_empty(key):
    """The structural-empty rule is not weakened for a direct provider: a catalog the owner
    saved empty beside the provider's key mints nothing and the pool is loudly empty."""
    from ouroboros import reviewer_slot_config as rs

    document = {key: "present", SUBAGENTS: catalog()}
    assert m.migration_trigger(document) == "" and m.migrate_review_lanes(dict(document)) is None
    loaded = cfg.normalize_settings_raw(dict(document))
    assert loaded[SUBAGENTS] == catalog() and rs.review_pool_rows(loaded) == []
    assert rs.review_pool_state(loaded[SUBAGENTS])["state"] == "empty" and cfg.review_pool_migrations_seen() == ()


@pytest.mark.parametrize("document", [
    {"OUROBOROS_MODEL": "x/y", "OPENROUTER_API_KEY": "present"},
    {"OUROBOROS_MODEL": "x/y", "OPENROUTER_API_KEY": "present", SUBAGENTS: ""},
    {"TOTAL_BUDGET": 1.0},
    {"OUROBOROS_REVIEW_MODELS": "a/one, b/two", "OPENROUTER_API_KEY": "present"},
    {SLOTS: "", SUBAGENTS: "", "OPENROUTER_API_KEY": "present"},
    {SLOTS: "  ", "OPENROUTER_API_KEY": "present"},
], ids=["provider-key-only", "blank-catalog-string", "no-provider", "retired-comma-keys", "ui-saved-blank-lanes",
        "whitespace-lanes"])
def test_m1_an_environment_catalog_wins_over_the_factory_rows_of_a_document_that_authored_none(
        tmp_path, monkeypatch, document):
    """The factory rows the seam mints for a document that authored NO review lanes and saved
    no catalog stand in for an ABSENT catalog; they are not the owner's disk value. That is
    the never-configured install, the pre-structured document whose retired comma keys never
    entered the panel (canon 11: such an install ran the shipped default rows), and the
    6.90+ document the UI saved with ``""`` for both keys (N1). A catalog the environment
    carries (``docker run -e OUROBOROS_SUBAGENTS=…`` over a mounted volume the wizard never
    saw) therefore wins over them, exactly as it won over the absence before M1 — without
    the environment catalog the same file still reads the factory pool (M1 stays)."""
    from ouroboros import reviewer_slot_config as rs

    path = tmp_path / "settings.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr(cfg, "SETTINGS_PATH", path)
    env_pool = catalog(api_row("mine", "env/model", "high", review_eligible=True))

    monkeypatch.setenv(SUBAGENTS, env_pool)
    settings = cfg.load_settings_lock_held(_settings_lock_held=False)
    assert settings[SUBAGENTS] == env_pool
    assert [row.slot_id for row in rs.review_pool_rows(settings)] == ["mine"]
    assert m.environment_overridable_keys(document) == {SUBAGENTS}

    monkeypatch.delenv(SUBAGENTS)
    factory = cfg.load_settings_lock_held(_settings_lock_held=False)
    assert _marked(json.loads(factory[SUBAGENTS])["items"]) == ["review-1", "review-2", "review-3"]
    assert all(row["minted_from"] == "factory_default" for row in json.loads(factory[SUBAGENTS])["items"])


@pytest.mark.parametrize("document", [
    {SLOTS: lanes(triad=[direct("t1", "lane/model", "high")], scope=[direct("s1", "lane/model", "high")]),
     "OPENROUTER_API_KEY": "present"},
    {SUBAGENTS: catalog(api_row("saved", "disk/model", "high", review_eligible=True))},
    {SLOTS: "", SUBAGENTS: catalog(api_row("helper", "disk/model", "high"))},
], ids=["authored-lane", "saved-catalog", "blank-lanes-over-a-saved-unmarked-catalog"])
def test_rows_the_owner_authored_on_disk_still_shadow_an_environment_catalog(tmp_path, monkeypatch, document):
    """The other side of the same rule: a catalog the owner saved (marked or not — the
    migration marks or mints INTO it), or rows minted from the owner's own lanes, ARE the
    document's decision and keep shadowing the environment the way every disk-authored key
    does."""
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr(cfg, "SETTINGS_PATH", path)
    assert m.environment_overridable_keys(document) == frozenset()
    monkeypatch.setenv(SUBAGENTS, catalog(api_row("mine", "env/model", "high", review_eligible=True)))
    settings = cfg.load_settings_lock_held(_settings_lock_held=False)
    assert "mine" not in settings[SUBAGENTS]
    assert settings[SUBAGENTS] == cfg.normalize_settings_raw(dict(document))[SUBAGENTS]


def test_n1_the_ui_saved_n_minus_1_document_does_not_shadow_an_environment_catalog(tmp_path, monkeypatch):
    """N1 (FIX4's disclosed residual): the real N-1 document the UI saved —
    ``OUROBOROS_REVIEWER_SLOTS: ""`` and ``OUROBOROS_SUBAGENTS: ""`` beside the retired comma
    keys — authored no lanes and no catalog, so the rows the seam mints for it are a default.
    Beside an environment catalog: the environment's rows are the pool (not lost); when none
    of them is marked the pool is loudly EMPTY (``pool_empty`` in the ``## Review`` block, no
    migration receipt claimed for it) — never silently the factory rows. Without the
    environment catalog the document still reads its factory rows (M1 stays)."""
    import os

    from ouroboros import reviewer_slot_config as rs
    from ouroboros import subagent_runtime
    from ouroboros.settings_integrity import task_settings_snapshot

    path = tmp_path / "settings.json"
    path.write_text(json.dumps(N1_DOC), encoding="utf-8")
    monkeypatch.setattr(cfg, "SETTINGS_PATH", path)
    for key in m._SHA_PRESENCE_KEYS:
        monkeypatch.delenv(key, raising=False)
    assert m.migration_trigger(N1_DOC) == m.TRIGGER_LANES_KEY, "the UI wrote the lanes key as \"\": the first trigger"

    def block(settings):
        snapshot = task_settings_snapshot(settings, {**os.environ, SUBAGENTS: settings[SUBAGENTS]})
        return json.loads(subagent_runtime.review_facts_block(snapshot).split("\n\n", 1)[1])

    marked = catalog(api_row("mine", "env/model", "high", review_eligible=True))
    monkeypatch.setenv(SUBAGENTS, marked)
    settings = cfg.load_settings_lock_held(_settings_lock_held=False)
    assert [row.slot_id for row in rs.review_pool_rows(settings)] == ["mine"], "the environment catalog is not lost"
    assert settings[SUBAGENTS] == marked and m.environment_overridable_keys(N1_DOC) == {SUBAGENTS}
    facts = block(settings)
    assert (facts["source"], facts["pool_empty"], [row["seat_id"] for row in facts["pool"]]) == ("structured", False, ["mine"])

    unmarked = catalog(api_row("helper", "env/model", "high"))
    monkeypatch.setenv(SUBAGENTS, unmarked)
    settings = cfg.load_settings_lock_held(_settings_lock_held=False)
    assert settings[SUBAGENTS] == unmarked and rs.review_pool_rows(settings) == []
    assert rs.review_pool_state(settings[SUBAGENTS]) == {"state": "empty", "error": ""}
    facts = block(settings)
    assert (facts["source"], facts["pool_empty"], facts["pool"], facts["error"]) == ("empty", True, [], "")
    assert "migration_snapshot" not in facts, "the environment's pool was not decided by the document's migration"

    monkeypatch.delenv(SUBAGENTS)
    factory = cfg.load_settings_lock_held(_settings_lock_held=False)
    items = json.loads(factory[SUBAGENTS])["items"]
    assert _marked(items) == ["review-1", "review-2", "review-3"]
    assert all(row["minted_from"] == "factory_default" for row in items if row.get("review_eligible"))
    assert [row.slot_id for row in rs.review_pool_rows(factory)] == ["review-1", "review-2", "review-3"]


def test_m1_a_retired_comma_keys_document_is_distinguished_from_a_fresh_install():
    outcome, _after = _migrated({"OUROBOROS_REVIEW_MODELS": "a/one, b/two", "OPENROUTER_API_KEY": "present"})
    assert outcome.trigger == m.TRIGGER_RETIRED_KEYS and outcome.slots_state == "absent"
    assert "shipped default review lanes" in m.owner_message(outcome, "snap.json")
    lanes_outcome, _after = _migrated({SLOTS: "", "OPENROUTER_API_KEY": "present"})
    assert lanes_outcome.trigger == m.TRIGGER_LANES_KEY


def test_a_pool_catalog_beside_the_empty_lanes_key_drops_it_without_a_rewrite():
    pool = catalog(api_row("r", "x/y", "high", review_eligible=True, minted_from="factory_default"))
    doc = {SUBAGENTS: pool, SLOTS: "", "OUROBOROS_EFFORT_REVIEW": "medium"}
    outcome = m.migrate_review_lanes(doc)
    assert outcome.noop and not outcome.error and outcome.catalog_after is None
    assert outcome.consumed_keys == (SLOTS, "OUROBOROS_EFFORT_REVIEW")
    loaded = cfg.normalize_settings_raw(dict(doc))
    assert loaded[SUBAGENTS] == pool and SLOTS not in loaded and "OUROBOROS_EFFORT_REVIEW" not in loaded
    assert m.owner_message(outcome, "x") == ""


def test_authored_lanes_beside_a_pool_catalog_are_refused_not_dropped_in_silence():
    """VD3-11 (P5b): two review configurations in one document. The catalog's pool is what
    runs; the lanes are neither applied nor consumed — the key stays, the snapshot holds
    both, the owner hears it, and the Settings → Agents save retires the lanes."""
    pool = catalog(api_row("r", "x/y", "high", review_eligible=True, minted_from="factory_default"))
    doc = {SUBAGENTS: pool, SLOTS: ANTON_LANES, "OUROBOROS_EFFORT_REVIEW": "medium"}
    outcome = m.migrate_review_lanes(doc)
    assert outcome.error and not outcome.noop and outcome.catalog_after is None
    assert "review lanes AND a subagent catalog that already holds review pool rows" in outcome.error
    assert outcome.retained_keys == (SLOTS, "OUROBOROS_EFFORT_REVIEW") and outcome.consumed_keys == ()
    assert (outcome.snapshot["before"][SLOTS], outcome.snapshot["before"][SUBAGENTS]) == (ANTON_LANES, pool)
    loaded = cfg.normalize_settings_raw(dict(doc))
    assert loaded[SUBAGENTS] == pool and loaded[SLOTS] == ANTON_LANES and loaded["OUROBOROS_EFFORT_REVIEW"] == "medium"
    assert "could not be migrated automatically" in m.owner_message(outcome, "x")


@pytest.mark.parametrize("value", [{"triad": []}, ["x/y"], 7, True])
def test_a_non_string_lanes_value_is_refused_with_the_key_retained(value):
    """VD3-11 (P4): the lane key held JSON text; a dict, list or number is garbage the
    frozen reader never accepted — not "no lanes" to be replaced by the factory rows."""
    doc = {SLOTS: value, "OPENROUTER_API_KEY": "present"}
    outcome = m.migrate_review_lanes(doc)
    assert outcome.error == f"{SLOTS} must be a JSON string, not {type(value).__name__}", outcome
    assert outcome.retained_keys == (SLOTS,) and outcome.catalog_after is None and outcome.slots_state == "invalid"
    loaded = cfg.normalize_settings_raw(dict(doc))
    assert loaded[SLOTS] == value and SUBAGENTS not in loaded, "nothing minted, nothing dropped"


def test_an_invalid_catalog_with_a_pool_marker_is_refused_not_passed_off_as_a_pool():
    """VD3-05 (P5): ``review_eligible`` on a row does not excuse a catalog this tree's
    parser rejects (two rows with one id); the lanes keys stay, the owner hears why."""
    broken = catalog(api_row("r", "x/y", "high", review_eligible=True), api_row("r", "x/z", "high", review_eligible=True))
    doc = {SUBAGENTS: broken, SLOTS: ""}
    outcome = m.migrate_review_lanes(doc)
    assert outcome.error.startswith("the subagent catalog is invalid, so the review lanes cannot be migrated: ")
    assert not outcome.noop and outcome.catalog_state == "invalid" and outcome.retained_keys == (SLOTS,)
    loaded = cfg.normalize_settings_raw(dict(doc))
    assert loaded[SUBAGENTS] == broken and loaded[SLOTS] == ""
    # The control: the same shape, valid, IS a pool — the "" key is dropped, nothing rewritten.
    assert m.migrate_review_lanes({SUBAGENTS: catalog(api_row("r", "x/y", "high", review_eligible=True)), SLOTS: ""}).noop


def test_a_deep_review_reference_and_a_disabled_advisory_mint_nothing():
    doc = {SUBAGENTS: catalog(api_row("helper", "a/one", "high"), api_row("deep", "b/two", "xhigh")),
           SLOTS: lanes(triad=[ref("t", "helper")], scope=[ref("s", "helper")],
                        advisory={"enabled": False, "route": {"kind": "api_chat", "target_id": "c/adv"}},
                        deep_review={"subagent_id": "deep"})}
    outcome, after = _migrated(doc)
    assert len(after["items"]) == 2 and _marked(after["items"]) == ["helper"]
    assert "the deep review reference to deep needed no row" in outcome.snapshot["notes"]
    assert any("disabled" in note for note in outcome.snapshot["notes"])
    assert outcome.snapshot["summary"]["helper_rows_minted"] == 0


def test_a_referenced_row_with_divergent_processing_or_pin_mints_from_the_source_row():
    doc = {SUBAGENTS: catalog(api_row("helper", "claudexor::codex=gpt-6-astra", "medium",
                                      processing_preference="economy")),
           SLOTS: lanes(triad=[ref("t", "helper", "xhigh")], scope=[ref("s", "helper", "xhigh")])}
    _outcome, after = _migrated(doc)
    assert _marked(after["items"]) == ["review-1"]
    minted = after["items"][1]
    assert minted["route"] == {"kind": "api_model", "target_id": "claudexor::codex=gpt-6-astra", "credential_profile_id": ""}
    assert minted["effort"] == "xhigh" and minted["processing_preference"] == "economy"
    assert "review_eligible" not in after["items"][0]


def test_package_a_factory_rows_seam_is_the_one_factory_source(monkeypatch):
    """``factory_review_rows(doc)`` (package A) mints the rows a fresh install's onboarding
    writes; the factory cell adopts EXACTLY those rows — the frozen factory seats are the
    same rows (``factory_lanes``), so no second table mints a seat beside them (D1-V04:
    a one-row seam gives a one-row pool, not that row plus the OpenRouter panel). The
    seam is a module attribute so the two packages meet without an import."""
    calls = []

    def factory_rows(document):
        calls.append(dict(document))
        return [{"subagent_id": "review-1", "recommended_use": "A's row",
                 "route": {"kind": "api_model", "target_id": "google/gemini-3.8-flash"}, "effort": "",
                 "review_eligible": True, "minted_from": "factory_default"}]

    monkeypatch.setattr(m, "factory_review_rows", factory_rows)
    assert [row.target_id for row in m.factory_lanes(N1_DOC).triad] == ["google/gemini-3.8-flash"]
    outcome, after = _migrated(N1_DOC)
    assert calls, "the seam was consulted"
    assert after["items"][0]["recommended_use"] == "A's row" and after["items"][0]["effort"] == "high"
    assert _marked(after["items"]) == ["review-1"]
    assert [r["route"]["target_id"] for r in after["items"]] == ["google/gemini-3.8-flash", "openai/gpt-5.6-sol-pro"]
    assert outcome.snapshot["summary"] == {"seats_before": 1, "rows_marked_after": 1, "distinct_models": 1,
                                           "helper_rows_minted": 1}


@pytest.mark.parametrize("install, main", [
    ("local-only", {"USE_LOCAL_MAIN": True, "LOCAL_MODEL_SOURCE": "owner/local.gguf", "OUROBOROS_MODEL": "owner/local-main"}),
    ("compatible-only", {"OPENAI_COMPATIBLE_BASE_URL": "https://llm.example/v1", "OUROBOROS_MODEL": "openai-compatible::glm-5.3"}),
])
def test_factory_cells_of_a_one_model_install_mint_the_three_runs_of_main(install, main):
    """I3-1: the factory pool of a local-only or compatible-only install (cells (A,0),
    (C,0), (D,0), (E,0)) is what those installs ran — three independent seats of
    Main (twins, quorum 2 of 3) — through package A's bound seam, so the migration
    and a fresh onboarding mint one shape; an OpenRouter install keeps three models."""
    from ouroboros.review_model_routes import adaptive_quorum

    outcome, after = _migrated({SLOTS: "", **main})
    marked = [row for row in after["items"] if row.get("review_eligible")]
    assert [row["subagent_id"] for row in marked] == ["review-1", "review-2", "review-3"], install
    assert [row["route"]["target_id"] for row in marked] == [main["OUROBOROS_MODEL"]] * 3, install
    assert all(row["minted_from"] == "factory_default" and row["effort"] == "high" for row in marked), install
    assert outcome.snapshot["summary"]["rows_marked_after"] == 3 and adaptive_quorum(len(marked)) == 2
    assert outcome.snapshot["summary"]["distinct_models"] == 1, "twins are one engine, disclosed"
    # The same seam on an existing catalog with one row of that engine: marked, not twinned (F6).
    twin = api_row("mine", main["OUROBOROS_MODEL"], "high")
    _outcome, after = _migrated({SLOTS: "", SUBAGENTS: catalog(twin), **main})
    assert _marked(after["items"]) == ["mine", "review-1", "review-2"], install
    # OpenRouter: three different models, as before.
    _outcome, after = _migrated({SLOTS: "", "OPENROUTER_API_KEY": "present"})
    assert [r["route"]["target_id"] for r in after["items"] if r.get("review_eligible")] == list(
        OPENROUTER_REVIEW_DEFAULTS["triad"])


@pytest.mark.parametrize("install, document", [
    ("local Main behind a dead OPENAI_BASE_URL (D1-V04)",
     {"USE_LOCAL_MAIN": True, "OUROBOROS_MODEL": "local-demo", "OPENAI_BASE_URL": "http://127.0.0.1:9/v1"}),
    ("one direct key beside an OpenRouter-style Main (the Colab re-run over N-1)",
     {"OPENAI_API_KEY": "present", "OUROBOROS_MODEL": "anthropic/claude-opus-5"}),
])
def test_d1_v04_a_document_without_lanes_gets_exactly_the_factory_rows_from_one_source(install, document):
    """D1-V04 / D1-V06: ``factory_lanes`` and ``factory_review_rows`` are ONE source. A
    second provider table once read these documents differently (a bare base URL made
    the panel "remote"; a Main not on the provider kept the OpenRouter ids), so the
    frozen seats minted three OpenRouter rows BESIDE the factory rows: six marked rows
    for the ``""`` lanes key against three for the absent key. Both cells are the same
    pool now — exactly the factory rows, no second table."""
    from ouroboros.subscription_install_presets import factory_review_rows

    templates = factory_review_rows(document)
    targets = [row["route"]["target_id"] for row in templates]
    assert len(targets) == 3 and [row.target_id for row in m.factory_lanes(document).triad] == targets, install
    for lanes_cell in ({}, {SLOTS: ""}):
        outcome, after = _migrated({**document, **lanes_cell})
        marked = [row for row in after["items"] if row.get("review_eligible")]
        assert [row["route"]["target_id"] for row in marked] == targets, (install, lanes_cell)
        assert len(after["items"]) == 3 and outcome.snapshot["summary"]["seats_before"] == 3, (install, lanes_cell)


# --- 5. the read seam -----------------------------------------------------------------


def test_normalize_settings_raw_migrates_once_per_document_digest(monkeypatch):
    doc = anton_document()
    loaded = cfg.normalize_settings_raw(dict(doc))
    assert SLOTS not in loaded and len(json.loads(loaded[SUBAGENTS])["items"]) == 11
    for key in REVIEW_POOL_MIGRATED_SETTING_KEYS:
        assert key not in loaded
    outcomes = cfg.review_pool_migrations_seen()
    assert len(outcomes) == 1 and outcomes[0].input_sha256 == m.input_sha256(doc)
    # The same document again replays the recorded outcome: the migration is not recomputed.
    monkeypatch.setattr(m, "migrate_review_lanes", lambda _doc: pytest.fail("recomputed"))
    assert cfg.normalize_settings_raw(dict(doc)) == loaded
    assert len(cfg.review_pool_migrations_seen()) == 1
    # The purge never reports the consumed lane keys as a loss.
    assert cfg.retired_key_sets_seen() == ()


def test_the_read_seam_keeps_the_lane_keys_of_a_migration_that_could_not_finish(caplog):
    import logging

    doc = {SUBAGENTS: catalog(), SLOTS: lanes(triad=[ref("t", "ghost")], scope=[]),
           "OUROBOROS_EFFORT_REVIEW": "medium", "OUROBOROS_SCOPE_REVIEW_FLOOR": "blocking_1m"}
    with caplog.at_level(logging.WARNING, logger="ouroboros.review_pool_migration"):
        loaded = cfg.normalize_settings_raw(dict(doc))
    assert loaded[SLOTS] == doc[SLOTS] and loaded["OUROBOROS_EFFORT_REVIEW"] == "medium"
    assert loaded[SUBAGENTS] == catalog(), "no partial migration"
    assert "OUROBOROS_SCOPE_REVIEW_FLOOR" not in loaded, "the ordinary retired purge still runs"
    assert cfg.retired_key_sets_seen() == (("OUROBOROS_SCOPE_REVIEW_FLOOR",),)
    (outcome,) = cfg.review_pool_migrations_seen()
    assert outcome.error and set(outcome.retained_keys) == {SLOTS, "OUROBOROS_EFFORT_REVIEW"}
    assert any("review lanes not migrated" in r.getMessage() for r in caplog.records)


def test_the_five_lane_keys_are_retired_with_the_catalog_as_successor():
    assert set(REVIEW_POOL_MIGRATED_SETTING_KEYS) <= set(RETIRED_SETTING_KEYS)
    for key in REVIEW_POOL_MIGRATED_SETTING_KEYS:
        assert key not in SETTINGS_DEFAULTS
        assert RETIRED_SETTING_SUCCESSORS[key] == (SUBAGENTS,)
    assert not set(REVIEW_POOL_MIGRATED_SETTING_KEYS) & set(RETIRED_COMMA_LIST_SETTING_KEYS)
    # A stray effort key without lanes on a pool document reaches the ordinary notice.
    pool = {SUBAGENTS: catalog(api_row("r", "x/y", "high", review_eligible=True)), "OUROBOROS_EFFORT_REVIEW": "low"}
    loaded = cfg.normalize_settings_raw(dict(pool))
    assert "OUROBOROS_EFFORT_REVIEW" not in loaded
    assert cfg.retired_key_sets_seen() == (("OUROBOROS_EFFORT_REVIEW",),)


def test_the_input_digest_reads_credentials_by_presence_only():
    doc = anton_document()
    other = dict(doc, OPENROUTER_API_KEY="another-value")
    assert m.input_sha256(doc) == m.input_sha256(other)
    assert m.input_sha256(doc) != m.input_sha256({k: v for k, v in doc.items() if k != "OPENROUTER_API_KEY"})
    assert m.input_sha256(doc) != m.input_sha256(dict(doc, OUROBOROS_EFFORT_REVIEW="high"))


def test_the_input_digest_covers_the_legacy_row_materialization_keys():
    """VD3-04 (P6): a legacy harness singleton materializes catalog rows from
    OUROBOROS_MODEL_HEAVY / USE_LOCAL_HEAVY / USE_LOCAL_LIGHT, so two documents that differ
    only there are two migration subjects — the per-digest cache must not replay the
    first document's catalog onto the second."""
    base = {SLOTS: "", "OUROBOROS_SUBAGENT_HARNESS": "codex", "OPENROUTER_API_KEY": "present",
            "OUROBOROS_MODEL": "x/main", "OUROBOROS_MODEL_HEAVY": "a/one"}
    for change in ({"OUROBOROS_MODEL_HEAVY": "b/two"}, {"USE_LOCAL_HEAVY": True}, {"USE_LOCAL_LIGHT": True}):
        assert m.input_sha256(base) != m.input_sha256({**base, **change}), change
    first = cfg.normalize_settings_raw(dict(base))
    second = cfg.normalize_settings_raw({**base, "OUROBOROS_MODEL_HEAVY": "b/two"})
    heavy = lambda loaded: [r["route"]["target_id"] for r in json.loads(loaded[SUBAGENTS])["items"]  # noqa: E731
                            if r["subagent_id"] == "legacy-heavy"]
    assert heavy(first) == ["a/one"] and heavy(second) == ["b/two"], (heavy(first), heavy(second))


# --- 6. the supervisor boot: snapshot once, owner told once -------------------------------


@pytest.fixture
def boot(tmp_path, monkeypatch):
    state.init(tmp_path)
    (tmp_path / "state").mkdir(parents=True, exist_ok=True)
    (tmp_path / "locks").mkdir(parents=True, exist_ok=True)
    state.save_state({})
    monkeypatch.setattr(server_maintenance, "DATA_DIR", tmp_path)
    sent: list = []
    monkeypatch.setattr(message_bus, "send_with_budget",
                        lambda chat_id, text, *args, **kwargs: sent.append((chat_id, text, kwargs)))
    return tmp_path, sent


def _snapshots(root):
    return sorted(p for p in (root / "state" / "review_migrations").glob("*-slots-to-pool.json")) \
        if (root / "state" / "review_migrations").exists() else []


def _served_from_disk(root, monkeypatch, document):
    """The server's read of ``document`` as the settings file at ``root``: the boot receipts what decides that file."""
    monkeypatch.setattr(cfg, "SETTINGS_PATH", root / "settings.json")
    cfg.SETTINGS_PATH.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.delenv(SUBAGENTS, raising=False)
    return cfg.load_settings_lock_held(_settings_lock_held=False)


def test_the_snapshot_is_written_once_and_the_owner_hears_once(boot, monkeypatch):
    root, sent = boot
    loaded = _served_from_disk(root, monkeypatch, anton_document())
    state.update_state(lambda st: st.__setitem__("owner_chat_id", 7))

    server_maintenance._startup_review_pool_notice(loaded)
    server_maintenance._startup_review_pool_notice(loaded)
    cfg.load_settings_lock_held(_settings_lock_held=False)  # the same document read again
    server_maintenance._startup_review_pool_notice(loaded)

    files = _snapshots(root)
    assert len(files) == 1
    name = files[0].name
    assert len(name) == len("20261007T214000Z-slots-to-pool.json") and name.endswith("Z-slots-to-pool.json")
    snapshot = json.loads(files[0].read_text(encoding="utf-8"))
    assert snapshot["schema"] == 1 and snapshot["ts"] == name.split("-", 1)[0]
    assert snapshot["input_sha256"] == m.input_sha256(anton_document())
    assert snapshot["before"]["OUROBOROS_REVIEWER_SLOTS"] == ANTON_LANES
    assert snapshot["before"]["OUROBOROS_EFFORT_REVIEW"] == "medium"
    assert snapshot["before"]["catalog_state"] == "configured" and snapshot["before"]["slots_state"] == "mixed"
    assert snapshot["effective_before"] == ANTON_EXPECTED_EXECUTIONS
    assert json.loads(loaded[SUBAGENTS]) == snapshot["after"][SUBAGENTS]
    assert len(snapshot["rows"]) == 4 and snapshot["error"] == ""

    assert len(sent) == 1
    chat_id, text, kwargs = sent[0]
    assert chat_id == 7 and kwargs == {"role": "system", "system_type": "review_pool_migration_notice"}
    assert text == m.owner_message(cfg.review_pool_migrations_seen()[0], f"state/review_migrations/{name}")
    assert f"Snapshot: state/review_migrations/{name}." in text

    records = server_maintenance.review_pool_migration_records()
    (record,) = records.values()
    assert record["snapshot"] == f"state/review_migrations/{name}" and record["reported"] and record["error"] == ""
    assert record["ts"] == snapshot["ts"]

    # A fresh process (empty in-process seam) reading the same document is quiet.
    m._MIGRATIONS_SEEN.clear()
    server_maintenance._startup_review_pool_notice(cfg.load_settings_lock_held(_settings_lock_held=False))
    assert len(_snapshots(root)) == 1 and len(sent) == 1


def test_without_an_owner_chat_the_snapshot_is_written_but_the_message_waits(boot, monkeypatch):
    root, sent = boot
    loaded = _served_from_disk(root, monkeypatch, dict(N1_DOC))
    server_maintenance._startup_review_pool_notice(loaded)
    assert len(_snapshots(root)) == 1 and sent == []
    (record,) = server_maintenance.review_pool_migration_records().values()
    assert record["reported"] is None

    state.update_state(lambda st: st.__setitem__("owner_chat_id", 3))
    server_maintenance._startup_review_pool_notice(loaded)
    assert len(_snapshots(root)) == 1 and [row[0] for row in sent] == [3]
    assert "shipped default review lanes" in sent[0][1]
    server_maintenance._startup_review_pool_notice(loaded)
    assert len(sent) == 1


@pytest.mark.parametrize("document", [N1_DOC, {"OUROBOROS_MODEL": "x/y", "OPENROUTER_API_KEY": "present"}],
                         ids=["ui-saved-n-minus-1", "never-configured"])
def test_n1_the_boot_notice_names_the_environment_pool_in_force_not_the_minted_rows(boot, monkeypatch, document):
    """N1: when the environment's catalog runs in place of the rows the seam minted for a
    document without review settings of its own, the owner is told THAT — which rows run,
    or that none is marked (``pool_empty``) — not that the factory rows run or that the
    default lanes became the pool. The snapshot is still written: it is the receipt of
    what the migration computed for the document. Without the environment catalog the
    message is the migration's own."""
    root, sent = boot
    state.update_state(lambda st: st.__setitem__("owner_chat_id", 7))
    path = root / "settings.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr(cfg, "SETTINGS_PATH", path)
    for key in m._SHA_PRESENCE_KEYS:
        monkeypatch.delenv(key, raising=False)
    minted_claims = ("factory reviewer rows run", "became one review pool", "— new row", "— marked")

    def boot_with(env_catalog):
        m._MIGRATIONS_SEEN.clear()
        state.update_state(lambda st: st.pop(server_maintenance.REVIEW_POOL_MIGRATION_STATE_KEY, None))
        if env_catalog is None:
            monkeypatch.delenv(SUBAGENTS, raising=False)
        else:
            monkeypatch.setenv(SUBAGENTS, env_catalog)
        settings = cfg.load_settings_lock_held(_settings_lock_held=False)
        before = len(sent)
        server_maintenance._startup_review_pool_notice(settings)
        assert len(sent) == before + 1
        return sent[-1][1]

    text = boot_with(catalog(api_row("mine", "env/model", "high", review_eligible=True)))
    assert "the subagent catalog set in the environment (OUROBOROS_SUBAGENTS) is in force" in text
    assert "1 reviewer rows, 1 distinct models: mine (env/model)." in text and "the factory rows do not" in text
    assert not any(claim in text for claim in minted_claims)
    assert len(_snapshots(root)) == 1 and _snapshots(root)[0].name in text

    text = boot_with(catalog(api_row("helper", "env/model", "high")))
    assert "is in force" in text and "the review pool is empty (pool_empty)" in text
    assert "will not run and will report not performed" in text and not any(claim in text for claim in minted_claims)

    text = boot_with(None)
    assert "is in force" not in text and "pool_empty" not in text
    assert text == m.owner_message(cfg.review_pool_migrations_seen()[0], f"state/review_migrations/{_snapshots(root)[-1].name}")
    assert ("Review pool initialized" in text) == (m.migration_trigger(document) == m.TRIGGER_NEVER_CONFIGURED)


def test_a_refused_migration_is_recorded_and_reported_with_its_error(boot, monkeypatch):
    root, sent = boot
    state.update_state(lambda st: st.__setitem__("owner_chat_id", 7))
    doc = {SUBAGENTS: catalog(), SLOTS: lanes(triad=[ref("t", "ghost")], scope=[])}
    loaded = _served_from_disk(root, monkeypatch, dict(doc))
    server_maintenance._startup_review_pool_notice(loaded)
    (path,) = _snapshots(root)
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    assert snapshot["error"] and snapshot["after"] is None and snapshot["summary"] is None
    assert len(sent) == 1 and sent[0][1].startswith("⚙️ Review settings could not be migrated automatically")
    assert "ghost" in sent[0][1] and path.name in sent[0][1]
    (record,) = server_maintenance.review_pool_migration_records().values()
    assert record["error"] == snapshot["error"]


def test_a_noop_outcome_leaves_no_receipt(boot, monkeypatch):
    root, sent = boot
    state.update_state(lambda st: st.__setitem__("owner_chat_id", 7))
    pool = catalog(api_row("r", "x/y", "high", review_eligible=True, minted_from="factory_default"))
    loaded = _served_from_disk(root, monkeypatch, {SUBAGENTS: pool, SLOTS: ""})
    assert cfg.review_pool_migrations_seen()[0].noop
    server_maintenance._startup_review_pool_notice(loaded)
    assert _snapshots(root) == [] and sent == []
    assert server_maintenance.review_pool_migration_records() == {}


# --- 7. the receipts belong to the SAVING process, not to the boot's memory (VD3-01) -----


def _other_root(tmp_path, name):
    """A second data root with its own supervisor state (the Drive root of a Colab install)."""
    root = tmp_path / name
    for sub in ("state", "locks"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def test_a_document_migrated_and_saved_by_another_process_still_gets_its_receipts_at_boot(boot, monkeypatch):
    """VD3-01: the Colab kernel reads the N-1 Drive document (the lanes become the pool in ITS
    process) and writes the Drive document back — the pre-image is gone before any server ran.
    The kernel's writer gives the migration its snapshot under the Drive root (its supervisor
    state is not bound there, so no state record); the server that later boots on that root
    computes NO migration (the document is a pool now), yet reconciles the record from the
    snapshot and tells the owner ONCE — and a second boot stays quiet."""
    from ouroboros.colab_bootstrap import build_colab_settings, write_colab_settings

    root, sent = boot  # the kernel's own process state is bound to ``root``, not to the Drive root
    drive = _other_root(root, "drive")
    kernel_view = build_colab_settings({"OPENROUTER_API_KEY": "present"}, existing=dict(N1_DOC))
    kernel_document = {**N1_DOC, "OPENROUTER_API_KEY": "present"}  # the Drive document plus the fresh secret
    assert [o.trigger for o in cfg.review_pool_migrations_seen()] == [m.TRIGGER_LANES_KEY]
    write_colab_settings(drive, kernel_view)
    (snapshot_file,) = _snapshots(drive)
    assert _snapshots(root) == [] and server_maintenance.review_pool_migration_records() == {}
    snapshot = json.loads(snapshot_file.read_text(encoding="utf-8"))
    assert snapshot["input_sha256"] == m.input_sha256(kernel_document) and snapshot["before"][SLOTS] == ""
    assert SLOTS not in json.loads((drive / "settings.json").read_text(encoding="utf-8"))

    # The server: a fresh process on the Drive root.
    m._MIGRATIONS_SEEN.clear()
    state.init(drive)
    state.save_state({"owner_chat_id": 7})
    monkeypatch.setattr(server_maintenance, "DATA_DIR", drive)
    monkeypatch.setattr(cfg, "SETTINGS_PATH", drive / "settings.json")
    served = cfg.load_settings_lock_held(_settings_lock_held=False)
    assert cfg.review_pool_migrations_seen() == ()
    server_maintenance._startup_review_pool_notice(served)
    server_maintenance._startup_review_pool_notice(served)

    assert _snapshots(drive) == [snapshot_file]
    (record,) = server_maintenance.review_pool_migration_records().values()
    assert record["snapshot"] == f"state/review_migrations/{snapshot_file.name}" and record["ts"] == snapshot["ts"]
    assert (record["trigger"], record["outcome"], record["error"]) == (m.TRIGGER_LANES_KEY, "factory", "")
    assert record["reported"]
    assert len(sent) == 1 and sent[0][0] == 7
    assert sent[0][1] == m.owner_message(m.migrate_review_lanes(kernel_document), record["snapshot"])


def test_the_saving_process_writes_the_receipts_before_its_write_and_the_boot_adds_nothing(boot, monkeypatch):
    """VD3-01: the UI's owner save (or the launcher menu) persists the migrated document BEFORE
    the supervisor generation starts. Every in-process writer passes the persistence prologue,
    which writes the snapshot AND the state record (this process's state is bound to the root)
    before the document write; the boot then finds the record, tells the owner once and
    writes no second snapshot — and a save after the boot (the normal order) adds nothing."""
    root, sent = boot
    monkeypatch.setattr(cfg, "DATA_DIR", root)
    monkeypatch.setattr(cfg, "SETTINGS_PATH", root / "settings.json")
    loaded = cfg.normalize_settings_raw(anton_document())
    assert _snapshots(root) == []

    cfg.save_settings(dict(loaded))

    (snapshot_file,) = _snapshots(root)
    (record,) = server_maintenance.review_pool_migration_records().values()
    assert record["snapshot"] == f"state/review_migrations/{snapshot_file.name}" and record["reported"] is None
    assert (record["trigger"], record["outcome"], record["error"]) == (m.TRIGGER_LANES_KEY, "converted", "")
    on_disk = json.loads((root / "settings.json").read_text(encoding="utf-8"))
    assert SLOTS not in on_disk and json.loads(on_disk[SUBAGENTS]) == json.loads(loaded[SUBAGENTS])

    state.update_state(lambda st: st.__setitem__("owner_chat_id", 7))
    server_maintenance._startup_review_pool_notice(loaded)
    cfg.save_settings(dict(loaded))
    server_maintenance._startup_review_pool_notice(loaded)
    assert _snapshots(root) == [snapshot_file] and len(sent) == 1
    (record,) = server_maintenance.review_pool_migration_records().values()
    assert record["reported"] and snapshot_file.name in sent[0][1]


def test_a_save_receipts_the_document_it_replaces_not_every_document_the_process_read(boot, monkeypatch):
    """FIX6F: the prologue gives receipts to the migration of the document the write REPLACES (the
    on-disk pre-image by its exact digest, even when the owner's save brings a new catalog instead
    of the migrated rows) and to one whose result it saves — not to every document this process
    normalized (a draft the wizard read, another root's document): those receipts described rows
    that never ran, and each snapshot cost the writer a UTC second under the settings lock."""
    root, _sent = boot
    path = root / "settings.json"
    monkeypatch.setattr(cfg, "DATA_DIR", root)
    monkeypatch.setattr(cfg, "SETTINGS_PATH", path)
    for unsaved in (dict(N1_DOC), {"OUROBOROS_MODEL": "x/y", "OPENROUTER_API_KEY": "present"}):
        cfg.normalize_settings_raw(unsaved)
    path.write_text(json.dumps(anton_document()), encoding="utf-8")
    loaded = cfg.normalize_settings_raw(anton_document())
    assert len(cfg.review_pool_migrations_seen()) == 3
    owner_catalog = catalog(api_row("mine", "x/y", "high", review_eligible=True))

    cfg.save_settings({**loaded, SUBAGENTS: owner_catalog})

    (snapshot_file,) = _snapshots(root)
    snapshot = json.loads(snapshot_file.read_text(encoding="utf-8"))
    assert snapshot["input_sha256"] == m.input_sha256(anton_document()) and snapshot["before"][SLOTS] == ANTON_LANES
    (record,) = server_maintenance.review_pool_migration_records().values()
    assert record["snapshot"] == f"state/review_migrations/{snapshot_file.name}"
    assert json.loads(path.read_text(encoding="utf-8"))[SUBAGENTS] == owner_catalog


def test_a_receipt_failure_never_blocks_the_save(boot, monkeypatch):
    from ouroboros import review_pool_receipts as receipts

    root, _sent = boot
    monkeypatch.setattr(cfg, "DATA_DIR", root)
    monkeypatch.setattr(cfg, "SETTINGS_PATH", root / "settings.json")
    loaded = cfg.normalize_settings_raw(anton_document())
    monkeypatch.setattr(receipts, "read_snapshots", lambda data_dir: (_ for _ in ()).throw(OSError("disk")))
    cfg.save_settings(dict(loaded))
    assert (root / "settings.json").exists() and _snapshots(root) == []
