"""The review pool (PR-3, contract §1.1–1.3, §2 F4/F5/F8): the catalog's marked rows ARE the panel.

`OUROBOROS_REVIEWER_SLOTS` lanes are gone as a review surface: every surface — the
commit gate, plan review, skill review, task acceptance, the author's own wave —
reads ONE builder (``review_pool_slots``) over the enabled catalog rows the owner
marked ``review_eligible``. A row rides its own identity (``slot_id`` is its
stored ``subagent_id``), its own delivery (an api row's ``delivery`` field, a
session always retrieves — F8: never inferred from an actor id) and its own
effort (row, then a compound slug, then the pool default). A malformed catalog
is a typed refusal on every reader; a catalog with no marked row is an EMPTY
pool, a loud configured fact — never a shipped default panel (F4/F5).
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from ouroboros import reviewer_slot_config as rsc
from ouroboros.configured_subagents import (
    MAX_CONFIGURED_SUBAGENTS,
    SUBAGENTS_SETTING,
    parse_configured_subagents,
    roster_handles,
)
from ouroboros.review_execution import ReviewRouteKind


def _row(subagent_id: str, target: str, *, kind: str = "api_model", effort: str = "", marked: bool = True,
         **extra) -> dict:
    row = {"subagent_id": subagent_id, "recommended_use": "Reviews diffs.",
           "route": {"kind": kind, "target_id": target}, **extra}
    if effort:
        row["effort"] = effort
    if marked:
        row["review_eligible"] = True
    return row


def _roster(*rows: dict, enabled: bool = True) -> str:
    return json.dumps({"enabled": enabled, "items": list(rows)})


_SESSION = {"kind": "agent_session", "target_id": "codex=gpt-5.6-sol", "credential_profile_id": "profile-1"}

_MIXED = _roster(
    _row("api-critic", "openai/gpt-5.6-terra", effort="medium"),
    _row("packet-critic", "openai/gpt-5.6-sol", delivery="packet"),
    {"subagent_id": "session-critic", "recommended_use": "Subscription reviewer.", "route": _SESSION,
     "effort": "high", "review_eligible": True},
    _row("helper", "openai/gpt-5.6-luna", marked=False),
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in (SUBAGENTS_SETTING, "OUROBOROS_REVIEWER_SLOTS", "OUROBOROS_REVIEW_MODELS",
                "OUROBOROS_SCOPE_REVIEW_MODELS", "OUROBOROS_SCOPE_REVIEW_MODEL", "OUROBOROS_EFFORT_REVIEW"):
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


def _handles() -> dict[str, str]:
    import os

    return roster_handles(parse_configured_subagents(os.environ[SUBAGENTS_SETTING]), dict(os.environ))


# --- membership -------------------------------------------------------------------------


def test_the_pool_is_the_marked_enabled_rows_in_catalog_order_regardless_of_the_catalog_switch(clean_env):
    clean_env.setenv(SUBAGENTS_SETTING, _roster(
        _row("b", "m/b"), _row("a", "m/a", marked=False), _row("c", "m/c"),
        _row("off", "m/off", enabled=False),
        enabled=False,  # delegation switched off does not switch review off (F6)
    ))
    rows = rsc.review_pool_rows()
    assert [(row.slot_id, row.subagent_id, row.target_id) for row in rows] == [("b", "b", "m/b"), ("c", "c", "m/c")]
    assert rsc.review_pool_state(_roster(_row("x", "m/x"), enabled=False)) == {"state": "structured", "error": ""}


def test_review_pool_state_is_three_valued():
    assert rsc.review_pool_state(_roster(_row("x", "m/x"))) == {"state": "structured", "error": ""}
    assert rsc.review_pool_state(_roster(_row("x", "m/x", marked=False))) == {"state": "empty", "error": ""}
    assert rsc.review_pool_state("") == {"state": "empty", "error": ""}
    assert rsc.review_pool_state(None) == {"state": "empty", "error": ""}
    broken = rsc.review_pool_state('{"enabled": true, "items": [{"subagent_id": "x"}]}')
    assert broken["state"] == "error" and broken["error"]
    assert rsc.review_pool_state("{nope")["state"] == "error"


def test_the_save_judge_refuses_an_unmarked_catalog_unless_the_owner_allows_an_empty_pool():
    unmarked = _roster(_row("x", "m/x", marked=False))
    assert "no reviewers marked" in rsc.review_pool_save_error(unmarked, allow_empty=False)
    assert rsc.review_pool_save_error(unmarked, allow_empty=True) == ""
    assert rsc.review_pool_save_error(_roster(_row("x", "m/x")), allow_empty=False) == ""
    # An absent catalog or one without rows is not "unmarked": nothing to judge.
    assert rsc.review_pool_save_error("", allow_empty=False) == ""
    assert rsc.review_pool_save_error(_roster(), allow_empty=False) == ""
    assert "not valid JSON" in rsc.review_pool_save_error("{nope", allow_empty=True)


def test_catalog_review_row_names_any_enabled_row_by_handle_or_stored_id(clean_env):
    clean_env.setenv(SUBAGENTS_SETTING, _roster(
        _row("critic", "openai/gpt-5.6-terra", marked=False), _row("gone", "m/gone", enabled=False)))
    handles = _handles()
    by_id, by_handle = rsc.catalog_review_row(None, "critic"), rsc.catalog_review_row(None, handles["critic"])
    assert by_id == by_handle and (by_id.slot_id, by_id.target_id, by_id.native_retrieval) == (
        "critic", "openai/gpt-5.6-terra", True)  # unmarked rows are the author's to name
    with pytest.raises(ValueError, match="switched off"):
        rsc.catalog_review_row(None, "gone")
    with pytest.raises(ValueError, match="unknown_subagent_id"):
        rsc.catalog_review_row(None, "ghost")
    clean_env.delenv(SUBAGENTS_SETTING)
    with pytest.raises(ValueError, match="not configured"):
        rsc.catalog_review_row(None, "critic")


def test_a_marked_session_row_without_a_concrete_harness_is_a_typed_refusal(clean_env):
    """The catalog parser refuses a session target that names no harness; the
    pool readers surface that typed error instead of a shared-route guess."""
    clean_env.setenv(SUBAGENTS_SETTING, _roster(
        {"subagent_id": "vague", "recommended_use": "r", "review_eligible": True,
         "route": {"kind": "agent_session", "target_id": "=gpt-5.6-sol"}}))
    with pytest.raises(ValueError, match=f"{SUBAGENTS_SETTING}: items\\[0\\] session harness"):
        rsc.review_pool_slots()
    assert rsc.review_pool_state(_roster(_row("v", "=x", kind="agent_session")))["state"] == "error"


# --- slots: identity, delivery, pin -------------------------------------------------------


def test_review_pool_slots_carry_identity_delivery_and_pin(clean_env):
    clean_env.setenv(SUBAGENTS_SETTING, _MIXED)
    api, packet, session = rsc.review_pool_slots(role_hint="task acceptance")

    assert [s.slot_id for s in (api, packet, session)] == ["api-critic", "packet-critic", "session-critic"]
    assert all(s.subagent_id == s.slot_id for s in (api, packet, session)), "slot_id IS the stored id"
    assert all(s.role_hint == "task acceptance" for s in (api, packet, session))
    assert (api.route, api.native_retrieval, api.retrieves, api.model) == (
        ReviewRouteKind.API_CHAT, True, True, "openai/gpt-5.6-terra")
    assert (packet.route, packet.native_retrieval, packet.retrieves) == (ReviewRouteKind.API_CHAT, False, False)
    assert (session.route, session.retrieves, session.native_retrieval) == (ReviewRouteKind.AGENT_SESSION, True, False)
    assert (session.session_target, session.session_profile, session.model) == (
        "codex=gpt-5.6-sol", "profile-1", "codex=gpt-5.6-sol")
    assert "helper" not in {s.slot_id for s in (api, packet, session)}


def test_a_malformed_catalog_raises_on_every_pool_reader_never_a_default_panel(clean_env):
    clean_env.setenv(SUBAGENTS_SETTING, "{broken")
    clean_env.setenv("OUROBOROS_REVIEW_MODELS", "m/one,m/two")  # the env plane is not a panel
    for reader in (rsc.review_pool_slots, rsc.triad_delivery_slots, rsc.commit_triad_delivery, rsc.review_pool_rows):
        with pytest.raises(ValueError, match="not valid JSON"):
            reader()
    assert "not valid JSON" in rsc.reviewer_slot_config_error()
    clean_env.setenv(SUBAGENTS_SETTING, _roster(_row("x", "m/x", marked=False)))
    assert rsc.review_pool_slots() == [] and rsc.reviewer_slot_config_error() == ""  # empty is not an error


def test_triad_delivery_slots_is_the_pool_under_its_historical_name(clean_env):
    clean_env.setenv(SUBAGENTS_SETTING, _MIXED)
    pool = rsc.review_pool_slots(role_hint="plan reviewer", timeout_sec=7)
    alias = rsc.triad_delivery_slots(role_hint="plan reviewer", timeout_sec=7)
    assert alias == pool and [s.timeout_sec for s in alias] == [7, 7, 7]


def test_commit_triad_delivery_projects_aligned_vectors_from_the_pool(clean_env):
    clean_env.setenv(SUBAGENTS_SETTING, _MIXED)
    plan = rsc.commit_triad_delivery()

    assert plan["slot_ids"] == plan["subagent_ids"] == ["api-critic", "packet-critic", "session-critic"]
    assert plan["models"] == ["openai/gpt-5.6-terra", "openai/gpt-5.6-sol", "codex=gpt-5.6-sol"]
    assert plan["routes"] == [ReviewRouteKind.API_CHAT, ReviewRouteKind.API_CHAT, ReviewRouteKind.AGENT_SESSION]
    assert plan["efforts"] == ["medium", "high", "high"]
    assert plan["session_targets"] == ["", "", "codex=gpt-5.6-sol"]
    assert plan["session_profiles"] == ["", "", "profile-1"]
    assert plan["retrieves"] == [True, False, True] and plan["use_local"] == [False, False, False]
    # The pool is always a configured panel: the pre-structured all-packet identity never applies.
    assert plan["legacy_skill_fingerprint"] is False
    assert [rsc.row_plan_retrieves(plan, i) for i in range(3)] == [True, False, True]
    # A plan without the vector is the route's own class, with no native retrieval (F8).
    legacy = {k: v for k, v in plan.items() if k != "retrieves"}
    assert [rsc.row_plan_retrieves(legacy, i) for i in range(4)] == [False, False, True, False]


def test_the_skill_fingerprint_names_native_delivery_only_where_a_row_reads(clean_env):
    from ouroboros.skill_review_cycles import skill_review_contract_fingerprint

    def fingerprint(roster):
        clean_env.setenv(SUBAGENTS_SETTING, roster)
        delivery = rsc.commit_triad_delivery()
        return skill_review_contract_fingerprint(delivery["models"], delivery=delivery,
                                                 required_items=("a",), review_profile="p")

    bare = fingerprint(_roster(_row("t", "m/one")))  # the catalog default is native
    native = fingerprint(_roster(_row("t", "m/one", delivery="native")))
    packet = fingerprint(_roster(_row("t", "m/one", delivery="packet")))
    assert bare == native != packet


# --- effort -------------------------------------------------------------------------------


def test_pool_effort_is_the_row_then_a_compound_slug_then_the_pool_default(clean_env):
    from ouroboros.config import REVIEW_POOL_DEFAULT_EFFORT

    clean_env.setenv("OUROBOROS_EFFORT_REVIEW", "low")  # the lane-era surface setting; not the pool's
    clean_env.setenv(SUBAGENTS_SETTING, _roster(
        _row("pinned", "m/one", effort="medium"),
        _row("cursor-row", "cursor=cursor-grok-4.6-xhigh-fast", kind="agent_session"),
        _row("bare", "m/two"),
        _row("plain-session", "codex=gpt-5.6-sol", kind="agent_session"),
    ))
    assert rsc.commit_triad_delivery()["efforts"] == ["medium", "xhigh", REVIEW_POOL_DEFAULT_EFFORT,
                                                      REVIEW_POOL_DEFAULT_EFFORT]
    assert REVIEW_POOL_DEFAULT_EFFORT == "high"
    assert [s.declared_effort for s in rsc.review_pool_slots()] == ["", "", "", ""]


def test_a_declared_plan_effort_outranks_row_pins_but_not_compound_slugs(clean_env):
    """The envelope's reviewer_effort is an ORDER for this plan alone: the commit
    gate, acceptance and skill-review identities are byte-identical around it."""
    from ouroboros.skill_review_cycles import skill_review_contract_fingerprint
    from ouroboros.tools import plan_review_runtime
    from ouroboros.tools.commit_gate import commit_review_contract_fingerprint

    clean_env.setenv(SUBAGENTS_SETTING, _roster(
        _row("cursor-row", "cursor=cursor-grok-4.6-xhigh-fast", kind="agent_session"),
        _row("plain-row", "codex=gpt-5.6-sol", kind="agent_session"),
        _row("pinned-row", "openai/gpt-5.6-sol", effort="low"),
    ))
    before = (rsc.commit_triad_delivery(), commit_review_contract_fingerprint(),
              skill_review_contract_fingerprint(["m"], delivery=rsc.commit_triad_delivery()))
    declared = plan_review_runtime.plan_review_slots(default_effort="max")
    assert [s.effort for s in declared] == ["xhigh", "max", "max"]  # the pinned `low` row runs the order
    assert [s.declared_effort for s in declared] == ["", "max", "max"]
    assert [s.effort for s in plan_review_runtime.plan_review_slots()] == ["xhigh", "high", "low"]
    assert [s.slot_id for s in declared] == ["cursor-row", "plain-row", "pinned-row"]
    after = (rsc.commit_triad_delivery(), commit_review_contract_fingerprint(),
             skill_review_contract_fingerprint(["m"], delivery=rsc.commit_triad_delivery()))
    assert before == after and before[0]["efforts"] == ["xhigh", "high", "low"]


def test_compound_effort_stabilizes_replay_identity_against_global_drift(clean_env):
    from ouroboros.skill_review_cycles import skill_review_contract_fingerprint

    def fingerprint():
        delivery = rsc.commit_triad_delivery()
        return delivery["efforts"], skill_review_contract_fingerprint(
            delivery["models"], required_items=("manifest_schema",), delivery=delivery)

    clean_env.setenv(SUBAGENTS_SETTING, _roster(_row("c", "cursor=cursor-grok-4.6-xhigh", kind="agent_session")))
    clean_env.setenv("OUROBOROS_EFFORT_REVIEW", "low")
    first = fingerprint()
    clean_env.setenv("OUROBOROS_EFFORT_REVIEW", "high")
    assert fingerprint() == first and first[0] == ["xhigh"]
    clean_env.setenv(SUBAGENTS_SETTING, _roster(_row("c", "cursor=cursor-grok-4.6-max", kind="agent_session")))
    changed = fingerprint()
    assert changed[0] == ["max"] and changed[1] != first[1]


# --- the surfaces -------------------------------------------------------------------------


def test_root_acceptance_runs_every_pool_row_in_order_and_a_child_takes_one(clean_env):
    """A session row beside api rows: acceptance carries them ALL, in the owner's
    order, each with its own delivery — no row is filtered out and no API
    default is substituted. A child task names at most one, by id or handle."""
    clean_env.setenv(SUBAGENTS_SETTING, _MIXED)
    slots = rsc.triad_delivery_slots(role_hint="task acceptance")
    assert [(s.slot_id, s.route.value, s.retrieves) for s in slots] == [
        ("api-critic", "api_chat", True), ("packet-critic", "api_chat", False), ("session-critic", "agent_session", True),
    ]
    assert rsc.child_acceptance_slots(slots)[1]["reason"] == "reviewer_selection_required"
    handles = _handles()
    for slot in slots:
        for selector in (slot.slot_id, handles[slot.slot_id]):
            chosen, refusal = rsc.child_acceptance_slots(slots, selector)
            assert not refusal and [s.slot_id for s in chosen] == [slot.slot_id]
    _, refusal = rsc.child_acceptance_slots(slots, "ghost")
    assert refusal["reason"] == "reviewer_slot_unknown"
    assert [row["delivery"] for row in refusal["reviewer_rows"]] == ["native", "packet", "agent_session"]
    only, none = rsc.child_acceptance_slots(slots[:1])
    assert [s.slot_id for s in only] == ["api-critic"] and none == {}


def test_plan_review_rows_are_the_pool_rows_and_duplicate_models_stay_distinct(tmp_path, clean_env):
    """Two marked rows on the SAME model keep their own ids all the way into the
    raw rows the engine records (identity is CARRIED, never re-derived)."""
    from ouroboros import review_substrate
    from ouroboros.tools import plan_review_runtime

    ran_as: list = []
    findings = [{"item": "x", "verdict": "PASS", "severity": "advisory", "reason": "checked"}]

    def run(request, *, slots, drive_root, llm, usage_ctx=None):
        ran_as.extend(slot.slot_id for slot in slots)
        return SimpleNamespace(actors=[{
            "slot_id": slot.slot_id, "model": slot.model, "status": "ok", "raw_text": json.dumps(findings),
            "usage": {}, "prompt_ref": {}, "response_ref": {},
        } for slot in slots])

    clean_env.setattr(review_substrate, "run_review_request", run)
    clean_env.setattr(plan_review_runtime, "LLMClient", lambda *a, **k: object())
    clean_env.setenv(SUBAGENTS_SETTING, _roster(_row("dup-a", "m/dup", delivery="packet"),
                                                _row("dup-b", "m/dup", delivery="packet")))
    ctx = SimpleNamespace(repo_dir=tmp_path, drive_root=tmp_path, task_id="pool-identity",
                          pending_events=[], drive_logs=lambda: tmp_path)
    slots = plan_review_runtime.plan_review_slots()
    assert [s.slot_id for s in slots] == ["dup-a", "dup-b"]
    raw = asyncio.run(plan_review_runtime.run_plan_review_slots(
        ctx, slots, system_prompt="system prompt", user_content="user content"))
    assert ran_as == ["dup-a", "dup-b"], ran_as
    assert [r.get("model") for r in raw] == ["m/dup", "m/dup"]
    assert [r.get("slot_id") for r in raw] == ["dup-a", "dup-b"]
    assert {r.get("route") for r in raw} == {"api_chat"}
    assert {r.get("host_file_read_attestation") for r in raw} == {"host_assembled_packet"}


def test_execution_records_are_keyed_by_the_catalog_id(clean_env, tmp_path):
    clean_env.setattr(rsc, "_last_execution_path", lambda: tmp_path / "last.json")
    clean_env.setenv(SUBAGENTS_SETTING, _MIXED)
    slots = {slot.slot_id: slot for slot in rsc.review_pool_slots()}
    actors = [SimpleNamespace(slot_id=sid, status="ok", usage={}) for sid in slots]
    rsc.record_reviewer_slot_executions("multi_model_review", actors, slots)
    last = rsc.reviewer_slot_last_executions()
    assert set(last) == {"api-critic", "packet-critic", "session-critic"}
    assert last["session-critic"]["requested"]["session_target"] == "codex=gpt-5.6-sol"
    assert last["session-critic"]["requested"]["profile_id"] == "profile-1"


# --- the composed wave ---------------------------------------------------------------------


def test_a_composed_wave_replaces_the_pool_for_its_readers_only(clean_env):
    from ouroboros.review_substrate import ReviewSlot

    clean_env.setenv(SUBAGENTS_SETTING, _MIXED)
    extra = ReviewSlot(slot_id="named-by-author", model="m/extra", effort="low", route=ReviewRouteKind.API_CHAT,
                       native_retrieval_override=False)
    configured = rsc.review_pool_slots()
    seats = (rsc.PoolSeat(configured[2], ("change", "coupling")), rsc.PoolSeat(extra, ("change",), additional=True))

    assert rsc.composed_pool_seats() is None
    with rsc.composed_review_pool(seats):
        assert rsc.composed_pool_seats() == seats
        assert [s.slot_id for s in rsc.review_pool_slots()] == ["session-critic", "named-by-author"]
        assert [s.role_hint for s in rsc.review_pool_slots(role_hint="author wave")] == ["author wave"] * 2
        assert rsc.commit_triad_delivery()["slot_ids"] == ["session-critic", "named-by-author"]
        assert rsc.commit_triad_delivery()["retrieves"] == [True, False]
    assert rsc.composed_pool_seats() is None
    assert [s.slot_id for s in rsc.review_pool_slots()] == ["api-critic", "packet-critic", "session-critic"]


def test_w1_the_gate_vectors_carry_a_composed_seats_own_parts_and_the_added_bit(clean_env):
    """A ``coupling_only`` seat the author added is asked Part 2 alone and is heard, not
    counted: ``commit_triad_delivery`` projects the composed seats' ``parts`` and
    ``additional`` on the same index as every other vector, so ``seat_vectors`` keeps them
    instead of re-deriving both parts from the seat's delivery."""
    from ouroboros.tools.review_admission import seat_vectors

    clean_env.setenv(SUBAGENTS_SETTING, _MIXED)
    configured = rsc.review_pool_slots()
    seats = (rsc.PoolSeat(configured[1], ("change",)), rsc.PoolSeat(configured[2], ("change", "coupling")),
             rsc.PoolSeat(configured[0], ("coupling",), additional=True))
    with rsc.composed_review_pool(seats):
        plan = seat_vectors(rsc.commit_triad_delivery())
    assert plan["slot_ids"] == ["packet-critic", "session-critic", "api-critic"]
    assert plan["retrieves"] == [False, True, True], "the retrieving critic would be asked both parts by delivery"
    assert plan["parts"] == [("change",), ("change", "coupling"), ("coupling",)]
    assert plan["additional"] == [False, False, True]
    # Outside a composed wave the vectors are the configured pool's: parts by delivery, no added seat.
    pool_plan = seat_vectors(rsc.commit_triad_delivery())
    assert pool_plan["parts"] == [("change", "coupling"), ("change",), ("change", "coupling")]
    assert pool_plan["additional"] == [False, False, False]


# --- ceilings pinned to their owners ------------------------------------------------------------


def test_the_commit_review_ceiling_is_the_catalog_ceiling():
    from ouroboros.tools.review import MAX_MODELS
    from ouroboros.tools.review_multi_model import MAX_MODELS as POOL_MAX

    assert MAX_MODELS == POOL_MAX == MAX_CONFIGURED_SUBAGENTS == 26


# --- the wait card's persistent choice names a catalog row -----------------------------


def test_a_persisted_reviewer_choice_changes_that_catalog_rows_route_and_nothing_else():
    """``reviewer:<id>`` on the wait card is the pool seat's own row: its route
    changes (pin included), while the mark, delivery, effort, the other rows and
    the catalog order are exactly as saved — and no review lanes key is authored."""
    from ouroboros.model_slots import apply_model_role_override

    before = {SUBAGENTS_SETTING: _MIXED, "OUROBOROS_MODEL": "openai/gpt-5.6-sol"}
    saved = apply_model_role_override(before, role="reviewer:packet-critic", model="claudexor::codex=gpt-5.6-luna",
                                      credential_profile_id="pin-2", use_local=False)

    assert before[SUBAGENTS_SETTING] == _MIXED, "the input document is never mutated"
    assert "OUROBOROS_REVIEWER_SLOTS" not in saved
    rows = json.loads(saved[SUBAGENTS_SETTING])["items"]
    assert [row["subagent_id"] for row in rows] == ["api-critic", "packet-critic", "session-critic", "helper"]
    changed = rows[1]
    assert changed["route"] == {"kind": "api_model", "target_id": "claudexor::codex=gpt-5.6-luna",
                                "credential_profile_id": "pin-2"}
    assert changed["delivery"] == "packet" and changed["review_eligible"] is True
    from ouroboros.configured_subagents import normalize_configured_subagents

    untouched = json.loads(normalize_configured_subagents(_MIXED)[1])["items"]
    assert [rows[index] for index in (0, 2, 3)] == [untouched[index] for index in (0, 2, 3)]
    seats = rsc.review_pool_slots(saved)
    assert [(seat.slot_id, seat.model, seat.session_profile) for seat in seats][1] == (
        "packet-critic", "claudexor::codex=gpt-5.6-luna", "pin-2")
    replayed = apply_model_role_override(saved, role="reviewer:packet-critic", model="claudexor::codex=gpt-5.6-luna",
                                         credential_profile_id="pin-2", use_local=False)
    assert replayed == saved, "replaying a saved choice changes nothing"


def test_the_review_commands_main_seat_persists_its_wait_card_as_mains_own_role():
    """``/review`` runs on the direct Main row when no reviewer is named (decision 3A,
    ``deep_self_review.main_review_row`` → ``slot_id == "main"``); its wait card
    carries ``reviewer:main``, which is Main's role — no catalog row is invented and
    the pool is untouched."""
    from ouroboros.model_slots import apply_model_role_override

    before = {SUBAGENTS_SETTING: _MIXED, "OUROBOROS_MODEL": "openai/gpt-5.6-sol", "USE_LOCAL_MAIN": False}
    saved = apply_model_role_override(before, role="reviewer:main", model="owner/local-main",
                                      credential_profile_id="", use_local=True)
    assert (saved["OUROBOROS_MODEL"], saved["USE_LOCAL_MAIN"]) == ("owner/local-main", True)
    assert saved[SUBAGENTS_SETTING] == _MIXED and "OUROBOROS_REVIEWER_SLOTS" not in saved
    assert json.loads(saved["OUROBOROS_MODEL_ACCOUNTS"])["main"] == ""
    pinned = apply_model_role_override(saved, role="reviewer:main", model="claudexor::codex=gpt-5.6-luna",
                                       credential_profile_id="pin-7", use_local=False)
    assert (pinned["OUROBOROS_MODEL"], pinned["USE_LOCAL_MAIN"]) == ("claudexor::codex=gpt-5.6-luna", False)
    assert json.loads(pinned["OUROBOROS_MODEL_ACCOUNTS"])["main"] == "pin-7"


def test_a_persisted_reviewer_choice_for_a_row_that_is_gone_is_a_typed_refusal_naming_the_pool():
    from ouroboros.model_slots import apply_model_role_override

    with pytest.raises(ValueError, match=r"no longer exists \(the review pool is: api-critic, packet-critic, session-critic\)"):
        apply_model_role_override({SUBAGENTS_SETTING: _MIXED}, role="reviewer:retired-critic",
                                  model="openai/gpt-5.6-terra", credential_profile_id="", use_local=False)
    with pytest.raises(ValueError, match=r"the review pool is: empty"):
        apply_model_role_override({SUBAGENTS_SETTING: _roster(_row("helper", "openai/gpt-5.6-luna", marked=False))},
                                  role="reviewer:retired-critic", model="openai/gpt-5.6-terra",
                                  credential_profile_id="", use_local=False)


def test_t3_plan_review_falls_back_to_the_pool_default_effort_and_the_canon_says_so(clean_env):
    """F6: a bare pool row under a plan with no order runs at the pool's
    ``REVIEW_POOL_DEFAULT_EFFORT`` — the lane-era ``OUROBOROS_EFFORT_REVIEW`` tunes
    nothing — and the architecture chapter's strength-axis paragraph plus the
    builder's own docstring name that fallback instead of the retired setting."""
    import pathlib

    from ouroboros.config import REVIEW_POOL_DEFAULT_EFFORT
    from ouroboros.tools.plan_review_runtime import plan_review_slots

    clean_env.setenv("OUROBOROS_EFFORT_REVIEW", "low")
    clean_env.setenv(SUBAGENTS_SETTING, _roster(_row("bare", "m/two"), _row("pinned", "m/one", effort="medium")))
    assert [s.effort for s in plan_review_slots("")] == [REVIEW_POOL_DEFAULT_EFFORT, "medium"]
    assert [s.effort for s in plan_review_slots("xhigh")] == ["xhigh", "xhigh"]

    chapter = (pathlib.Path(__file__).resolve().parents[1] / "docs" / "architecture" / "06-agent-core.md"
               ).read_text(encoding="utf-8")
    paragraph = chapter[chapter.index("The ONE caller-facing strength axis"):].split("\n\n", 1)[0]
    assert "`REVIEW_POOL_DEFAULT_EFFORT`" in paragraph
    assert "then the owner's `OUROBOROS_EFFORT_REVIEW`" not in paragraph
    assert "REVIEW_POOL_DEFAULT_EFFORT" in plan_review_slots.__doc__
    assert "owner's review-effort setting" not in plan_review_slots.__doc__
