"""#1334 / #1116 / #1335: reviewer delivery defaults and the one-time notices.

A pool row's saved ``delivery`` is the one ``retrieves`` fact every consumer
reads (F8), the triad consumer hands a native row the compact work order rather
than the assembled packet, a compatible-only install reviews on Main (the
factory pool rows), and an upgraded install hears once, factually, about any
finite task limit it still runs under. The pool's own delivery/order/quorum
tests live in ``tests/test_review_pool.py``.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from ouroboros import reviewer_slot_config as rsc
from tests.test_owner_settings_write_seam import isolated_settings  # noqa: F401  (fixture)


@pytest.fixture
def clean_env(monkeypatch):
    for key in ("OUROBOROS_SUBAGENTS", "OUROBOROS_REVIEW_MODELS", "OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                "OPENAI_COMPATIBLE_BASE_URL", "OPENAI_BASE_URL", "OUROBOROS_MODEL", "OUROBOROS_MODEL_LIGHT",
                "MINIMAX_API_KEY", "DEEPSEEK_API_KEY", "ZAI_API_KEY", "CLOUDRU_FOUNDATION_MODELS_API_KEY",
                "GIGACHAT_CREDENTIALS", "USE_LOCAL_MAIN"):
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


# --- the saved delivery field --------------------------------------------------------


def test_the_triad_multi_model_row_runs_its_native_delivery(monkeypatch, tmp_path):
    """The commit/skill triad consumer hands a native row the compact work order
    and the native episode — never the assembled packet messages."""
    import ouroboros.review_substrate as rs
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.tools import review as review_module
    from ouroboros.tools import review_multi_model as rmm

    seen = []

    def run(request, *, slots, **_kw):
        seen.append((request, slots[0]))
        return SimpleNamespace(actors=[{"slot_id": slots[0].slot_id, "status": "ok", "raw_text": "[]"}])

    monkeypatch.setattr(rs, "run_review_request", run)
    monkeypatch.setattr(review_module, "review_drive_root", lambda ctx: tmp_path)
    plan = {"slot_ids": ["n"], "subagent_ids": [""], "retrieves": [True], "session_profiles": [""],
            "efforts": [""], "session_targets": [""], "use_local": [False]}
    asyncio.run(rmm._multi_model_review_async(
        "content", "", ["m/one"], None, routes=[ReviewRouteKind.API_CHAT], row_plan=plan,
        session_task="WORK ORDER", session_root="/repo"))
    (request, slot), = seen
    assert slot.native_retrieval and request.messages == [] and request.session_task.startswith("WORK ORDER")


def test_a_pool_rows_delivery_is_native_unless_the_row_says_packet(monkeypatch, isolated_settings):  # noqa: F811
    """The catalog's reading of delivery, through the owner's save and the pool's own
    endpoint: an api row saved without ``delivery`` reads natively (the catalog
    default — the lane reader's packet default does not carry over); a row saved
    with ``delivery: packet`` packs the brief; a session row always reads (F8)."""
    from starlette.testclient import TestClient

    from ouroboros import server_maintenance
    from ouroboros.gateway import settings as settings_mod
    from tests.test_owner_settings_write_seam import _settings_app

    catalog = {"enabled": True, "items": [
        {"subagent_id": "bare", "recommended_use": "Reads the work itself.", "review_eligible": True,
         "route": {"kind": "api_model", "target_id": "openai/gpt-5.6-sol"}},
        {"subagent_id": "packed", "recommended_use": "Reads the brief.", "review_eligible": True,
         "route": {"kind": "api_model", "target_id": "openai/gpt-5.6-luna"}, "delivery": "packet"},
        {"subagent_id": "session", "recommended_use": "Reads the repository.", "review_eligible": True,
         "route": {"kind": "agent_session", "target_id": "cursor=gpt-5.6-sol-xhigh"}},
    ]}
    saved = TestClient(_settings_app(monkeypatch, isolated_settings)).post(
        "/api/settings", json={"OUROBOROS_SUBAGENTS": catalog})
    assert saved.status_code == 200, saved.text
    stored = json.loads(isolated_settings.read_text(encoding="utf-8"))["OUROBOROS_SUBAGENTS"]
    assert "delivery" not in json.loads(stored)["items"][0], "an absent field stays absent on disk"

    monkeypatch.setenv("OUROBOROS_SUBAGENTS", stored)  # the applied settings plane the pool reads
    monkeypatch.setattr(rsc, "reviewer_slot_last_executions", lambda: {})
    monkeypatch.setattr(server_maintenance, "review_pool_migration_records", lambda: {})
    monkeypatch.setattr(settings_mod, "_review_pool_costs", lambda items, env: {})
    body = json.loads(asyncio.run(settings_mod.api_review_pool(None)).body)
    assert [(row["subagent_id"], row["delivery"]) for row in body["pool"]] == [
        ("bare", "native"), ("packed", "packet"), ("session", "session")]
    # The same fact on the rows every review surface runs.
    assert [(slot.slot_id, slot.retrieves) for slot in rsc.review_pool_slots()] == [
        ("bare", True), ("packed", False), ("session", True)]


# --- #1116: an OpenAI-compatible-only install ------------------------------------------


def test_a_compatible_only_install_reviews_on_main(clean_env):
    from ouroboros.deep_self_review import deep_review_route
    from ouroboros.subscription_install_presets import factory_review_rows

    doc = {"OPENAI_COMPATIBLE_BASE_URL": "https://llm.example/v1", "OUROBOROS_MODEL": "openai-compatible::glm-5.3"}
    for key, value in doc.items():
        clean_env.setenv(key, value)
    # The factory POOL is the shipped panel's three seats on the one reachable
    # model (three independent runs of Main, quorum 2 of 3 — what this install
    # ran); a configured catalog can select a different explicit panel.
    assert [row["route"]["target_id"] for row in factory_review_rows(doc)] == ["openai-compatible::glm-5.3"] * 3
    # The deep self-review runs on Main too (decision 3A), on that route.
    clean_env.setenv("OPENAI_COMPATIBLE_API_KEY", "test-only-key")
    assert deep_review_route() == ("", "openai-compatible::glm-5.3")
    from ouroboros.reviewer_slot_config import review_pool_slots
    from tests.review_pool_rosters import set_review_pool

    models = ["openai-compatible::a", "openai-compatible::b"]
    set_review_pool(clean_env, models)
    assert [slot.model for slot in review_pool_slots()] == models


@pytest.mark.parametrize("other_key", ["OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_BASE_URL"])
def test_another_remote_route_keeps_the_existing_defaults(clean_env, other_key):
    from ouroboros.settings_defaults import OPENROUTER_REVIEW_DEFAULTS
    from ouroboros.subscription_install_presets import factory_review_rows

    doc = {"OPENAI_COMPATIBLE_BASE_URL": "https://llm.example/v1", "OUROBOROS_MODEL": "openai-compatible::glm-5.3",
           other_key: "x-key" if other_key != "OPENAI_BASE_URL" else "https://base.example"}
    models = [row["route"]["target_id"] for row in factory_review_rows(doc)]
    assert models != ["openai-compatible::glm-5.3"] * 3
    if other_key == "OPENROUTER_API_KEY":
        assert models == list(OPENROUTER_REVIEW_DEFAULTS["triad"])


# --- one-time notices (#1334, #1335) --------------------------------------------------


@pytest.mark.parametrize(("document", "settings", "expected"), [
    (None, {"OUROBOROS_MAX_ROUNDS": "unlimited", "OUROBOROS_TASK_ABS_CEILING_SEC": "unlimited"}, []),  # fresh
    ({}, {"OUROBOROS_MAX_ROUNDS": 200, "OUROBOROS_TASK_ABS_CEILING_SEC": 21600},
     [("OUROBOROS_MAX_ROUNDS", 200, "document_absent_key"),
      ("OUROBOROS_TASK_ABS_CEILING_SEC", 21600, "document_absent_key")]),
    ({"OUROBOROS_MAX_ROUNDS": "unlimited", "OUROBOROS_TASK_ABS_CEILING_SEC": "unlimited"}, {}, []),
    ({"OUROBOROS_MAX_ROUNDS": 150, "OUROBOROS_TASK_ABS_CEILING_SEC": "unlimited"}, {},
     [("OUROBOROS_MAX_ROUNDS", 150, "saved")]),
    ({"OUROBOROS_MAX_ROUNDS": "abc", "OUROBOROS_TASK_ABS_CEILING_SEC": "unlimited"}, {"OUROBOROS_MAX_ROUNDS": 200},
     [("OUROBOROS_MAX_ROUNDS", 200, "saved_invalid")]),
    ({"OUROBOROS_TASK_ABS_CEILING_SEC": "unlimited"}, {"OUROBOROS_MAX_ROUNDS": 300},
     [("OUROBOROS_MAX_ROUNDS", 300, "env")]),  # only the launch environment explains 300
    (None, {"OUROBOROS_MAX_ROUNDS": 120, "OUROBOROS_TASK_ABS_CEILING_SEC": "unlimited"},
     [("OUROBOROS_MAX_ROUNDS", 120, "env")]),
])
def test_finite_bound_facts_name_only_what_the_document_and_env_show(document, settings, expected):
    from ouroboros.upgrade_notices import optional_bound_facts, optional_bounds_notice

    facts = optional_bound_facts(document, settings)
    assert [(f["key"], f["value"], f["origin"]) for f in facts] == expected
    text = optional_bounds_notice(facts)
    assert bool(text) == bool(expected)
    if expected:
        assert "nothing was changed here" in text and "chose" not in text


def test_the_classifier_ignores_the_process_environment_startup_projects_settings_into(monkeypatch):
    from ouroboros.upgrade_notices import optional_bound_facts

    # apply_settings_to_env has already exported the document default before the notice runs.
    monkeypatch.setenv("OUROBOROS_MAX_ROUNDS", "200")
    facts = optional_bound_facts({}, {"OUROBOROS_MAX_ROUNDS": 200, "OUROBOROS_TASK_ABS_CEILING_SEC": "unlimited"})
    assert [(f["value"], f["origin"]) for f in facts] == [(200, "document_absent_key")]


@pytest.mark.serial
@pytest.mark.parametrize(("document", "launch_env", "expected", "origin", "raw"), [
    ({"OUROBOROS_MAX_ROUNDS": None}, "77", 200, "saved_invalid", None),
    ({"OUROBOROS_MAX_ROUNDS": ""}, "77", 200, "saved_invalid", ""),
    ({"OUROBOROS_MAX_ROUNDS": "bad"}, "77", 200, "saved_invalid", "bad"),
    ({"OUROBOROS_MAX_ROUNDS": 42}, "77", 42, "saved", 42),
    ({}, None, 200, "document_absent_key", 200),
    ({}, "77", 77, "env", 77),
    (None, "77", 77, "env", 77),
    ({"OUROBOROS_MAX_ROUNDS": "unlimited"}, "77", None, None, None),
    ({"OUROBOROS_TASK_ABS_CEILING_SEC": None}, "60", 21600, "saved_invalid", None),
    ({"OUROBOROS_TASK_ABS_CEILING_SEC": ""}, "60", 21600, "saved_invalid", ""),
    ({"OUROBOROS_TASK_ABS_CEILING_SEC": 60}, None, 300, "saved", 60),
    ({}, "60", 300, "env", 60),
])
def test_startup_notice_matches_loaded_runtime_bounds_and_raw_presence(
        monkeypatch, tmp_path, document, launch_env, expected, origin, raw):
    from ouroboros import config as cfg, upgrade_notices as notices
    from ouroboros.settings_integrity import task_settings_scope, task_settings_snapshot
    from ouroboros.utils import iter_jsonl_chain_objects
    from supervisor import message_bus as bus, state as ss

    lifetime = "OUROBOROS_TASK_ABS_CEILING_SEC"
    rounds = "OUROBOROS_MAX_ROUNDS"
    key = lifetime if (document and lifetime in document) or launch_env == "60" else rounds
    other = rounds if key == lifetime else lifetime
    path = tmp_path / "settings.json"
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    monkeypatch.setattr(cfg, "SETTINGS_PATH", path)
    for setting in cfg.SETTINGS_DEFAULTS:
        monkeypatch.delenv(setting, raising=False)
    if launch_env is not None:
        monkeypatch.setenv(key, launch_env)
    monkeypatch.setenv(other, "unlimited")
    if document is not None:
        path.write_text(json.dumps({other: "unlimited", **document}), encoding="utf-8")
    before = path.read_bytes() if path.exists() else None
    for name, value in {"DRIVE_ROOT": tmp_path, "STATE_PATH": tmp_path / "state/state.json",
                        "STATE_LAST_GOOD_PATH": tmp_path / "state/state.last_good.json",
                        "STATE_LOCK_PATH": tmp_path / "locks/state.lock"}.items():
        monkeypatch.setattr(ss, name, value)
    monkeypatch.setattr(bus, "DATA_DIR", tmp_path)
    from ouroboros.startup_migrations import prepare_startup_state
    prepare_startup_state(tmp_path)
    monkeypatch.setattr(bus, "get_bridge", lambda: None)
    ss.save_state({"owner_chat_id": 7, "owner_id": 1})

    settings = cfg.load_settings()
    projected = {}
    cfg.apply_settings_to_env(settings, environ=projected)
    with task_settings_scope(task_settings_snapshot(settings, projected)):
        effective = cfg.get_task_abs_ceiling_sec() if key == lifetime else cfg.get_max_rounds()
        assert effective == expected
        notices.startup_upgrade_notices(settings)
        notices.startup_upgrade_notices(settings)
    assert (path.read_bytes() if path.exists() else None) == before
    facts = notices.optional_bound_facts(notices._raw_settings_document(), settings)
    rows = [r for r in iter_jsonl_chain_objects(tmp_path / "logs/chat.jsonl")
            if r.get("type") == "optional_bounds_notice"]
    if expected is None:
        assert facts == [] and rows == []
        return
    assert facts == [{"key": key, "value": effective, "origin": origin, "raw": raw}]
    assert len(rows) == 1 and rows[0]["direction"] == "system"
    text = rows[0]["text"]
    assert f"= {effective} {'seconds' if key == lifetime else 'rounds'}" in text
    assert ("key is absent" in text) == (origin == "document_absent_key")
    assert ("finite fallback" in text) == (origin == "saved_invalid")
    if key == lifetime and raw == 60:
        assert "as 60; the runtime minimum applies" in text
    assert "nothing was changed here" in text and "chose" not in text


@pytest.fixture
def notice_world(monkeypatch, tmp_path):
    import supervisor.message_bus as bus
    import supervisor.state as sstate
    from ouroboros import upgrade_notices

    monkeypatch.setattr(bus, "DATA_DIR", tmp_path)
    from ouroboros.startup_migrations import prepare_startup_state
    prepare_startup_state(tmp_path)
    state = {"owner_chat_id": 7}
    sent = []

    def update(fn):
        fn(state)

    monkeypatch.setattr(sstate, "load_state", lambda: dict(state))
    monkeypatch.setattr(sstate, "update_state", update)
    monkeypatch.setattr(bus, "send_with_budget", lambda chat, text, **kw: sent.append((chat, text, kw)))
    monkeypatch.setattr(upgrade_notices, "_raw_settings_document", lambda: {})
    monkeypatch.delenv("OUROBOROS_MAX_ROUNDS", raising=False)
    monkeypatch.delenv("OUROBOROS_TASK_ABS_CEILING_SEC", raising=False)
    return SimpleNamespace(state=state, sent=sent, notices=upgrade_notices)


def test_an_upgraded_untouched_install_hears_each_notice_once(notice_world):
    # Which reviewers run is the review-pool migration's own report
    # (server_maintenance._startup_review_pool_notice), not an upgrade notice.
    notice_world.notices.startup_upgrade_notices({})
    assert [kw["system_type"] for _chat, _text, kw in notice_world.sent] == ["optional_bounds_notice"]
    assert all(kw["require_write"] and kw["role"] == "system" for _c, _t, kw in notice_world.sent)
    assert "200 rounds" in notice_world.sent[0][1] and "21600 seconds" in notice_world.sent[0][1]
    notice_world.notices.startup_upgrade_notices({})
    assert len(notice_world.sent) == 1  # never repeated
    assert "optional_bounds_notified" in notice_world.state
    assert not hasattr(notice_world.notices, "REVIEWER_DEFAULT_NOTICE")


def test_no_owner_chat_or_a_failed_write_leaves_the_notice_owed(notice_world, monkeypatch):
    import supervisor.message_bus as bus

    notice_world.state["owner_chat_id"] = 0
    notice_world.notices.startup_upgrade_notices({})
    assert notice_world.sent == [] and "optional_bounds_notified" not in notice_world.state
    notice_world.state["owner_chat_id"] = 7
    monkeypatch.setattr(bus, "send_with_budget", lambda *a, **k: (_ for _ in ()).throw(OSError("disk")))
    notice_world.notices.startup_upgrade_notices({})
    assert "optional_bounds_notified" not in notice_world.state


@pytest.mark.parametrize("quality", ["unavailable", "recovered", "recovered_transient"])
def test_notice_never_uses_unconfirmed_display_owner(notice_world, quality):
    notice_world.state["_state_read"] = {"quality": quality, "source": "backup",
                                         "unconfirmed": ["owner_chat_id"]}
    notice_world.notices.startup_upgrade_notices({})
    assert not notice_world.sent
    assert "optional_bounds_notified" not in notice_world.state


def test_notice_bookkeeping_preserves_recovered_control_uncertainty(monkeypatch, tmp_path):
    from ouroboros import upgrade_notices as notices
    from supervisor import message_bus as bus, state as ss

    for name, path in {"DRIVE_ROOT": tmp_path, "STATE_PATH": tmp_path / "state/state.json",
                       "STATE_LAST_GOOD_PATH": tmp_path / "state/state.last_good.json",
                       "STATE_LOCK_PATH": tmp_path / "locks/state.lock"}.items():
        monkeypatch.setattr(ss, name, path)
    monkeypatch.setattr(bus, "DATA_DIR", tmp_path)
    from ouroboros.startup_migrations import prepare_startup_state
    prepare_startup_state(tmp_path)
    monkeypatch.setattr(bus, "get_bridge", lambda: SimpleNamespace(send_message=lambda *a, **kw: None))
    monkeypatch.setattr(notices, "_raw_settings_document", lambda: {})
    ss.save_state({"owner_chat_id": 7, "owner_id": 1, "bg_consciousness_enabled": True})
    ss.STATE_PATH.write_bytes(b'{broken')
    before = ss.load_state()
    assert ss.control_value(before, "owner_chat_id") == (True, 7)
    assert ss.control_value(before, "bg_consciousness_enabled") == (False, None)
    notices.startup_upgrade_notices({})
    after = ss.load_state()
    assert after[notices.OPTIONAL_BOUNDS_NOTICE_KEY]
    assert ss.control_value(after, "bg_consciousness_enabled") == (False, None)
    assert ss.control_value(after, "owner_chat_id") == (True, 7)


@pytest.mark.parametrize("failure", ["chat_append", "state_marker", "bridge"])
def test_notice_recovers_through_real_chat_and_state_writers(monkeypatch, tmp_path, failure):
    from ouroboros import upgrade_notices as notices
    from ouroboros.utils import iter_jsonl_chain_objects
    from supervisor import message_bus as bus, state as ss

    for name, path in {"DRIVE_ROOT": tmp_path, "STATE_PATH": tmp_path / "state/state.json",
                       "STATE_LAST_GOOD_PATH": tmp_path / "state/state.last_good.json",
                       "STATE_LOCK_PATH": tmp_path / "locks/state.lock"}.items():
        monkeypatch.setattr(ss, name, path)
    monkeypatch.setattr(bus, "DATA_DIR", tmp_path)
    from ouroboros.startup_migrations import prepare_startup_state
    prepare_startup_state(tmp_path)
    monkeypatch.setattr(notices, "_raw_settings_document", lambda: {"OUROBOROS_MAX_ROUNDS": 150,
                        "OUROBOROS_TASK_ABS_CEILING_SEC": "unlimited"})
    ss.save_state({"owner_chat_id": 7, "owner_id": 1})
    real_update, real_append = ss.update_state, bus.append_jsonl
    delivered = []
    failed = False

    def fail_once():
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("injected persistence/delivery failure")

    def update(fn):
        if failure == "state_marker":
            fail_once()
        return real_update(fn)

    def append(*a, **kw):
        if failure == "chat_append":
            fail_once()
        return real_append(*a, **kw)

    def send(*a, **kw):
        if failure == "bridge":
            fail_once()
        delivered.append(a)

    monkeypatch.setattr(ss, "update_state", update)
    monkeypatch.setattr(bus, "append_jsonl", append)
    monkeypatch.setattr(bus, "get_bridge", lambda: SimpleNamespace(send_message=send))
    settings = {}
    notices.startup_upgrade_notices(settings)
    assert not ss.load_state().get(notices.OPTIONAL_BOUNDS_NOTICE_KEY)
    notices.startup_upgrade_notices(settings)
    notices.startup_upgrade_notices(settings)
    assert ss.load_state()[notices.OPTIONAL_BOUNDS_NOTICE_KEY]
    rows = [r for r in iter_jsonl_chain_objects(tmp_path / "logs/chat.jsonl") if r.get("type") == "optional_bounds_notice"]
    assert len(rows) == 1 and "150 rounds" in rows[0]["text"]
    # Durable web history is the owner delivery witness. External bridge loss
    # does not erase that row or require another visible notice on restart.
    assert len(delivered) == (0 if failure == "bridge" else 1)

