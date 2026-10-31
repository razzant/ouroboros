"""Legacy question addresses bind only from positive history evidence; a wait with no automatic
route keeps Main's real retry."""
import json
import time
from types import SimpleNamespace

import pytest

from ouroboros import obligations, model_wait, startup_migrations as migrations
from ouroboros.gateway.main_decisions import main_decisions_snapshot
from ouroboros.memory import Memory
from ouroboros.owner_quiz import quiz_states
from ouroboros.utils import append_jsonl
from tests.test_main_decisions import older_body_ask, project_task
from tests.test_model_wait import _action_for, live_wait as wait_fixture
from tests.test_llm_claudexor import setup as gateway_fixture
from tests.test_quota_rotation_fallback import MODEL, ROUTE_B, _history, _vendor_quota, _waits, result

setup, live_wait = gateway_fixture, wait_fixture


def _legacy(root, *ids):
    project = project_task(root)
    for qid in ids:
        older_body_ask(root, "task", qid, chat_id=1)
    return project


def _published(root, qid, chat_id):
    append_jsonl(root / "logs/chat.jsonl", {"direction": "out", "type": "quiz",
                 "task_id": "task", "chat_id": chat_id, "quiz": {"quiz_id": qid}})


def test_mixed_history_keeps_unique_publications_and_specific_unknowns(tmp_path, monkeypatch):
    project = _legacy(tmp_path, "known", "conflict", "missing")
    for qid, chat_id in [("known", project["chat_id"]), ("known", project["chat_id"]),
                         ("conflict", project["chat_id"]), ("conflict", 1)]:
        _published(tmp_path, qid, chat_id)
    append_jsonl(tmp_path / "logs/chat.jsonl", {"direction": "out", "type": "quiz", "quiz": "malformed"})
    with (tmp_path / "logs/chat.jsonl").open("a") as stream:
        stream.write("unrelated malformed row\n")
    (tmp_path / "memory").mkdir(exist_ok=True)
    (tmp_path / "memory/dialogue_blocks.json").write_text(json.dumps([
        {"gap_id": "older-unrelated-consolidation", "content": "[MEMORY GAP]"}]))
    assert migrations.prepare_startup_state(tmp_path)["question_publications"] == {"bound": 1, "unknown": 2}
    blocks = quiz_states(tmp_path, "task")
    assert blocks["known"]["publication_chat_id"] == project["chat_id"]
    assert blocks["known"]["chat_id"] == 1 and blocks["known"]["publication_evidence"] == "history"
    assert all("publication_chat_id" not in blocks[qid] for qid in ("conflict", "missing"))
    unknowns = obligations.members(tmp_path, "unknowns")
    assert unknowns["question:task:conflict"]["reason"] == "publication_address_ambiguous"
    assert unknowns["question:task:missing"]["reason"] == "publication_address_missing"
    assert {gap["kind"] for gap in unknowns["question:task:missing"]["history_gaps"]} == {
        "jsonl_malformed", "durable_consolidation_gap"}
    snapshot = main_decisions_snapshot(tmp_path, [], {})
    assert [row["quiz_id"] for row in snapshot["current_questions"]] == ["known"]
    assert {gap["quiz_id"] for gap in snapshot["current_questions_gaps"]} == {"conflict", "missing"}
    monkeypatch.setattr(Memory, "read_chat_generations", lambda *_a, **_kw: pytest.fail("permanent gap replay"))
    assert migrations.prepare_startup_state(tmp_path)["question_publications"] == {}
    assert main_decisions_snapshot(tmp_path, [], {}) == snapshot


@pytest.mark.parametrize("failure", ["unstable", "unreadable", "exception"])
def test_transient_history_repairs_in_maintenance_and_clears_only_its_unknown(tmp_path, monkeypatch, failure):
    project = _legacy(tmp_path, "older")
    _published(tmp_path, "older", project["chat_id"])
    original = Memory.read_chat_generations
    reads = []

    def unavailable(self, **kwargs):
        reads.append(1)
        if failure == "exception":
            raise OSError("temporarily unreadable")
        rows, coverage = original(self, **kwargs)
        if failure == "unstable":
            return rows, {**coverage, "snapshot_stable": False}
        return [], {**coverage, "gaps": [{"kind": "generation_unreadable"}]}

    monkeypatch.setattr(Memory, "read_chat_generations", unavailable)
    migrations.prepare_startup_state(tmp_path)
    assert "publication_chat_id" not in quiz_states(tmp_path, "task")["older"]
    assert migrations.watermarks(tmp_path)["question_publication_retry"] == {"attempts": 1, "state": "pending"}
    for _ in range(3):
        assert not main_decisions_snapshot(tmp_path, [], {})["current_questions_complete"]
    assert reads == [1], "GET never repairs or scans history"
    obligations.add(tmp_path, "unknowns", "unrelated", {"reason": "retained uncertainty"})
    monkeypatch.setattr(Memory, "read_chat_generations", original)
    assert migrations.prepare_startup_state(tmp_path)["question_publications"] == {}
    from ouroboros.store_import import run_legacy_reconcile
    assert run_legacy_reconcile(tmp_path)["question_publications"] == {"bound": 1, "unknown": 0}
    assert quiz_states(tmp_path, "task")["older"]["publication_chat_id"] == project["chat_id"]
    unknowns = obligations.members(tmp_path, "unknowns")
    assert "question:task:older" not in unknowns and "unrelated" in unknowns
    assert migrations.watermarks(tmp_path)["question_publication_retry"] is None
    monkeypatch.setattr(Memory, "read_chat_generations", lambda *_a, **_kw: pytest.fail("repeat scan"))
    assert migrations.prepare_startup_state(tmp_path)["question_publications"] == {}


def test_persistent_unreadable_history_stops_boot_replay_until_explicit_rebuild(tmp_path, monkeypatch):
    project = _legacy(tmp_path, "older")
    _published(tmp_path, "older", project["chat_id"])
    original = Memory.read_chat_generations
    monkeypatch.setattr(Memory, "read_chat_generations", lambda *_a, **_kw: (
        [], {"snapshot_stable": True, "gaps": [{"kind": "generation_unreadable"}]}))
    assert migrations.prepare_startup_state(tmp_path)["question_publications"] == {"bound": 0, "unknown": 1}
    assert migrations.prepare_startup_state(tmp_path)["question_publications"] == {}
    from ouroboros.store_import import run_legacy_reconcile
    assert run_legacy_reconcile(tmp_path)["question_publications"] == {"bound": 0, "unknown": 1}
    assert migrations.watermarks(tmp_path)["question_publication_retry"] == {"attempts": 2, "state": "rebuild_required"}
    monkeypatch.setattr(Memory, "read_chat_generations", lambda *_a, **_kw: pytest.fail("unbounded boot replay"))
    assert migrations.prepare_startup_state(tmp_path)["question_publications"] == {}
    assert not main_decisions_snapshot(tmp_path, [], {})["current_questions_complete"]
    monkeypatch.setattr(Memory, "read_chat_generations", original)
    migrations.prepare_startup_state(tmp_path, rebuild=True)
    assert quiz_states(tmp_path, "task")["older"]["publication_chat_id"] == project["chat_id"]


@pytest.mark.parametrize("alternate", [False, True])
def test_no_reset_wait_exposes_main_retry_until_an_automatic_route_can_resume(live_wait, monkeypatch, alternate):
    root, gateway, client, _owner, events, decide = live_wait
    project = project_task(root, task_id="task-one")
    gateway.results = [_vendor_quota(reset=False), _vendor_quota(reset=False), result(route=ROUTE_B) if alternate else result()]
    gateway.dispatch = ["response_received"] * 3
    clock = SimpleNamespace(now=time.monotonic())
    seen = []

    def activity():
        waits = _waits(events)
        return {"activity_id": "task-one", "project_id": project["id"], "phase": "working",
                "task_attempt": 1, "model_waits": {waits[-1]["wait_id"]: waits[-1]}}

    def sleep(seconds):
        clock.now += seconds
        assert len(gateway.accepted_operations) == 2, "metadata checking never resends the refused operation"
        [attention] = main_decisions_snapshot(root, [activity()], {})["main_attention"]
        [wait] = attention["model_waits"].values()
        assert wait["requires_owner_action"] is True and wait["auto_continue"] is True
        if not alternate:
            assert decide(_action_for({**wait, "task_id": "task-one"}, "retry")).status_code == 202

    def catalog(*_a, **_kw):
        account = "account-b" if alternate and seen else "account-a"
        seen.append(account)
        return {"source": "codex", "credentialProfileId": account, "models": [{"id": "exact-model"}]}

    monkeypatch.setattr(model_wait, "time", SimpleNamespace(monotonic=lambda: clock.now, time=time.time, sleep=sleep))
    monkeypatch.setattr(client, "claudexor_model_catalog", catalog)
    client.chat(_history(), MODEL, model_role="main", cache_affinity="affinity")
    assert _waits(events)[-1]["resolution"] == ("resource_available" if alternate else "retry_requested")
    assert _waits(events)[-1]["requires_owner_action"] is False
    assert main_decisions_snapshot(root, [activity()], {})["main_attention"] == []
    assert seen == (["account-a", "account-b"] if alternate else ["account-a"])
    assert len(gateway.accepted_operations) == 3
