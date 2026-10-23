"""Phase 5A: destructive learning sees complete source state or abstains."""

from __future__ import annotations

import json
import pathlib

from tests import _dialogue as dialogue_test

_RECENT = json.dumps({"chat_id": 1, "direction": "in", "text": "complete recent chat"})


def _chain(tmp_path, rows):
    """The older install's live chat chain, written once (a legacy cursor signs these exact bytes);
    the dialogue import reads it before any context is built."""
    chat = tmp_path / "logs" / "chat.jsonl"
    if not chat.exists():
        dialogue_test.seed(tmp_path, live={"chat": [row + "\n" for row in rows]})
    return chat


def test_closed_objective_before_old_horizon_reaches_chooser(tmp_path, monkeypatch):
    from ouroboros import post_task_evolution as pte

    state = tmp_path / "state"
    state.mkdir(parents=True)
    old_objective = "OLD CLOSED OBJECTIVE MUST NOT BE PROMOTED AGAIN"
    rows = [{"task_id": "old", "kind": "cycle_outcome", "cycle_outcome": "absorbed",
             "campaign_objective": old_objective}]
    rows.extend({"task_id": f"new-{i}", "kind": "cycle_outcome", "cycle_outcome": "absorbed",
                 "campaign_objective": f"new objective {i}"} for i in range(230))
    (state / "evolution_checkpoints.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows), encoding="utf-8",
    )
    captured = {}

    def fake_chat(*args, **kwargs):
        captured["prompt"] = kwargs["messages"][0]["content"]
        return ({"content": '{"promote": false, "objective": ""}'}, {})

    monkeypatch.setattr("ouroboros.llm_observability.chat_observed", fake_chat)
    monkeypatch.setattr(pte, "_active_campaign_objective", lambda: "")
    env = type("Env", (), {"drive_root": tmp_path})()
    decision = pte._decide_promotion(env, {"id": "root"}, None, object(), force=False)
    assert decision and decision["promote"] is False
    assert old_objective in captured["prompt"]


def test_closed_objective_unavailable_abstains_before_chooser(tmp_path, monkeypatch):
    from ouroboros import post_task_evolution as pte

    state = tmp_path / "state"
    state.mkdir(parents=True)
    ledger = state / "evolution_checkpoints.jsonl"
    ledger.write_text("{}\n", encoding="utf-8")
    real_read_text = pathlib.Path.read_text

    def unreadable(self, *args, **kwargs):
        if self == ledger:
            raise PermissionError("ledger unavailable")
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "read_text", unreadable)
    called = []

    def chooser(*args, **kwargs):
        called.append(True)
        return ({"content": '{"promote": true, "objective": "unsafe"}'}, {})

    monkeypatch.setattr("ouroboros.llm_observability.chat_observed", chooser)
    env = type("Env", (), {"drive_root": tmp_path})()
    assert pte._decide_promotion(env, {"id": "root"}, None, object(), force=False) is None
    assert called == []


def _wake_context(tmp_path, *, chat_rows=None):
    """The system text a consciousness wake-up gets: Main's own builder (build_llm_messages)
    over a wake-shaped task and the real repository prompts — the typed gap facts the
    context builders disclose (recent-chat, dialogue-history and schedule digests) are
    what the wake reads; nothing consciousness-specific is layered on top."""
    from ouroboros.context import build_llm_messages
    from ouroboros.memory import Memory

    repo_dir = pathlib.Path(__file__).parents[1]

    class FakeEnv:
        def drive_path(self, p):
            return tmp_path / p

        def repo_path(self, p):
            return repo_dir / p

        @property
        def repo_dir(self):
            return repo_dir

        @property
        def drive_root(self):
            return tmp_path

    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "state").mkdir(parents=True, exist_ok=True)
    (tmp_path / "memory").mkdir(parents=True, exist_ok=True)
    if not (tmp_path / "state" / "state.json").exists():
        (tmp_path / "state" / "state.json").write_text("{}", encoding="utf-8")
    _chain(tmp_path, chat_rows if chat_rows is not None else [_RECENT])
    dialogue_test.import_all(tmp_path)
    task = {"id": "wake1", "type": "task", "text": "[Wake-up · heartbeat]", "_is_direct_chat": True,
            "metadata": {"initiator": "consciousness", "usage_category": "consciousness",
                         "consciousness_autonomy": "act"}}
    messages, _cap = build_llm_messages(env=FakeEnv(), memory=Memory(drive_root=tmp_path, repo_dir=repo_dir), task=task)
    return "\n\n".join(block["text"] for block in messages[0]["content"])


def _write_schedules(tmp_path, count):
    tasks = [{
        "id": f"schedule-{idx}", "name": f"schedule {idx}", "enabled": True,
        "trigger": {"type": "cron", "expr": "0 * * * *"},
    } for idx in range(count)]
    (tmp_path / "state").mkdir(parents=True, exist_ok=True)
    (tmp_path / "state" / "scheduled_tasks.json").write_text(
        json.dumps({"tasks": tasks}), encoding="utf-8",
    )


def test_wake_context_reads_main_beside_a_malformed_chat_line_and_the_reader_discloses_it(tmp_path):
    """A torn chat line takes no position and breaks nothing: the wake reads Main as its own room, the
    one complete row verbatim and counted in its facts; the deliberate reader discloses the malformed line."""
    from ouroboros.memory import Memory

    context = _wake_context(tmp_path, chat_rows=[
        json.dumps({"chat_id": 1, "direction": "in", "text": "complete recent chat"}),
        '{"direction":"in","text":"broken"',
    ])
    room = context.split("\n## This room (Main) — head ", 1)[1].split("\n## ", 1)[0]  # the heading, not the books' mention
    assert "people 1, mine 0" in room and "complete recent chat" in room.split("### Open conversation", 1)[1]
    assert "\n## Live rooms" not in context and '"text":"broken"' not in context  # Main is the room, not a live line
    assert "jsonl_malformed" in Memory(drive_root=tmp_path).chat_history(count=20)


def _legacy_memory(tmp_path, block):
    """One legacy block covering the one chat row, with the old cursor after it: no cursor gap."""
    from ouroboros.utils import jsonl_generation_signature

    (tmp_path / "memory").mkdir(parents=True, exist_ok=True)
    chat = _chain(tmp_path, [_RECENT])  # the same bytes _wake_context reads
    (tmp_path / "memory" / "dialogue_blocks.json").write_text(json.dumps([{
        "ts": "2026-08-21T00:00:00Z", "source": "consolidator", "message_count": 1, **block,
    }]), encoding="utf-8")
    (tmp_path / "memory" / "dialogue_meta.json").write_text(json.dumps({
        "last_consolidated_offset": 1, "chat_log_signature": jsonl_generation_signature(chat)}), encoding="utf-8")


def _story(context):
    return context.split("## My story", 1)[1].split("\n## ", 1)[0]


def test_wake_context_carries_a_complete_dialogue_block_without_a_gap(tmp_path):
    _legacy_memory(tmp_path, {"content": "Complete consolidated biography block."})
    _write_schedules(tmp_path, 8)
    story = _story(_wake_context(tmp_path))
    # The wake integrates my life (its room is Main): this unfolded mixed retelling
    # stays whole in my story, headed by its id and provenance, never a gap.
    assert "#### legacy-b00-rlegacy — " in story and " — Unknown provenance [legacy mixed record] — " in story
    assert "\n  Complete consolidated biography block." in story
    assert "memory gap" not in story and "[MEMORY GAP]" not in story
    assert "memory_read(node_id='legacy-b00-rlegacy')" not in story  # whole, so no pointer line repeats it


def test_wake_context_discloses_a_durable_dialogue_gap(tmp_path):
    _legacy_memory(tmp_path, {"gap_id": "dialogue-gap-123",
                              "content": "[MEMORY GAP] A durable biography interval is unavailable."})
    context = _wake_context(tmp_path)
    assert "## Dialogue History" not in context
    assert "- memory gap: Unknown provenance [legacy mixed record]; " in _story(context)
    assert "memory_read(node_id='legacy-b00-rlegacy')" in _story(context)


def test_wake_context_discloses_an_omitted_schedule_count(tmp_path):
    _write_schedules(tmp_path, 9)
    context = _wake_context(tmp_path)
    runtime = json.loads(context.split("## Runtime context\n\n", 1)[1].split("\n\n## ", 1)[0])
    assert runtime["scheduled_tasks"]["omitted_count"] == 1
