"""Phase 5A: destructive learning sees complete source state or abstains."""

from __future__ import annotations

import concurrent.futures
import json
import pathlib
import threading


def test_pattern_register_rewrite_receives_complete_tail(tmp_path, monkeypatch):
    from ouroboros import reflection

    knowledge = tmp_path / "memory" / "knowledge"
    knowledge.mkdir(parents=True)
    tail = "DECISIVE_PATTERN_TAIL_MUST_SURVIVE"
    current = reflection._PATTERNS_HEADER + ("| old | 1 | cause | fix | open |\n" * 600) + tail
    path = knowledge / "patterns.md"
    path.write_text(current, encoding="utf-8")
    captured = {}
    monkeypatch.setattr("ouroboros.config.get_light_model", lambda: "light")
    monkeypatch.setattr("ouroboros.llm.LLMClient", lambda: object())

    def fake_chat(*args, **kwargs):
        captured["prompt"] = kwargs["messages"][0]["content"]
        return ({"content": current}, {})

    monkeypatch.setattr("ouroboros.llm_observability.chat_observed", fake_chat)
    reflection._update_patterns(tmp_path, {
        "task_id": "task-pattern", "goal": "keep complete patterns",
        "key_markers": ["TOOL_ERROR"], "reflection": "A new occurrence.",
    })
    assert tail in captured["prompt"]
    assert tail in path.read_text(encoding="utf-8")


def test_pattern_register_receives_the_whole_reflection_and_the_exact_goal(tmp_path, monkeypatch):
    """A correction past character 500 decides the register's row, so it must arrive.

    The incident: the Pattern Register writer received ``reflection[:500]``. The
    clip landed inside the exculpatory clause of a reflection whose EARLIER
    sentence said the opposite, so the register recorded the inverse conclusion
    and kept bumping its count. The goal was clipped the same way at 200 chars.
    Both are decision inputs of a DESTRUCTIVE rewrite (this call replaces the
    whole register), so both arrive complete. Asserted on the messages actually
    composed for the Light model, and on the producer's own entry rather than a
    hand-built one.
    """
    from ouroboros import reflection

    (tmp_path / "memory" / "knowledge").mkdir(parents=True)
    goal_tail = "GOAL TAIL: the owner asked for the release notes, not a branch cleanup."
    goal = "Ship the release. " + ("Background context sentence. " * 30) + goal_tail
    early = "EARLY READING: the host should fail closed on REVIEW_REQUIRED."
    correction = (
        "DECISIVE CORRECTION: this install runs advisory enforcement, so failing closed "
        "on REVIEW_REQUIRED would be the inverse of the configured rule."
    )
    reflection_body = early + " " + ("Padding sentence about the trace. " * 30) + correction
    assert len(goal) > 200
    assert reflection_body.index(correction) > 500

    captured = {}
    replies = [{"content": reflection_body
                + "\nMEMORY_ACTIONS_JSON: []\nBACKLOG_CANDIDATES_JSON: []"}]

    def fake_chat(*args, **kwargs):
        if kwargs.get("call_type") == "pattern_register_update":
            captured["prompt"] = kwargs["messages"][0]["content"]
            return ({"content": reflection._PATTERNS_HEADER
                     + "| advisory misreading | 1 | clipped input | pass whole input | open |\n"}, {})
        return (replies.pop(0), {})

    monkeypatch.setattr("ouroboros.config.get_light_model", lambda: "light")
    monkeypatch.setattr("ouroboros.llm.LLMClient", lambda: object())
    monkeypatch.setattr("ouroboros.llm_observability.chat_observed", fake_chat)

    entry = reflection.generate_reflection(
        {"id": "task-learn", "text": goal, "drive_root": str(tmp_path)},
        {"tool_calls": [{"tool": "write_file", "is_error": True, "status": "error",
                         "tool_result_code": "TOOL_REPORTED_FAILURE", "result": "boom"}]},
        "trace", object(), {"rounds": 3, "cost": 0.0},
    )
    # The bounded display field survives beside the exact one; neither replaces the other.
    assert entry["goal_exact"] == goal
    assert entry["goal"].startswith("Ship the release.") and goal_tail not in entry["goal"]
    assert entry["reflection"] == reflection_body

    reflection.append_reflection(tmp_path, entry)

    prompt = captured["prompt"]
    assert reflection_body in prompt
    assert prompt.index(correction) > prompt.index(early)
    assert goal_tail in prompt
    assert "OMISSION NOTE" not in prompt


def test_backlog_fingerprint_uses_unsanitized_canonical_fields(tmp_path, monkeypatch):
    from ouroboros.improvement_backlog import append_backlog_items, load_backlog_items

    monkeypatch.setattr("ouroboros.semantic_dedup.find_semantic_duplicate_id", lambda *a, **k: None)
    prefix = "x" * 300
    assert append_backlog_items(tmp_path, [{
        "summary": prefix + "A" * 40, "category": "process", "source": "reflection",
    }]) == 1
    assert append_backlog_items(tmp_path, [{
        "summary": prefix + "B" * 40, "category": "process", "source": "reflection",
    }]) == 1
    items = load_backlog_items(tmp_path)
    assert len(items) == 2
    assert len({item["fingerprint"] for item in items}) == 2


def test_generate_reflection_carries_raw_backlog_identity_through_append(tmp_path, monkeypatch):
    from ouroboros import reflection
    from ouroboros.improvement_backlog import append_backlog_items, load_backlog_items

    prefix = "decisive-prefix-" + ("x" * 300)
    summaries = [prefix + (tail * 80) for tail in ("A", "B")]
    replies = []
    for summary in summaries:
        candidate = {
            "summary": summary,
            "category": "process",
            "source": "execution_reflection",
            "evidence": "production reflection evidence",
        }
        replies.append({
            "content": (
                "Reflection body\n"
                "MEMORY_ACTIONS_JSON: []\n"
                "BACKLOG_CANDIDATES_JSON: " + json.dumps([candidate])
            )
        })

    monkeypatch.setattr("ouroboros.config.get_light_model", lambda: "light")
    monkeypatch.setattr(
        "ouroboros.semantic_dedup.find_semantic_duplicate_id", lambda *a, **k: None,
    )

    def fake_chat(*args, **kwargs):
        return replies.pop(0), {}

    monkeypatch.setattr("ouroboros.llm_observability.chat_observed", fake_chat)
    generated = [
        reflection.generate_reflection(
            {"id": f"task-{idx}", "text": "reflect", "drive_root": str(tmp_path)},
            {}, "trace", object(), {"rounds": 1, "cost": 0.0},
        )
        for idx in range(2)
    ]
    assert generated[0]["backlog_candidates"][0]["summary"] == generated[1]["backlog_candidates"][0]["summary"]
    for entry in generated:
        assert append_backlog_items(tmp_path, entry["backlog_candidates"]) == 1

    items = load_backlog_items(tmp_path)
    assert len(items) == 2
    assert len({item["fingerprint"] for item in items}) == 2


def test_backlog_semantic_redirect_skips_known_partial_query_and_candidate(tmp_path, monkeypatch):
    from ouroboros.improvement_backlog import append_backlog_items, load_backlog_items

    semantic_calls = []

    def redirect_to_first(query, candidates, **kwargs):
        semantic_calls.append((query, candidates))
        return candidates[0]["id"]

    monkeypatch.setattr(
        "ouroboros.semantic_dedup.find_semantic_duplicate_id", redirect_to_first,
    )

    query_root = tmp_path / "partial-query"
    assert append_backlog_items(query_root, [{
        "summary": "complete existing item", "category": "process", "source": "reflection",
    }]) == 1
    assert append_backlog_items(query_root, [{
        "summary": "partial query " + ("q" * 400),
        "category": "process", "source": "reflection",
    }]) == 1

    candidate_root = tmp_path / "partial-candidate"
    assert append_backlog_items(candidate_root, [{
        "summary": "partial candidate " + ("c" * 400),
        "category": "process", "source": "reflection",
    }]) == 1
    assert append_backlog_items(candidate_root, [{
        "summary": "complete new item", "category": "process", "source": "reflection",
    }]) == 1

    assert semantic_calls == []
    assert len(load_backlog_items(query_root)) == 2
    assert len(load_backlog_items(candidate_root)) == 2


def test_groom_receives_complete_records_and_preserves_on_unavailable(tmp_path, monkeypatch):
    from ouroboros import improvement_backlog as ib

    monkeypatch.setattr("ouroboros.semantic_dedup.find_semantic_duplicate_id", lambda *a, **k: None)
    for idx in range(35):
        ib.append_backlog_items(tmp_path, [{
            "id": f"ibl-{idx}", "fingerprint": f"fp-{idx}", "summary": f"item {idx}",
            "category": "process", "source": "reflection",
            "evidence": f"complete-evidence-{idx}", "context": f"complete-context-{idx}",
            "proposed_next_step": f"complete-next-step-{idx}",
        }])
    items = ib.load_backlog_items(tmp_path)
    keep = [{"id": i["id"], "fingerprint": i["fingerprint"], "summary": i["summary"]}
            for i in items[:20]]
    captured = {}
    monkeypatch.setattr("ouroboros.config.get_light_model", lambda: "light")
    monkeypatch.setattr("ouroboros.llm.LLMClient", lambda: object())

    def fake_chat(*args, **kwargs):
        captured["prompt"] = kwargs["messages"][0]["content"]
        return ({"content": json.dumps(keep)}, {})

    monkeypatch.setattr("ouroboros.llm_observability.chat_observed", fake_chat)
    assert ib.groom_backlog(tmp_path, cap=30) == 20
    for expected in ("complete-evidence-34", "complete-context-34", "complete-next-step-34"):
        assert expected in captured["prompt"]

    before = ib.backlog_path(tmp_path).read_text(encoding="utf-8")
    real_locked = ib._locked_text_file

    def unavailable(path, mode, *, shared=False):
        if mode == "r":
            raise PermissionError("backlog unavailable")
        return real_locked(path, mode, shared=shared)

    monkeypatch.setattr(ib, "_locked_text_file", unavailable)
    assert ib.groom_backlog(tmp_path, cap=10) == 0
    assert ib.backlog_path(tmp_path).read_text(encoding="utf-8") == before


def test_groom_preserves_annotated_fingerprinted_survivor_verbatim(tmp_path, monkeypatch):
    from ouroboros import improvement_backlog as ib

    monkeypatch.setattr(
        "ouroboros.semantic_dedup.find_semantic_duplicate_id", lambda *a, **k: None,
    )
    assert ib.append_backlog_items(tmp_path, [{
        "id": f"ibl-{idx}", "fingerprint": f"fp-{idx}", "summary": f"item {idx}",
        "category": "process", "source": "reflection",
    } for idx in range(35)]) == 35
    path = ib.backlog_path(tmp_path)
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "- summary: item 0\n",
        "- summary: item 0\n- owner_note: keep this exact owner byte\n"
        "freeform decisive survivor tail\n",
        1,
    )
    path.write_text(text, encoding="utf-8")
    before_items = ib.load_backlog_items(tmp_path)
    annotated_before = next(item for item in before_items if item["id"] == "ibl-0")["_raw"]
    keep = [{
        "id": item["id"], "fingerprint": item["fingerprint"], "summary": item["summary"],
    } for item in before_items[:20]]

    monkeypatch.setattr("ouroboros.config.get_light_model", lambda: "light")
    monkeypatch.setattr("ouroboros.llm.LLMClient", lambda: object())

    def fake_chat(*args, **kwargs):
        prompt = kwargs["messages"][0]["content"]
        assert "owner_note" in prompt and "freeform decisive survivor tail" in prompt
        return ({"content": json.dumps(keep)}, {})

    monkeypatch.setattr("ouroboros.llm_observability.chat_observed", fake_chat)
    assert ib.groom_backlog(tmp_path, cap=30) == 20
    annotated_after = next(
        item for item in ib.load_backlog_items(tmp_path) if item["id"] == "ibl-0"
    )
    assert annotated_after["_raw"] == annotated_before


def test_concurrent_pattern_updates_commit_one_cas_winner(tmp_path, monkeypatch):
    from ouroboros import reflection

    knowledge = tmp_path / "memory" / "knowledge"
    knowledge.mkdir(parents=True)
    path = knowledge / "patterns.md"
    initial = reflection._PATTERNS_HEADER
    path.write_text(initial, encoding="utf-8")
    llm_gate = threading.Barrier(2)
    legacy_write_gate = threading.Barrier(2)
    monkeypatch.setattr("ouroboros.config.get_light_model", lambda: "light")
    monkeypatch.setattr("ouroboros.llm.LLMClient", lambda: object())

    def fake_chat(*args, **kwargs):
        llm_gate.wait(timeout=10)
        task_id = kwargs["task_id"]
        return ({
            "content": initial + f"| {task_id} | 1 | cause | fix | open |\n",
        }, {})

    monkeypatch.setattr("ouroboros.llm_observability.chat_observed", fake_chat)
    real_write_text = pathlib.Path.write_text

    def synchronize_legacy_replace(self, data, *args, **kwargs):
        if self == path:
            legacy_write_gate.wait(timeout=10)
        return real_write_text(self, data, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "write_text", synchronize_legacy_replace)
    entries = [{
        "task_id": f"writer-{idx}", "goal": "concurrent pattern update",
        "key_markers": ["TOOL_ERROR"], "reflection": f"writer {idx}",
    } for idx in range(2)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(reflection._update_patterns, tmp_path, entry) for entry in entries]
        for future in futures:
            future.result(timeout=15)

    final = path.read_text(encoding="utf-8")
    history_path = knowledge / "patterns_history.jsonl"
    history = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines()]
    assert sum(f"writer-{idx}" in final for idx in range(2)) == 1
    assert len(history) == 1
    assert history[0]["new_content"] == final


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
    decision = pte._decide_promotion(env, {"id": "root"}, {"reflection": "done"}, object(), force=False)
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
    assert pte._decide_promotion(env, {"id": "root"}, {"reflection": "done"}, object(), force=False) is None
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
    rows = chat_rows if chat_rows is not None else [json.dumps({"chat_id": 1, "direction": "in", "text": "complete recent chat"})]
    (tmp_path / "logs" / "chat.jsonl").write_text("".join(row + "\n" for row in rows), encoding="utf-8")
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

    for name in ("memory", "logs"):
        (tmp_path / name).mkdir(parents=True, exist_ok=True)
    chat = tmp_path / "logs" / "chat.jsonl"
    chat.write_text(json.dumps({"chat_id": 1, "direction": "in", "text": "complete recent chat"}) + "\n",
                    encoding="utf-8")  # the same bytes _wake_context writes
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
    # The wake integrates my life (its room is Main): the retold first block (no rooms: an
    # old mixed record) is whole in my story, headed by its id and provenance, never a gap.
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
    assert '"omitted_count": 1' in context
