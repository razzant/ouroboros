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


def _bg_fixture(tmp_path, *, backlog_count=10):
    from ouroboros.consciousness import BackgroundConsciousness
    from ouroboros.improvement_backlog import append_backlog_items

    repo_dir = pathlib.Path(__file__).parents[1]
    (tmp_path / "logs").mkdir(parents=True)
    (tmp_path / "logs" / "chat.jsonl").write_text(
        json.dumps({"chat_id": 1, "direction": "in", "text": "complete recent chat"}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "state").mkdir(parents=True)
    (tmp_path / "state" / "state.json").write_text("{}", encoding="utf-8")
    for idx in range(backlog_count):
        append_backlog_items(tmp_path, [{
            "id": f"ibl-bg-{idx}", "fingerprint": f"fp-bg-{idx}",
            "summary": f"background item {idx}", "category": "identity", "source": "reflection",
        }])
    return BackgroundConsciousness(tmp_path, repo_dir, None, lambda: None)


def _tool_call(name, args, call_id):
    return {"id": call_id, "function": {"name": name, "arguments": json.dumps(args)}}


def _write_schedules(tmp_path, count):
    tasks = [{
        "id": f"schedule-{idx}", "name": f"schedule {idx}", "enabled": True,
        "trigger": {"type": "cron", "expr": "0 * * * *"},
    } for idx in range(count)]
    (tmp_path / "state" / "scheduled_tasks.json").write_text(
        json.dumps({"tasks": tasks}), encoding="utf-8",
    )


def test_bgc_direct_identity_update_requires_complete_named_omission(tmp_path):
    bc = _bg_fixture(tmp_path)
    try:
        # The knowledge header is separate from the exact body; both the
        # delivered body and revision must preserve a CRLF source.
        backlog = tmp_path / "memory" / "knowledge" / "improvement-backlog.md"
        backlog.write_bytes(
            backlog.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
        )
        context = bc._build_context()
        assert "knowledge_read" in context and "improvement-backlog" in context
        content = "I remain directly self-authoring after complete source materialization."
        blocked = bc._execute_tool(_tool_call("update_identity", {"content": content}, "u1"), [])
        assert "IDENTITY_UPDATE_ABSTAINED" in blocked
        read = bc._execute_tool(_tool_call("knowledge_read", {"topic": "improvement-backlog"}, "r1"), [])
        assert "background item 9" in read
        updated = bc._execute_tool(_tool_call("update_identity", {"content": content}, "u2"), [])
        assert updated.startswith("OK: identity updated")
        journal = tmp_path / "memory" / "identity_journal.jsonl"
        assert journal.exists() and content in journal.read_text(encoding="utf-8")
    finally:
        bc._tool_executor.shutdown(wait=False, cancel_futures=True)


def test_bgc_source_mutation_after_read_cannot_authorize_identity_rewrite(tmp_path, monkeypatch):
    bc = _bg_fixture(tmp_path)
    try:
        bc._build_context()
        backlog = tmp_path / "memory" / "knowledge" / "improvement-backlog.md"
        changed = backlog.read_bytes() + (
            b"\n### ibl-concurrent\n- summary: changed after materialization\n"
        )
        real_read_bytes = pathlib.Path.read_bytes
        target_reads = 0

        def mutate_before_snapshot(self, *args, **kwargs):
            nonlocal target_reads
            if self == backlog:
                target_reads += 1
                if target_reads == 1:
                    captured = real_read_bytes(self, *args, **kwargs)
                    self.write_bytes(changed)
                    return captured
            return real_read_bytes(self, *args, **kwargs)

        monkeypatch.setattr(pathlib.Path, "read_bytes", mutate_before_snapshot)
        bc._execute_tool(_tool_call("knowledge_read", {"topic": "improvement-backlog"}, "r1"), [])
        result = bc._execute_tool(
            _tool_call("update_identity", {"content": "must remain blocked"}, "u1"), [],
        )
        assert target_reads >= 2  # reader capture plus current-source validation
        assert "IDENTITY_UPDATE_ABSTAINED" in result
    finally:
        bc._tool_executor.shutdown(wait=False, cancel_futures=True)


def test_bgc_source_snapshot_cannot_mix_text_and_digest(tmp_path, monkeypatch):
    bc = _bg_fixture(tmp_path)
    try:
        backlog = tmp_path / "memory" / "knowledge" / "improvement-backlog.md"
        backlog.write_bytes(
            backlog.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
        )
        bc._build_context()
        original = backlog.read_bytes()
        changed = original + b"\r\n### ibl-concurrent\r\n- summary: changed bytes\r\n"
        real_read_bytes = pathlib.Path.read_bytes
        target_reads = 0

        def oscillate_during_validation(self, *args, **kwargs):
            nonlocal target_reads
            if self == backlog:
                target_reads += 1
                if target_reads == 1:
                    self.write_bytes(changed)
                    raw = real_read_bytes(self, *args, **kwargs)
                    self.write_bytes(original)
                    return raw
            return real_read_bytes(self, *args, **kwargs)

        monkeypatch.setattr(pathlib.Path, "read_bytes", oscillate_during_validation)
        materialized = bc._execute_tool(
            _tool_call("knowledge_read", {"topic": "improvement-backlog"}, "r1"), [],
        )
        assert materialized.endswith(changed.decode("utf-8"))
        assert materialized.startswith("[Knowledge source]")
        assert "improvement-backlog" not in bc._identity_source_reads
        backlog.write_bytes(changed)
        result = bc._execute_tool(
            _tool_call("update_identity", {"content": "must remain blocked"}, "u1"), [],
        )
        assert "IDENTITY_UPDATE_ABSTAINED" in result
    finally:
        bc._tool_executor.shutdown(wait=False, cancel_futures=True)


def test_bgc_unavailable_named_omission_abstains_without_approval_flow(tmp_path):
    bc = _bg_fixture(tmp_path)
    try:
        bc._build_context()
        (tmp_path / "memory" / "knowledge" / "improvement-backlog.md").unlink()
        content = "I retain direct authority but abstain when the named source is unavailable."
        result = bc._execute_tool(_tool_call("update_identity", {"content": content}, "u1"), [])
        assert "IDENTITY_UPDATE_ABSTAINED" in result
        assert "approval" not in result.lower()
        assert not (tmp_path / "memory" / "identity_journal.jsonl").exists()
    finally:
        bc._tool_executor.shutdown(wait=False, cancel_futures=True)


def test_bgc_malformed_recent_chat_gap_blocks_direct_identity_update(tmp_path):
    bc = _bg_fixture(tmp_path, backlog_count=0)
    try:
        chat = tmp_path / "logs" / "chat.jsonl"
        chat.write_text(
            chat.read_text(encoding="utf-8") + '{"direction":"in","text":"broken"\n',
            encoding="utf-8",
        )
        context = bc._build_context()
        assert "jsonl_malformed" in context
        content = "I must not rewrite identity from a context with a known chat gap."
        result = bc._execute_tool(_tool_call("update_identity", {"content": content}, "u1"), [])
        assert "IDENTITY_UPDATE_ABSTAINED" in result
        assert not (tmp_path / "memory" / "identity_journal.jsonl").exists()
    finally:
        bc._tool_executor.shutdown(wait=False, cancel_futures=True)


def test_bgc_complete_recent_chat_keeps_direct_identity_update_available(tmp_path):
    bc = _bg_fixture(tmp_path, backlog_count=0)
    try:
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)
        (memory_dir / "dialogue_blocks.json").write_text(json.dumps([{
            "ts": "2026-08-21T00:00:00Z", "source": "consolidator",
            "content": "Complete consolidated biography block.",
        }]), encoding="utf-8")
        _write_schedules(tmp_path, 8)
        bc._build_context()
        content = "I retain direct identity authority with complete ordinary context."
        result = bc._execute_tool(_tool_call("update_identity", {"content": content}, "u1"), [])
        assert result.startswith("OK: identity updated")
        assert content in (tmp_path / "memory" / "identity_journal.jsonl").read_text(encoding="utf-8")
    finally:
        bc._tool_executor.shutdown(wait=False, cancel_futures=True)


def test_bgc_durable_dialogue_gap_blocks_direct_identity_update(tmp_path):
    bc = _bg_fixture(tmp_path, backlog_count=0)
    try:
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)
        (memory_dir / "dialogue_blocks.json").write_text(json.dumps([{
            "ts": "2026-08-21T00:00:00Z", "source": "consolidator",
            "gap_id": "dialogue-gap-123",
            "content": "[MEMORY GAP] A durable biography interval is unavailable.",
        }]), encoding="utf-8")
        context = bc._build_context()
        assert "## Dialogue History" in context and "[MEMORY GAP]" in context
        content = "I must not rewrite identity across a known durable biography gap."
        result = bc._execute_tool(_tool_call("update_identity", {"content": content}, "u1"), [])
        assert "IDENTITY_UPDATE_ABSTAINED" in result
        assert "dialogue-gap-123" in result
        assert not (memory_dir / "identity_journal.jsonl").exists()
    finally:
        bc._tool_executor.shutdown(wait=False, cancel_futures=True)


def test_bgc_scheduled_tasks_omission_blocks_direct_identity_update(tmp_path):
    bc = _bg_fixture(tmp_path, backlog_count=0)
    try:
        _write_schedules(tmp_path, 9)
        context = bc._build_context()
        assert '"omitted_count": 1' in context
        content = "I must not rewrite identity from an incomplete standing-schedule digest."
        result = bc._execute_tool(_tool_call("update_identity", {"content": content}, "u1"), [])
        assert "IDENTITY_UPDATE_ABSTAINED" in result
        assert "scheduled-tasks" in result
        assert not (tmp_path / "memory" / "identity_journal.jsonl").exists()
    finally:
        bc._tool_executor.shutdown(wait=False, cancel_futures=True)


def test_bgc_observation_gap_blocks_direct_identity_update(tmp_path):
    bc = _bg_fixture(tmp_path, backlog_count=0)
    try:
        store = tmp_path / "state" / "consciousness_observations.jsonl"
        store.write_text(
            '{"op":"enqueue","id":"valid","source":"runtime",'
            '"kind":"text","time":"2026-08-21T00:00:00Z",'
            '"payload":"known row","ref":null}\n'
            '{"op":"enqueue","id":"broken"\n',
            encoding="utf-8",
        )
        context = bc._build_context()
        assert "source_complete=False" in context
        assert "state/consciousness_observations.jsonl" in context
        assert "background-observations" in context
        content = "I must not rewrite identity across a known observation-source gap."
        result = bc._execute_tool(_tool_call("update_identity", {"content": content}, "u1"), [])
        assert "IDENTITY_UPDATE_ABSTAINED" in result
        assert "background-observations" in result
        assert not (tmp_path / "memory" / "identity_journal.jsonl").exists()
    finally:
        bc._tool_executor.shutdown(wait=False, cancel_futures=True)


def test_bgc_omitted_observation_rows_block_direct_identity_update(tmp_path):
    bc = _bg_fixture(tmp_path, backlog_count=0)
    try:
        for index in range(11):
            assert bc.inject_observation(
                f"observation-{index}", observation_id=f"omitted-{index}"
            )
        context = bc._build_context()
        assert "omitted=1" in context
        assert "source_complete=False" in context
        assert "background-observations" in context
        result = bc._execute_tool(
            _tool_call(
                "update_identity",
                {"content": "Do not rewrite identity from omitted observations."},
                "u1",
            ),
            [],
        )
        assert "IDENTITY_UPDATE_ABSTAINED" in result
        assert "background-observations" in result
        assert not (tmp_path / "memory" / "identity_journal.jsonl").exists()
    finally:
        bc._tool_executor.shutdown(wait=False, cancel_futures=True)


def test_bgc_payload_truncation_blocks_direct_identity_update(tmp_path):
    bc = _bg_fixture(tmp_path, backlog_count=0)
    try:
        assert bc.inject_observation(
            "p" * 2_000, observation_id="payload-truncated"
        )
        context = bc._build_context()
        assert "payload omitted" in context
        assert "source_complete=False" in context
        assert "background-observations" in context
        result = bc._execute_tool(
            _tool_call(
                "update_identity",
                {"content": "Do not rewrite identity from a truncated payload."},
                "u1",
            ),
            [],
        )
        assert "IDENTITY_UPDATE_ABSTAINED" in result
        assert "background-observations" in result
        assert not (tmp_path / "memory" / "identity_journal.jsonl").exists()
    finally:
        bc._tool_executor.shutdown(wait=False, cancel_futures=True)


def test_bgc_aggregate_projection_truncation_blocks_direct_identity_update(tmp_path):
    bc = _bg_fixture(tmp_path, backlog_count=0)
    try:
        for index in range(10):
            assert bc.inject_observation(
                "small",
                observation_id=f"aggregate-{index}",
                source="s" * 1_400,
            )
        context = bc._build_context()
        assert "projection truncated" in context
        assert "source_complete=False" in context
        assert "background-observations" in context
        result = bc._execute_tool(
            _tool_call(
                "update_identity",
                {"content": "Do not rewrite identity from an aggregate projection."},
                "u1",
            ),
            [],
        )
        assert "IDENTITY_UPDATE_ABSTAINED" in result
        assert "background-observations" in result
        assert not (tmp_path / "memory" / "identity_journal.jsonl").exists()
    finally:
        bc._tool_executor.shutdown(wait=False, cancel_futures=True)


def test_bgc_complete_observation_source_keeps_direct_identity_update_available(tmp_path):
    bc = _bg_fixture(tmp_path, backlog_count=0)
    try:
        assert bc.inject_observation(
            "complete observation", observation_id="complete-observation"
        )
        context = bc._build_context()
        assert "source_complete=True" in context
        assert "complete-observation" in context
        content = "I retain direct identity authority with a complete observation source."
        result = bc._execute_tool(_tool_call("update_identity", {"content": content}, "u1"), [])
        assert result.startswith("OK: identity updated")
        assert content in (tmp_path / "memory" / "identity_journal.jsonl").read_text(
            encoding="utf-8"
        )
    finally:
        bc._tool_executor.shutdown(wait=False, cancel_futures=True)
