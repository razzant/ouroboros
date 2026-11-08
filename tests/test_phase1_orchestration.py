"""Phase 1 golden coverage: honest runtime digest, activity-based timeout model,
and the task-tree coordination ledger."""

from types import SimpleNamespace



def _patch_queue(queue_module, workers_module, monkeypatch, tmp_path, workers):
    monkeypatch.setattr(queue_module, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(queue_module, "PENDING", [])
    monkeypatch.setattr(queue_module, "RUNNING", {})
    monkeypatch.setattr(queue_module, "FINALIZATION_GRACE_SEC", 0)
    monkeypatch.setattr(queue_module, "QUEUE_MAX_RETRIES", 1)
    monkeypatch.setattr(queue_module, "load_state", lambda: {})
    monkeypatch.setattr(queue_module, "append_jsonl", lambda *a, **k: None)
    monkeypatch.setattr(queue_module, "persist_queue_snapshot", lambda reason="": None)
    monkeypatch.setattr(queue_module, "_ensure_reaper_started", lambda: None)
    monkeypatch.setattr(queue_module, "_reap_queue", queue_module._stdqueue.Queue())
    monkeypatch.setattr(workers_module, "WORKERS", workers)
    monkeypatch.setattr(workers_module, "respawn_worker", lambda worker_id: None)


class _FakeProc:
    pid = 0

    def is_alive(self):
        return False

    def join(self, timeout=None):
        return None


def test_tree_ledger_scope_and_attention(monkeypatch, tmp_path):
    # Point the ledger at tmp_path WITHOUT a global config reload (which would leave
    # ouroboros.config.DATA_DIR stuck on this tmp_path and pollute later tests): the ledger
    # reads its module-level DATA_DIR, so monkeypatch THAT (auto-restored after the test).
    import ouroboros.task_tree_ledger as L
    monkeypatch.setattr(L, "DATA_DIR", str(tmp_path))

    assert L.tree_ledger_append("rootA", "contract", "API: f()->g", task_id="rootA", role="lead").startswith("OK")
    assert L.tree_ledger_append("rootA", "question", "lib X or Y?", task_id="c1", role="scout").startswith("OK")
    # blocker/question imply parent attention
    att = L.tree_ledger_attention_after("rootA", "")
    assert len(att) == 1 and att[0]["kind"] == "question"
    # bad kind rejected; empty text rejected
    assert "TOOL_ARG_ERROR" in L.tree_ledger_append("rootA", "bogus", "x")
    assert "TOOL_ARG_ERROR" in L.tree_ledger_append("rootA", "note", "")
    # scope isolation: a different tree has its own ledger
    assert L.tree_ledger_rows("rootB") == []
    # interface_contract is a beacon that ALSO flags parent attention (the shared seam must change)
    assert L.tree_ledger_append("rootA", "interface_contract", "f() now returns h, not g",
                                task_id="c1", role="scout").startswith("OK")
    att2 = L.tree_ledger_attention_after("rootA", "")
    assert any(a["kind"] == "interface_contract" for a in att2), \
        "an interface_contract beacon must surface as a parent early-return"
    # strict root_id validation: a malformed scope is rejected on write; reads soft-fail to []
    assert "TOOL_ARG_ERROR" in L.tree_ledger_append("bad/scope", "note", "x")
    assert "TOOL_ARG_ERROR" in L.tree_ledger_append("", "note", "x")
    assert L.tree_ledger_rows("bad/scope") == []
    digest = L.tree_ledger_tail_digest("rootA")
    assert "contract" in digest and "needs_parent_attention" in digest


def test_review_requested_preserves_typed_concerns_and_exact_hash(monkeypatch, tmp_path):
    import ouroboros.task_tree_ledger as L
    from ouroboros.tools.control import _wait_attention_poll

    monkeypatch.setattr(L, "DATA_DIR", str(tmp_path))
    payload = {"evidence_ref": "artifact:research-notes", "evidence_sha256": "a" * 64}
    first = L.tree_ledger_append(
        "rootA", "review_requested", "Challenge the source coverage.",
        task_id="childA", role="researcher", payload=payload,
    )
    assert first.startswith("OK:")
    second = L.tree_ledger_append(
        "rootA", "review_requested", "Same bytes, different security concern.",
        task_id="childB", role="critic", payload={**payload, "evidence_ref": "artifact:copy"},
    )
    assert second.startswith("OK:")
    rows = [row for row in L.tree_ledger_rows("rootA") if row["kind"] == "review_requested"]
    assert len(rows) == 2
    assert all(row["needs_parent_attention"] is True for row in rows)
    assert rows[0]["payload"] == payload
    # Recovery/context projections retain the exact 64-character binding, not
    # merely a prefix that cannot identify the evidence after wake acknowledgement.
    assert "a" * 64 in L.tree_ledger_page("rootA", data_root=tmp_path)
    assert "a" * 64 in L.tree_ledger_tail_digest("rootA", data_root=tmp_path)

    # Ordinary sliced waits are branch-local: waiting for childA must not surface
    # childB's request merely because both share the same tree ledger.
    ctx = SimpleNamespace(task_id="parentA", task_metadata={"root_task_id": "rootA"})
    early = _wait_attention_poll(ctx, "", ["childA"])({}, {})
    assert early is not None
    assert [row["task_id"] for row in early["beacons"]] == ["childA"]
    assert "TOOL_ARG_ERROR" in L.tree_ledger_append(
        "rootA", "review_requested", "missing hash", task_id="childA",
        payload={"evidence_ref": "artifact:x"},
    )


def test_delegation_constraint_payload_and_override(monkeypatch, tmp_path):
    import ouroboros.task_tree_ledger as L

    monkeypatch.setattr(L, "DATA_DIR", str(tmp_path))
    result = L.tree_ledger_append(
        "rootA",
        "delegation_constraint",
        "stop fanning out until git evidence is gathered",
        task_id="scout1",
        role="scout",
        payload={"directive": "halt_fanout", "scope": {"role": "release-band"}, "rationale": "readonly child cannot run git"},
    )
    assert result.startswith("OK")
    open_rows = L.open_delegation_constraints("rootA")
    assert len(open_rows) == 1
    cid = open_rows[0]["payload"]["constraint_id"]

    L.tree_ledger_append(
        "rootA",
        "decision",
        "override after parent gathered git evidence",
        task_id="parent",
        role="lead",
        payload={"constraint_id": cid, "decision": "overridden", "reason": "parent owns raw git data"},
        allow_constraint_override=True,
    )
    assert L.open_delegation_constraints("rootA") == []
    L.tree_ledger_append(
        "rootA",
        "delegation_constraint",
        "same id raised later should be open again",
        task_id="scout1",
        role="scout",
        payload={"constraint_id": cid, "directive": "halt_fanout", "scope": {}, "rationale": "new evidence"},
    )
    assert len(L.open_delegation_constraints("rootA")) == 1
    assert "TOOL_ARG_ERROR" in L.tree_ledger_append(
        "rootA", "delegation_constraint", "bad", payload={"directive": "unknown"}
    )
    assert "TOOL_ARG_ERROR" in L.tree_ledger_append(
        "rootA",
        "decision",
        "forged override",
        payload={"constraint_id": "forged", "decision": "overridden", "reason": "no"},
    )


def test_effective_delegation_budget_honors_require_lane_and_scope():
    from ouroboros.tools.control_delegation import effective_delegation_budget

    row = {
        "payload": {
            "constraint_id": "c1",
            "directive": "require_lane",
            "scope": {"role": "critic", "lane": "heavy"},
            "rationale": "needs stronger coding lane",
        }
    }
    ignored = effective_delegation_budget({}, unresolved_constraints=[row], role="researcher", requested_lane="light")
    assert ignored.ok is True

    # The INTENDED lane decides, not the literal request (v6.87.28: this gate runs at
    # ADMISSION, before the child is dispatched, so no effective lane exists yet —
    # `subagents.intended_lane` is what a request MEANS, and its one owner).
    requested_does_not_count = effective_delegation_budget(
        {},
        unresolved_constraints=[row],
        role="critic",
        requested_lane="auto",
        intended_lane="light",
    )
    assert requested_does_not_count.ok is False

    blocked = effective_delegation_budget({}, unresolved_constraints=[row], role="critic", requested_lane="light", intended_lane="light")
    assert blocked.ok is False
    assert blocked.reason_code == "delegation_constraint_require_lane"

    allowed = effective_delegation_budget({}, unresolved_constraints=[row], role="critic", requested_lane="heavy", intended_lane="heavy")
    assert allowed.ok is True
    # F9 (sol #1): a PASSING admission under an applicable require_lane carries
    # the verified lane out, so the scheduler can stamp it onto the child record
    # and the dispatch-time policy default cannot override it.
    assert allowed.required_lane == "heavy"
    # A non-applicable constraint (role mismatch) contributes no required lane.
    assert ignored.required_lane == ""


def test_task_is_readonly_subagent_gate():
    """The single SSOT gate the task_done path and the reaper both read (a re-derivation
    drift of this rule is what stranded the reaper's artifact finalization)."""
    from ouroboros.headless import task_is_readonly_subagent

    assert task_is_readonly_subagent(
        {"delegation_role": "subagent", "task_constraint": {"mode": "local_readonly_subagent"}}) is True
    # constraint nested under metadata is honored too
    assert task_is_readonly_subagent(
        {"delegation_role": "subagent", "metadata": {"task_constraint": {"mode": "local_readonly_subagent"}}}) is True
    # acting subagents and plain tasks are NOT readonly → they DO finalize artifacts
    assert task_is_readonly_subagent(
        {"delegation_role": "subagent", "task_constraint": {"mode": "acting_subagent"}}) is False
    assert task_is_readonly_subagent({"id": "root", "type": "task"}) is False
    assert task_is_readonly_subagent(None) is False
