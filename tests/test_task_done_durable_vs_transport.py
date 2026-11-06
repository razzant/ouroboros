"""The durable/transport split of the task_done delegation enrichment.

Adversarial-wave B-PY-1 pin: only call-site ordering protected the durable log
shape — this asserts it at the byte level so a refactor cannot silently start
persisting the transport-only keys.

Verifies Lane B claims at the byte level:
  1. events.jsonl task_done row does NOT carry the three delegation keys
     (durable format unchanged — enrichment is post-append, transport-only).
  2. bridge.push_log receives the SAME event WITH the keys (live log channel).
  3. send_with_budget fires exactly once for chat 0 with the truth meta.
  4. Negative (A2A) chat: no chat frame, log channel still enriched.
  5. Non-subagent task: push_log event carries NO delegation keys.
"""
import json


def _mk_ctx(tmp_path, sent, pushed):
    class Bridge:
        @staticmethod
        def push_log(data):
            # snapshot the dict AT PUSH TIME (later mutation would not count)
            pushed.append(json.loads(json.dumps(data)))

    class Ctx:
        DRIVE_ROOT = tmp_path
        RUNNING = {}
        WORKERS = {}
        PENDING = []
        bridge = Bridge()

        @staticmethod
        def persist_queue_snapshot(reason=""):
            pass

        @staticmethod
        def send_with_budget(chat_id, text, **kw):
            sent.append((chat_id, text, json.loads(json.dumps(kw.get("progress_meta") or {}))))

        @staticmethod
        def load_state():
            return {}

        @staticmethod
        def save_state(st):
            pass

        @staticmethod
        def append_jsonl(path, data):
            from ouroboros.utils import append_jsonl
            append_jsonl(path, data)

        @staticmethod
        def sort_pending():
            pass

    return Ctx()


def _seed(tmp_path, task_id, chat_id, delegation_role="subagent"):
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    from ouroboros.task_results import write_task_result
    fields = dict(
        result="done",
        chat_id=chat_id,
        parent_task_id="root-9",
        root_task_id="root-9",
        delegation_role=delegation_role,
        executor_route="codex=gpt-5.6-sol",
        subagent_envelope={
            "execution_evidence": {
                "delegated_runs_started": 2,
                "delegated_runs_settled": 2,
                "delegated_runs_succeeded": 1,
                "delegated_runs_failed": 1,
            },
            "actual_substrate": "harness_used",
        },
    )
    write_task_result(tmp_path, task_id, "completed", **fields)


def _run(tmp_path, task_id, chat_id, delegation_role="subagent"):
    sent, pushed = [], []
    ctx = _mk_ctx(tmp_path, sent, pushed)
    task = {
        "id": task_id, "type": "task", "chat_id": chat_id,
        "delegation_role": delegation_role, "role": "researcher",
        "parent_task_id": "root-9", "root_task_id": "root-9",
    }
    ctx.RUNNING[task_id] = {"task": task}
    _seed(tmp_path, task_id, chat_id, delegation_role)
    from ouroboros.headless import prepare_terminal_task_files
    from supervisor.events import _handle_task_done
    assert not prepare_terminal_task_files(tmp_path, task)["error"]
    evt = {
        "type": "task_done", "task_id": task_id, "task_type": "task",
        "worker_id": 0, "chat_id": chat_id,
        "ts": "2026-08-29T00:00:00Z",
        "_files_prepared_attempt": int(task.get("_attempt") or 1),
    }
    _handle_task_done(evt, ctx)
    events_file = tmp_path / "logs" / "events.jsonl"
    rows = [json.loads(l) for l in events_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    done_rows = [r for r in rows if r.get("type") == "task_done" and r.get("task_id") == task_id]
    return sent, pushed, done_rows


KEYS = ("executor_route", "execution_evidence", "actual_substrate")


def test_chat0_subagent_durable_vs_transport(tmp_path):
    sent, pushed, done_rows = _run(tmp_path, "child-a", 0)
    # 1. durable byte-level: no delegation keys in events.jsonl
    assert len(done_rows) == 1, done_rows
    for k in KEYS:
        assert k not in done_rows[0], f"durable task_done gained {k}: FORMAT CHANGE"
    # 2. live log channel enriched
    pushed_done = [p for p in pushed if p.get("type") == "task_done"]
    assert len(pushed_done) == 1
    assert pushed_done[0]["executor_route"] == "codex=gpt-5.6-sol"
    assert pushed_done[0]["execution_evidence"]["delegated_runs_failed"] == 1
    assert pushed_done[0]["actual_substrate"] == "harness_used"
    # 3. chat frame delivered exactly once, to chat 0
    assert [c for c, _t, _m in sent] == [0], sent
    meta = sent[0][2]
    assert meta["executor_route"] == "codex=gpt-5.6-sol"
    assert meta["execution_evidence"]["delegated_runs_succeeded"] == 1
    assert meta["actual_substrate"] == "harness_used"


def test_negative_a2a_subagent_no_chat_frame_but_log_enriched(tmp_path):
    sent, pushed, done_rows = _run(tmp_path, "child-b", -1002)
    assert sent == []
    pushed_done = [p for p in pushed if p.get("type") == "task_done"]
    assert len(pushed_done) == 1
    assert pushed_done[0]["executor_route"] == "codex=gpt-5.6-sol"
    for k in KEYS:
        assert k not in done_rows[0]


def test_a_child_refused_at_dispatch_names_the_session_route_its_row_named(tmp_path):
    """Live, its card said ``{harness} · blocked`` from the dispatch caps; the record keeps no route (nothing ran
    on it), so the terminal carriers take it from the configured row. An API-model row names none, as live."""
    from ouroboros.task_results import write_task_result
    from supervisor.subagent_task_truth import terminal_executor_route

    def refused(kind, target):
        return dict(result="⚠️ SUBAGENT_UNAVAILABLE: this subagent could not start.", chat_id=0,
                    parent_task_id="root-9", root_task_id="root-9", delegation_role="subagent",
                    effective_executor="blocked", executor_route="", reason_code="subagent_executor_unavailable",
                    configured_subagent={"route": {"kind": kind, "target_id": target}})

    session, api = refused("agent_session", "claude=claude-opus-5-5:high"), refused("api_model", "openai/gpt-6")
    assert (terminal_executor_route(session), terminal_executor_route(api)) == ("claude", "")
    # A recorded route stays the record's, and a child that ran natively names none.
    assert terminal_executor_route({**session, "executor_route": "codex=gpt-6-astra"}) == "codex=gpt-6-astra"
    assert terminal_executor_route({**session, "effective_executor": "native"}) == ""
    for task_id, fields, route in (("child-r", session, "claude"), ("child-q", api, None)):
        (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
        sent, pushed = [], []
        ctx = _mk_ctx(tmp_path, sent, pushed)
        task = {"id": task_id, "type": "task", "chat_id": 0, "delegation_role": "subagent", "role": "critic",
                "parent_task_id": "root-9", "root_task_id": "root-9"}
        ctx.RUNNING[task_id] = {"task": task}
        write_task_result(tmp_path, task_id, "failed", **fields)
        from ouroboros.headless import prepare_terminal_task_files
        from supervisor.events import _handle_task_done
        assert not prepare_terminal_task_files(tmp_path, task)["error"]
        _handle_task_done({"type": "task_done", "task_id": task_id, "task_type": "task", "worker_id": 0, "chat_id": 0,
                           "ts": "2026-11-06T00:00:00Z", "_files_prepared_attempt": 1}, ctx)
        pushed_done = [p for p in pushed if p.get("type") == "task_done"]
        assert pushed_done[0].get("executor_route") == route, task_id
        assert [meta["executor_route"] for _c, _t, meta in sent] == [route or ""], task_id


def test_non_subagent_log_channel_not_enriched(tmp_path):
    sent, pushed, done_rows = _run(tmp_path, "root-task", 1, delegation_role="")
    pushed_done = [p for p in pushed if p.get("type") == "task_done"]
    assert len(pushed_done) == 1
    for k in KEYS:
        assert k not in pushed_done[0], f"non-subagent push_log gained {k}"
        assert k not in done_rows[0]
