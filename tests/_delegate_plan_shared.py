"""Claudexor v4.5.0 (2b2dfc15) plan/thread fixture; all roots are disposable.

Wire sources: packages/schema/src/plan.ts, control-run-detail.ts and
packages/control-api/src/thread-turn-routes.ts in razzant/claudexor.
"""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from ouroboros import delegate_collection, delegate_custody as custody, delegate_custody_current as current
from ouroboros.gateways.claudexor import ClaudexorGateway, ClaudexorUnavailable
from ouroboros.subagent_runtime import exact_start
from ouroboros.subagents import DelegationRoute
from tests._delegated_transport_shared import _delegating_ctx, _transport_snapshot


class PlanEngine:
    engine_version = "4.5.0"
    start_run = ClaudexorGateway.start_run
    create_thread = ClaudexorGateway.create_thread
    start_thread_turn = ClaudexorGateway.start_thread_turn

    def __init__(self, root):
        self.root, self.calls, self.threads, self.runs, self.accepted = root, [], {}, {}, {}
        self.modes = ["ask", "plan", "agent"]
        self.route_modes = self.modes[:]
        self.fail_next = False
        self.questions = {"parse": "found", "questions": [{"id": "q1", "kind": "text",
            "prompt": "Which file?", "options": [], "allow_text": True}]}
        self.plan = b"# Frozen plan\nEdit README.md after the starter answers.\n"

    def agent_capabilities(self):
        return {"modes": self.modes, "harnesses": [{"id": "some-route", "enabled": True,
            "status": "ok", "modes": self.route_modes,
            "accessProfilesSupported": ["readonly", "workspace_write", "full"]}]}

    def find_project_id(self, root): return "existing-project"
    def ensure_full_access(self, root): return {}
    def close(self): pass

    def _request(self, method, path, *, json_body, headers):
        key, body = headers["Idempotency-Key"], copy.deepcopy(json_body)
        assert "planSha256" not in body and "planHash" not in body, "The server alone freezes the hash"
        self.calls.append((path, body, key))
        if path == "/v2/threads":
            tid = f"thread-{len(self.threads) + 1}"
            self.threads[tid] = {"thread": {"id": tid, "headRunId": ""}, "turns": []}
            return {"id": tid}
        pending = custody.invocation_record(self.root, key)
        assert pending and pending["state"] == "pending", "Physical starts require durable intent first"
        if key not in self.accepted:
            rid = f"run-{len(self.accepted) + 1}"
            self.accepted[key] = {"runId": rid}
            self.runs[rid] = {"summary": {"runId": rid, "state": "running",
                "effectiveAccess": body["access"], "spendUsd": 0, "spendEstimated": False},
                "primaryOutput": {"text": "Plan complete", "truncated": False}}
            if path.startswith("/v2/threads/"):
                thread = self.threads[path.split("/")[3]]
                thread["thread"]["headRunId"] = rid
                thread["turns"].append({"runId": rid})
        if self.fail_next:
            self.fail_next = False
            raise ClaudexorUnavailable("transport_unavailable", "Lost response after acceptance")
        return self.accepted[key]

    def get_thread(self, thread_id): return copy.deepcopy(self.threads[thread_id])
    def get_run(self, run_id, **kwargs):
        result = copy.deepcopy(self.runs[run_id])
        artifact = self.questions
        found = isinstance(artifact, dict) and artifact.get("parse") == "found"
        questions = artifact.get("questions", []) if found else []
        result.update(planQuestions=questions, planReadiness={"state": (
            "needs_answers" if questions else "ready") if found else "unverified", "questionCount": len(questions)})
        return result
    def get_run_artifact(self, run_id, path):
        assert run_id in self.runs
        if path == "final/plan.md":
            return self.plan
        if path == "final/questions.json":
            return self.questions if isinstance(self.questions, bytes) else json.dumps(self.questions).encode()
        raise ClaudexorUnavailable("artifact_not_found", path, status_code=404)


@pytest.fixture
def plan_setup(tmp_path, monkeypatch):
    from ouroboros import claudexor_daemon, usage_accounting

    ctx = _delegating_ctx(tmp_path, acting=True)
    snapshot = _transport_snapshot(DelegationRoute("some-route", "model"), "workspace_write")
    engine = PlanEngine(tmp_path)
    monkeypatch.setenv("OUROBOROS_SUBAGENT_WORKTREE_ROOT", str(tmp_path / "snapshots"))
    monkeypatch.setattr(claudexor_daemon, "ensure_owned_gateway", lambda: engine)
    monkeypatch.setattr(claudexor_daemon, "read_owned_gateway", lambda **kw: engine)
    monkeypatch.setattr(usage_accounting, "record_subscription_session", lambda *a, **kw: None)
    custody._CUSTODY.clear()

    def start(prompt="Make a plan", **options):
        if not options.get("retry_of"):
            options.setdefault("snapshot", copy.deepcopy(snapshot))
        return json.loads(exact_start(ctx, prompt, options).text)

    def finish(rid):
        engine.runs[rid]["summary"]["state"] = "succeeded"
        result = delegate_collection.collect_terminal(tmp_path, rid, detail=engine.get_run(rid), gateway=engine)
        assert result["status"] == "collected", result
        return current.run(tmp_path, rid), delegate_collection.retained_payload(tmp_path, current.run(tmp_path, rid))

    yield SimpleNamespace(root=tmp_path, ctx=ctx, snapshot=snapshot, engine=engine, start=start, finish=finish)
    custody._CUSTODY.clear()
