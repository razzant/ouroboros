"""Shared engine harness for the plan-review contract suites.

The fake review substrate, the ToolContext factory and the small readers over the
recorded wave, in ONE place: ``tests/test_plan_review_engine.py`` and its themed
sibling ``tests/test_plan_review_health.py`` both drive the engine through this
exact harness, so a contract proved in one file cannot be proved against a
different fake in the other. Non-test module (no ``test_`` prefix) so pytest
collects it only through the importers.
"""
from __future__ import annotations

import json
import queue
from types import SimpleNamespace

import pytest

from ouroboros.tools import plan_review as pr
from ouroboros.tools.registry import ToolContext
from ouroboros.tools.review_synthesis import PLAN_REVIEW_CONTROL_PREFIX

FP_LEN = 64

CLEAN = "[]\nNO_FINDINGS"

def _finding(fid, klass, *, breaks="", locator="", summary="something", rec="fix it"):
    return {"id": fid, "class": klass, "breaks": breaks, "locator": locator,
            "summary": summary, "recommendation": rec}

def _slots(*specs):
    """``specs`` = (slot_id, model[, "session"]) tuples → ReviewSlot list."""
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_substrate import ReviewSlot

    out = []
    for spec in specs:
        sid, model = spec[0], spec[1]
        session = len(spec) > 2 and spec[2] == "session"
        out.append(ReviewSlot(
            slot_id=sid, model=model, effort="high", role_hint="plan reviewer",
            route=ReviewRouteKind.AGENT_SESSION if session else ReviewRouteKind.API_CHAT,
            session_target="cursor=grok" if session else "",
        ))
    return out

class _Substrate:
    """Fake ``run_review_request``: answers per slot id (str or callable(request))."""

    def __init__(self, answers):
        self.answers = answers
        self.calls: list = []

    def __call__(self, request, *, slots, drive_root, llm, usage_ctx=None):
        self.calls.append({"request": request, "slots": list(slots)})
        actors = []
        for slot in slots:
            answer = self.answers.get(slot.slot_id, CLEAN)
            text = answer(request) if callable(answer) else answer
            actors.append({
                "slot_id": slot.slot_id, "model": slot.model, "status": "ok" if text else "error",
                "raw_text": text or "", "error": "" if text else "transport died",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "resolved_model": slot.model},
                "prompt_ref": {}, "response_ref": {},
            })
        return SimpleNamespace(actors=actors)

@pytest.fixture
def harness(tmp_path, monkeypatch):
    system = tmp_path / "repo"
    system.mkdir()
    (system / "BIBLE.md").write_text(
        "# BIBLE.md\n\n## Principle 0: Agency\n\nbe.\n\n## Principle 3: Immune Integrity\n\nreview.\n",
        encoding="utf-8",
    )
    (system / "docs").mkdir()
    (system / "docs" / "ARCHITECTURE.md").write_text(
        "# Ouroboros vX — Architecture & Reference\n\n## 1. Runtime\n\nthe loop.\n\n"
        "## 2. Review organ\n\nslots and quorum.\n",
        encoding="utf-8",
    )
    (system / "ouroboros").mkdir()
    (system / "ouroboros" / "loop.py").write_text("x = 1\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "notes.md").write_text("deck notes\n", encoding="utf-8")
    drive = tmp_path / "data"
    drive.mkdir()
    events: queue.Queue = queue.Queue()
    progress: list = []

    def make_ctx(*, active_workspace=True, task_id="task-1", messages=None, force_plan=False):
        ctx = ToolContext(
            repo_dir=system, system_repo_dir=system, drive_root=drive, task_id=task_id,
            workspace_root=workspace if active_workspace else None,
            workspace_mode="external" if active_workspace else "",
            task_metadata={"root_task_id": task_id, **({"force_plan": True} if force_plan else {})},
            task_contract={"objective": "Deliver the thing"},
            event_queue=events,
        )
        ctx.emit_progress_fn = progress.append
        ctx.messages = messages
        return ctx

    state = {"enforcement": "blocking", "slots": _slots(("s1", "m/a"), ("s2", "m/b"), ("s3", "m/c"))}
    monkeypatch.setattr(pr, "get_review_enforcement", lambda: state["enforcement"])
    monkeypatch.setattr(pr, "_plan_review_slots", lambda: state["slots"])
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "2")

    def install(answers):
        import ouroboros.review_substrate as rs

        sub = _Substrate(answers)
        monkeypatch.setattr(rs, "run_review_request", sub)
        return sub

    return SimpleNamespace(
        system=system, workspace=workspace, drive=drive, events=events, progress=progress,
        make_ctx=make_ctx, state=state, install=install,
    )

DECK_SPEC = {
    "in_scope": ["a 5-slide deck on the Q3 roadmap"],
    "non_goals": ["speaker notes"],
    "acceptance_claims": ["exactly 5 slides", "every slide has a title and one chart"],
    "invariants": ["deliver by Friday", "no confidential numbers"],
    "decisions": [{"choice": "one chart per slide", "rejected": ["tables"], "why": "audience"}],
    "deferred": [{"what": "color palette", "why_safe_to_defer": "cosmetic"}],
    "affected_resources": [],
    "evidence": [],
}

def _call(ctx, spec=None, *, goal="Ship the deck", plan="Outline first, then draft each slide.", **kw):
    return pr._handle_plan_task(ctx, goal=goal, plan=plan, spec=dict(spec or DECK_SPEC), **kw)

def _control(text):
    lines = [line for line in text.splitlines() if line.startswith(PLAN_REVIEW_CONTROL_PREFIX)]
    assert len(lines) == 1, text
    return json.loads(lines[0][len(PLAN_REVIEW_CONTROL_PREFIX):])

def _state(h, task_id="task-1"):
    from ouroboros.task_results import load_plan_review_state

    return load_plan_review_state(h.drive, task_id)
