"""Running works shaped like an owner's install, through the real roster producers (task store, queue, fences).

Synthetic words only: what each kind of row records (a dated focus with its retained source, an owner question, an
owner's Pause with its words, a model wait, a sleep, a held Continue, a project-bound work) at its real field sizes.
"""
from __future__ import annotations

from ouroboros.task_results import write_task_result
from ouroboros.utils import utc_now_iso
from tests._shared import write_queue_document

# Kinds in the proportions a long-running install keeps them; 39 rows in all (the owner's count on 25.11.2026).
KINDS = ("focused",) * 20 + ("paused",) * 6 + ("question",) * 4 + ("model",) * 3 + ("sleep",) * 2 + ("held",) * 2 + (
    "project",) * 2


def _words(seed: str, size: int) -> str:
    text = (seed + " ") * (size // (len(seed) + 1) + 1)
    return text[:size].strip()


def focus(task_id: str, n: int) -> dict:
    return {"text": _words(f"Checking step {n} of the release notes against the owner's last words", 260),
            "author_task_id": task_id, "authored_at": f"2026-11-{10 + n % 15:02d}T1{n % 10}:00:00Z",
            "source_ref": {"reader": "get_task_result", "task_id": task_id},
            "source_handle": {"kind": "task_source", "root": "artifact_store",
                              "path": f"source_handles/context_checkpoints/focus-{n:04d}.json",
                              "size": 1200 + n, "sha256": f"{n:064x}"}}


def build(root, kinds=KINDS) -> list:
    """Write every row's task result and one queue snapshot; returns the root ids in queue order."""
    from ouroboros.projects_registry import bind_task_to_project, create_project

    running, pending, ids = [], [], []
    project = None
    for n, kind in enumerate(kinds):
        task_id = f"w{n:02d}{kind[:3]}"
        ids.append(task_id)
        title = _words(f"Work {n}: {kind} on the owner's request", 70)
        task = {"id": task_id, "chat_id": 1, "title": title, "focus": focus(task_id, n)}
        stored = {}
        if kind == "paused":
            by = {"actor": "owner", "text": _words(f"wait until I answer about the budget {n}", 90),
                  "at": f"2026-11-2{n % 6}T09:00:00Z"}
            stored["owner_pause"] = {"state": "paused", "requested_by": "owner", "by": by, "fence_id": f"f{n}",
                                     "generation": 1, "request_id": f"r{n}", "requested_at": by["at"],
                                     "paused_at": by["at"], "root_task_id": task_id,
                                     "requests": {f"r{n}": {"by": by, "fence_id": f"f{n}", "generation": 1}}}
        elif kind == "question":
            stored["owner_quiz"] = {f"q{n}": {"state": "open", "question": _words(f"Which option for step {n}?", 120),
                                              "options": ["A: keep the horizon", "B: narrow it"],
                                              "asked_at": "2026-11-24T10:00:00Z", "wait_for_answer": True}}
            stored["owner_wait"] = {"state": "waiting", "quiz_id": f"q{n}", "task_attempt": 1, "reason": "owner",
                                    "parked_at": "2026-11-24T10:00:00Z"}
        elif kind == "model":
            stored["model_waits"] = {f"m{n}": {"state": "waiting", "task_attempt": 1, "reason": "quota",
                                               "role": "main", "started_at": "2026-11-25T08:00:00Z",
                                               "reset_at": "2026-11-25T13:00:00Z"}}
        elif kind == "sleep":
            stored["owner_wait"] = {"state": "waiting", "task_attempt": 1, "reason": "sleep", "quiz_id": "",
                                    "parked_at": "2026-11-25T07:00:00Z",
                                    "sleep": {"mode": "warm", "wake_at": "2026-11-27T07:00:00Z",
                                              "tasks": [f"w{(n + 1) % 10:02d}foc"]}}
        elif kind == "held":
            task["_budget_pause_hold"] = {"reason": "continuation_writer_unsettled", "held_at": "2026-11-23T11:00:00Z"}
            task["metadata"] = {"continuation": {"predecessor_task_id": f"w{n - 1:02d}hel"}}
        elif kind == "project":
            if project is None:
                project = create_project(root, "release", name="Release 8.0")
            bind_task_to_project(root, task_id, project["id"], project["chat_id"], origin={"absent": "system"})
        write_task_result(root, task_id, "running", title=title, chat_id=1, focus=task["focus"],
                          started_at=f"2026-11-{10 + n % 15:02d}T08:00:00Z", **stored)
        row = {"id": task_id, "attempt": 1, "started_at": 1793000000.0 + n, "task": task}
        (pending if kind == "held" else running).append(row)
    write_queue_document(root / "state/queue_snapshot.json", {"ts": utc_now_iso(), "running": running,
                                                               "pending": pending})
    return ids
