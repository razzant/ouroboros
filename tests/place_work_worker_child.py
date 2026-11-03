"""One real pool worker process for the place-conversation work consumer; never an installation.

Started by ``tests/test_place_conversation_work.py`` with ``multiprocessing``'s ``spawn`` context, as the
supervisor spawns its workers: the production ``supervisor.worker_process.worker_main`` reads its input queue,
runs the promoted root in its own process and writes its frames to its event queue. Only the model transport is
scripted (``tests.test_presence_continuation_bootstrap``) and extension reload is skipped. Each scripted call
writes what it saw to ``<facts_dir>/call-<n>.json``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


def run(in_q, out_q, repo: str, data: str, facts_dir: str) -> None:
    import pytest

    from ouroboros import agent as agent_module, extension_loader
    from supervisor import worker_process
    from tests.test_presence_continuation_bootstrap import install_bootstrap_harness

    patch = pytest.MonkeyPatch()
    h = install_bootstrap_harness(patch, Path(data), repo=Path(repo), review_mode="off")
    patch.setattr(agent_module, "make_agent", h.factory)  # the production constructor, observed
    patch.setattr(extension_loader, "reload_all", lambda *_a, **_kw: None)
    patch.setattr(worker_process, "_adopt_published_extensions", lambda *_a: None)
    facts, calls = Path(facts_dir), []

    def tool(name, arguments):
        return {"content": None, "tool_calls": [{"id": f"{name}-{len(calls)}", "type": "function",
                                                  "function": {"name": name, "arguments": json.dumps(arguments)}}]}

    def model(messages):
        calls.append(1)
        ctx = h.agents[-1].tools._ctx
        row = {"pid": os.getpid(), "task": str(ctx.task_id), "attempt": ctx.task_attempt,
               "in_worker": os.environ.get("OUROBOROS_IN_WORKER"), "messages": json.dumps(messages, ensure_ascii=False)}
        tmp = facts / f"call-{len(calls)}.json.tmp"
        tmp.write_text(json.dumps(row), encoding="utf-8")
        tmp.replace(facts / f"call-{len(calls)}.json")
        if len(calls) == 1:
            return tool("await_messages", {"timeout_sec": 90})  # an ordinary wait in its slot
        return tool("finish_task", {"action": "finish", "answer": "Noted: the backup job too."})

    h.script = model
    worker_process.worker_main(0, in_q, out_q, repo, data)


def run_carrier(in_q, out_q, repo: str, data: str, facts_dir: str, key: str) -> None:
    """The production worker carrying a real Presence turn: its first call asks a private question and waits."""
    import pytest

    from ouroboros import agent as agent_module, extension_loader
    from ouroboros.conversation_lease import live_author
    from supervisor import worker_process
    from tests.test_presence_continuation_bootstrap import install_bootstrap_harness

    patch = pytest.MonkeyPatch()
    h = install_bootstrap_harness(patch, Path(data), repo=Path(repo), review_mode="off")
    patch.setattr(agent_module, "make_agent", h.factory)
    patch.setattr(extension_loader, "reload_all", lambda *_a, **_kw: None)
    patch.setattr(worker_process, "_adopt_published_extensions", lambda *_a: None)
    patch.setattr("supervisor.workers.get_event_q", lambda *, create=True: type("Sink", (), {"put": lambda self, _event: None})())
    facts, calls = Path(facts_dir), []

    def tool(name, arguments):
        return {"content": None, "tool_calls": [{"id": f"{name}-{len(calls)}", "type": "function",
                                                  "function": {"name": name, "arguments": json.dumps(arguments)}}]}

    def model(messages):
        calls.append(1)
        row = {"pid": os.getpid(), "holder": live_author(key, drive_root=Path(data)),
               "messages": json.dumps(messages, ensure_ascii=False)}
        tmp = facts / f"carrier-{len(calls)}.json.tmp"
        tmp.write_text(json.dumps(row), encoding="utf-8")
        tmp.replace(facts / f"carrier-{len(calls)}.json")
        if len(calls) == 1:
            return tool("escalate", {"question": "May I start?", "options": [{"label": "Yes"}], "wait_for_answer": True,
                                     "max_wait_minutes": 2, "request_id": "q1"})
        return tool("finish_task", {"action": "finish", "answer": "Noted."})

    h.script = model
    worker_process.worker_main(7, in_q, out_q, repo, data)
