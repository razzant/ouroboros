"""A tool call through the real executor, for the suites that read what it journals and emits.

Moved verbatim out of ``tests/test_tool_call_log.py``: ``_Registry`` is a registry double whose
context captures the live frames the executor emits (``frames``); ``_call`` runs one call through
``loop_tool_execution._execute_with_timeout`` into ``<tmp_path>/child/logs``; ``_rows`` reads that
``tools.jsonl`` back.
"""
from __future__ import annotations

import json
import pathlib
from types import SimpleNamespace

import ouroboros.loop_tool_execution as execution


def _rows(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class _Registry:
    CODE_TOOLS = frozenset()

    def __init__(self, tmp_path, handler, *, round_id="exec:round:1"):
        self.handler = handler
        self.frames = []
        queue = SimpleNamespace(put_nowait=lambda env: self.frames.append(env["data"]), put=None)
        self._ctx = SimpleNamespace(
            task_attempt=2, event_queue=queue, task_metadata={"budget_drive_root": str(tmp_path / "canonical")},
            _current_llm_call_meta={"execution_id": "exec", "round_id": round_id, "llm_call_id": "llm-1"})

    def execute_result(self, name, args):
        return self.handler(name, args)


def _call(registry, tmp_path, call_id="call_0", name="read_file", args=None, timeout=5):
    logs = tmp_path / "child" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    tc = {"id": call_id, "function": {"name": name, "arguments": json.dumps(args or {"path": "a.txt"})}}
    return execution._execute_with_timeout(registry, tc, logs, timeout, "task-1"), logs
