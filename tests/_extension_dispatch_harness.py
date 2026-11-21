"""Real loader → ToolRegistry → tool-batch harness for extension dispatch tests.

This process acts as the server that publishes and loads extensions, settings and the skills repository are pinned,
one batch runs through the real loop executor, and only the Safety Supervisor's physical model call is scripted.
Shared by the generation-adoption and connector target-group tests; ``_restore_server_pid`` applies on import.
"""
from __future__ import annotations

import json
import os
import pathlib

import pytest

from ouroboros.extension_companion import init_server_process_pid


@pytest.fixture(autouse=True)
def _restore_server_pid():
    original = os.environ.get("OUROBOROS_SERVER_PROCESS_PID")
    yield
    init_server_process_pid(int(original) if original else -1)
    if original is None:
        os.environ.pop("OUROBOROS_SERVER_PROCESS_PID", None)


def as_server() -> None:
    init_server_process_pid(os.getpid())


def settings_and_repo(monkeypatch, repo_root: pathlib.Path) -> None:
    from ouroboros import config

    monkeypatch.setattr(config, "load_settings", lambda: {})
    monkeypatch.setattr(config, "get_skills_repo_path", lambda: str(repo_root))


def batch(registry, drive_root, *calls, task_id: str = "genprobe-task"):
    """Run ``calls`` as one model batch through the real executor; returns its results and their start rows."""
    from ouroboros.loop_tool_execution import execute_tool_batch

    logs = drive_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    results, control = execute_tool_batch(list(calls), registry, logs, task_id, None, {})
    assert control is None
    starts = {}
    for line in (logs / "tools.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("type") == "tool_call_started":
            starts[row["tool_call_id"]] = row
    return results, starts


def supervisor(monkeypatch, verdict="SAFE", runtime="advanced"):
    """The real Safety path; only the physical model call is recorded and answers ``verdict`` (an exception: no answer)."""
    from ouroboros import safety
    from tests._safety_fixtures import reset_safety_stores

    prompts: list[str] = []
    reset_safety_stores()
    monkeypatch.setattr(safety, "_SAFETY_STORM_UNTIL", 0)
    monkeypatch.setattr(safety, "_CHAIN_STORMS", {})
    monkeypatch.setattr(safety, "_resolve_safety_routing", lambda: (False, False, None))
    monkeypatch.setattr(safety, "get_safety_model", lambda: "fixture/checker")
    monkeypatch.setattr(safety, "get_safety_mode", lambda *_a: "full")
    monkeypatch.setattr(safety, "get_runtime_mode", lambda: runtime)
    monkeypatch.setattr(safety, "_emit_durable_safety_event", lambda *_a, **_k: None)

    def model(**kwargs):
        prompts.append(kwargs["user_prompt"])
        if isinstance(verdict, Exception):
            raise verdict
        return {"content": json.dumps({"status": verdict, "reason": "fixture verdict"})}, {}

    monkeypatch.setattr(safety, "_safety_model_call", model)
    return prompts


def brief_evidence(prompt: str) -> dict:
    """The host evidence of one Supervisor prompt."""
    evidence = prompt.split("Host evidence (provenance is not consent):\n", 1)[1].split("\nProposed tool call:\n", 1)[0]
    return json.loads(evidence)


def brief_place(prompt: str) -> dict:
    """The ``subject.place`` host facts of one Supervisor prompt."""
    return brief_evidence(prompt)["subject"]["place"]
