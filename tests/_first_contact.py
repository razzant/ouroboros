"""Shared first-contact episode using the real loop and a keyless HTTP model."""
import json
import queue
import shutil
from pathlib import Path

from tests.system_e2e.harness import MOCK_SLUG, keyless_settings

REPO = Path(__file__).resolve().parents[1]
ANSWER = "Привет! Я на месте и отвечаю обычным сообщением."


def _episode(tmp_path, monkeypatch, server):
    from ouroboros import config, context, context_health, usage_accounting as ua
    from ouroboros.memory import Memory
    from ouroboros.tools.registry import ToolRegistry

    repo, drive = tmp_path / "repo", tmp_path / "data"
    for name in ("prompts/SYSTEM.md", "BIBLE.md", "docs/ARCHITECTURE.md", "docs/DEVELOPMENT.md"):
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        if (REPO / name).exists():
            shutil.copyfile(REPO / name, repo / name)
    if (REPO / "docs/architecture").is_dir():
        shutil.copytree(REPO / "docs/architecture", repo / "docs/architecture")
    cfg = keyless_settings(server, OUROBOROS_CONTEXT_MODE="max", OUROBOROS_REASONING_SUMMARY="off",
                           OUROBOROS_CONSCIOUSNESS_SAFETY_MODE="off")
    for key, value in cfg.items():
        if isinstance(value, (str, int, float)):
            monkeypatch.setenv(key, str(value))
    drive.mkdir(parents=True)
    (drive / "settings.json").write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setattr(config, "SETTINGS_PATH", drive / "settings.json")
    monkeypatch.setattr(config, "DATA_DIR", drive)
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(drive / "settings.json"))
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(drive))
    monkeypatch.setenv("OUROBOROS_REPO_DIR", str(repo))
    monkeypatch.setattr(ua, "estimate_cost_optional", lambda *_a, **_kw: 0.0)
    monkeypatch.setattr(context_health, "_stray_server_note", lambda *_: "")
    (drive / "memory").mkdir(parents=True)
    (drive / "memory/WORLD.md").write_text("A test machine.", encoding="utf-8")  # no operator profiler at ensure_files
    memory = Memory(drive_root=drive, repo_dir=repo)
    memory.ensure_files()
    registry = ToolRegistry(repo_dir=repo, drive_root=drive)
    task = {"id": "first-contact", "type": "task", "text": "Привет! Ты на связи?", "model": MOCK_SLUG,
            "budget_drive_root": str(drive)}
    registry._ctx.task_id = task["id"]
    registry._ctx.task_metadata = task
    registry._ctx.budget_drive_root = drive
    from types import SimpleNamespace

    env = SimpleNamespace(repo_dir=repo, drive_root=drive, repo_path=lambda p: repo / p, drive_path=lambda p: drive / p)
    from ouroboros.tool_policy import initial_tool_schemas

    plan = context.build_context_fit_plan(env, memory, task, lambda: "", ctx=registry._ctx, preferred_mode="max",
                                          tool_schemas=initial_tool_schemas(registry, context_mode="max"),
                                          tool_registry=registry)
    registry._ctx.context_fit_plan = plan
    return SimpleNamespace(task=task, tools=registry, drive=drive, plan=plan)


def _run(episode):
    from ouroboros import loop, usage_accounting as ua
    from ouroboros.llm import LLMClient

    plan = episode.plan
    scope = ua.UsageScope(drive_root=episode.drive, task_id=episode.task["id"], root_task_id=episode.task["id"],
                          category="task", source="test.first_contact")
    with ua.usage_scope(scope):
        return loop.run_llm_loop(messages=plan.messages_for(plan.initial_mode), tools=episode.tools,
                                 llm=LLMClient(api_key="unused-keyless"), drive_logs=episode.drive / "logs",
                                 drive_root=episode.drive, task_id=episode.task["id"], task_type="task",
                                 incoming_messages=queue.Queue(), emit_progress=lambda *_a, **_kw: None)
