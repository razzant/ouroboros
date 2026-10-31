"""Hermetic Supervisor harness: real check path, fake physical model and place seams."""
import json
from types import SimpleNamespace

import pytest


@pytest.fixture
def supervisor(monkeypatch, tmp_path):
    from ouroboros import safety
    from ouroboros import tool_effects, place_guard

    values = {"OUROBOROS_MODEL_SAFETY": "", "OUROBOROS_SUBAGENTS": ""}
    modes = {"owner": "full", "consciousness": "full"}
    calls, events, replies = [], [], []
    ctx = SimpleNamespace(task_id="work", task_metadata={"work_kind": "owner"}, drive_root=tmp_path,
                          task_contract={"unbounded": "MUST NOT BE INCLUDED"})
    safety._PRIOR_VERDICTS.clear()
    monkeypatch.setattr(safety, "_SAFETY_STORM_UNTIL", 0)
    monkeypatch.setattr(safety, "runtime_setting", lambda key, default=None: values.get(key, default))
    monkeypatch.setattr(safety, "get_safety_mode", lambda kind="owner": modes[kind])
    monkeypatch.setattr(safety, "get_runtime_mode", lambda: "advanced")
    monkeypatch.setattr(safety, "get_safety_model", lambda: values.get("OUROBOROS_MODEL_SAFETY") or "test/checker")
    monkeypatch.setattr(safety, "_resolve_safety_routing", lambda: (False, False, None))
    monkeypatch.setattr(safety, "_emit_durable_safety_event", lambda ctx, event: events.append(event))
    monkeypatch.setattr(safety, "_emit_safety_mode_skip", lambda *a: events.append({"type": "skip"}))
    monkeypatch.setattr(place_guard, "call_place_facts", lambda *a, **kw: {})
    monkeypatch.setattr(tool_effects, "effect_of", lambda name, **kw: tool_effects.BUILTIN)
    def call(**kwargs):
        calls.append(kwargs)
        reply = replies.pop(0) if replies else {"status": "SAFE", "reason": "fine"}
        if isinstance(reply, Exception):
            raise reply
        return {"content": reply if isinstance(reply, str) else json.dumps(reply)}, {}
    monkeypatch.setattr(safety, "_safety_model_call", call)
    yield SimpleNamespace(module=safety, ctx=ctx, calls=calls, events=events, replies=replies, values=values, modes=modes)
    safety._PRIOR_VERDICTS.clear()


def disable_safety(monkeypatch):
    """Keep unrelated dispatch tests independent of both assessment levels."""
    for key in ("OUROBOROS_SAFETY_MODE", "OUROBOROS_CONSCIOUSNESS_SAFETY_MODE"):
        monkeypatch.setenv(key, "off")


def pin_safety_model():
    """Let defaults tests change provider access without changing their checker."""
    from ouroboros.gateway.owner_settings import _owner_update_settings, fields_on_saved_document
    _owner_update_settings(fields_on_saved_document({"OUROBOROS_MODEL_SAFETY": "fixture/checker"}),
                           allow_safety_lowering=True)


def owner_registry(tmp_path):
    from ouroboros.tools.registry import ToolRegistry

    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_metadata = {"work_kind": "owner"}
    return registry


def clear_safety_provider_env(monkeypatch) -> None:
    """Keep post-check tests from depending on live safety LLM credentials."""
    for key in (
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENAI_COMPATIBLE_API_KEY",
        "CLOUDRU_FOUNDATION_MODELS_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def settings_install(tmp_path, monkeypatch):
    """An isolated data root; no provider key, model slot or legacy alias leaks in from the shell."""
    from ouroboros import config as cfg, review_pool_migration as m

    data = tmp_path / "data"
    data.mkdir()
    path = data / "settings.json"
    monkeypatch.setattr(cfg, "DATA_DIR", data)
    monkeypatch.setattr(cfg, "SETTINGS_PATH", path)
    for key in (*cfg.SETTINGS_DEFAULTS, "OUROBOROS_MODEL_FALLBACK"):
        monkeypatch.delenv(key, raising=False)
    m._MIGRATIONS_SEEN.clear()
    yield path
    m._MIGRATIONS_SEEN.clear()
