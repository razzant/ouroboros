"""Retired register bodies stay at their ordinary knowledge reader in every mode.

Synthetic data only. Capture and route changes retain the index's source revision;
reading a note returns its current complete bytes, including malformed metadata.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


HEADER = "## Known error patterns (Pattern Register)"
ROWS = 40
REGISTER = ("---\ntitle: Pattern Register\nsummary: Synthetic error classes\n---\n# Pattern Register\n\n"
            "| Error class | Root cause | Fix |\n|---|---|---|\n"
            + "".join(f"| synthetic-class-{n:02d} | cause {n} | fix {n} |\n" for n in range(ROWS))
            + "SYNTHETIC REGISTER END\n")
TEXT_TASK = {"id": "t-text", "type": "task", "text": "Fix the flaky test.", "_is_direct_chat": True}


def _route(*_args, **_kwargs):
    return ({"model": "m", "provider": "p"},
            SimpleNamespace(route_fp="r", status="asserted", stale=False, window_tokens=1_000_000))


def _world(tmp_path, monkeypatch, register: bytes | None = REGISTER.encode()):
    import ouroboros.config as cfg
    from ouroboros import context_fit
    from tests.test_cache_optimization import _make_env_and_memory

    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(context_fit, "_route_calibration_ratio", lambda *_a, **_kw: 1.0)
    env, memory = _make_env_and_memory(tmp_path)
    shelf = env.drive_root / "memory" / "knowledge"
    (shelf / "overview.md").write_text("---\nsummary: Synthetic orientation\n---\nSYNTHETIC OVERVIEW.\n", encoding="utf-8")
    (shelf / "runbook.md").write_text("---\ntitle: Runbook\nsummary: Synthetic topic\n---\n# Runbook\n", encoding="utf-8")
    if register is not None:
        (shelf / "patterns.md").write_bytes(register)
    return env, memory


def _plan(env, memory, task=TEXT_TASK, preferred="nano", core=None):
    from ouroboros import context

    core = core or context._capture_context_core(env, memory, task, None, None)
    return core, context._build_context_fit_plan(env, core, task, preferred_mode=preferred, route_resolver=_route)


def _changing(plan, mode):
    return plan.projection(mode).system_message()["content"][-1]["text"]


def _files(root):
    return {path: path.read_bytes() for path in sorted((root / "memory" / "knowledge").rglob("*")) if path.is_file()}


def test_every_mode_keeps_the_summary_without_a_resident_register(tmp_path, monkeypatch):
    env, memory = _world(tmp_path, monkeypatch)
    before = _files(env.drive_root)
    core, plan = _plan(env, memory)
    for mode in ("max", "low", "nano"):
        text = _changing(plan, mode)
        assert text.startswith(core.dynamic_head_text)
        assert HEADER not in text and "synthetic-class-" not in text and "SYNTHETIC REGISTER END" not in text
        assert "SYNTHETIC OVERVIEW." in text
        assert "- **patterns**: [Pattern Register](<patterns.md>)" in text and "Synthetic error classes" in text
    assert not any(part.section_id == "pattern_register" for part in core.dynamic_head_sections)
    assert _files(env.drive_root) == before


def test_text_and_call_share_the_ordinary_knowledge_view(tmp_path, monkeypatch):
    env, memory = _world(tmp_path, monkeypatch)
    call = {"id": "t-call", "type": "presence", "source": "presence", "_presence_turn": True,
            "context_requires_development": False, "chat_id": 777, "text": "Fix the flaky test."}
    heads = []
    for task in (TEXT_TASK, call):
        core, plan = _plan(env, memory, task)
        for mode in ("max", "low", "nano"):
            assert "synthetic-class-" not in _changing(plan, mode)
            assert "Synthetic error classes" in _changing(plan, mode)
        heads.append(core.dynamic_head_text)
    assert heads[0] == heads[1]


def test_route_changes_and_refusal_modes_do_not_restore_the_retired_body(tmp_path, monkeypatch):
    env, memory = _world(tmp_path, monkeypatch)
    core, plan = _plan(env, memory, preferred="max")
    (env.drive_root / "memory" / "knowledge" / "patterns.md").write_text("# Pattern Register\n\nEDITED LATER\n")
    tight = plan.projection("max").estimated_tokens + plan.output_reserve_tokens - 1
    for rebound in (
        plan.reproject_for_route(window_tokens=2_000_000, known_window=True, ratio=1.0,
                                 output_reserve=plan.output_reserve_tokens, tool_schemas=None),
        plan.reproject_for_route(window_tokens=tight, known_window=True, ratio=1.0,
                                 output_reserve=plan.output_reserve_tokens, tool_schemas=None, select_mode="low"),
    ):
        for mode in ("max", "low", "nano"):
            text = _changing(rebound, mode)
            assert "synthetic-class-" not in text and "EDITED LATER" not in text
            assert "patterns" in text
            assert all(row["section_id"] != "pattern_register" for row in
                       rebound.projection(mode).memory_facts.get("host_sections_addressed", ()))


@pytest.mark.parametrize("register", [None, b"  \n\t\n", b"---\ntitle: Register\n---\nbad bytes \xff\n"],
                         ids=["missing", "blank", "undecodable"])
def test_missing_or_unreadable_register_has_no_resident_section(tmp_path, monkeypatch, register):
    env, memory = _world(tmp_path, monkeypatch, register)
    core, plan = _plan(env, memory)
    assert HEADER not in core.dynamic_head_text
    for mode in ("max", "low", "nano"):
        assert HEADER not in _changing(plan, mode)
        assert _changing(plan, mode).startswith(core.dynamic_head_text)


def test_malformed_metadata_stays_at_the_reader_which_returns_the_original(tmp_path, monkeypatch):
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    malformed = "---\ntitle: [unclosed\n---\n# Pattern Register\n\n| synthetic-class-00 | cause | fix |\n"
    env, memory = _world(tmp_path, monkeypatch, malformed.encode())
    core, plan = _plan(env, memory)
    assert all("synthetic-class-00" not in _changing(plan, mode) for mode in ("max", "low", "nano"))
    registry = ToolRegistry(repo_dir=env.repo_dir, drive_root=env.drive_root)
    registry.set_context(ToolContext(repo_dir=env.repo_dir, drive_root=env.drive_root, task_id="t-text"))
    read = registry.execute("knowledge_read", {"topic": "patterns", "scope": "global"})
    assert "[Metadata unavailable; the complete original Markdown follows.]" in read and read.endswith(malformed)


def test_the_reader_returns_the_current_register_whole_beyond_the_nano_target(tmp_path, monkeypatch):
    """The reader is the current note; the capture holds its summary in every mode."""
    from ouroboros import knowledge as store
    from ouroboros.context_budget import OWNER_NANO_TARGET_TOKENS
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    large = REGISTER + "".join(f"| synthetic-bulk-{n:05d} | {'cause ' * 12}| fix |\n" for n in range(4_400))
    assert len(large) > 4 * OWNER_NANO_TARGET_TOKENS  # above the target by the host's chars/4 estimate
    env, memory = _world(tmp_path, monkeypatch, large.encode())
    core, plan = _plan(env, memory)
    assert all("synthetic-bulk-" not in _changing(plan, mode) for mode in ("max", "low", "nano"))
    registry = ToolRegistry(repo_dir=env.repo_dir, drive_root=env.drive_root)
    registry.set_context(ToolContext(repo_dir=env.repo_dir, drive_root=env.drive_root, task_id="t-text"))
    note = store.read_knowledge_note(store.resolve_knowledge_address(env.drive_root, "patterns", "global"))
    edited = registry.execute("knowledge_write", {
        "topic": "patterns", "mode": "edit", "old_str": "SYNTHETIC REGISTER END\n",
        "content": "SYNTHETIC REGISTER END (edited after capture)\n", "expected_revision": note.revision})
    assert edited.startswith("✅"), edited
    read = registry.execute("knowledge_read", {"topic": "patterns", "scope": "global"})
    current = (env.drive_root / "memory" / "knowledge" / "patterns.md").read_text(encoding="utf-8")
    source = json.loads(read.split("\n", 1)[0].removeprefix("[Knowledge source] "))
    assert source["complete_chars"] == source["end_char"] == len(current) and read.endswith(current)
    assert "(edited after capture)" in read and "(edited after capture)" not in core.dynamic_head_text
    assert _changing(plan, "max").count("(edited after capture)") == 0
