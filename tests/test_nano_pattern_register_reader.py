"""Nano carries the Pattern Register's reader line, never its body; the capture keeps the body.

Owner decision 2026-10-20 (mailbox row 2302258): take the register out of Nano and keep a link.
The capture reads the register once and keeps it whole in the immutable core, so Max, Low,
a roomy route change and refusal-selected mode retain it; physical window pressure may
address host sections in any mode. Nano always shows the existing reader line in its place,
for every role alike (text and call; no voice or provider condition). The index keeps its ``patterns`` line and the reader reads
the current note whole. Fixture: a SYNTHETIC drive under ``tmp_path``; no live data.
"""
from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ouroboros.context_fit import PATTERN_REGISTER_POINTER

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


def test_nano_shows_the_reader_where_max_and_low_show_the_captured_register(tmp_path, monkeypatch):
    env, memory = _world(tmp_path, monkeypatch)
    before = _files(env.drive_root)
    core, plan = _plan(env, memory)
    section = f"{HEADER}\n\n{REGISTER}"
    assert core.pattern_register_text == section and core.dynamic_head_text.count(section) == 1
    nano_head = core.dynamic_head_text.replace(section, PATTERN_REGISTER_POINTER)
    for mode in ("max", "low"):
        assert _changing(plan, mode).startswith(core.dynamic_head_text), mode
    nano = _changing(plan, "nano")
    assert nano.startswith(nano_head) and nano.count(PATTERN_REGISTER_POINTER) == 1
    assert "synthetic-class-" not in nano and "SYNTHETIC REGISTER END" not in nano
    # The overview and the knowledge index (its ``patterns`` line included) ride unchanged.
    assert "SYNTHETIC OVERVIEW." in nano_head and "- **patterns**: [Pattern Register](<patterns.md>)" in nano_head
    assert nano_head.index(PATTERN_REGISTER_POINTER) == core.dynamic_head_text.index(section)
    # Max and Low are the bytes of a capture without the field (the rendering before this change);
    # Nano differs only in its changing block, and its governance is the same bytes.
    _, unchanged = _plan(env, memory, core=replace(core, pattern_register_text=""))
    for mode in ("max", "low"):
        assert plan.projection(mode).system_content_json == unchanged.projection(mode).system_content_json, mode
    ours, theirs = (json.loads(p.projection("nano").system_content_json) for p in (plan, unchanged))
    assert ours[:-1] == theirs[:-1] and section in theirs[-1]["text"] and section not in ours[-1]["text"]
    assert _files(env.drive_root) == before  # capture and rendering write nothing


def test_text_and_call_share_one_nano_rendering_of_the_register(tmp_path, monkeypatch):
    env, memory = _world(tmp_path, monkeypatch)
    call = {"id": "t-call", "type": "presence", "source": "presence", "_presence_turn": True,
            "context_requires_development": False, "chat_id": 777, "text": "Fix the flaky test."}
    heads = []
    for task in (TEXT_TASK, call):
        core, plan = _plan(env, memory, task)
        nano = _changing(plan, "nano")
        assert nano.count(PATTERN_REGISTER_POINTER) == 1 and "synthetic-class-" not in nano, task["id"]
        assert all("synthetic-class-07" in _changing(plan, mode) for mode in ("max", "low")), task["id"]
        heads.append(core.dynamic_head_text[:core.dynamic_head_text.index(HEADER)] + PATTERN_REGISTER_POINTER)
    assert heads[0] == heads[1]


def test_a_route_change_and_a_refusal_selected_mode_render_the_captured_body(tmp_path, monkeypatch):
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
        for mode in ("max", "low"):
            addressed = [row for row in rebound.projection(mode).memory_facts.get("host_sections_addressed", ())
                         if row["section_id"] == "pattern_register"]
            if addressed:
                from ouroboros.artifacts import read_actor_source_bytes
                retained = read_actor_source_bytes(env.drive_root, core.section_task_id, addressed[0]["source_ref"])
                assert retained.decode() == core.pattern_register_text
                assert PATTERN_REGISTER_POINTER in _changing(rebound, mode)
            else:
                assert core.pattern_register_text in _changing(rebound, mode), mode
        nano = _changing(rebound, "nano")
        assert nano.count(PATTERN_REGISTER_POINTER) == 1 and "synthetic-class-" not in nano
        assert all("EDITED LATER" not in _changing(rebound, mode) for mode in ("max", "low", "nano"))


@pytest.mark.parametrize("register", [None, b"  \n\t\n", b"---\ntitle: Register\n---\nbad bytes \xff\n"],
                         ids=["missing", "blank", "undecodable"])
def test_a_register_max_does_not_carry_leaves_nano_identical_to_max(tmp_path, monkeypatch, register):
    env, memory = _world(tmp_path, monkeypatch, register)
    core, plan = _plan(env, memory)
    assert core.pattern_register_text == "" and HEADER not in core.dynamic_head_text
    assert PATTERN_REGISTER_POINTER not in _changing(plan, "nano")
    assert _changing(plan, "nano").startswith(core.dynamic_head_text)


def test_malformed_metadata_still_swaps_the_body_and_the_reader_returns_the_original(tmp_path, monkeypatch):
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    malformed = "---\ntitle: [unclosed\n---\n# Pattern Register\n\n| synthetic-class-00 | cause | fix |\n"
    env, memory = _world(tmp_path, monkeypatch, malformed.encode())
    core, plan = _plan(env, memory)
    assert core.pattern_register_text == f"{HEADER}\n\n{malformed}"
    assert "synthetic-class-00" in _changing(plan, "max") and "synthetic-class-00" not in _changing(plan, "nano")
    registry = ToolRegistry(repo_dir=env.repo_dir, drive_root=env.drive_root)
    registry.set_context(ToolContext(repo_dir=env.repo_dir, drive_root=env.drive_root, task_id="t-text"))
    read = registry.execute("knowledge_read", {"topic": "patterns", "scope": "global"})
    assert "[Metadata unavailable; the complete original Markdown follows.]" in read and read.endswith(malformed)


def test_the_reader_returns_the_current_register_whole_beyond_the_nano_target(tmp_path, monkeypatch):
    """The reader is the CURRENT note; the captured body stays what the task's Max/Low show."""
    from ouroboros import knowledge as store
    from ouroboros.context_budget import OWNER_NANO_TARGET_TOKENS
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    large = REGISTER + "".join(f"| synthetic-bulk-{n:05d} | {'cause ' * 12}| fix |\n" for n in range(4_400))
    assert len(large) > 4 * OWNER_NANO_TARGET_TOKENS  # above the target by the host's chars/4 estimate
    env, memory = _world(tmp_path, monkeypatch, large.encode())
    core, plan = _plan(env, memory)
    assert "synthetic-bulk-04399" in _changing(plan, "max") and "synthetic-bulk-" not in _changing(plan, "nano")
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
    assert "(edited after capture)" in read and "(edited after capture)" not in core.pattern_register_text
    assert _changing(plan, "max").count("(edited after capture)") == 0
