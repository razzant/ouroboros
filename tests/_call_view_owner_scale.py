"""A data root at the sizes measured on a mature install, with synthetic Russian and English text only.

The composition follows a mature install whose whole call context measured about 184k tokens:
an identity of ~41 KB, a ten-block scratchpad, a knowledge index of ~74k characters over
~120 notes with authored summaries, a ~120k-character shared understanding in dated sections, a
~50k-character Pattern Register, 207 delegated-run obligations in Health Invariants, a subagent
catalog, execution reflections, fifteen other live rooms with a note each, a long Main room and
a story of sealed pages. No owner content: every text is generated from the word lists below.
"""
from __future__ import annotations

import itertools
import json
import pathlib
from types import SimpleNamespace
from typing import Any, Dict, List

RU = ("память", "задача", "комната", "звонок", "контекст", "решение", "проверка", "владелец", "история",
      "страница", "граница", "причина", "заметка", "работа", "вопрос", "ответ", "сессия", "модель")
EN = ("memory", "task", "room", "call", "context", "decision", "review", "owner", "story", "page", "window",
      "pointer", "section", "facts", "record", "view", "engine", "route")
MIND = {"kind": "mind", "task_id": "t-owner-scale", "focus": {"role": "root", "task_id": "t-owner-scale"}}
CALL_TASK_ID = "call-owner-scale"


def text(chars: int, seed: int = 0) -> str:
    """Deterministic mixed Russian and English prose of about ``chars`` characters."""
    words = itertools.cycle(itertools.chain.from_iterable(zip(RU[seed % 7:] + RU[:seed % 7], EN)))
    out, size = [], 0
    for index, word in enumerate(words):
        piece = word + ("." if index % 11 == 10 else ",") if index % 5 == 4 else word
        out.append(piece)
        size += len(piece) + 1
        if size >= chars:
            break
    return " ".join(out)


def _write(path: pathlib.Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _knowledge(root: pathlib.Path) -> None:
    shelf = root / "memory" / "knowledge"
    for i in range(120):
        summary = json.dumps(f"Note {i}: " + text(540, i))
        _write(shelf / f"topic-{i:03d}.md", f"---\ntype: note\nsummary: {summary}\n---\n\n# Topic {i}\n\n{text(900, i)}\n")
    sections = "".join(f"\n## 2026-10-{day:02d} — {text(30, day)}\n\n{text(7_000, day)}\n"
                       f"\n### {text(24, day + 1)}\n\n{text(7_600, day + 2)}\n" for day in range(1, 9))
    summary = json.dumps("Where we stand: " + text(700, 99))
    _write(shelf / "overview.md", f"---\ntype: note\nsummary: {summary}\n---\n\n# Shared understanding\n\n{sections}")
    rows = "\n".join(f"| P-{i:03d} | {text(160, i)} | {text(160, i + 3)} |" for i in range(140))
    _write(shelf / "patterns.md", "## Current reading boundary — 16 October 2026\n\n| id | pattern | lesson |\n"
                                  f"|---|---|---|\n{rows}\n")


def _memory(root: pathlib.Path) -> None:
    _write(root / "memory" / "identity.md", "# Who I am\n\n" + "\n\n".join(
        f"## {text(20, i)}\n\n{text(3_400, i)}" for i in range(12)))
    _write(root / "memory" / "scratchpad.md", "\n\n".join(
        f"### Block {i} — 2026-10-{i + 1:02d}\n\n{text(1_750, i + 40)}" for i in range(10)))
    _write(root / "memory" / "WORLD.md", "\n\n".join(
        f"## {name}\n\n{text(1_400, i)}" for i, name in enumerate(("System", "Available Tools", "File System Rules",
                                                                    "Network"))))
    reflections = [{"ts": f"2026-10-{1 + i % 15:02d}T10:{i:02d}:00Z", "type": "acceptance_late_settlement",
                    "task_type": "task", "task_id": f"r{i:04d}", "reflection": text(360, i)} for i in range(20)]
    _write(root / "logs" / "task_reflections.jsonl", "".join(json.dumps(row, ensure_ascii=False) + "\n"
                                                             for row in reflections))


def _rows(root: pathlib.Path, rooms: List[int]) -> List[Dict[str, Any]]:
    rows = []
    for i in range(44):  # the long Main room: the owner and my replies
        owner = i % 2 == 0
        rows.append({"chat_id": 1, "direction": "in" if owner else "out", "ts": f"2026-10-16T{8 + i // 6:02d}:"
                     f"{i % 6 * 9:02d}:00+00:00", "text": text(900 if owner else 2_200, i),
                     **({"client_message_id": f"m{i}"} if owner else {"task_id": f"t{i:03d}"})})
    for j, chat_id in enumerate(rooms):
        rows.append({"chat_id": chat_id, "direction": "in", "ts": f"2026-10-15T{6 + j % 12:02d}:00:00+00:00",
                     "text": text(300, j), "client_message_id": f"p{j}"})
    _write(root / "logs" / "chat.jsonl", "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    return rows


def build(root: pathlib.Path) -> SimpleNamespace:
    """The data root, activated; returns the rooms and the call task."""
    from ouroboros.chronicle_store import ChronicleStore
    from ouroboros.projects_registry import create_project
    from tests._memory_production_shape import page

    for name in ("state", "logs", "memory/knowledge"):
        (root / name).mkdir(parents=True, exist_ok=True)
    _write(root / "state" / "state.json", '{"spent_usd": 0}')
    _knowledge(root)
    _memory(root)
    rooms = [int(create_project(root, f"room-{j:02d}", name=f"Project {j} {text(20, j)}")["chat_id"]) for j in range(15)]
    _rows(root, rooms)
    store = ChronicleStore(root)
    assert store.ensure_activated()["kind"] == "activation"
    for j, chat_id in enumerate(rooms):
        assert store.write_note(room_id=chat_id, task_id=f"pt{j}", text=text(1_450, j), author=MIND).ok
    for k in range(10):  # sealed pages over the older Main rows, two rows each
        page(root, "1", first=2 * k, last=2 * k + 1, text=text(2_100, k + 60))
    return SimpleNamespace(root=root, rooms=rooms)


def call_task() -> Dict[str, Any]:
    from ouroboros.call_protocol import CallSpec
    from ouroboros.contracts.task_contract import attach_task_contract

    spec = CallSpec("call-owner-scale", 1, "google", "gemini-3.8-live", "Kore", opened_by="owner",
                    presence_binding_id="", max_minutes=0, label="Owner-scale call", client_message_id="cm-1")
    task = {"id": CALL_TASK_ID, "type": "task", "chat_id": 1, "_is_direct_chat": True,
            "text": "[Live call] A live voice call is open in this room.",
            "metadata": {"live_call": spec.as_metadata()}}
    attach_task_contract(task)
    return task


def obligations(monkeypatch) -> None:
    """The 207 delegated-run obligations of the owner's Health Invariants (same shape as their lines)."""
    from tests.test_context_operative_views import _obligations

    _obligations(monkeypatch)


def subagents(monkeypatch) -> None:
    catalog = {"rows": [{"subagent_id": f"helper-{i:02d}", "route_class": "Agent session" if i % 3 else "API model",
                         "effort": "auto", "requested_model": f"model-{i}", "review_eligible": i % 4 == 0,
                         "delivery": "native", "recommended_use": text(700, i)} for i in range(22)]}
    monkeypatch.setattr("ouroboros.subagent_runtime.current_model_visible_subagent_catalog", lambda: catalog)
