"""CPL4-C17 pins: knowledge history appends through the sidecar-lock seam.

``knowledge_history.jsonl`` (and the since-removed ``knowledge_journal.jsonl``
size telemetry) used raw ``open("a")`` — the one torn-line hazard left among
the memory journals. Every history row is now the common note writer's, so the
backlog has no history writer of its own either.
"""

from __future__ import annotations

import inspect


def test_knowledge_module_has_no_raw_journal_appends():
    import ouroboros.tools.knowledge as knowledge
    import ouroboros.knowledge as knowledge_store

    sources = [inspect.getsource(knowledge), inspect.getsource(knowledge_store)]
    assert all('open(history_path, "a"' not in src for src in sources)
    assert all('open(journal_path, "a"' not in src for src in sources)
    assert inspect.getsource(knowledge_store).count("append_jsonl(") >= 2
    assert "append_jsonl" not in inspect.getsource(knowledge)  # the tool writes through the store only


def test_knowledge_history_rows_land_via_locked_seam(tmp_path, monkeypatch):
    import ouroboros.knowledge as knowledge_store
    import ouroboros.tools.knowledge as knowledge
    from ouroboros.tools.registry import ToolContext

    calls = []
    real = knowledge_store.append_jsonl

    def spy(path, obj, **kw):
        calls.append(path.name)
        return real(path, obj, **kw)

    monkeypatch.setattr(knowledge_store, "append_jsonl", spy)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="t1")
    assert knowledge._knowledge_write(ctx, "improvement-backlog", "- row", mode="overwrite").startswith("✅")

    assert calls == ["knowledge_history.jsonl"]
    line = (tmp_path / "memory" / "knowledge_history.jsonl").read_text(encoding="utf-8")
    assert '"topic": "improvement-backlog"' in line and line.endswith("\n")
