"""CPL4-C17 pins: knowledge history appends through the sidecar-lock seam.

``knowledge_history.jsonl`` (and the since-removed ``knowledge_journal.jsonl``
size telemetry) used raw ``open("a")`` — the one torn-line hazard left among
the memory journals.
"""

from __future__ import annotations

import inspect


def test_knowledge_module_has_no_raw_journal_appends():
    import ouroboros.tools.knowledge as knowledge
    import ouroboros.knowledge as knowledge_store

    sources = [inspect.getsource(knowledge), inspect.getsource(knowledge_store)]
    assert all('open(history_path, "a"' not in src for src in sources)
    assert all('open(journal_path, "a"' not in src for src in sources)
    assert sum(src.count("append_jsonl(") for src in sources) >= 3


def test_knowledge_history_rows_land_via_locked_seam(tmp_path, monkeypatch):
    import ouroboros.tools.knowledge as knowledge

    calls = []
    real = knowledge.append_jsonl

    def spy(path, obj, **kw):
        calls.append(path.name)
        return real(path, obj, **kw)

    monkeypatch.setattr(knowledge, "append_jsonl", spy)
    backlog = tmp_path / "memory" / "knowledge" / "backlog.md"
    backlog.parent.mkdir(parents=True)
    backlog.write_text("row", encoding="utf-8")

    knowledge._record_backlog_history(backlog, "topic-x", "append", "t1")

    assert calls == ["knowledge_history.jsonl"]
    line = (tmp_path / "memory" / "knowledge_history.jsonl").read_text(encoding="utf-8")
    assert '"topic": "topic-x"' in line and line.endswith("\n")
