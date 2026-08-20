"""The health environment builder shared by the context suites.

Split out of ``tests/test_context.py`` when that module was divided by theme; the
builder is verbatim, so every sibling suite keeps the exact drive layout and state it
was written against.
"""

from __future__ import annotations





def _make_health_env(tmp_path, events_lines=None):
    class FakeEnv:
        def drive_path(self, p):
            return tmp_path / p

        def repo_path(self, p):
            return tmp_path / "repo" / p

        @property
        def repo_dir(self):
            return tmp_path / "repo"

        @property
        def drive_root(self):
            return tmp_path

    (tmp_path / "state").mkdir(parents=True, exist_ok=True)
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "memory").mkdir(parents=True, exist_ok=True)
    (tmp_path / "repo" / "docs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "repo" / "prompts").mkdir(parents=True, exist_ok=True)
    (tmp_path / "archive" / "rescue").mkdir(parents=True, exist_ok=True)
    (tmp_path / "repo" / "VERSION").write_text("1.2.3", encoding="utf-8")
    (tmp_path / "repo" / "pyproject.toml").write_text('version = "1.2.3"', encoding="utf-8")
    (tmp_path / "repo" / "web").mkdir(parents=True, exist_ok=True)
    (tmp_path / "repo" / "web" / "package.json").write_text('{"version": "1.2.3"}', encoding="utf-8")
    (tmp_path / "repo" / "README.md").write_text('version-1.2.3', encoding="utf-8")
    (tmp_path / "repo" / "docs" / "ARCHITECTURE.md").write_text('# Ouroboros v1.2.3', encoding="utf-8")
    (tmp_path / "repo" / "docs" / "DEVELOPMENT.md").write_text('# Dev', encoding="utf-8")
    (tmp_path / "repo" / "prompts" / "CONSCIOUSNESS.md").write_text('Prompt text', encoding="utf-8")
    (tmp_path / "state" / "state.json").write_text('{"spent_usd": 0, "budget_drift_alert": false}', encoding="utf-8")
    (tmp_path / "memory" / "identity.md").write_text('x' * 300, encoding="utf-8")
    (tmp_path / "memory" / "scratchpad.md").write_text('x' * 300, encoding="utf-8")
    event_lines = events_lines or []
    (tmp_path / "logs" / "events.jsonl").write_text("\n".join(event_lines) + ("\n" if event_lines else ""), encoding="utf-8")
    return FakeEnv()
