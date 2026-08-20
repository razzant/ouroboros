"""The skill preflight: what it reports, what it tolerates, and what it degrades.

Split out of ``tests/test_skill_exec.py`` by theme: the clean run that leaves no pycache,
the python syntax error it reports, the file-limit omission that is degraded rather than
blocked, the missing validator runtime it tolerates, the literal widget schema it
validates, the dynamic one it degrades, and the missing PluginAPI permissions it names.
"""

from __future__ import annotations

import json

from tests._skill_exec_shared import (
    _build_skill,
    _make_ctx,
)
from tests._skill_exec_shared import (  # noqa: F401  (autouse fixture applies on import)
    _clean_extension_runtime,
)


def test_skill_preflight_success_and_no_pycache(tmp_path, monkeypatch):
    ctx = _make_ctx(tmp_path)
    skills_root = tmp_path / "skills"
    skills_root.mkdir()
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    skill_dir = _build_skill(skills_root, "alpha", script_body="print('ok')\n")

    from ouroboros.tools.skill_preflight import _handle_skill_preflight

    result = json.loads(_handle_skill_preflight(ctx, skill="alpha"))

    assert result["ok"] is True
    assert result["files_checked"] >= 1
    assert result["files_failed"] == 0
    assert not (skill_dir / "scripts" / "__pycache__").exists()


def test_skill_preflight_reports_python_syntax_error(tmp_path, monkeypatch):
    ctx = _make_ctx(tmp_path)
    skills_root = tmp_path / "skills"
    skills_root.mkdir()
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    _build_skill(skills_root, "alpha", script_body="def broken(:\n")

    from ouroboros.tools.skill_preflight import _handle_skill_preflight

    result = json.loads(_handle_skill_preflight(ctx, skill="alpha"))

    assert result["ok"] is False
    assert result["files_failed"] == 1
    assert "SyntaxError" in result["files"][0]["stderr"]


def test_skill_preflight_file_limit_omission_is_degraded_not_blocked(tmp_path, monkeypatch):
    # A file count beyond the syntax-check headroom is a DEGRADED note, NOT a hard block:
    # the skill-review pass now reads every file under a pack-level token budget (chunked
    # when oversized), so preflight must not re-introduce an arbitrary file-count gate.
    ctx = _make_ctx(tmp_path)
    skills_root = tmp_path / "skills"
    skills_root.mkdir()
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    skill_dir = _build_skill(skills_root, "alpha")
    scripts = skill_dir / "scripts"
    from ouroboros.tools import skill_preflight as sp

    for idx in range(sp._PREFLIGHT_HARD_FILE_LIMIT + 2):
        (scripts / f"extra_{idx}.py").write_text("print('ok')\n", encoding="utf-8")

    result = json.loads(sp._handle_skill_preflight(ctx, skill="alpha"))

    assert result["ok"] is True  # proceeds to the authoritative token-budgeted review
    assert result["omitted_count"] > 0
    assert result.get("degraded") is True
    assert "token budget" in result.get("degraded_note", "")


def test_skill_preflight_missing_validator_runtime_is_tolerated(tmp_path, monkeypatch):
    # A missing external runtime (e.g. node not installed, or a Homebrew node
    # code-signing-killed by macOS) is an environment gap, not a syntax verdict.
    # Preflight must skip it rather than block; tri-model review stays authoritative.
    ctx = _make_ctx(tmp_path)
    skills_root = tmp_path / "skills"
    skills_root.mkdir()
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    skill_dir = _build_skill(skills_root, "alpha")
    (skill_dir / "scripts" / "check.js").write_text("console.log('ok')\n", encoding="utf-8")

    from ouroboros.tools import skill_preflight as sp
    monkeypatch.setattr(sp, "_resolve_runtime", lambda runtime: None if runtime == "node" else "/bin/echo")

    result = json.loads(sp._handle_skill_preflight(ctx, skill="alpha", paths=["scripts/check.js"]))

    assert result["ok"] is True
    assert result.get("degraded") is True
    js = next(f for f in result["files"] if f["path"].endswith("check.js"))
    assert js.get("skipped") is True
    assert js.get("skip_reason") == "runtime_unavailable"


def test_skill_preflight_validates_literal_widget_schema(tmp_path, monkeypatch):
    ctx = _make_ctx(tmp_path)
    skills_root = tmp_path / "skills"
    skills_root.mkdir()
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    manifest = (
        "---\n"
        "name: alpha\n"
        "description: widget test\n"
        "version: 0.1.0\n"
        "type: extension\n"
        "entry: plugin.py\n"
        "permissions: [widget, route]\n"
        "---\n"
        "body\n"
    )
    skill_dir = skills_root / "alpha"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(manifest, encoding="utf-8")
    (skill_dir / "plugin.py").write_text(
        "_UI_RENDER = {\n"
        "    'kind': 'declarative',\n"
        "    'schema_version': 1,\n"
        "    'components': [\n"
        "        {'type': 'form', 'action_route': 'generate', 'fields': [{'name': 'prompt'}]},\n"
        "    ],\n"
        "}\n"
        "def register(api):\n"
        "    api.register_ui_tab('main', 'Main', render=_UI_RENDER)\n",
        encoding="utf-8",
    )

    from ouroboros.tools import skill_preflight as sp

    result = json.loads(sp._handle_skill_preflight(ctx, skill="alpha"))

    assert result["ok"] is False
    assert any("requires route or api_route" in item["detail"] for item in result["widgets"])


def test_skill_preflight_reports_dynamic_widget_schema_as_degraded(tmp_path, monkeypatch):
    ctx = _make_ctx(tmp_path)
    skills_root = tmp_path / "skills"
    skills_root.mkdir()
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    manifest = (
        "---\n"
        "name: alpha\n"
        "description: dynamic widget test\n"
        "version: 0.1.0\n"
        "type: extension\n"
        "entry: plugin.py\n"
        "permissions: [widget]\n"
        "---\n"
        "body\n"
    )
    skill_dir = skills_root / "alpha"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(manifest, encoding="utf-8")
    (skill_dir / "plugin.py").write_text(
        "def make_render(mode):\n"
        "    return {'kind': 'declarative', 'components': []}\n"
        "def register(api):\n"
        "    api.register_ui_tab('main', 'Main', render=make_render('full'))\n",
        encoding="utf-8",
    )

    from ouroboros.tools import skill_preflight as sp

    result = json.loads(sp._handle_skill_preflight(ctx, skill="alpha"))

    assert result["ok"] is True
    assert result["degraded"] is True
    assert "dynamic UI schema" in result["degraded_note"]
    assert result["widgets"][0]["verified"] is False
    assert result["widgets"][0]["skip_reason"] == "dynamic_ui_schema"


def test_skill_preflight_reports_missing_pluginapi_permissions(tmp_path, monkeypatch):
    ctx = _make_ctx(tmp_path)
    skills_root = tmp_path / "skills"
    skills_root.mkdir()
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    manifest = (
        "---\n"
        "name: alpha\n"
        "description: permissions test\n"
        "version: 0.1.0\n"
        "type: extension\n"
        "entry: plugin.py\n"
        "permissions: [net]\n"
        "env_from_settings: [OPENROUTER_API_KEY]\n"
        "---\n"
        "body\n"
    )
    skill_dir = skills_root / "alpha"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(manifest, encoding="utf-8")
    (skill_dir / "plugin.py").write_text(
        "def register(api):\n"
        "    api.register_route('status', lambda request: {})\n"
        "    api.register_ui_tab('main', 'Main', render={'kind':'declarative','schema_version':1,'components': []})\n"
        "    api.get_settings(['OPENROUTER_API_KEY'])\n",
        encoding="utf-8",
    )

    from ouroboros.tools import skill_preflight as sp

    result = json.loads(sp._handle_skill_preflight(ctx, skill="alpha"))

    assert result["ok"] is False
    missing = {item["permission"] for item in result["permissions"] if not item["ok"]}
    assert {"route", "widget", "read_settings"} <= missing
