"""Real example routes use their host request root in both dispatch modes."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import shutil

import httpx
import pytest

from tests._extension_loader_shared import (
    _add_fake_native_dep,
    _clear_loader_state as _clear_loader_state,
    _mark_isolated_deps_installed,
    _write_ext_skill,
)

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "docs/examples/author_ui_kit"


def _copy_installed_runtime(installed):
    # The real OOP route worker starts python -m from request.app.state.repo_dir.
    for name in ("ouroboros", "supervisor"):
        shutil.copytree(REPO / name, installed / name, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(REPO / "VERSION", installed / "VERSION")


@pytest.mark.parametrize("out_of_process", [False, True])
def test_author_routes_read_request_repo_at_each_mount(tmp_path, monkeypatch, out_of_process):
    from starlette.applications import Starlette
    from starlette.routing import Route
    from ouroboros import extension_loader
    from ouroboros.gateway.extensions import api_extension_dispatch
    from ouroboros.skill_loader import find_skill, save_enabled, save_review_state, SkillReviewState

    drive, skills, installed = tmp_path / "drive", tmp_path / "skills", tmp_path / "installed"
    drive.mkdir()
    (installed / "web/modules").mkdir(parents=True)
    _copy_installed_runtime(installed)
    css = installed / "web/ui.css"
    javascript = installed / "web/modules/ui_primitives.js"
    css.write_text('.ouro-ui { --fixture-installed: 1; } /* </style><script>bad()</script> */', encoding="utf-8")
    javascript.write_text('export const example = "</script><script>bad()</script>";', encoding="utf-8")
    monkeypatch.setattr("ouroboros.config.get_skills_repo_path", lambda: str(skills))
    skill_dir = _write_ext_skill(
        skills, "author_ui_kit", plugin_body=(EXAMPLE / "plugin.py").read_text(encoding="utf-8"),
        permissions=["route", "widget"],
        extra_frontmatter='plugin_api: "2.0"\n' + ("dependencies:\n  - dummy_pkg\n" if out_of_process else ""),
    )
    (skill_dir / "widget.js").write_text(
        (EXAMPLE / "widget.js").read_text(encoding="utf-8"), encoding="utf-8",
    )
    loaded = find_skill(drive, "author_ui_kit", repo_path=str(skills))
    save_enabled(drive, loaded.name, True)
    save_review_state(drive, loaded.name, SkillReviewState(status="pass", content_hash=loaded.content_hash))
    loaded = find_skill(drive, loaded.name, repo_path=str(skills))
    if out_of_process:
        _add_fake_native_dep(loaded)
        _mark_isolated_deps_installed(drive, loaded)
    assert extension_loader.load_extension(loaded, lambda: {}, drive_root=drive, repo_path=str(skills)) is None
    spec = extension_loader.list_routes()["/api/extensions/author_ui_kit/author-kit"]
    assert bool(spec.get("out_of_process")) is out_of_process
    app = Starlette(routes=[Route("/api/extensions/{skill}/{rest:path}", api_extension_dispatch)])
    app.state.drive_root, app.state.repo_dir = drive, installed

    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://fixture") as client:
            response = await client.get("/api/extensions/author_ui_kit/author-kit")
            assert response.status_code == 200, response.text
            assert response.headers["cache-control"] == "no-store"
            assert response.json() == {
                "css": css.read_text(encoding="utf-8"),
                "javascript": javascript.read_text(encoding="utf-8"),
            }
            page = await client.get("/api/extensions/author_ui_kit/page")
            assert page.status_code == 200, page.text
            assert page.headers["cache-control"] == "no-store"
            assert "script-src 'nonce-" in page.headers["content-security-policy"]
            embedded = page.text.split('<script id="author-kit-source" type="application/json">', 1)[1].split("</script>", 1)[0]
            assert "<" not in embedded
            source = json.loads(embedded)
            assert source["css"] == css.read_text(encoding="utf-8")
            assert source["javascript"] == javascript.read_text(encoding="utf-8")
            assert source["application"] == (EXAMPLE / "widget.js").read_text(encoding="utf-8")
            css.write_text(".ouro-ui { --fixture-installed: 2; }", encoding="utf-8")
            current = await client.get("/api/extensions/author_ui_kit/author-kit")
            fresh_page = await client.get("/api/extensions/author_ui_kit/page")
            assert current.json()["css"] == css.read_text(encoding="utf-8")
            assert "--fixture-installed: 2" in fresh_page.text

    asyncio.run(check())
