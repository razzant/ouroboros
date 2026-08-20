"""The extension dispatcher and the assets it serves.

Split verbatim out of ``tests/test_extensions_api.py`` by theme. This module owns the
route that reaches a registered handler, the module entry served only while the
extension is live, the settings section scoped to one skill, and the dispatcher's
answers to a HEAD request, an unknown route, a lazy-load error, a not-live route and a
stale live route.
"""

from __future__ import annotations





from tests._extensions_api_shared import (  # noqa: F401  (autouse fixture applies on import)
    _clean_extensions,
    _make_client,
    _stop_patches,
    _write_ext,
)


def test_api_extension_dispatcher_routes_to_registered_handler(tmp_path, monkeypatch):
    from ouroboros import extension_loader
    from ouroboros.skill_loader import (
        SkillReviewState,
        compute_content_hash,
        save_enabled,
        save_review_state,
    )

    skills_root = tmp_path / "skills"
    plugin = (
        "from starlette.responses import JSONResponse\n"
        "def _hello(request):\n"
        "    return JSONResponse({'hello': 'world'})\n"
        "def register(api):\n"
        "    api.register_route('greet', _hello, methods=('GET',))\n"
    )
    skill_dir = _write_ext(skills_root, "ext_route", permissions=["route"], plugin=plugin)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    client, drive_root, patches = _make_client(tmp_path, monkeypatch)
    try:
        content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
        save_enabled(drive_root, "ext_route", True)
        save_review_state(
            drive_root,
            "ext_route",
            SkillReviewState(status="pass", content_hash=content_hash),
        )
        from ouroboros.skill_loader import find_skill
        from ouroboros.config import load_settings
        refreshed = find_skill(drive_root, "ext_route", repo_path=str(skills_root))
        err = extension_loader.load_extension(refreshed, load_settings, drive_root=drive_root)
        assert err is None, err

        resp = client.get("/api/extensions/ext_route/greet")
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"hello": "world"}
    finally:
        _stop_patches(patches)


def test_api_extension_module_serves_only_live_declared_entry(tmp_path, monkeypatch):
    from ouroboros import extension_loader
    from ouroboros.skill_loader import (
        SkillReviewState,
        compute_content_hash,
        find_skill,
        save_enabled,
        save_review_state,
    )

    skills_root = tmp_path / "skills"
    plugin = (
        "def register(api):\n"
        "    api.register_ui_tab('module', 'Module', render={'kind': 'module', 'entry': 'widget.js'})\n"
    )
    skill_dir = _write_ext(skills_root, "ext_module", permissions=["widget"], plugin=plugin)
    (skill_dir / "widget.js").write_text("window.__ok = true;\n", encoding="utf-8")
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    client, drive_root, patches = _make_client(tmp_path, monkeypatch)
    try:
        content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
        save_enabled(drive_root, "ext_module", True)
        save_review_state(
            drive_root,
            "ext_module",
            SkillReviewState(status="pass", content_hash=content_hash),
        )
        loaded = find_skill(drive_root, "ext_module", repo_path=str(skills_root))
        assert loaded is not None
        err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
        assert err is None, err

        ok = client.get("/api/extensions/ext_module/module/widget.js")
        assert ok.status_code == 200, ok.text
        assert "window.__ok" in ok.text
        assert ok.headers["cache-control"] == "no-store"

        assert client.get("/api/extensions/ext_module/module/other.js").status_code == 404
        assert client.get("/api/extensions/ext_module/module/../widget.js").status_code in {400, 404}
    finally:
        _stop_patches(patches)


def test_api_extension_module_rejects_non_live_extension(tmp_path, monkeypatch):
    skills_root = tmp_path / "skills"
    plugin = (
        "def register(api):\n"
        "    api.register_ui_tab('module', 'Module', render={'kind': 'module', 'entry': 'widget.js'})\n"
    )
    _write_ext(skills_root, "ext_module", permissions=["widget"], plugin=plugin)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    client, _, patches = _make_client(tmp_path, monkeypatch)
    try:
        resp = client.get("/api/extensions/ext_module/module/widget.js")
        assert resp.status_code == 409
    finally:
        _stop_patches(patches)


def test_api_extension_settings_section_returns_only_requested_skill(tmp_path, monkeypatch):
    from ouroboros import extension_loader
    from ouroboros.skill_loader import (
        SkillReviewState,
        compute_content_hash,
        find_skill,
        save_enabled,
        save_review_state,
    )

    skills_root = tmp_path / "skills"
    plugin_a = (
        "def register(api):\n"
        "    api.register_settings_section('config', 'Config A', schema={'components': [\n"
        "        {'type': 'markdown', 'text': 'A'}\n"
        "    ]})\n"
    )
    plugin_b = (
        "def register(api):\n"
        "    api.register_settings_section('config', 'Config B', schema={'components': [\n"
        "        {'type': 'markdown', 'text': 'B'}\n"
        "    ]})\n"
    )
    skill_a = _write_ext(skills_root, "settings_a", permissions=["widget"], plugin=plugin_a)
    skill_b = _write_ext(skills_root, "settings_b", permissions=["widget"], plugin=plugin_b)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    client, drive_root, patches = _make_client(tmp_path, monkeypatch)
    try:
        for name, skill_dir in {"settings_a": skill_a, "settings_b": skill_b}.items():
            content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
            save_enabled(drive_root, name, True)
            save_review_state(drive_root, name, SkillReviewState(status="pass", content_hash=content_hash))
            loaded = find_skill(drive_root, name, repo_path=str(skills_root))
            assert loaded is not None
            err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
            assert err is None, err

        resp = client.get("/api/extensions/settings_a/settings_section")
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["skill"] == "settings_a"
        assert [section["skill"] for section in data["sections"]] == ["settings_a"]
        assert data["sections"][0]["title"] == "Config A"
    finally:
        _stop_patches(patches)


def test_api_extension_dispatcher_allows_head_for_get_route(tmp_path, monkeypatch):
    from ouroboros import extension_loader
    from ouroboros.skill_loader import (
        SkillReviewState,
        compute_content_hash,
        find_skill,
        save_enabled,
        save_review_state,
    )

    skills_root = tmp_path / "skills"
    skill_dir = _write_ext(
        skills_root,
        "ext_head",
        permissions=["route"],
        plugin=(
            "from starlette.responses import JSONResponse\n"
            "def _hello(request):\n"
            "    return JSONResponse({'hello': 'world'})\n"
            "def register(api):\n"
            "    api.register_route('greet', _hello, methods=('GET',))\n"
        ),
    )
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    client, drive_root, patches = _make_client(tmp_path, monkeypatch)
    try:
        content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
        save_enabled(drive_root, "ext_head", True)
        save_review_state(
            drive_root,
            "ext_head",
            SkillReviewState(status="pass", content_hash=content_hash),
        )
        loaded = find_skill(drive_root, "ext_head", repo_path=str(skills_root))
        assert loaded is not None
        err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
        assert err is None, err

        resp = client.head("/api/extensions/ext_head/greet")
        assert resp.status_code == 200, resp.text
    finally:
        _stop_patches(patches)


def test_api_extension_dispatcher_404_for_unknown_route(tmp_path, monkeypatch):
    client, _, patches = _make_client(tmp_path, monkeypatch)
    try:
        resp = client.get("/api/extensions/nope/xyz")
        assert resp.status_code == 404
    finally:
        _stop_patches(patches)


def test_api_extension_dispatcher_surfaces_lazy_load_error(tmp_path, monkeypatch):
    from ouroboros.skill_loader import (
        SkillReviewState,
        compute_content_hash,
        save_enabled,
        save_review_state,
    )

    skills_root = tmp_path / "skills"
    plugin = (
        "def _hello(request):\n"
        "    return {'hello': 'world'}\n"
        "def register(api):\n"
        "    api.register_route('/absolute', _hello, methods=('GET',))\n"
    )
    skill_dir = _write_ext(skills_root, "ext_broken", permissions=["route"], plugin=plugin)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    client, drive_root, patches = _make_client(tmp_path, monkeypatch)
    try:
        content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
        save_enabled(drive_root, "ext_broken", True)
        save_review_state(
            drive_root,
            "ext_broken",
            SkillReviewState(status="pass", content_hash=content_hash),
        )

        resp = client.get("/api/extensions/ext_broken/greet")
        assert resp.status_code == 409, resp.text
        data = resp.json()
        assert data["state"]["action"] == "extension_load_error"
        assert data["state"]["reason"] == "load_error"
    finally:
        _stop_patches(patches)


def test_api_extension_dispatcher_rejects_not_live_route(tmp_path, monkeypatch):
    from ouroboros import extension_loader
    from ouroboros.skill_loader import (
        SkillReviewState,
        compute_content_hash,
        save_enabled,
        save_review_state,
    )

    skills_root = tmp_path / "skills"
    plugin = (
        "from starlette.responses import JSONResponse\n"
        "def _hello(request):\n"
        "    return JSONResponse({'hello': 'world'})\n"
        "def register(api):\n"
        "    api.register_route('greet', _hello, methods=('GET',))\n"
    )
    skill_dir = _write_ext(skills_root, "ext_guarded", permissions=["route"], plugin=plugin)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    client, drive_root, patches = _make_client(tmp_path, monkeypatch)
    try:
        content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
        save_enabled(drive_root, "ext_guarded", True)
        save_review_state(
            drive_root,
            "ext_guarded",
            SkillReviewState(status="pass", content_hash=content_hash),
        )
        from ouroboros.skill_loader import find_skill

        loaded = find_skill(drive_root, "ext_guarded", repo_path=str(skills_root))
        assert loaded is not None
        err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
        assert err is None, err
        assert "ext_guarded" in extension_loader.snapshot()["extensions"]

        # Leave stale registrations in memory but mark the skill disabled on disk.
        save_enabled(drive_root, "ext_guarded", False)

        resp = client.get("/api/extensions/ext_guarded/greet")
        assert resp.status_code == 409, resp.text
        data = resp.json()
        assert data["state"]["reason"] == "disabled"
        assert "ext_guarded" not in extension_loader.snapshot()["extensions"]
    finally:
        _stop_patches(patches)


def test_api_extension_dispatcher_reloads_stale_live_route(tmp_path, monkeypatch):
    from ouroboros import extension_loader
    from ouroboros.skill_loader import (
        SkillReviewState,
        compute_content_hash,
        find_skill,
        save_enabled,
        save_review_state,
    )

    skills_root = tmp_path / "skills"
    skill_dir = _write_ext(
        skills_root,
        "ext_route_reload",
        permissions=["route"],
        plugin=(
            "from starlette.responses import JSONResponse\n"
            "def _hello(request):\n"
            "    return JSONResponse({'hello': 'v1'})\n"
            "def register(api):\n"
            "    api.register_route('greet', _hello, methods=('GET',))\n"
        ),
    )
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    client, drive_root, patches = _make_client(tmp_path, monkeypatch)
    try:
        content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
        save_enabled(drive_root, "ext_route_reload", True)
        save_review_state(
            drive_root,
            "ext_route_reload",
            SkillReviewState(status="pass", content_hash=content_hash),
        )
        loaded = find_skill(drive_root, "ext_route_reload", repo_path=str(skills_root))
        assert loaded is not None
        err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
        assert err is None, err

        (skill_dir / "plugin.py").write_text(
            (
                "from starlette.responses import JSONResponse\n"
                "def _hello(request):\n"
                "    return JSONResponse({'hello': 'v2'})\n"
                "def register(api):\n"
                "    api.register_route('greet', _hello, methods=('GET',))\n"
            ),
            encoding="utf-8",
        )
        refreshed = find_skill(drive_root, "ext_route_reload", repo_path=str(skills_root))
        assert refreshed is not None
        save_review_state(
            drive_root,
            "ext_route_reload",
            SkillReviewState(status="pass", content_hash=refreshed.content_hash),
        )

        resp = client.get("/api/extensions/ext_route_reload/greet")
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"hello": "v2"}
    finally:
        _stop_patches(patches)
