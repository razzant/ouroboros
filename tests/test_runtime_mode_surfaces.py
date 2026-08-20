"""Where a runtime mode is published and what a generic POST may do to it.

Split verbatim out of ``tests/test_runtime_mode_core.py`` by theme. This module owns
the ``/api/state`` keys and their TypedDict mirror, the settings/onboarding/skills web
copy that must match the shipped runtime, and the ``/api/settings`` POST that clamps an
unknown mode and silently drops a mode change.
"""

from __future__ import annotations

import ast
import pathlib


from ouroboros.onboarding_wizard import build_onboarding_html

REPO = pathlib.Path(__file__).resolve().parent.parent


# ===========================================================================
# Part 3: server.py /api/state surfaces + TypedDict
# ===========================================================================


def test_api_state_declares_phase2_keys():
    tree = ast.parse((REPO / "ouroboros" / "gateway" / "state.py").read_text(encoding="utf-8"))
    api_state_fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "api_state":
            api_state_fn = node
            break
    assert api_state_fn is not None

    for node in ast.walk(api_state_fn):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Name) and func.id == "JSONResponse"):
            continue
        if not node.args or not isinstance(node.args[0], ast.Dict):
            continue
        keys = {
            k.value for k in node.args[0].keys
            if isinstance(k, ast.Constant) and isinstance(k.value, str)
        }
        if keys == {"error"}:
            continue
        assert "runtime_mode" in keys
        assert "skills_repo_configured" in keys
        return
    raise AssertionError("api_state exposes no happy-path JSONResponse literal")


def test_state_response_typeddict_declares_phase2_keys():
    from ouroboros.gateway.contracts import StateResponse

    keys = set(StateResponse.__annotations__.keys())
    assert "runtime_mode" in keys
    assert "skills_repo_configured" in keys


# ===========================================================================
# Part 4: Web UI substrings
# ===========================================================================


def test_settings_ui_renders_runtime_mode_and_skills_path():
    src = (REPO / "web" / "modules" / "settings_ui.js").read_text(encoding="utf-8")
    assert 'id="s-runtime-mode"' in src
    # Runtime-mode segmented control is built by the renderSegmentedField SSOT
    # (C7.1): the column-override modifier and the light/advanced/pro options come
    # through its params, not inline data-effort-value markup.
    assert "modifier: 'data-runtime-mode-group'" in src
    for mode in ("light", "advanced", "pro"):
        assert f"value: '{mode}'" in src
    assert 'id="s-skills-repo-path"' in src


def test_settings_js_reads_and_writes_phase2_keys():
    src = (REPO / "web" / "modules" / "settings.js").read_text(encoding="utf-8")
    assert "OUROBOROS_RUNTIME_MODE" in src
    assert "OUROBOROS_CONTEXT_MODE_DRAFT" in src
    assert "OUROBOROS_SKILLS_REPO_PATH" in src
    assert "['s-runtime-mode', 'OUROBOROS_RUNTIME_MODE', 'advanced']" in src
    assert "['s-context-mode', 'OUROBOROS_CONTEXT_MODE', 'max']" in src
    assert "['s-skills-repo-path', 'OUROBOROS_SKILLS_REPO_PATH']" in src
    assert "fieldValue(id).trim()" in src


def test_chat_context_mode_toggle_reports_owner_endpoint_errors():
    src = (REPO / "web" / "modules" / "chat.js").read_text(encoding="utf-8")
    assert "/api/owner/context-mode" in src
    assert "resp.json()" in src
    assert "showToast(message, 'error')" in src


def test_onboarding_js_has_runtime_mode_selector_and_save_payload():
    src = (REPO / "web" / "modules" / "onboarding_wizard.js").read_text(encoding="utf-8")
    html = build_onboarding_html({})
    for mode in ("light", "advanced", "pro"):
        assert f'"value": "{mode}"' in html
    assert "data-runtime-mode" in src
    assert "OUROBOROS_RUNTIME_MODE" in src
    assert "OUROBOROS_SKILLS_REPO_PATH" in src


def test_phase4_ui_copy_matches_shipped_runtime():
    settings_ui = (REPO / "web" / "modules" / "settings_ui.js").read_text(encoding="utf-8")
    onboarding_html = build_onboarding_html({})

    assert "Phase 2 plumbing only" not in settings_ui
    assert "land in Phase 3" not in settings_ui
    assert "data/skills/" in settings_ui
    assert "Pick both review enforcement and the initial runtime mode" in onboarding_html
    assert "normal triad + scope review" in onboarding_html
    assert "Phase 6+:" not in onboarding_html


def test_skills_ui_reads_live_extension_state_fields():
    renderer = (REPO / "web" / "modules" / "skill_card_renderer.js").read_text(encoding="utf-8")
    orchestration = (REPO / "web" / "modules" / "skills.js").read_text(encoding="utf-8")
    src = renderer + "\n" + orchestration
    assert "live_loaded" in src
    assert "review_gate?.executable_review" in src or "review_gate.executable_review" in src
    assert "executable_review" in src
    assert "skill.review_status === 'blockers' && !reviewReady(skill)" in src
    assert "function statusBadge(status, gate = null, profile = '')" in src
    assert "statusBadge(skill.review_status, skill.review_gate, skill.review_profile)" in src
    assert "Open widgets" in src
    assert "retry_install" in src
    assert "Retry install" in src
    assert "result.error" in src


def test_onboarding_js_exposes_skills_repo_path_input_and_binding():
    src = (REPO / "web" / "modules" / "onboarding_wizard.js").read_text(encoding="utf-8")
    assert 'id="skills-repo-path"' in src
    assert 'data-clear="skills-repo-path"' in src
    assert "state.skillsRepoPath = skillsInput.value" in src
    assert "'skills-repo-path': () => { state.skillsRepoPath = ''; }" in src


def test_onboarding_css_has_three_column_variant():
    src = (REPO / "web" / "onboarding.css").read_text(encoding="utf-8")
    assert ".wizard-choice-grid.three" in src


# ===========================================================================
# Part 5: /api/settings POST elevation + clamp behavior
# ===========================================================================


def test_api_settings_post_clamps_unknown_runtime_mode(tmp_path, monkeypatch):
    """POSTing an invalid runtime mode must be normalized to 'advanced'
    before save — so /api/settings and /api/state can never disagree."""
    import server as srv
    from starlette.testclient import TestClient
    from unittest.mock import patch

    saved: dict = {}

    def fake_load_settings():
        from ouroboros.config import SETTINGS_DEFAULTS
        out = dict(SETTINGS_DEFAULTS)
        out.update(saved)
        return out

    def fake_save_settings(payload, *, allow_elevation: bool = False, allow_context_lowering: bool = False,
                           authored_keys=(), boundary=None):
        # Stands in for both save_settings (allow_elevation) and _owner_write_settings
        # (allow_context_lowering, added in v6.33.0 P4; authored_keys in v6.80.0 — the caller
        # names the disk-authored keys it really authors, see prepare_settings_for_persist;
        # boundary marks the commit point, so the stub marks it as the real writer would).
        saved.clear()
        saved.update(payload)
        if boundary is not None:
            boundary.commit()

    with patch.object(srv, "load_settings", side_effect=fake_load_settings), \
            patch.object(srv, "save_settings", side_effect=fake_save_settings), \
            patch.object(srv._gateway_settings, "_owner_read_settings_raw", side_effect=fake_load_settings), \
            patch.object(srv._gateway_settings, "_owner_write_settings", side_effect=fake_save_settings), \
            patch.object(srv, "_start_supervisor_if_needed", lambda *_a, **_k: None), \
            patch.object(srv, "_apply_settings_to_env", lambda *_a, **_k: None), \
            patch.object(srv, "apply_runtime_provider_defaults", lambda s: (s, False, [])), \
            patch("ouroboros.server_auth.get_configured_network_password", return_value=""):
        client = TestClient(srv.app)
        resp = client.post(
            "/api/settings",
            json={"OUROBOROS_RUNTIME_MODE": "turbo"},
        )
        assert resp.status_code == 200, resp.text
        # /api/settings drops OUROBOROS_RUNTIME_MODE entirely — even invalid
        # inputs do not reach the body merge. The persisted value equals the
        # SETTINGS_DEFAULTS baseline ("advanced") via the belt-and-braces
        # revert in api_settings_post.
        assert saved["OUROBOROS_RUNTIME_MODE"] == "advanced"


def test_api_settings_post_silently_drops_runtime_mode_changes():
    """v5.1.2 elevation ratchet: even a VALID runtime_mode in the body
    is silently dropped — the API never accepts mode changes."""
    import server as srv
    from starlette.testclient import TestClient
    from unittest.mock import patch

    saved: dict = {}

    def fake_load_settings():
        from ouroboros.config import SETTINGS_DEFAULTS
        out = dict(SETTINGS_DEFAULTS)
        out["OUROBOROS_RUNTIME_MODE"] = "light"
        out.update(saved)
        return out

    def fake_save_settings(payload, *, allow_elevation: bool = False, allow_context_lowering: bool = False,
                           authored_keys=(), boundary=None):
        # Stands in for both save_settings (allow_elevation) and _owner_write_settings
        # (allow_context_lowering, added in v6.33.0 P4; authored_keys in v6.80.0 — the caller
        # names the disk-authored keys it really authors, see prepare_settings_for_persist;
        # boundary marks the commit point, so the stub marks it as the real writer would).
        saved.clear()
        saved.update(payload)
        if boundary is not None:
            boundary.commit()

    with patch.object(srv, "load_settings", side_effect=fake_load_settings), \
            patch.object(srv, "save_settings", side_effect=fake_save_settings), \
            patch.object(srv._gateway_settings, "_owner_read_settings_raw", side_effect=fake_load_settings), \
            patch.object(srv._gateway_settings, "_owner_write_settings", side_effect=fake_save_settings), \
            patch.object(srv, "_start_supervisor_if_needed", lambda *_a, **_k: None), \
            patch.object(srv, "_apply_settings_to_env", lambda *_a, **_k: None), \
            patch.object(srv, "apply_runtime_provider_defaults", lambda s: (s, False, [])), \
            patch("ouroboros.server_auth.get_configured_network_password", return_value=""):
        client = TestClient(srv.app)
        resp = client.post(
            "/api/settings",
            json={"OUROBOROS_RUNTIME_MODE": "pro", "OUROBOROS_SKILLS_REPO_PATH": "  /tmp/sk  "},
        )
        assert resp.status_code == 200, resp.text
        assert saved["OUROBOROS_RUNTIME_MODE"] == "light"
        assert saved["OUROBOROS_SKILLS_REPO_PATH"] == "/tmp/sk"
