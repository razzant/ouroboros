"""WS5 — v6.33.0 review carryover fixes (v6.34.0)."""

from __future__ import annotations


# --- CW1 lineage: owner-only settings stay out of the generic settings write.      ---
# --- ABI 7.0 (owner Q10=A): OUROBOROS_SCOPE_REVIEW_FLOOR and its whole surface are  ---
# --- REMOVED (see tests/test_abi5_q10_removals.py); scope-review applicability      ---
# --- comes solely from the owner context mode.                                      ---

def test_context_mode_is_owner_only_not_generic_settings():
    from ouroboros.gateway.settings import _merge_settings_payload

    current = {"OUROBOROS_CONTEXT_MODE": "max"}
    merged = _merge_settings_payload(current, {"OUROBOROS_CONTEXT_MODE": "low"})
    # The generic /api/settings merge must NOT narrow the horizon: the setting is
    # the owner's own working window and only the owner endpoint authors it.
    assert merged["OUROBOROS_CONTEXT_MODE"] == "max"


# --- CW1: the owner-control mention family and its shared read-carve ---

def test_stored_singular_scope_pin_is_ghost_purged(monkeypatch, tmp_path):
    """ABI 7.0 (ABI-10): both comma spellings are RETIRED settings keys — a
    stored pin (singular or plural) is ghost-purged on load, never promoted.
    (The pre-7.0 singular→plural promotion left with the migration read.)"""
    import json

    import ouroboros.config as cfg

    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps({"OUROBOROS_SCOPE_REVIEW_MODEL": "anthropic/claude-opus-4.8"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(cfg, "SETTINGS_PATH", settings_path)
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    monkeypatch.delenv("OUROBOROS_SCOPE_REVIEW_MODEL", raising=False)
    monkeypatch.delenv("OUROBOROS_SCOPE_REVIEW_MODELS", raising=False)

    loaded = cfg.load_settings()
    assert "OUROBOROS_SCOPE_REVIEW_MODEL" not in loaded
    assert "OUROBOROS_SCOPE_REVIEW_MODELS" not in loaded


# --- CW4: the external-shell secret guard catches relative interpreter paths ---

# --- Exact-route context fitting honours USE_LOCAL_MAIN ---

def test_active_main_route_honours_use_local_main():
    from ouroboros.gateway.settings import _active_main_route

    local = _active_main_route({"OUROBOROS_MODEL": "openai/gpt-5.5", "USE_LOCAL_MAIN": True})
    assert local["use_local"] is True and local["provider"] == "local"
    remote = _active_main_route({"OUROBOROS_MODEL": "openai/gpt-5.5", "USE_LOCAL_MAIN": False})
    assert remote["use_local"] is False and remote["provider"] != "local"


# --- CW9: the pacing-interval timeout constant lives in the SETTINGS_DEFAULTS SSOT ---

def test_pacing_interval_in_settings_defaults():
    from ouroboros.config import PACING_INTERVAL_DEFAULT_SEC, SETTINGS_DEFAULTS

    assert SETTINGS_DEFAULTS.get("OUROBOROS_PACING_INTERVAL_SEC") == PACING_INTERVAL_DEFAULT_SEC


# === Triad+scope review-fix regressions (v6.34.0) ===

# Predicted route evidence is measurement input, never a global-mode writer or
# an initial Max-to-Low authority. Functional fit cases are pinned in the Phase
# 2 matrix; this carryover suite guards deletion of the old compatibility seam.
def test_predicted_route_downgrade_seam_stays_deleted():
    from ouroboros import loop as loopmod

    assert not hasattr(loopmod, "_maybe_downgrade_max_unconfirmed")


def test_switch_model_does_not_blanket_gate_on_context_window(monkeypatch, tmp_path):
    """The loop rebinds/fits the exact route after the override; the tool only selects it."""
    from ouroboros.tools import control
    from ouroboros.tools.registry import ToolContext

    monkeypatch.setattr(
        "ouroboros.llm.LLMClient.available_models",
        lambda self: ["small-model"],
    )
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx.active_context_mode = "max"

    out = control._switch_model(ctx, model="small-model")

    assert "SWITCH_BLOCKED" not in out
    assert ctx.active_model_override == "small-model"


# --- running_tasks routing context never silently truncates (codex no-[:N] rule) ---

def test_running_tasks_clip_marker_is_explicit():
    import server

    assert server._clip_marked("short objective", 600) == "short objective"
    clipped = server._clip_marked("x" * 1000, 600)
    assert clipped.startswith("x" * 600)
    assert "chars omitted]" in clipped  # explicit omission marker, not a silent cut



def test_capability_evidence_is_route_aware_not_model_aware(monkeypatch, tmp_path):
    """A base-URL change is a NEW ROUTE and must reprobe.

    Capability is a property of provider+base_url+model, and evidence is stored
    under that route fingerprint. A lazy probe memoised by MODEL NAME serves a
    hot `OPENAI_BASE_URL` change (or the openai-compatible / cloudru / gigachat
    equivalents) from the previous route's record, so the scope reviewer is sized
    from a window that belongs to a different endpoint."""

    import ouroboros.config as cfg
    from ouroboros import capability_evidence as ce
    from ouroboros.tools import scope_window as sr

    model = "openai::gpt-5.5-pinned"
    base_urls = {"OPENAI_BASE_URL": "https://route-a.example/v1"}
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    monkeypatch.setattr(cfg, "load_settings", lambda: dict(base_urls))

    fetched: list = []

    # The REAL probe, so what is counted is real network work: a route whose stored
    # record is current is served from the cache, which is the rate limit — not a
    # process memo that outlives the record and can never re-source it.
    def fake_metadata(_provider, _model, base_url, allow_fetch, **_kw):
        fetched.append(str(base_url or ""))
        return 200_000

    monkeypatch.setattr(ce, "_provider_metadata_window", fake_metadata)

    sr.scope_window(model)
    sr.scope_window(model)
    assert fetched == ["https://route-a.example/v1"], "one probe per route, not per call"

    # Same model, DIFFERENT base URL: a new route, so the lazy probe must run again.
    base_urls["OPENAI_BASE_URL"] = "https://route-b.example/v1"
    sr.scope_window(model)
    assert fetched == [
        "https://route-a.example/v1", "https://route-b.example/v1",
    ], "a base-URL change is a new route fingerprint and must be probed"


def test_no_scope_row_shape_is_asked_to_confirm_a_context_window(monkeypatch, tmp_path):
    """Owner decision 2026-09-17: window size is not a condition of scope authority.

    Saving a scope row used to probe that row's route and, whenever the reading was
    below the packet floor or the retrieving floor, return a `needs_ack` notice that
    Settings turned into "confirm this reviewer's context window" — the only path by
    which the row could sign a blocking verdict. Scope review now reads the
    repository itself, and authority rests on the declared required-source manifest
    and the reads recorded against it, so no reviewer row of any shape is asked about a
    window and the save performs no capability probe for one.
    """
    import json

    import pytest
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros import capability_evidence as ce
    from ouroboros import config as cfg
    from ouroboros.gateway import settings as smod

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    settings_path = data_dir / "settings.json"
    settings_path.write_text(json.dumps({"OUROBOROS_MODEL": "openai::gpt-main"}), encoding="utf-8")
    monkeypatch.setattr(cfg, "DATA_DIR", data_dir)
    monkeypatch.setattr(cfg, "SETTINGS_PATH", settings_path)
    monkeypatch.setattr(smod, "apply_runtime_provider_defaults", lambda s: (s, False, []))
    monkeypatch.setattr(smod, "_start_supervisor_if_needed_for_request", lambda *_a, **_k: False)
    monkeypatch.setattr(smod, "_apply_settings_to_env", lambda *_a, **_k: None)
    monkeypatch.setattr(smod, "_apply_settings_save_side_effects", lambda *_a, **_k: None)
    # The unknown-model-id warning is a catalog lookup, not a window probe; it is
    # the other consumer of the candidate rows and is out of this test's claim.
    monkeypatch.setattr(smod, "_unrecognised_review_models", lambda _models: [])
    monkeypatch.setattr(ce, "probe", lambda *_a, **_kw: pytest.fail(
        "a settings save probed a reviewer's context window"))

    # Reviewers are catalog rows marked Reviewer; one of every shape a row can take.
    roster = {"enabled": True, "items": [
        {"subagent_id": "bare-critic", "recommended_use": "Reads the packet.", "review_eligible": True,
         "route": {"kind": "api_model", "target_id": "openai::gpt-bare"}, "delivery": "packet"},
        {"subagent_id": "api-critic", "name": "API critic", "recommended_use": "Bounded inspection episode.",
         "review_eligible": True, "route": {"kind": "api_model", "target_id": "openai::gpt-critic"}, "effort": "medium"},
        {"subagent_id": "session-critic", "recommended_use": "Reads the repository.", "review_eligible": True,
         "route": {"kind": "agent_session", "target_id": "codex=gpt-5.6-sol"}},
    ]}
    app = Starlette(routes=[Route("/api/settings", smod.api_settings_post, methods=["POST"])])
    app.state.drive_root = data_dir
    app.state.repo_dir = data_dir
    response = TestClient(app).post("/api/settings", json={
        "OUROBOROS_SUBAGENTS": json.dumps(roster),
        "OPENAI_BASE_URL": "https://route-b.example/v1",
    })

    assert response.status_code == 200, response.text
    body = response.json()
    assert "review_capability_notices" not in body, body
    assert "needs_ack" not in response.text
    stored = json.loads(json.loads(settings_path.read_text(encoding="utf-8"))["OUROBOROS_SUBAGENTS"])
    assert [row["subagent_id"] for row in stored["items"]] == ["bare-critic", "api-critic", "session-critic"]
    assert not hasattr(smod, "_review_capability_notices")
    # The generic ack endpoint stays: the main model's own window evidence is
    # recorded through it (tests/test_owner_settings_write_seam.py
    # ::test_the_capability_ack_is_not_a_settings_writer drives it end to end).
    assert callable(smod.api_acknowledge_capability)


def test_unrecognised_review_model_ids_are_reported_loudly(monkeypatch):
    """RS5: a truncated slot value (the owner's `-5`) used to surface only as three
    waves of `400 ... is not a valid model ID`, destroying the review quorum. It is
    reported at save time — evidence-based (absent from a SUCCESSFULLY fetched
    catalog), never a guess, and never a save rejection."""
    from ouroboros.gateway import settings as smod
    from ouroboros.llm import LLMClient

    monkeypatch.setattr(LLMClient, "openrouter_context_length", classmethod(lambda cls, m, **k: 0))
    monkeypatch.setattr(LLMClient, "_CAPABILITIES_FETCH_OK", True, raising=False)
    monkeypatch.setattr(
        LLMClient, "_CONTEXT_LENGTH_CACHE",
        {"anthropic/claude-fable-5": 1_000_000}, raising=False,
    )

    unknown = smod._unrecognised_review_models(["anthropic/claude-fable-5", "-5"])
    assert unknown == ["-5"]

    # Without an authoritative catalog nothing may be CLAIMED unknown.
    monkeypatch.setattr(LLMClient, "_CAPABILITIES_FETCH_OK", False, raising=False)
    assert smod._unrecognised_review_models(["-5"]) == []
