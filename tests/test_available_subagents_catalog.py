"""Focused prompt-catalog projection tests for Available subagents."""

from __future__ import annotations

import json

import pytest


def _row(
    row_id: str,
    *,
    kind: str,
    target: str,
    recommendation: str,
    effort: str = "",
    profile: str = "",
) -> dict:
    route = {"kind": kind, "target_id": target}
    if kind == "agent_session":
        route["credential_profile_id"] = profile
    return {
        "subagent_id": row_id,
        "name": row_id.replace("-", " ").title(),
        "recommended_use": recommendation,
        "route": route,
        "effort": effort,
    }


def _settings(*rows: dict, enabled: bool = True) -> dict:
    return {
        "OUROBOROS_SUBAGENTS": json.dumps({"enabled": enabled, "items": list(rows)}),
    }


def test_catalog_projects_every_saved_row_in_owner_order_verbatim():
    from ouroboros.subagent_runtime import model_visible_subagent_catalog

    verbatim = "Use exact owner wording.\nKeep punctuation: a/b, quotes, and cost $0."
    settings = _settings(
        _row(
            "api-scout",
            kind="api_model",
            target="google/gemini-3.7-flash",
            recommendation=verbatim,
            effort="low",
        ),
        _row(
            "auto-session",
            kind="agent_session",
            target="claude=claude-fable-5",
            recommendation="Use the automatic account pool.",
            effort="high",
        ),
        _row(
            "pinned-session",
            kind="agent_session",
            target="cursor=cursor-grok-4.6-high",
            recommendation="Use this pinned account.",
            profile="cursor-owner",
        ),
    )

    catalog = model_visible_subagent_catalog(settings)

    # Facts only: selection prose lives in prompts/SYSTEM.md, and the list
    # fingerprint and provenance stay host-side (snapshots carry them).
    assert set(catalog) == {"rows"}
    # Rows are named by their handle - the route plus the row's own set facets.
    assert [row["subagent_id"] for row in catalog["rows"]] == [
        "google/gemini-3.7-flash/low", "claude=claude-fable-5/high",
        "cursor=cursor-grok-4.6-high/@cursor-owner",
    ]
    assert catalog["rows"][0]["recommended_use"] == verbatim
    assert list(catalog["rows"][0])[-1] == "recommended_use"
    assert catalog["rows"][0]["route_class"] == "API model"
    assert catalog["rows"][0]["requested_model"] == "google/gemini-3.7-flash"
    assert catalog["rows"][0]["requested_effort"] == "low"
    assert catalog["rows"][1]["route_class"] == "Agent session"
    assert catalog["rows"][1]["requested_target"] == "claude=claude-fable-5"
    assert catalog["rows"][1]["mutating_access"] == "full"
    assert "credential_profile_id" not in catalog["rows"][1]
    assert catalog["rows"][2]["requested_effort"] == "(not explicitly set)"
    assert catalog["rows"][2]["credential_profile_id"] == "cursor-owner"
    assert not any("account_policy" in row for row in catalog["rows"])


@pytest.mark.parametrize(
    "settings",
    [
        {"OUROBOROS_SUBAGENTS": "not-json"},
        {
            **_settings(enabled=False),
            "OUROBOROS_SUBAGENT_HARNESS": "codex=gpt-5.6-sol:high",
        },
        _settings(),
    ],
)
def test_catalog_omits_unsaved_invalid_disabled_and_empty(settings):
    from ouroboros.subagent_runtime import model_visible_subagent_catalog

    assert model_visible_subagent_catalog(settings) == {}


def test_catalog_omits_a_real_undecided_candidate_rejected_by_new_id_dispatch():
    from ouroboros.configured_subagents import SOURCE_UNDECIDED, resolve_configured_subagents
    from ouroboros.subagent_runtime import (
        SubagentSelectionError,
        model_visible_subagent_catalog,
        select_subagent_snapshot,
    )

    settings = {"OUROBOROS_MODEL_HEAVY": "owner/unsaved-candidate"}
    resolution = resolve_configured_subagents(settings)
    assert resolution.source == SOURCE_UNDECIDED
    assert resolution.config is not None and resolution.config.items
    assert model_visible_subagent_catalog(settings) == {}
    with pytest.raises(SubagentSelectionError) as refused:
        select_subagent_snapshot(settings, subagent_id="legacy-heavy")
    assert refused.value.code == "subagent_configuration_unsaved"


def _context_env(tmp_path):
    class FakeEnv:
        def drive_path(self, path):
            return tmp_path / path

        def repo_path(self, path):
            return tmp_path / "repo" / path

        @property
        def repo_dir(self):
            return tmp_path / "repo"

        @property
        def drive_root(self):
            return tmp_path

    for path in ("state", "logs", "memory", "repo/docs", "repo/prompts", "repo/web"):
        (tmp_path / path).mkdir(parents=True, exist_ok=True)
    (tmp_path / "repo/prompts/SYSTEM.md").write_text("System", encoding="utf-8")
    (tmp_path / "repo/BIBLE.md").write_text("Bible", encoding="utf-8")
    (tmp_path / "repo/docs/ARCHITECTURE.md").write_text("Architecture", encoding="utf-8")
    (tmp_path / "repo/docs/DEVELOPMENT.md").write_text("Development", encoding="utf-8")
    (tmp_path / "state/state.json").write_text("{}", encoding="utf-8")
    (tmp_path / "logs/events.jsonl").write_text("", encoding="utf-8")
    return FakeEnv()


def test_catalog_is_semi_stable_while_dated_history_stays_dynamic(tmp_path, monkeypatch):
    from ouroboros.context import _capture_context_core
    from ouroboros.context_fit import _render_context_system_content
    from ouroboros.memory import Memory

    env = _context_env(tmp_path)
    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    owner_text = "Use this exact owner description, verbatim.\nSecond line stays intact."
    saved = _settings(_row(
        "builder",
        kind="agent_session",
        target="codex=gpt-5.6-sol",
        recommendation=owner_text,
        effort="high",
    ))
    monkeypatch.setattr("ouroboros.config.load_settings", lambda: saved)
    (tmp_path / "state/reviewer_slot_last_execution.json").write_text(json.dumps({
        "triad": {
            "ts": "2026-08-18T01:02:03+00:00",
            "status": "ok",
            "requested": {"profile_id": "review-requested"},
            "effective": {"profile_id": "review-applied"},
        },
    }), encoding="utf-8")
    (tmp_path / "state/subagent_last_delegation.json").write_text(json.dumps({
        "ts": "2026-08-18T02:00:00+00:00",
        "route": "codex",
        "requested_model": "gpt-5.6-sol",
        "applied_model": "gpt-5.6-sol",
        "requested_profile": "delegate-requested",
        "applied_profile": "delegate-applied",
        "selected_subagent_id": "builder",
        "run_id": "run-1",
    }), encoding="utf-8")

    core = _capture_context_core(
        env, Memory(drive_root=tmp_path),
        {"id": "task-1", "type": "task", "text": "work"},
        None, None,
    )
    blocks = _render_context_system_content(env, core, mode="max")
    catalog_text = core.semi_stable_text.split("## Available subagents\n\n", 1)[1]
    catalog, _end = json.JSONDecoder().raw_decode(catalog_text)

    assert blocks[1]["text"] == core.semi_stable_text
    assert blocks[1]["cache_control"] == {"type": "ephemeral"}
    assert blocks[2]["text"] == core.dynamic_head_text + "\n\n" + core.dynamic_text  # knowledge leads block 2
    assert "cache_control" not in blocks[2]
    # A projection's story closes block 1 after the catalog; its rooms open block 2 after the knowledge.
    viewed = _render_context_system_content(env, core, mode="max", story="## My story\n\nS", room="## This room (R)")
    assert viewed[1]["text"] == core.semi_stable_text + "\n\n## My story\n\nS"
    assert viewed[2]["text"] == core.dynamic_head_text + "\n\n## This room (R)\n\n" + core.dynamic_text
    assert catalog["rows"][0]["recommended_use"] == owner_text
    assert '"subagent_id": "codex=gpt-5.6-sol/high"' in core.semi_stable_text
    assert "builder" not in core.semi_stable_text, "the stored key is not model-facing"
    assert "2026-08-18T01:02:03+00:00" not in core.semi_stable_text
    assert "reviewer_slots_last" not in core.semi_stable_text
    assert "subagent_last_delegation" not in core.semi_stable_text
    assert "2026-08-18T01:02:03+00:00" in core.dynamic_text
    assert "reviewer_slots_last" in core.dynamic_text
    assert "subagent_last_delegation" in core.dynamic_text
    # An old receipt without a typed identity shows its recorded route target.
    assert '"selected_subagent_id": "codex=gpt-5.6-sol"' in core.dynamic_text
    for profile in (
        "review-requested", "review-applied", "delegate-requested", "delegate-applied",
    ):
        assert profile in core.dynamic_text
    assert "configured_route" not in core.dynamic_text


def test_catalog_hides_an_owner_disabled_row_and_selection_refuses_it_typed(monkeypatch):
    """Owner-disabled is a THIRD axis, distinct from the list-level switch and
    from live availability: the row stays saved and complete, the model never
    sees it, and an explicit selection of it refuses with its own code rather
    than `unknown_subagent_id` or a substitute actor."""
    from ouroboros.subagent_runtime import (
        SubagentSelectionError,
        current_subagent_alternatives,
        model_visible_subagent_catalog,
        select_subagent_snapshot,
    )

    settings = _settings(
        _row("builder", kind="api_model", target="openai/gpt-5.6-sol",
             recommendation="Use for implementation."),
        {**_row("paused", kind="api_model", target="openai/gpt-5.6-luna",
                recommendation="Use for scouting."), "enabled": False},
    )

    catalog = model_visible_subagent_catalog(settings)
    assert [row["subagent_id"] for row in catalog["rows"]] == ["openai/gpt-5.6-sol"]  # named by handle

    # Resolution comes first, the row switch second: the same typed refusal by
    # the row's stored id and by its handle, never `unknown_subagent_id`.
    for selector in ("paused", "openai/gpt-5.6-luna"):
        with pytest.raises(SubagentSelectionError) as refused:
            select_subagent_snapshot(settings, subagent_id=selector)
        assert refused.value.code == "subagent_disabled", selector
        assert "switched off" in refused.value.detail
    # The enabled sibling is unaffected by its neighbour's switch.
    assert select_subagent_snapshot(settings, subagent_id="builder")[0][
        "selected_subagent_id"] == "builder"

    import ouroboros.config as config_module

    monkeypatch.setattr(config_module, "runtime_settings", lambda: dict(settings))
    assert [row["subagent_id"] for row in current_subagent_alternatives()] == ["openai/gpt-5.6-sol"]


def test_a_roster_whose_every_row_is_switched_off_projects_no_catalog():
    from ouroboros.subagent_runtime import model_visible_subagent_catalog

    settings = _settings({
        **_row("paused", kind="api_model", target="openai/gpt-5.6-sol",
               recommendation="Use for implementation."),
        "enabled": False,
    })
    assert model_visible_subagent_catalog(settings) == {}


def test_a_captured_snapshot_stays_valid_after_its_row_is_switched_off():
    """Existing task and review snapshots are immutable intent: validation must
    not consult live settings, so disabling the row later cannot invalidate a
    task that is already running on it."""
    from ouroboros.subagent_runtime import select_subagent_snapshot, validate_subagent_snapshot

    enabled = _settings(_row("builder", kind="api_model", target="openai/gpt-5.6-sol",
                             recommendation="Use for implementation."))
    snapshot, _legacy = select_subagent_snapshot(enabled, subagent_id="builder")

    switched_off = _settings({
        **_row("builder", kind="api_model", target="openai/gpt-5.6-sol",
               recommendation="Use for implementation."),
        "enabled": False,
    })
    assert switched_off != enabled
    assert validate_subagent_snapshot(snapshot) == snapshot
    assert "enabled" not in snapshot


def test_schedule_subagent_reaches_the_model_with_the_typed_disabled_refusal(
    monkeypatch, tmp_path,
):
    """The delegation consumer, end to end: the model sees only the enabled row
    in its catalog, and selecting the switched-off one comes back as a typed
    refusal it can act on — never a substituted actor."""
    from ouroboros.tools import control
    from ouroboros.tools.registry import ToolContext, ToolRegistry
    from ouroboros.subagent_runtime import model_visible_subagent_catalog

    settings = _settings(
        _row("builder", kind="api_model", target="openai/gpt-5.6-sol",
             recommendation="Use for implementation."),
        {**_row("paused", kind="api_model", target="openai/gpt-5.6-luna",
                recommendation="Use for scouting."), "enabled": False},
    )
    monkeypatch.setattr(control, "load_settings", lambda: settings)
    assert "paused" not in json.dumps(model_visible_subagent_catalog(settings))

    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry.set_context(ToolContext(repo_dir=tmp_path, drive_root=tmp_path))
    result = registry.execute("schedule_subagent", {
        "subagent_id": "paused",
        "objective": "Implement it",
        "expected_output": "Patch",
    })
    assert "subagent_disabled" in result
    assert "switched off" in result
    assert "unknown_subagent_id" not in result
