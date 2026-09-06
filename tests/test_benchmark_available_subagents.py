"""Fixed-model benchmark profiles use the canonical Available-subagents wire."""

from __future__ import annotations

import json
import pathlib
import sys
import types

import pytest

from devtools.benchmarks.common.manifests import MODEL_SLOT_KEYS, model_slot_snapshot
from devtools.benchmarks.common.model_slots import (
    configured_subagents_snapshot,
    disabled_subagents_setting,
    fixed_model_actor_snapshot,
    pin_single_model,
    runtime_actor_snapshot,
    single_model_reviewer_slots_setting,
    single_model_slot_snapshot,
    single_model_subagents_setting,
)
from devtools.benchmarks.common.server_runner import (
    STALE_INHERITED_ENV_KEYS,
    build_isolated_settings,
)
from ouroboros.config import SETTINGS_DEFAULTS
from ouroboros.configured_subagents import (
    parse_configured_subagents,
    serialize_configured_subagents,
)
from ouroboros.provider_models import provider_for_model, review_model_uses_local
from ouroboros.reviewer_slot_config import REVIEWER_SLOTS_ENV, parse_reviewer_slots

REPO = pathlib.Path(__file__).resolve().parents[1]
PROFILE_TARGETS = {
    "devtools/benchmarks/gaia/settings_base.json": "google/gemini-2.5-pro",
    "devtools/benchmarks/osworld/settings_base.json": "anthropic/claude-sonnet-4.6",
    "devtools/benchmarks/programbench/settings_base.json": "openai/gpt-5.5",
    "devtools/benchmarks/continual_learning/settings_base.json": "anthropic/claude-sonnet-4.6",
    "devtools/benchmarks/cybergym/settings_base.json": "deepseek/deepseek-v4-flash-0731",
    "devtools/benchmarks/swe_bench_pro/e1v2/settings_base.json": "anthropic/claude-sonnet-4.5",
    "devtools/benchmarks/swe_bench_pro/e1v2/settings_sonnet46_probe.json":
        "anthropic/claude-sonnet-4.6",
    "devtools/benchmarks/swe_bench_pro/e1v2/_run_settings.example.json":
        "anthropic/claude-sonnet-4.5",
    "devtools/benchmarks/swe_bench_pro/e1v2/profiles/light_subagents_gpt55.json":
        "openai/gpt-5.5",
}

# Registry-derived so a newly registered provider can never leak ambient routing.
from ouroboros.provider_models import PROVIDER_CREDENTIAL_GROUPS as _CRED_GROUPS

_PROVIDER_ROUTE_ENV_KEYS = tuple(
    key for group in _CRED_GROUPS.values() for key in group
)


def _only_target(raw: object) -> str:
    config = parse_configured_subagents(raw)
    assert config.enabled is True
    assert len(config.items) == 1
    row = config.items[0]
    assert row.subagent_id == "benchmark-model"
    assert row.route.kind == "api_model"
    assert row.route.credential_profile_id == ""
    return row.route.target_id


def _fixed_actor_settings(model: str, *, review_slots: int = 1) -> dict[str, str]:
    settings: dict[str, str] = {}
    pin_single_model(model, review_slots=review_slots, target=settings)
    return settings


def _scrub_model_route_env(monkeypatch) -> None:
    from devtools.benchmarks.common.manifests import MODEL_SLOT_KEYS

    for key in (*_PROVIDER_ROUTE_ENV_KEYS, *MODEL_SLOT_KEYS):
        monkeypatch.delenv(key, raising=False)


def test_single_model_encoder_round_trips_one_exact_api_actor():
    raw = single_model_subagents_setting("openai::gpt-5.6-sol")
    assert _only_target(raw) == "openai::gpt-5.6-sol"
    assert serialize_configured_subagents(parse_configured_subagents(raw)) == raw


def test_pin_single_model_replaces_legacy_heavy_and_prior_actor_list():
    target = {
        "OUROBOROS_MODEL_HEAVY": "decoy/heavy",
        "USE_LOCAL_HEAVY": "true",
        "USE_LOCAL_MAIN": "true",
        "USE_LOCAL_LIGHT": "true",
        "USE_LOCAL_FALLBACK": "true",
        "USE_LOCAL_CONSCIOUSNESS": "true",
        "CLAUDE_CODE_MODEL": "foreign-sdk-model",
        REVIEWER_SLOTS_ENV: json.dumps({
            "triad": [{
                "slot_id": "foreign-triad",
                "route": {"kind": "agent_session", "target_id": "codex=gpt-5.6-sol"},
            }],
            "scope": [{
                "slot_id": "foreign-scope",
                "route": {"kind": "api_chat", "target_id": "foreign/scope"},
            }],
        }),
        "OUROBOROS_SUBAGENTS": single_model_subagents_setting("decoy/actor"),
    }
    snapshot = fixed_model_actor_snapshot("openai/gpt-5.5", target=target)
    assert "OUROBOROS_MODEL_HEAVY" not in target
    assert "USE_LOCAL_HEAVY" not in target
    assert _only_target(target["OUROBOROS_SUBAGENTS"]) == "openai/gpt-5.5"
    assert all(target[key] == "false" for key in (
        "USE_LOCAL_MAIN", "USE_LOCAL_LIGHT", "USE_LOCAL_FALLBACK",
        "USE_LOCAL_CONSCIOUSNESS",
    ))
    # Retired Claude-SDK setting: stale bytes, not execution authority — the
    # pin no longer rewrites it (nothing reads it).
    assert target["CLAUDE_CODE_MODEL"] == "foreign-sdk-model"
    reviewers = parse_reviewer_slots(target[REVIEWER_SLOTS_ENV])
    assert [row.target_id for row in reviewers.triad] == ["openai/gpt-5.5"]
    assert [row.target_id for row in reviewers.scope] == ["openai/gpt-5.5"]
    assert all(not row.is_session for row in (*reviewers.triad, *reviewers.scope))
    assert reviewers.advisory.enabled is False
    assert snapshot["mismatches"] == []
    assert snapshot["reviewer_slots"]["advisory"]["enabled"] is False


def test_pin_single_model_preserves_canonical_local_route_semantics():
    model = "owner/model (local)"
    target = _fixed_actor_settings(model, review_slots=2)
    assert provider_for_model(model) == "local"
    assert review_model_uses_local(model) is True
    assert all(target[key] == "true" for key in (
        "USE_LOCAL_MAIN", "USE_LOCAL_LIGHT", "USE_LOCAL_FALLBACK",
        "USE_LOCAL_CONSCIOUSNESS",
    ))
    reviewers = parse_reviewer_slots(target[REVIEWER_SLOTS_ENV])
    assert [row.target_id for row in reviewers.triad] == [model, model]
    assert [row.target_id for row in reviewers.scope] == [model]
    assert all(not row.is_session for row in (*reviewers.triad, *reviewers.scope))
    assert reviewers.advisory.enabled is False
    actor = runtime_actor_snapshot(target, expected_model=model)
    assert actor["mismatches"] == []
    assert all(actor["local_routes"].values())


@pytest.mark.parametrize(
    "light_model,uses_local_light",
    (
        ("anthropic/claude-sonnet-4.6", False),
        ("owner/light (local)", True),
    ),
)
def test_fixed_model_actor_compiles_explicit_light_route(
    light_model, uses_local_light
):
    model = "openai/gpt-5.5"
    target = {
        "OUROBOROS_MODEL_LIGHT": "foreign/light",
        "USE_LOCAL_LIGHT": "true",
    }
    actor = fixed_model_actor_snapshot(
        model,
        light_model=light_model,
        target=target,
    )

    assert actor["mismatches"] == []
    assert actor["model_slots"]["OUROBOROS_MODEL"] == model
    assert actor["model_slots"]["OUROBOROS_MODEL_LIGHT"] == light_model
    assert actor["model_slots"]["OUROBOROS_MODEL_FALLBACKS"] == model
    assert actor["local_routes"] == {
        "USE_LOCAL_MAIN": False,
        "USE_LOCAL_LIGHT": uses_local_light,
        "USE_LOCAL_FALLBACK": False,
        "USE_LOCAL_CONSCIOUSNESS": False,
    }
    assert _only_target(target["OUROBOROS_SUBAGENTS"]) == model
    target["USE_LOCAL_LIGHT"] = str(not uses_local_light).lower()
    drifted = runtime_actor_snapshot(
        target,
        expected_model=model,
        expected_light_model=light_model,
    )
    assert any("USE_LOCAL_LIGHT" in item for item in drifted["mismatches"])


def test_disabled_encoder_is_explicit_empty_off():
    config = parse_configured_subagents(disabled_subagents_setting())
    assert config.enabled is False
    assert config.items == ()


def test_disabled_encoder_can_retain_one_exact_measured_actor():
    config = parse_configured_subagents(disabled_subagents_setting("openai/gpt-5.5"))
    assert config.enabled is False
    assert len(config.items) == 1
    assert config.items[0].route.target_id == "openai/gpt-5.5"


def test_runtime_actor_snapshot_compares_main_and_canonical_actor():
    model = "openai/gpt-5.5"
    settings = _fixed_actor_settings(model)
    settings["OUROBOROS_MODEL_FALLBACKS"] = f"{model}, {model}"
    exact = runtime_actor_snapshot(settings, expected_model=model)
    assert exact["mismatches"] == []
    assert exact["model_slots"]["OUROBOROS_MODEL_FALLBACKS"] == f"{model}, {model}"
    assert not any(exact["local_routes"].values())
    assert exact["reviewer_slots"]["advisory"]["enabled"] is False
    assert _only_target(json.dumps(exact["available_subagents"])) == model

    contaminated_settings = dict(settings)
    contaminated_settings["OUROBOROS_MODEL_LIGHT"] = "anthropic/foreign-light"
    contaminated_settings["OUROBOROS_MODEL_FALLBACKS"] = (
        f"{model},anthropic/foreign-fallback"
    )
    contaminated = runtime_actor_snapshot(contaminated_settings, expected_model=model)
    assert any("OUROBOROS_MODEL_LIGHT" in item for item in contaminated["mismatches"])
    assert any("OUROBOROS_MODEL_FALLBACKS" in item for item in contaminated["mismatches"])

    other = "anthropic/claude-fable-5"
    drifted = runtime_actor_snapshot(_fixed_actor_settings(other), expected_model=model)
    assert any("OUROBOROS_MODEL:" in item for item in drifted["mismatches"])
    assert any("OUROBOROS_SUBAGENTS" in item for item in drifted["mismatches"])
    assert _only_target(json.dumps(drifted["available_subagents"])) == (
        other
    )


@pytest.mark.parametrize(
    "kind,target,needle",
    (
        ("api_chat", "foreign/reviewer", "foreign/reviewer"),
        ("agent_session", "codex=gpt-5.6-sol-high", "agent_session"),
    ),
)
def test_runtime_actor_snapshot_refuses_foreign_or_session_reviewer_rows(
    kind, target, needle
):
    model = "openai/gpt-5.5"
    settings = _fixed_actor_settings(model)
    payload = json.loads(settings[REVIEWER_SLOTS_ENV])
    payload["triad"][0]["route"] = {"kind": kind, "target_id": target}
    settings[REVIEWER_SLOTS_ENV] = json.dumps(payload)
    snapshot = runtime_actor_snapshot(settings, expected_model=model)
    assert any(needle in item for item in snapshot["mismatches"])


def test_runtime_actor_snapshot_refuses_a_retrieving_row_even_on_the_measured_model(monkeypatch):
    """A configured-subagent api row on the measured model RETRIEVES the subject
    (native tool rounds): a different delivery class from the packet panel every
    published number was produced with, so provenance must refuse it too."""
    from devtools.benchmarks.common.model_slots import BENCHMARK_SUBAGENT_ID

    model = "openai/gpt-5.5"
    settings = _fixed_actor_settings(model)
    monkeypatch.setenv("OUROBOROS_SUBAGENTS", settings["OUROBOROS_SUBAGENTS"])
    payload = json.loads(settings[REVIEWER_SLOTS_ENV])
    payload["triad"][0] = {"slot_id": "native-t", "subagent_id": BENCHMARK_SUBAGENT_ID}
    settings[REVIEWER_SLOTS_ENV] = json.dumps(payload)
    snapshot = runtime_actor_snapshot(settings, expected_model=model)
    assert any(
        "native-t" in item and "native_tool_rounds" in item and "packet delivery" in item
        for item in snapshot["mismatches"]
    )


def test_runtime_actor_snapshot_refuses_enabled_foreign_advisory():
    model = "openai/gpt-5.5"
    settings = _fixed_actor_settings(model)
    payload = json.loads(settings[REVIEWER_SLOTS_ENV])
    payload["advisory"] = {
        "enabled": True,
        "route": {"kind": "api_chat", "target_id": "foreign-sdk-model"},
        "effort": "high",
    }
    settings[REVIEWER_SLOTS_ENV] = json.dumps(payload)
    snapshot = runtime_actor_snapshot(settings, expected_model=model)
    assert any("advisory is enabled" in item for item in snapshot["mismatches"])


def test_runtime_actor_snapshot_uses_structured_reviewers_not_stale_legacy_strings():
    model = "openai/gpt-5.5"
    settings = _fixed_actor_settings(model, review_slots=3)
    settings.update({
        "OUROBOROS_REVIEW_MODELS": "foreign/stale-triad",
        "OUROBOROS_SCOPE_REVIEW_MODELS": "foreign/stale-scope",
        "OUROBOROS_SCOPE_REVIEW_MODEL": "foreign/stale-singular",
        "CLAUDE_CODE_MODEL": "foreign-stale-sdk-model",
    })
    snapshot = runtime_actor_snapshot(settings, expected_model=model)
    assert snapshot["mismatches"] == []
    assert [row["route"]["target_id"] for row in snapshot["reviewer_slots"]["triad"]] == [
        model, model, model,
    ]
    assert snapshot["reviewer_slots"]["advisory"]["enabled"] is False


@pytest.mark.parametrize(
    "local_key",
    ("USE_LOCAL_MAIN", "USE_LOCAL_LIGHT", "USE_LOCAL_FALLBACK", "USE_LOCAL_CONSCIOUSNESS"),
)
def test_runtime_actor_snapshot_refuses_each_remote_to_local_route_drift(local_key):
    model = "openai/gpt-5.5"
    settings = _fixed_actor_settings(model)
    settings[local_key] = True
    snapshot = runtime_actor_snapshot(settings, expected_model=model)
    assert any(local_key in item for item in snapshot["mismatches"])


def test_single_model_slot_snapshot_is_cli_derived_and_has_no_heavy():
    slots = single_model_slot_snapshot("openai/gpt-5.6-sol", review_slots=2)
    assert slots["OUROBOROS_MODEL"] == "openai/gpt-5.6-sol"
    assert slots["OUROBOROS_REVIEW_MODELS"] == (
        "openai/gpt-5.6-sol,openai/gpt-5.6-sol"
    )
    assert "OUROBOROS_MODEL_HEAVY" not in slots


@pytest.mark.parametrize("relative,expected", PROFILE_TARGETS.items())
def test_committed_single_model_profiles_use_one_canonical_actor(relative: str, expected: str):
    payload = json.loads((REPO / relative).read_text(encoding="utf-8"))
    raw = payload["OUROBOROS_SUBAGENTS"]
    assert _only_target(raw) == expected == payload["OUROBOROS_MODEL"]
    assert serialize_configured_subagents(parse_configured_subagents(raw)) == raw
    assert "OUROBOROS_MODEL_HEAVY" not in payload
    assert "USE_LOCAL_HEAVY" not in payload


@pytest.mark.parametrize(
    "relative,expected,triad_count,scope_count",
    (
        ("devtools/benchmarks/programbench/settings_base.json", "openai/gpt-5.5", 3, 3),
        ("devtools/benchmarks/osworld/settings_base.json", "anthropic/claude-sonnet-4.6", 3, 1),
    ),
)
def test_target_attached_profiles_override_foreign_runtime_defaults(
        relative, expected, triad_count, scope_count):
    payload = json.loads((REPO / relative).read_text(encoding="utf-8"))
    # ABI 7.0 (ABI-10): the comma keys are retired — the structured slots value
    # is the template's ONE reviewer configuration surface. The slot counts are
    # the committed template shape (programbench 3 triad + 3 scope, osworld
    # 3 triad + 1 scope), pinned as literals: derived from the payload they
    # would certify whatever count the file happens to carry.
    slots = json.loads(payload[REVIEWER_SLOTS_ENV])
    assert (len(slots["triad"]), len(slots["scope"])) == (triad_count, scope_count)
    assert payload[REVIEWER_SLOTS_ENV] == single_model_reviewer_slots_setting(
        expected,
        review_slots=triad_count,
        scope_slots=scope_count,
        review_effort=payload["OUROBOROS_EFFORT_REVIEW"],
        scope_effort=payload["OUROBOROS_EFFORT_SCOPE_REVIEW"],
    )
    assert "CLAUDE_CODE_MODEL" not in payload  # retired setting: dropped from templates
    assert all(payload[key] is False for key in (
        "USE_LOCAL_MAIN", "USE_LOCAL_LIGHT", "USE_LOCAL_FALLBACK",
        "USE_LOCAL_CONSCIOUSNESS",
    ))
    effective = dict(SETTINGS_DEFAULTS)
    effective.update(payload)
    actor = runtime_actor_snapshot(effective, expected_model=expected)
    assert actor["mismatches"] == []
    assert not any(actor["local_routes"].values())
    assert actor["reviewer_slots"]["advisory"]["enabled"] is False
    assert all(
        item.strip() == expected
        for raw in actor["model_slots"].values()
        for item in raw.split(",")
        if item.strip()
    )


def test_benchmark_snapshot_records_canonical_actor_and_refuses_malformed(tmp_path):
    settings = tmp_path / "settings.json"
    raw = single_model_subagents_setting("anthropic/claude-fable-5")
    settings.write_text(json.dumps({"OUROBOROS_SUBAGENTS": raw}), encoding="utf-8")
    snapshot = configured_subagents_snapshot(settings, env_overrides=False)
    assert [row["route"]["target_id"] for row in snapshot["items"]] == [
        "anthropic/claude-fable-5"
    ]

    settings.write_text(json.dumps({"OUROBOROS_SUBAGENTS": "not-json"}), encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        configured_subagents_snapshot(settings, env_overrides=False)


def test_isolated_settings_copy_active_actor_but_not_legacy_heavy():
    model = "openai/gpt-5.5"
    raw = single_model_subagents_setting(model)
    reviewers = single_model_reviewer_slots_setting(model)
    isolated = build_isolated_settings({
        "OUROBOROS_MODEL": model,
        "OUROBOROS_MODEL_HEAVY": "decoy/heavy",
        "OUROBOROS_SUBAGENTS": raw,
        REVIEWER_SLOTS_ENV: reviewers,
    })
    assert isolated["OUROBOROS_SUBAGENTS"] == raw
    assert isolated[REVIEWER_SLOTS_ENV] == reviewers
    assert "OUROBOROS_MODEL_HEAVY" not in isolated
    assert {REVIEWER_SLOTS_ENV, "USE_LOCAL_CONSCIOUSNESS"}.issubset(
        STALE_INHERITED_ENV_KEYS
    )


def test_legacy_heavy_read_vocabulary_does_not_leak_into_new_projection(
    tmp_path, monkeypatch
):
    old = tmp_path / "old-settings.json"
    old.write_text(json.dumps({"OUROBOROS_MODEL_HEAVY": "legacy/measured"}), encoding="utf-8")
    assert "OUROBOROS_MODEL_HEAVY" in MODEL_SLOT_KEYS
    old_manifest = {"model_slots": {"OUROBOROS_MODEL_HEAVY": "legacy/measured"}}
    assert old_manifest["model_slots"]["OUROBOROS_MODEL_HEAVY"] == "legacy/measured"
    assert "OUROBOROS_MODEL_HEAVY" not in model_slot_snapshot(old, env_overrides=False)
    monkeypatch.setenv("OUROBOROS_MODEL_HEAVY", "ambient/heavy")
    assert "OUROBOROS_MODEL_HEAVY" not in model_slot_snapshot(old)


def test_clb_operator_adapters_transport_canonical_actor_without_active_heavy():
    patch_root = REPO / "devtools/benchmarks/continual_learning/operator_patches"
    host_patch = (patch_root / "_launcher.v6560.patch").read_text(encoding="utf-8")
    docker_patch = (patch_root / "clb_env_campaign_overrides.v6745.patch").read_text(
        encoding="utf-8"
    )
    official_patch = (patch_root / "adapter_official_submission.v681.patch").read_text(
        encoding="utf-8"
    )
    assert "Canonical benchmark actor bytes are authored by run_clb.py" in host_patch
    assert '"OUROBOROS_SUBAGENTS"):' in docker_patch
    assert "OUROBOROS_SUBAGENTS=os.environ.get" in official_patch
    assert "OUROBOROS_MODEL_HEAVY" not in official_patch
    assert "OUROBOROS_MODEL_CODE" not in official_patch


def test_programbench_preflight_requires_exact_benchmark_actor(tmp_path, monkeypatch):
    from devtools.benchmarks.programbench.run_programbench_e2e import preflight_model_slots

    _scrub_model_route_env(monkeypatch)
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({
        "OPENROUTER_API_KEY": "test-key",
        "OUROBOROS_MODEL": "openai/gpt-5.5",
    }), encoding="utf-8")
    with pytest.raises(SystemExit, match="OUROBOROS_SUBAGENTS"):
        preflight_model_slots(settings, solve_model="openai/gpt-5.5")


def test_programbench_preflight_requires_a_declared_measured_model(tmp_path, monkeypatch):
    from devtools.benchmarks.programbench.run_programbench_e2e import preflight_model_slots

    _scrub_model_route_env(monkeypatch)
    settings = tmp_path / "settings.json"
    settings.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="measured model must be declared"):
        preflight_model_slots(settings)


def test_programbench_binds_manifest_to_target_actor_and_refuses_before_discovery(
    tmp_path, monkeypatch
):
    from devtools.benchmarks.programbench import run_programbench_e2e as e2e

    measured = "openai/gpt-5.5"
    actual = "anthropic/claude-fable-5"
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps({
            "OUROBOROS_MODEL": measured,
            "OUROBOROS_SUBAGENTS": single_model_subagents_setting(measured),
        }),
        encoding="utf-8",
    )
    out = tmp_path / "run"

    def fake_admit(path, **_kwargs):
        manifest = {"harness": {}, "extra": {}, "output_paths": {}}
        e2e.write_json(path, manifest)
        return manifest

    monkeypatch.setattr(e2e, "run_root", lambda *_a, **_k: out)
    monkeypatch.setattr(e2e, "assert_outside_repo", lambda path, _repo: path)
    monkeypatch.setattr(e2e, "admit_benchmark_run", fake_admit)
    monkeypatch.setattr(
        e2e,
        "preflight_model_slots",
        lambda *_a, **_k: {"OUROBOROS_MODEL": measured},
    )
    monkeypatch.setattr(
        e2e,
        "ouroboros_api_request",
        lambda *_a, **_k: _fixed_actor_settings(actual),
    )
    monkeypatch.setattr(
        e2e,
        "_load_instances",
        lambda **_k: (_ for _ in ()).throw(AssertionError("paid discovery ran")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_programbench_e2e.py",
            "--repo-dir",
            str(tmp_path / "repo"),
            "--settings-path",
            str(settings),
            "--solve-model",
            measured,
        ],
    )
    assert e2e.main() == 2
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert _only_target(json.dumps(manifest["available_subagents"])) == actual
    assert manifest["extra"]["refusal"]["reason"] == "target_actor_mismatch"


def test_programbench_target_actor_is_durable_before_discovery_and_first_task_crash(
    tmp_path, monkeypatch
):
    from devtools.benchmarks.programbench import run_programbench_e2e as e2e

    measured = "openai/gpt-5.5"
    target_settings = _fixed_actor_settings(measured)
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps(target_settings), encoding="utf-8")
    out = tmp_path / "run"

    def fake_admit(path, **_kwargs):
        manifest = {"harness": {}, "extra": {}, "output_paths": {}}
        e2e.write_json(path, manifest)
        return manifest

    def assert_durable_actor():
        durable = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
        actor = durable["harness"]["target_runtime_actor"]
        assert actor["mismatches"] == []
        assert actor["model"] == measured
        assert actor["model_slots"]["OUROBOROS_MODEL"] == measured
        assert not any(actor["local_routes"].values())
        assert actor["reviewer_slots"]["advisory"]["enabled"] is False
        assert durable["available_subagents"] == actor["available_subagents"]

    observed = {"discovery": False, "first_task": False}

    def load_instances(**_kwargs):
        assert_durable_actor()
        observed["discovery"] = True
        return [{"instance_id": "inst-a", "image_name": "img-a"}]

    def crash_first_task(_instance, _config):
        assert_durable_actor()
        observed["first_task"] = True
        raise RuntimeError("synthetic first-task crash")

    monkeypatch.setattr(e2e, "_ensure_docker_host", lambda: None)
    monkeypatch.setattr(e2e, "run_root", lambda *_a, **_k: out)
    monkeypatch.setattr(e2e, "assert_outside_repo", lambda path, _repo: path)
    monkeypatch.setattr(e2e, "admit_benchmark_run", fake_admit)
    monkeypatch.setattr(e2e, "preflight_model_slots", lambda *_a, **_k: target_settings)
    monkeypatch.setattr(e2e, "ouroboros_api_request", lambda *_a, **_k: target_settings)
    monkeypatch.setattr(e2e, "runtime_attestation", lambda *_a, **_k: {"ok": True})
    monkeypatch.setattr(e2e, "_load_instances", load_instances)
    monkeypatch.setattr(e2e, "_process_instance", crash_first_task)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_programbench_e2e.py",
            "--repo-dir",
            str(tmp_path / "repo"),
            "--settings-path",
            str(settings),
            "--solve-model",
            measured,
        ],
    )

    with pytest.raises(RuntimeError, match="synthetic first-task crash"):
        e2e.main()
    assert observed == {"discovery": True, "first_task": True}
    assert_durable_actor()
    final = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert final["extra"]["outcome"] == "crashed"
    assert final["extra"]["error"]["type"] == "RuntimeError"


def test_osworld_allowed_target_mismatch_records_the_actual_actor():
    from devtools.benchmarks.osworld import run_step_agent as rsa

    actual = single_model_subagents_setting("anthropic/claude-fable-5")
    manifest = {"harness": {}}
    preflight = {
        "details": {
            "scaffold_mismatch_allowed": ["actor drift"],
            "target_runtime_actor": runtime_actor_snapshot(
                {
                    "OUROBOROS_MODEL": "anthropic/claude-fable-5",
                    "OUROBOROS_SUBAGENTS": actual,
                },
                expected_model="openai/gpt-5.5",
            ),
        }
    }
    rsa._bind_target_actor(manifest, preflight)
    assert _only_target(json.dumps(manifest["available_subagents"])) == (
        "anthropic/claude-fable-5"
    )
    assert manifest["harness"]["target_runtime_actor"]["mismatches"]


def test_osworld_cu_bridge_validates_declared_and_target_actor(tmp_path, monkeypatch):
    from devtools.benchmarks.osworld import run_cu_bridge_agent as cu

    model = "openai/gpt-5.5"
    exact = _fixed_actor_settings(model)
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps(exact), encoding="utf-8")
    monkeypatch.setattr(
        cu,
        "_api",
        lambda *_a, **_k: dict(exact),
    )
    assert cu._cu_actor_preflight(settings, "http://target")["ok"] is True

    contaminated_settings = dict(exact)
    contaminated_settings.update({
        "OUROBOROS_MODEL_LIGHT": "anthropic/foreign-light",
        "OUROBOROS_MODEL_FALLBACKS": "anthropic/foreign-fallback",
    })
    monkeypatch.setattr(cu, "_api", lambda *_a, **_k: contaminated_settings)
    contaminated = cu._cu_actor_preflight(settings, "http://target")
    assert contaminated["ok"] is False
    assert any("OUROBOROS_MODEL_LIGHT" in item for item in contaminated["failures"])
    assert any("OUROBOROS_MODEL_FALLBACKS" in item for item in contaminated["failures"])

    other = "anthropic/claude-fable-5"
    monkeypatch.setattr(
        cu,
        "_api",
        lambda *_a, **_k: _fixed_actor_settings(other),
    )
    drifted = cu._cu_actor_preflight(settings, "http://target")
    assert drifted["ok"] is False
    assert _only_target(json.dumps(drifted["target"]["available_subagents"])) == other
    manifest = {"harness": {}}
    cu._bind_cu_actor(manifest, drifted)
    assert _only_target(json.dumps(manifest["available_subagents"])) == other
    assert manifest["harness"]["actor_preflight"] is drifted


def test_harness_and_harbor_manifests_use_exact_cli_model(tmp_path, monkeypatch):
    from devtools.benchmarks.harness_bench_fast import run_harness_bench_fast as hbf
    from devtools.benchmarks.terminal_bench import run_harbor_smoke as smoke

    measured = "openai/gpt-5.6-sol"
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps({
            "OUROBOROS_MODEL": "decoy/template",
            "OUROBOROS_MODEL_HEAVY": "decoy/heavy",
            "OUROBOROS_SUBAGENTS": single_model_subagents_setting("decoy/actor"),
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv("OUROBOROS_MODEL", "decoy/ambient")
    monkeypatch.setenv("OUROBOROS_MODEL_HEAVY", "decoy/ambient-heavy")
    monkeypatch.setenv("CLAUDE_CODE_MODEL", "foreign-sdk-model")
    monkeypatch.setenv(REVIEWER_SLOTS_ENV, json.dumps({
        "triad": [{"slot_id": "foreign-t", "route": {
            "kind": "agent_session", "target_id": "codex=gpt-5.6-sol"}}],
        "scope": [{"slot_id": "foreign-s", "route": {
            "kind": "api_chat", "target_id": "foreign/scope"}}],
        "advisory": {"enabled": True, "route": {
            "kind": "api_chat", "target_id": "foreign-advisory"}},
    }))
    for key in ("USE_LOCAL_MAIN", "USE_LOCAL_LIGHT", "USE_LOCAL_FALLBACK",
                "USE_LOCAL_CONSCIOUSNESS"):
        monkeypatch.setenv(key, "true")

    def assert_complete_actor(manifest):
        actor = manifest["harness"]["fixed_model_actor"]
        assert actor["mismatches"] == []
        assert not any(actor["local_routes"].values())
        assert actor["reviewer_slots"]["advisory"]["enabled"] is False
        assert {row["route"]["target_id"] for row in actor["reviewer_slots"]["triad"]} == {measured}

    hbf_root = tmp_path / "hbf"

    def discover_hbf(*_a, **_k):
        assert_complete_actor(json.loads(
            (hbf_root / "run_manifest.json").read_text(encoding="utf-8")
        ))
        return ["task_1"]

    monkeypatch.setattr(hbf, "_read_task_ids", discover_hbf)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_harness_bench_fast.py",
            "--repo-dir",
            str(REPO),
            "--bench-root",
            str(tmp_path / "bench"),
            "--run-root",
            str(hbf_root),
            "--settings-path",
            str(settings),
            "--model",
            measured,
            "--allow-dirty-seed",
            "--dry-run",
        ],
    )
    assert hbf.main() == 0
    hbf_manifest = json.loads((hbf_root / "run_manifest.json").read_text(encoding="utf-8"))
    assert set(hbf_manifest["model_slots"].values()) == {measured}
    assert "OUROBOROS_MODEL_HEAVY" not in hbf_manifest["model_slots"]
    assert _only_target(json.dumps(hbf_manifest["available_subagents"])) == measured
    assert_complete_actor(hbf_manifest)

    smoke_root = tmp_path / "smoke"
    monkeypatch.setattr(smoke, "repo_root_from_devtools", lambda: REPO)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_harbor_smoke.py",
            "--run-root",
            str(smoke_root),
            "--settings-path",
            str(settings),
            "--model",
            measured,
            "--allow-dirty-seed",
        ],
    )
    assert smoke.main() == 0
    smoke_manifest = json.loads(
        (smoke_root / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert set(smoke_manifest["model_slots"].values()) == {measured}
    assert "OUROBOROS_MODEL_HEAVY" not in smoke_manifest["model_slots"]
    assert _only_target(json.dumps(smoke_manifest["available_subagents"])) == measured
    assert_complete_actor(smoke_manifest)


def test_swe_pro_derived_profile_overrides_actor_without_heavy(tmp_path):
    from devtools.benchmarks.swe_bench_pro.e1v2.run_pro import derive_run_settings

    template = tmp_path / "template.json"
    template.write_text(json.dumps({
        "OUROBOROS_MODEL": "decoy/main",
        "OUROBOROS_MODEL_HEAVY": "decoy/heavy",
        "OUROBOROS_SUBAGENTS": single_model_subagents_setting("decoy/actor"),
    }), encoding="utf-8")
    out = tmp_path / "run"
    out.mkdir()
    derived = derive_run_settings(str(template), out, "openai/gpt-5.6-sol", 10.0, 5.0)
    payload = json.loads(derived.read_text(encoding="utf-8"))
    assert _only_target(payload["OUROBOROS_SUBAGENTS"]) == "openai/gpt-5.6-sol"
    assert "OUROBOROS_MODEL_HEAVY" not in payload


def test_swe_pro_all_resume_manifest_keeps_exact_actor_and_slots(tmp_path, monkeypatch):
    from devtools.benchmarks.swe_bench_pro.e1v2 import run_pro

    out = tmp_path / "run"
    ids = ["inst__a", "inst__b"]
    for cid in ids:
        task_dir = out / cid
        task_dir.mkdir(parents=True)
        (task_dir / "patch.diff").write_text("diff --git a/x b/x\n", encoding="utf-8")
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps({
            "OUROBOROS_MODEL": "decoy/template",
            "OUROBOROS_MODEL_HEAVY": "decoy/heavy",
        }),
        encoding="utf-8",
    )
    args = types.SimpleNamespace(
        full_set=True,
        csv="",
        start=1,
        limit=2,
        allow_dirty_seed=True,
        solve_timeout=0,
        settings=str(settings),
        solve_model="openai/gpt-5.6-sol",
        self_improve=False,
        cadence="off",
        reset_state=False,
        model_name="ouroboros-test",
        review_slots=1,
        review_effort="",
        runtime_mode="",
        image_input_mode="",
        total_budget=100.0,
        per_task_cost=10.0,
        pretask_evolution=False,
        pause_on_api_err=-1,
    )
    row = {"dockerhub_tag": "unused"}
    monkeypatch.setattr(run_pro, "read_full_order", lambda: ids)
    monkeypatch.setattr(run_pro, "load_pro_rows", lambda selected: {i: row for i in selected})
    monkeypatch.setattr(run_pro, "assert_seed_is_git_directory", lambda _path: None)
    monkeypatch.setattr(run_pro, "ensure_util_image", lambda: None)
    monkeypatch.setattr(
        run_pro,
        "derive_run_settings",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("resume derived settings")),
    )
    monkeypatch.setattr(
        run_pro.subprocess,
        "run",
        lambda *_a, **_k: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    assert run_pro._run_schedule(args, out, "", "key") == 0
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_slots"]["OUROBOROS_MODEL"] == "openai/gpt-5.6-sol"
    assert "OUROBOROS_MODEL_HEAVY" not in manifest["model_slots"]
    assert _only_target(json.dumps(manifest["available_subagents"])) == (
        "openai/gpt-5.6-sol"
    )


def test_editbench_seed_disables_but_records_effective_main_actor(tmp_path, monkeypatch):
    from devtools.benchmarks.editbench import run_editbench

    fake_home = tmp_path / "home"
    (fake_home / "Ouroboros" / "data").mkdir(parents=True)
    (fake_home / "Ouroboros" / "data" / "settings.json").write_text(
        json.dumps({"OUROBOROS_MODEL": "anthropic/claude-fable-5"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(run_editbench.pathlib.Path, "home", lambda: fake_home)
    settings_path = run_editbench._seed_settings(tmp_path / "data")
    payload = json.loads(settings_path.read_text(encoding="utf-8"))
    config = parse_configured_subagents(payload["OUROBOROS_SUBAGENTS"])
    assert config.enabled is False
    assert len(config.items) == 1
    assert config.items[0].route.target_id == payload["OUROBOROS_MODEL"]
    assert payload["OUROBOROS_MODEL"] == "anthropic/claude-fable-5"
    assert "OUROBOROS_MODEL_HEAVY" not in payload
