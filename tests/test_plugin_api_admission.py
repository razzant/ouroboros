"""ABI-1 packaged-artifact admission suite (v7next Ф3.1-B).

The NEW-PASS admission predicate is one function, common to every PASS-minting
path — LLM review, owner attestation and native-seed trust — and lives OUTSIDE
the deterministic preflight. Every test here operates on a real packaged
payload directory (bytes on disk), not a synthetic in-memory manifest: a
field-less extension artifact is refused a NEW PASS at issuance (not at load),
the hash-bound grandfather PASS survives a repeat review of the same bytes,
the preflight itself now fails closed, grants ride a native re-seed with
unchanged requested sets, and the declared-dependency fingerprint (6.2=A)
keeps `.ouroboros_env` bytes out of the hash while hashing the declared names.
"""

from __future__ import annotations

import json
import logging
import pathlib
import types
from unittest.mock import patch

from ouroboros.skill_loader import (
    SkillReviewState,
    compute_content_hash,
    load_review_state,
    load_skill,
    save_review_state,
)
from ouroboros.skill_review import review_skill

from tests._skill_review_shared import _make_actor, _make_ctx, _pass_array_for_script_skill

log = logging.getLogger(__name__)


def _build_extension_payload(
    skills_root: pathlib.Path,
    name: str,
    *,
    plugin_api_line: str = "",
    extra_frontmatter: str = "",
) -> pathlib.Path:
    skill_dir = skills_root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        (
            "---\n"
            f"name: {name}\n"
            "description: Packaged admission-test extension.\n"
            "version: 0.1.0\n"
            "type: extension\n"
            "entry: plugin.py\n"
            "permissions: [tool]\n"
            f"{plugin_api_line}"
            f"{extra_frontmatter}"
            "---\n"
            "body\n"
        ),
        encoding="utf-8",
    )
    (skill_dir / "plugin.py").write_text(
        "def register(api):\n"
        "    api.register_tool('t1', lambda **kw: 'ok', description='d', schema={})\n",
        encoding="utf-8",
    )
    return skill_dir


def _pass_panel():
    canned = json.dumps({"results": [
        _make_actor("reviewer-a", _pass_array_for_script_skill()),
        _make_actor("reviewer-b", _pass_array_for_script_skill()),
    ]})
    return patch("ouroboros.tools.review._handle_multi_model_review", return_value=canned)


def _refuse_dispatch():
    return patch(
        "ouroboros.tools.review._handle_multi_model_review",
        side_effect=AssertionError("an inadmissible payload must not dispatch a paid panel"),
    )


# --- review path -------------------------------------------------------------


def test_fieldless_extension_artifact_is_refused_a_new_pass_before_dispatch(
    tmp_path, monkeypatch,
):
    skills_root = tmp_path / "skills"
    _build_extension_payload(skills_root, "noapifield")
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    ctx = _make_ctx(tmp_path)
    with _refuse_dispatch():
        outcome = review_skill(ctx, "noapifield")
    assert outcome.status == "pending"
    assert "PluginAPI 2.0 admission" in outcome.error
    assert outcome.paid is False and outcome.wave_id == ""
    assert [f["item"] for f in outcome.findings] == ["plugin_api_admission"]
    persisted = load_review_state(ctx.drive_root, "noapifield")
    assert persisted.status == "pending"
    assert [f["item"] for f in persisted.findings] == ["plugin_api_admission"]


def test_declared_2_0_extension_artifact_still_earns_a_pass(tmp_path, monkeypatch):
    skills_root = tmp_path / "skills"
    _build_extension_payload(skills_root, "withapifield", plugin_api_line='plugin_api: "2.0"\n')
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    ctx = _make_ctx(tmp_path)
    with _pass_panel():
        outcome = review_skill(ctx, "withapifield")
    assert outcome.status == "clean", outcome.error
    assert load_review_state(
        ctx.drive_root, "withapifield", skill_type="extension"
    ).status == "clean"


def test_repeat_review_of_grandfathered_bytes_never_clobbers_the_live_pass(
    tmp_path, monkeypatch,
):
    """Clobber guard: the admission refusal must not destroy the hash-bound
    PASS the grandfather construction depends on."""
    skills_root = tmp_path / "skills"
    skill_dir = _build_extension_payload(skills_root, "oldtrusted")
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    ctx = _make_ctx(tmp_path)
    content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
    save_review_state(ctx.drive_root, "oldtrusted", SkillReviewState(
        status="clean",
        content_hash=content_hash,
        findings=[{"item": "bug_hunting", "verdict": "PASS", "severity": "critical", "reason": "ok"}],
    ))
    with _refuse_dispatch():
        outcome = review_skill(ctx, "oldtrusted")
    assert outcome.status == "pending"
    assert "grandfather" in outcome.error and "preserved" in outcome.error
    survived = load_review_state(ctx.drive_root, "oldtrusted", skill_type="extension")
    assert survived.status == "clean"
    assert survived.content_hash == content_hash


# --- preflight fail-open fix -------------------------------------------------


def test_preflight_infrastructure_failure_fails_closed_without_persisting(
    tmp_path, monkeypatch,
):
    skills_root = tmp_path / "skills"
    _build_extension_payload(skills_root, "preflightfail", plugin_api_line='plugin_api: "2.0"\n')
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    ctx = _make_ctx(tmp_path)
    with patch(
        "ouroboros.tools.skill_preflight._handle_skill_preflight",
        side_effect=RuntimeError("gate machinery exploded"),
    ), _refuse_dispatch():
        outcome = review_skill(ctx, "preflightfail")
    assert outcome.status == "pending"
    assert "fail-closed" in outcome.error and "gate machinery exploded" in outcome.error
    # Nothing persisted: a transient infra failure must not clobber review state.
    assert load_review_state(ctx.drive_root, "preflightfail").status == "pending"
    assert load_review_state(ctx.drive_root, "preflightfail").content_hash == ""


def test_owner_attestation_shares_the_fail_closed_preflight(tmp_path, monkeypatch):
    """In the attest path the deterministic preflight is the ENTIRE
    replacement for the LLM review; a broken gate must never mint trust."""
    from ouroboros.skill_owner_attestation import run_owner_attestation

    skills_root = tmp_path / "skills"
    skill_dir = _build_extension_payload(skills_root, "attestinfra", plugin_api_line='plugin_api: "2.0"\n')
    skill = load_skill(skill_dir, tmp_path / "drive")
    assert skill is not None and not skill.load_error
    ctx = types.SimpleNamespace(drive_root=str(tmp_path / "drive"))
    with patch(
        "ouroboros.tools.skill_preflight._handle_skill_preflight",
        side_effect=RuntimeError("gate machinery exploded"),
    ):
        outcome = run_owner_attestation(ctx, tmp_path / "drive", skill, skill.content_hash)
    assert outcome.status == "pending"
    assert "fail-closed" in outcome.error


# --- owner attestation admission ---------------------------------------------


def test_owner_attestation_refuses_a_fieldless_extension(tmp_path, monkeypatch):
    import ouroboros.skill_review as sr
    from ouroboros.skill_owner_attestation import run_owner_attestation

    skills_root = tmp_path / "skills"
    skill_dir = _build_extension_payload(skills_root, "attestnoapi")
    drive_root = tmp_path / "drive"
    skill = load_skill(skill_dir, drive_root)
    assert skill is not None and not skill.load_error
    monkeypatch.setattr(sr, "_run_deterministic_preflight", lambda *a, **k: None)
    ctx = types.SimpleNamespace(drive_root=str(drive_root))
    outcome = run_owner_attestation(ctx, drive_root, skill, skill.content_hash)
    assert outcome.status == "pending"
    assert "PluginAPI 2.0 admission" in outcome.error
    # Nothing persisted: no attestation marker, no verdict.
    from ouroboros.skill_loader import skill_state_dir

    assert not (skill_state_dir(drive_root, "attestnoapi") / "owner_attestation.json").exists()


def test_owner_attestation_admits_a_declared_2_0_extension(tmp_path, monkeypatch):
    import ouroboros.skill_review as sr
    from ouroboros.skill_owner_attestation import run_owner_attestation

    skills_root = tmp_path / "skills"
    skill_dir = _build_extension_payload(skills_root, "attestapi", plugin_api_line='plugin_api: "2.0"\n')
    drive_root = tmp_path / "drive"
    skill = load_skill(skill_dir, drive_root)
    assert skill is not None and not skill.load_error
    monkeypatch.setattr(sr, "_run_deterministic_preflight", lambda *a, **k: None)
    ctx = types.SimpleNamespace(drive_root=str(drive_root))
    outcome = run_owner_attestation(ctx, drive_root, skill, skill.content_hash)
    assert outcome.status == "clean"
    assert outcome.review_profile == "owner_attested"


# --- native seed closed + grants resync --------------------------------------


def _seed_pair(tmp_path, *, plugin_api_line: str, extra_frontmatter: str = ""):
    seed_root = tmp_path / "repo_seed"
    seed_dir = _build_extension_payload(
        seed_root, "nativeext",
        plugin_api_line=plugin_api_line, extra_frontmatter=extra_frontmatter,
    )
    drive_root = tmp_path / "drive"
    native_root = drive_root / "skills" / "native"
    native_root.mkdir(parents=True, exist_ok=True)
    return seed_dir, native_root / "nativeext", drive_root


def test_native_seed_trust_is_closed_to_fieldless_extensions(tmp_path):
    import shutil

    from ouroboros.launcher_bootstrap import _stamp_native_seed_trust

    seed_dir, target, drive_root = _seed_pair(tmp_path, plugin_api_line="")
    shutil.copytree(seed_dir, target)
    (target / ".seed-origin").write_text("seeded_from=test\n", encoding="utf-8")
    _stamp_native_seed_trust(drive_root, target, log)
    assert load_review_state(drive_root, "nativeext").status == "pending"

    # The SAME payload with the field earns the native-trust verdict.
    shutil.rmtree(target)
    seed_dir2, target2, _ = _seed_pair(tmp_path, plugin_api_line='plugin_api: "2.0"\n')
    shutil.copytree(seed_dir2, target2)
    (target2 / ".seed-origin").write_text("seeded_from=test\n", encoding="utf-8")
    _stamp_native_seed_trust(drive_root, target2, log)
    stamped = load_review_state(
        drive_root, "nativeext",
        skill_type="extension", skill_dir=target2,
    )
    assert stamped.status == "clean"
    assert stamped.review_profile == "native_seed"


def test_native_reseed_with_unchanged_requested_sets_carries_grants(tmp_path):
    """Owner «A» (§6.1-Δ): a bundled version bump with identical requested
    key/permission sets must not orphan the owner's grants (the telegram
    precedent); a CHANGED requested set keeps the ordinary re-grant flow."""
    import shutil

    from ouroboros.launcher_bootstrap import _reseed_native_skill_in_place
    from ouroboros.skill_loader import load_skill_grants, save_skill_grants

    extra = "env_from_settings: [TELEGRAM_BOT_TOKEN]\n"
    seed_root_v1 = tmp_path / "seed_v1"
    installed = _build_extension_payload(
        seed_root_v1, "nativeext",
        plugin_api_line='plugin_api: "2.0"\n', extra_frontmatter=extra,
    )
    drive_root = tmp_path / "drive"
    native_root = drive_root / "skills" / "native"
    native_root.mkdir(parents=True)
    target = native_root / "nativeext"
    shutil.copytree(installed, target)
    (target / ".seed-origin").write_text("seeded_from=test\n", encoding="utf-8")
    old = load_skill(target, drive_root)
    assert old is not None and old.content_hash
    save_skill_grants(
        drive_root, "nativeext", ["TELEGRAM_BOT_TOKEN"],
        content_hash=old.content_hash, requested_keys=["TELEGRAM_BOT_TOKEN"],
    )

    # v2 seed: same requested sets, different bytes (version bump).
    seed_root_v2 = tmp_path / "seed_v2"
    seed_v2 = _build_extension_payload(
        seed_root_v2, "nativeext",
        plugin_api_line='plugin_api: "2.0"\n', extra_frontmatter=extra,
    )
    manifest = (seed_v2 / "SKILL.md").read_text(encoding="utf-8")
    (seed_v2 / "SKILL.md").write_text(
        manifest.replace("version: 0.1.0", "version: 0.2.0"), encoding="utf-8",
    )
    assert _reseed_native_skill_in_place(seed_v2, target, log, drive_root=drive_root)
    new = load_skill(target, drive_root)
    assert new is not None and new.content_hash != old.content_hash
    grants = load_skill_grants(drive_root, "nativeext")
    assert grants["content_hash"] == new.content_hash
    assert grants["granted_keys"] == ["TELEGRAM_BOT_TOKEN"]

    # v3 seed: the requested set CHANGES -> no carry, ordinary re-grant flow.
    seed_root_v3 = tmp_path / "seed_v3"
    seed_v3 = _build_extension_payload(
        seed_root_v3, "nativeext",
        plugin_api_line='plugin_api: "2.0"\n',
        extra_frontmatter="env_from_settings: [TELEGRAM_BOT_TOKEN, GITHUB_TOKEN]\n",
    )
    manifest = (seed_v3 / "SKILL.md").read_text(encoding="utf-8")
    (seed_v3 / "SKILL.md").write_text(
        manifest.replace("version: 0.1.0", "version: 0.3.0"), encoding="utf-8",
    )
    assert _reseed_native_skill_in_place(seed_v3, target, log, drive_root=drive_root)
    third = load_skill(target, drive_root)
    grants = load_skill_grants(drive_root, "nativeext")
    assert grants["content_hash"] != third.content_hash, (
        "a changed requested set must not silently carry old grants to the new hash"
    )


def test_bundled_extension_seeds_declare_the_plugin_api_field():
    """Q6/Q-B: bundled skills ship the field in the same release that closes
    native-seed trust to field-less extensions."""
    repo_root = pathlib.Path(__file__).parents[1]
    for name in ("telegram", "unix_computer_use"):
        from ouroboros.contracts.skill_manifest import parse_skill_manifest_text

        manifest = parse_skill_manifest_text(
            (repo_root / "skills" / name / "SKILL.md").read_text(encoding="utf-8")
        )
        assert manifest.is_extension()
        assert manifest.plugin_api == {"version": "2.0", "capabilities": []}, name


# --- 6.2=A: the declarative dependency fingerprint ---------------------------


def test_ouroboros_env_bytes_stay_outside_the_review_hash(tmp_path):
    skills_root = tmp_path / "skills"
    skill_dir = _build_extension_payload(skills_root, "depsext", plugin_api_line='plugin_api: "2.0"\n')
    before = compute_content_hash(skill_dir, manifest_entry="plugin.py")
    site = skill_dir / ".ouroboros_env" / "python" / "lib" / "python3.11" / "site-packages" / "evil"
    site.mkdir(parents=True)
    (site / "__init__.py").write_text("VALUE = 'unreviewed executable bytes'\n", encoding="utf-8")
    assert compute_content_hash(skill_dir, manifest_entry="plugin.py") == before


def test_new_declared_dependency_name_changes_the_hash_and_forces_rereview(tmp_path):
    skills_root = tmp_path / "skills"
    skill_dir = _build_extension_payload(skills_root, "depsext2", plugin_api_line='plugin_api: "2.0"\n')
    before = compute_content_hash(skill_dir, manifest_entry="plugin.py")
    sidecar = {
        "source": "clawhub",
        "install_specs": {"auto": [
            {"kind": "pip", "package": "requests", "bins": [], "mode": "auto", "raw": {}},
        ]},
    }
    (skill_dir / ".clawhub.json").write_text(json.dumps(sidecar), encoding="utf-8")
    after = compute_content_hash(skill_dir, manifest_entry="plugin.py")
    assert after != before, "a newly declared dependency name must stale the review hash"


def test_state_plane_dependency_desync_is_a_typed_load_refusal(tmp_path, monkeypatch):
    """The EFFECTIVE dependency set (state-plane provenance for ClawHub
    payloads) must match the names declared by hash-covered payload carriers;
    an unhashed record can never widen the surface silently (6.2=A)."""
    from ouroboros.extension_liveness import _deps_block_reason

    skills_root = tmp_path / "skills"
    skill_dir = _build_extension_payload(skills_root, "depsext3", plugin_api_line='plugin_api: "2.0"\n')
    drive_root = tmp_path / "drive"
    skill = load_skill(skill_dir, drive_root)
    assert skill is not None
    skill.source = "clawhub"
    monkeypatch.setattr(
        "ouroboros.marketplace.provenance.read_provenance",
        lambda dr, name: {"install_specs": {"auto": [
            {"kind": "pip", "package": "totally_unreviewed", "bins": [], "mode": "auto", "raw": {}},
        ]}},
    )
    assert _deps_block_reason(drive_root, skill) == "deps_declaration_desync"

    # A matching payload declaration clears the refusal (ordinary deps gates
    # take over: here the env is simply not installed yet).
    sidecar = {
        "source": "clawhub",
        "install_specs": {"auto": [
            {"kind": "pip", "package": "totally_unreviewed", "bins": [], "mode": "auto", "raw": {}},
        ]},
    }
    (skill_dir / ".clawhub.json").write_text(json.dumps(sidecar), encoding="utf-8")
    skill = load_skill(skill_dir, drive_root)
    skill.source = "clawhub"
    assert _deps_block_reason(drive_root, skill) == "deps_missing"
