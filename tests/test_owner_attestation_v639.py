"""Owner attestation ("Skip review" / "Turn on anyway"): the owner skips the LLM review for
the exact bytes of a skill from any origin, the deterministic preflight floor still gates
it, and the verdict is valid only while the owner-issued marker is present."""

from __future__ import annotations

import json



def _write_owner_attested_review(state_dir, content_hash, with_marker):
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "review.json").write_text(json.dumps({
        "status": "clean",
        "content_hash": content_hash,
        "review_profile": "owner_attested",
        "reviewer_models": ["owner_attestation"],
        "findings": [{"item": "owner_attestation", "verdict": "PASS", "severity": "info",
                      "reason": "owner attested"}],
    }), encoding="utf-8")
    if with_marker:
        (state_dir / "owner_attestation.json").write_text(
            json.dumps({"attested_at": "now", "content_hash": content_hash}), encoding="utf-8")


def test_owner_attest_refuses_invalid_manifest(monkeypatch, tmp_path):
    # The deterministic floor must cover what the SKIPPED LLM manifest reviewer would catch:
    # a parsed-but-invalid manifest (validate() warnings) is NOT attestable.
    import types
    import ouroboros.skill_review as sr
    import ouroboros.skill_owner_attestation as soa

    class _Manifest:
        def validate(self):
            return ["unknown type 'extension'", "type=extension requires a non-empty entry"]

    class _Skill:
        name = "s"
        manifest = _Manifest()

    # Preflight passes (soa references skill_review._run_deterministic_preflight module-qualified,
    # so patching it on skill_review takes effect); the manifest-validate floor must still refuse.
    monkeypatch.setattr(sr, "_run_deterministic_preflight", lambda *a, **k: None)
    ctx = types.SimpleNamespace(drive_root=str(tmp_path))
    out = soa.run_owner_attestation(ctx, tmp_path, _Skill(), "hash")
    assert out.status == sr.STATUS_PENDING
    assert "validation" in str(out.error or "").lower()


def test_real_instruction_manifest_with_scripts_is_flagged():
    # A REAL parsed instruction manifest that declares executable scripts is a structural
    # mismatch the owner-attestation deterministic floor (via SkillManifest.validate) catches.
    from ouroboros.contracts.skill_manifest import parse_skill_manifest_text
    m = parse_skill_manifest_text('{"name": "s", "type": "instruction", "scripts": ["run.py"]}')
    warnings = m.validate()
    assert any("must not declare executable" in w for w in warnings), warnings


def test_owner_attested_verdict_valid_only_with_marker(tmp_path):
    from ouroboros.skill_loader import load_review_state, skill_state_dir
    drive_root = tmp_path / "data"
    name = "myskill"
    state_dir = skill_state_dir(drive_root, name)

    # With the owner-issued marker present -> the attested verdict loads (clean, profile kept).
    _write_owner_attested_review(state_dir, "hash123", with_marker=True)
    state = load_review_state(drive_root, name, skill_type="script")
    assert state.review_profile == "owner_attested"
    assert state.content_hash == "hash123"
    assert state.status == "clean"

    # Remove the marker -> the verdict is INVALIDATED (fail-safe: drops to a blank/pending
    # state), exactly like native_seed provenance.
    (state_dir / "owner_attestation.json").unlink()
    state2 = load_review_state(drive_root, name, skill_type="script")
    assert state2.review_profile == ""
    assert state2.content_hash == ""
