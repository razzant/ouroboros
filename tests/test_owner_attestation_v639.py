"""Owner attestation ("Skip review" / "Turn on anyway"): the owner skips the LLM review for
the exact bytes of a skill from any origin, the deterministic preflight floor still gates
it, and the verdict is valid only while the owner-issued marker is present."""

from __future__ import annotations

import json

from ouroboros.contracts.skill_payload_policy import (
    SKILL_OWNER_STATE_FILENAMES,
    SKILL_OWNER_STATE_STEMS,
    is_skill_owner_state_target,
)
from ouroboros.skill_loader import skill_state_dir


def test_owner_attestation_marker_is_owner_state():
    # The marker is an OWNER-STATE file: the agent can never forge it (that would
    # self-bypass the immune system's review).
    assert "owner_attestation.json" in SKILL_OWNER_STATE_FILENAMES
    assert "owner_attestation" in SKILL_OWNER_STATE_STEMS


def test_owner_attestation_target_is_agent_write_protected(tmp_path):
    # is_skill_owner_state_target recognizes the marker under data/state/skills/<name>/.
    data_root = tmp_path / "data"
    target = data_root / "state" / "skills" / "myskill" / "owner_attestation.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    assert is_skill_owner_state_target(target, data_root) is True


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


def test_owner_attest_route_level_post_blocked():
    # The Playwright route-level guard aborts a click/form-triggered POST (not just an
    # evaluate fetch) to the owner-only endpoint.
    from ouroboros.browser_policy import _is_owner_skill_attest_post

    class _Req:
        def __init__(self, method, url):
            self.method = method
            self.url = url

    assert _is_owner_skill_attest_post(_Req("POST", "http://127.0.0.1:8765/api/owner/skills/x/attest-review")) is True
    # The broad route glob reaches the handler for an encoded path; the handler decodes it.
    assert _is_owner_skill_attest_post(_Req("POST", "http://127.0.0.1:8765/api/owner/skills/x/attest%2Dreview")) is True
    assert _is_owner_skill_attest_post(_Req("GET", "http://127.0.0.1:8765/api/owner/skills/x/attest-review")) is False
    assert _is_owner_skill_attest_post(_Req("POST", "http://127.0.0.1:8765/api/skills/x/review")) is False


def test_lifecycle_invokes_attest_impl_positionally(tmp_path):
    # The owner endpoint routes through run_skill_review_lifecycle, which invokes
    # review_impl(ctx, skill_name) POSITIONALLY (no persist=/review_rebuttal=). Prove that
    # review_skill_owner_attest's signature matches exactly, so the integration cannot
    # TypeError at runtime.
    import inspect
    import types
    from ouroboros.skill_review_runner import _call_review_with_lifecycle_guard
    from ouroboros.skill_owner_attestation import review_skill_owner_attest
    from ouroboros.skill_review import SkillReviewOutcome, STATUS_PENDING

    seen = {}

    def _impl(ctx, skill_name):
        seen["args"] = (ctx, skill_name)
        return SkillReviewOutcome(skill_name=skill_name, status=STATUS_PENDING)

    fake_ctx = types.SimpleNamespace(drive_root=str(tmp_path))
    _call_review_with_lifecycle_guard(_impl, fake_ctx, "myskill")
    assert seen["args"] == (fake_ctx, "myskill")
    params = list(inspect.signature(review_skill_owner_attest).parameters)
    assert params == ["ctx", "skill_name"]


def _attestable_skill(tmp_path, *, name, source):
    class _Manifest:
        entry = "plugin.py"
        scripts = []
        env_from_settings = []
        permissions = []
        subscribe_events = []

        def validate(self):
            return []

    skill_dir = tmp_path / "skills" / source / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "plugin.py").write_text("def run(): pass\n", encoding="utf-8")

    class _Skill:
        load_error = None
        is_self_authored = False
        manifest = _Manifest()

    from ouroboros.skill_loader import compute_content_hash

    skill = _Skill()
    skill.name, skill.source, skill.skill_dir = name, source, skill_dir
    skill.content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
    return skill


def test_owner_attest_accepts_every_origin(monkeypatch, tmp_path):
    """Skip review is the owner's decision for this install, whatever the origin:
    built-in, a hub source, ClawHub, a local folder (the deterministic floor still runs)."""
    import types
    import ouroboros.skill_owner_attestation as soa
    import ouroboros.skill_review as sr

    monkeypatch.setattr(sr, "_run_deterministic_preflight", lambda *a, **k: None)
    for source in ("native", "ouroboroshub", "clawhub", "user_repo", "external"):
        skill = _attestable_skill(tmp_path, name=f"s-{source}", source=source)
        monkeypatch.setattr(soa, "find_skill", lambda dr, n, _skill=skill: _skill)
        outcome = soa.review_skill_owner_attest(types.SimpleNamespace(drive_root=str(tmp_path)), skill.name)
        assert outcome.status == sr.STATUS_CLEAN, (source, outcome.error)
        assert outcome.review_profile == "owner_attested"
        assert (skill_state_dir(tmp_path, skill.name) / "owner_attestation.json").exists()


def test_owner_attest_still_refuses_a_failed_deterministic_floor(monkeypatch, tmp_path):
    """Widening the origins does not widen the floor: a failed preflight is not attestable."""
    import types
    import ouroboros.skill_owner_attestation as soa
    import ouroboros.skill_review as sr

    skill = _attestable_skill(tmp_path, name="hub-broken", source="ouroboroshub")
    monkeypatch.setattr(soa, "find_skill", lambda dr, n: skill)
    refused = sr.SkillReviewOutcome(skill_name=skill.name, status=sr.STATUS_PENDING, error="preflight failed")
    monkeypatch.setattr(sr, "_run_deterministic_preflight", lambda *a, **k: refused)
    outcome = soa.review_skill_owner_attest(types.SimpleNamespace(drive_root=str(tmp_path)), skill.name)
    assert outcome.status == sr.STATUS_PENDING
    assert not (skill_state_dir(tmp_path, skill.name) / "owner_attestation.json").exists()


def test_extensions_review_fields_offer_skip_for_every_readable_origin():
    """The listing hint offers Skip review for any origin with readable bytes and
    never for a broken payload; no hub catalog view is consulted any more."""
    import ouroboros.gateway.extensions as ext
    from ouroboros.skill_loader import SkillReviewState

    class _Skill:
        name = "hub"
        is_self_authored = False
        load_error = ""
        content_hash = "a" * 64
        review = SkillReviewState(status="pending", content_hash="")

    for source in ("native", "ouroboroshub", "clawhub", "external", "user_repo"):
        skill = _Skill()
        skill.source = source
        fields = ext._review_fields(skill)
        assert fields["owner_attestable"] is True, source
        assert "official_hub_verified" not in fields
    broken = _Skill()
    broken.source, broken.load_error, broken.content_hash = "clawhub", "payload unreadable", ""
    assert ext._review_fields(broken)["owner_attestable"] is False


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
