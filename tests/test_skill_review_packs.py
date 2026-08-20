"""The review prompt and the payload packs behind it, and the payloads that are refused outright.

Split out of ``tests/test_skill_review.py`` by theme: the rebuttal, history and governance
artifacts the prompt loads, the quorum failure on a single responder, the malformed and
non-JSON reviewer output, the missing or unreadable skill, the native binaries blocked
before any reviewer sees them, the pack chunking under budget and the single file over it,
and the run that must not persist.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from ouroboros.skill_loader import (
    SkillReviewState,
    compute_content_hash,
    load_review_state,
    save_review_state,
)
from ouroboros.skill_review import review_skill

from tests._skill_review_shared import (
    _build_skill,
    _make_actor,
    _make_ctx,
    _pass_array_for_script_skill,
    _patch_review,
)


def test_review_skill_prompt_includes_rebuttal_and_history(tmp_path, monkeypatch):
    skills_root = _build_skill(tmp_path)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    ctx = _make_ctx(tmp_path)
    captured = {}
    pass_array = _pass_array_for_script_skill()
    canned = json.dumps({"results": [
        _make_actor("openai/gpt-5.5", pass_array),
        _make_actor("openai/gpt-5.5", pass_array),
    ]})

    def fake_review(_ctx, **kwargs):
        captured["prompt"] = kwargs["prompt"]
        return canned

    from ouroboros.skill_review import _append_skill_review_history
    _append_skill_review_history(
        ctx.drive_root,
        "weather",
        status="warnings",
        content_hash="old",
        findings=[{"item": "error_handling", "verdict": "FAIL", "severity": "advisory"}],
    )
    monkeypatch.setattr("ouroboros.tools.review._handle_multi_model_review", fake_review)

    outcome = review_skill(ctx, "weather", review_rebuttal="Already fixed in plugin.py.")

    assert outcome.status == "clean"
    assert "Developer's rebuttal" in captured["prompt"]
    assert "Already fixed in plugin.py." in captured["prompt"]
    assert "Previous skill review attempts" in captured["prompt"]


def test_review_skill_quorum_failure_on_one_responder(tmp_path, monkeypatch):
    import ouroboros.skill_review_prompt as skill_review_prompt

    skills_root = _build_skill(tmp_path)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    monkeypatch.setattr(
        "ouroboros.config.get_review_models",
        lambda: [
            "openai/gpt-5.5",
            "google/gemini-3.5-flash",
            "anthropic/claude-opus-4.6",
        ],
    )
    ctx = _make_ctx(tmp_path)
    advisory_evidence = {
        "status": "completed",
        "model": "claude-opus",
        "session_id": "sess-skill",
        "raw_result": "advisory raw",
    }
    # The advisory pre-review moved to the prompt owner with the per-attempt
    # assembly that calls it; patch it where that caller reads it.
    monkeypatch.setattr(
        skill_review_prompt,
        "_run_skill_advisory_pre_review",
        lambda *args, **kwargs: dict(advisory_evidence),
    )
    prior_hash = compute_content_hash(skills_root / "weather")
    save_review_state(
        ctx.drive_root,
        "weather",
        SkillReviewState(
            status="clean",
            content_hash=prior_hash,
            findings=_pass_array_for_script_skill(),
        ),
    )
    # Only one responder, two ERROR legs.
    canned = json.dumps(
        {
            "results": [
                _make_actor("openai/gpt-5.5", _pass_array_for_script_skill()),
                {
                    "model": "google/gemini-3.5-flash",
                    "request_model": "google/gemini-3.5-flash",
                    "verdict": "ERROR",
                    "text": "OpenRouter 404",
                    "tokens_in": 0, "tokens_out": 0,
                },
                {
                    "model": "anthropic/claude-opus-4.6",
                    "request_model": "anthropic/claude-opus-4.6",
                    "verdict": "ERROR",
                    "text": "OpenRouter 429",
                    "tokens_in": 0, "tokens_out": 0,
                },
            ]
        }
    )
    with _patch_review(canned):
        outcome = review_skill(ctx, "weather")
    assert outcome.status == "pending"
    assert "quorum" in outcome.error.lower()
    assert outcome.advisory_result == advisory_evidence
    persisted = load_review_state(ctx.drive_root, "weather")
    assert persisted.status == "clean"
    assert persisted.content_hash == prior_hash
    history = (ctx.drive_root / "state" / "skills" / "weather" / "review_history.jsonl").read_text(encoding="utf-8")
    assert '"raw_actor_records"' in history
    assert '"status": "error"' in history


def test_review_skill_error_on_non_json_top_level(tmp_path, monkeypatch):
    """A non-JSON top-level response from ``_handle_multi_model_review``
    must surface as status=pending with the error populated, not crash
    and not be mistaken for a successful review."""
    skills_root = _build_skill(tmp_path)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    ctx = _make_ctx(tmp_path)
    with _patch_review("not json"):
        outcome = review_skill(ctx, "weather")
    assert outcome.status == "pending"
    assert "non-JSON" in outcome.error


def test_review_skill_missing_skill_returns_pending_with_error(tmp_path, monkeypatch):
    skills_root = _build_skill(tmp_path)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    ctx = _make_ctx(tmp_path)
    outcome = review_skill(ctx, "does-not-exist")
    assert outcome.status == "pending"
    assert "not found" in outcome.error


def test_review_skill_malformed_reviewer_slots_block_before_any_reviewer(tmp_path, monkeypatch):
    """#116: a malformed OUROBOROS_REVIEWER_SLOTS keeps the skill honestly
    PENDING with the precise parse error — the reviewer wave is never
    dispatched on the silently projected default panel."""
    skills_root = _build_skill(tmp_path)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    monkeypatch.setenv("OUROBOROS_REVIEWER_SLOTS", "{broken")
    ctx = _make_ctx(tmp_path)

    with patch(
        "ouroboros.tools.review._handle_multi_model_review",
        side_effect=AssertionError("no reviewer dispatch on a malformed slot config"),
    ):
        outcome = review_skill(ctx, "weather")

    assert outcome.status == "pending"
    assert "invalid reviewer-slot configuration blocks skill review" in outcome.error
    assert "not valid JSON" in outcome.error


def test_skill_review_hard_blocks_extensionless_binary(tmp_path, monkeypatch):
    """Phase 3 round 15 regression: ANY non-UTF8 file in the runtime-
    reachable surface is a hard-block, not just extension-matched
    loadable formats. An extensionless disguised binary must still
    raise ``_SkillBinaryPayload`` so raw bytes never reach reviewer
    models and no PASS verdict ships over an opaque hash."""
    from ouroboros.skill_review import _read_skill_text, _SkillBinaryPayload

    skills_root = tmp_path / "skills"
    skill_dir = skills_root / "bin1"
    skill_dir.mkdir(parents=True)
    # Invalid UTF-8 bytes, no telltale extension (could be a Mach-O or
    # ELF blob disguised with a misleading ``.dat`` suffix).
    payload = b"\xff\xfeBEGIN CERT leak-me-please\xff\xc0\xc1\xfe\xff"
    (skill_dir / "cert.dat").write_bytes(payload)

    with pytest.raises(_SkillBinaryPayload):
        _read_skill_text(skill_dir / "cert.dat", relpath="cert.dat")


def test_skill_review_blocks_loadable_native_binaries(tmp_path):
    """Phase 3 round 13 regression: loadable native code
    (``.so``/``.dylib``/``.pyc``/``.node``/``.wasm``) must hard-block
    review. The subprocess could otherwise ``ctypes.CDLL`` / import /
    require the blob and execute never-reviewed code even under a
    PASS verdict."""
    from ouroboros.skill_review import _read_skill_text, _SkillBinaryPayload

    skills_root = tmp_path / "skills"
    skill_dir = skills_root / "nativelink"
    skill_dir.mkdir(parents=True)
    target = skill_dir / "evil.so"
    target.write_bytes(b"\x7fELF" + b"\x00" * 128)
    with pytest.raises(_SkillBinaryPayload):
        _read_skill_text(target, relpath="evil.so")


def test_review_skill_fails_closed_on_unreadable_payload(tmp_path, monkeypatch):
    """Phase 3 round 18 regression: an unreadable payload file must
    fail review CLOSED (pending + error) instead of letting the
    placeholder slip past the gate. Regression for the old behaviour
    where ``_read_skill_text`` returned a string on OSError and
    ``compute_content_hash`` silently skipped the file."""
    import os, platform
    if platform.system() == "Windows":
        pytest.skip("chmod-based permission test not portable to Windows")
    if os.geteuid() == 0:  # pragma: no cover
        pytest.skip("root user bypasses 0o000 chmod")
    skills_root = _build_skill(tmp_path)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    script = skills_root / "weather" / "scripts" / "fetch.py"
    original = script.stat().st_mode
    os.chmod(script, 0o000)
    try:
        ctx = _make_ctx(tmp_path)
        with patch(
            "ouroboros.tools.review._handle_multi_model_review",
            side_effect=AssertionError("must not call reviewer on unreadable payload"),
        ):
            outcome = review_skill(ctx, "weather")
    finally:
        os.chmod(script, original)
    assert outcome.status == "pending"
    assert "unreadable" in outcome.error.lower()


def test_review_skill_refuses_when_payload_contains_native_binary(tmp_path, monkeypatch):
    """End-to-end regression for loadable-binary block: ``review_skill``
    returns ``pending`` with an actionable error instead of persisting a
    verdict over a content hash that covers opaque machine code."""
    skills_root = tmp_path / "skills"
    skill_dir = skills_root / "nativepack"
    (skill_dir / "scripts").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: nativepack\ntype: script\nversion: 0.1.0\nruntime: python3\ntimeout_sec: 30\nscripts:\n  - name: main.py\n---\nbody\n",
        encoding="utf-8",
    )
    (skill_dir / "scripts" / "main.py").write_text("print('ok')\n", encoding="utf-8")
    (skill_dir / "libevil.dylib").write_bytes(b"\xca\xfe\xba\xbe" + b"\x00" * 64)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    ctx = _make_ctx(tmp_path)
    with patch(
        "ouroboros.tools.review._handle_multi_model_review",
        side_effect=AssertionError("must not call reviewer when native blob present"),
    ):
        outcome = review_skill(ctx, "nativepack")
    assert outcome.status == "pending"
    assert "binary" in outcome.error.lower()
    assert "opaque" in outcome.error.lower()


def test_skill_pack_includes_large_individual_file(tmp_path):
    """A large legitimate data file (e.g. references/destinations.json — the 76 KB
    file that used to hard-fail the per-file byte cap and lock the skill) is now
    bound by ONE pack-level token budget, so it is reviewed in FULL instead of
    dead-ending the skill at 'pending' (P5 token-budget gate)."""
    from ouroboros.skill_review import _build_skill_file_packs

    skill_dir = tmp_path / "whale"
    (skill_dir / "references").mkdir(parents=True)
    big = "x" * (80 * 1024)  # well over the old 64 KiB per-file byte cap
    (skill_dir / "references" / "destinations.json").write_text(big, encoding="utf-8")
    (skill_dir / "SKILL.md").write_text("# whale\n", encoding="utf-8")

    packs = _build_skill_file_packs(skill_dir)
    assert len(packs) == 1  # well under the 800K-token budget -> a single pass
    assert "references/destinations.json" in packs[0]
    assert big in packs[0]  # full content, never silently truncated


def test_skill_packs_chunks_when_over_budget(tmp_path, monkeypatch):
    """When the WHOLE skill payload exceeds the reviewer TOKEN budget, the files are
    split into multiple budget-sized packs (every byte reviewed in a separate pass),
    NOT refused — the P5 over-budget fallback. No silent truncation."""
    # The pack budget and its only reader moved together to the pack owner, so
    # the budget seam is patched where _build_skill_file_packs reads it.
    import ouroboros.skill_review_packs as sr
    from ouroboros.skill_review import _build_skill_file_packs

    skill_dir = tmp_path / "huge"
    skill_dir.mkdir()
    for i in range(6):
        (skill_dir / f"f_{i}.py").write_text("# pad line\n" * 30, encoding="utf-8")
    # Each file's block fits, but a few together exceed this tiny budget -> chunking.
    monkeypatch.setattr(sr, "_skill_pack_token_budget", lambda: 200)

    packs = _build_skill_file_packs(skill_dir)
    assert len(packs) > 1  # split into chunks, not refused
    combined = "\n\n".join(packs)
    for i in range(6):
        assert f"f_{i}.py" in combined  # every file reviewed across the chunks


def test_skill_packs_single_file_over_budget_refused(tmp_path, monkeypatch):
    """A SINGLE file that alone exceeds the budget cannot be chunked without truncating
    it, so review fails closed loudly (_SkillFileOverBudget) — never silent truncation."""
    # The pack budget and its only reader moved together to the pack owner, so
    # the budget seam is patched where _build_skill_file_packs reads it.
    import ouroboros.skill_review_packs as sr
    from ouroboros.skill_review import _SkillFileOverBudget, _build_skill_file_packs

    skill_dir = tmp_path / "mono"
    skill_dir.mkdir()
    (skill_dir / "mono.py").write_text("payload " * 4000, encoding="utf-8")
    monkeypatch.setattr(sr, "_skill_pack_token_budget", lambda: 10)

    with pytest.raises(_SkillFileOverBudget):
        _build_skill_file_packs(skill_dir)


def test_review_skill_prompt_loads_core_governance_artifacts(tmp_path, monkeypatch):
    """DEVELOPMENT.md 'When adding a new reasoning flow' rule requires
    ARCHITECTURE.md and DEVELOPMENT.md to appear in the assembled skill
    review prompt. Regression guard for Phase 3 round 6 finding."""
    skills_root = _build_skill(tmp_path)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    ctx = _make_ctx(tmp_path)

    captured = {}

    def fake_review(ctx_, *, content, prompt, models, stable_prefix_len=0):
        captured["prompt"] = prompt
        captured["stable_prefix_len"] = stable_prefix_len
        return json.dumps(
            {
                "results": [
                    _make_actor("openai/gpt-5.5", _pass_array_for_script_skill()),
                    _make_actor("google/gemini-3.5-flash", _pass_array_for_script_skill()),
                ]
            }
        )

    with patch("ouroboros.tools.review._handle_multi_model_review", side_effect=fake_review):
        review_skill(ctx, "weather")

    prompt = captured.get("prompt", "")
    assert prompt, "review_skill did not invoke _handle_multi_model_review"
    assert "docs/ARCHITECTURE.md" in prompt, (
        "skill review prompt must cite ARCHITECTURE.md as governance context"
    )
    assert "docs/DEVELOPMENT.md" in prompt, (
        "skill review prompt must cite DEVELOPMENT.md as governance context"
    )
    # Phase 3 round 10 regression: BIBLE.md must also be loaded so the
    # reviewer has constitutional tie-breaker context.
    assert "BIBLE.md" in prompt, (
        "skill review prompt must cite BIBLE.md for constitutional context"
    )
    # Minimal content-presence check: Section 10 key-invariants header is
    # referenced by label, and the actual body should appear (shipping
    # repo has the canonical text there).
    assert "Key Invariants" in prompt


def test_review_skill_persist_false_does_not_write(tmp_path, monkeypatch):
    skills_root = _build_skill(tmp_path)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    ctx = _make_ctx(tmp_path)
    pass_array = _pass_array_for_script_skill()
    canned = json.dumps(
        {
            "results": [
                _make_actor("openai/gpt-5.5", pass_array),
                _make_actor("google/gemini-3.5-flash", pass_array),
            ]
        }
    )
    with _patch_review(canned):
        outcome = review_skill(ctx, "weather", persist=False)
    assert outcome.status == "clean"
    persisted = load_review_state(ctx.drive_root, "weather")
    # Default state: nothing written.
    assert persisted.status == "pending"
    assert persisted.content_hash == ""
