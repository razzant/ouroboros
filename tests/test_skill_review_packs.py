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


def test_non_magic_binary_becomes_a_typed_descriptor(tmp_path, monkeypatch):
    """#447 X4/В21: judgment moved from FILENAME to CONTENT.

    A non-UTF-8 blob that carries no loader magic is no longer a hard block —
    raw bytes still never reach the reviewer, but the pack carries a typed
    {path,size,mime_from_name,sha256} descriptor so the reviewer can judge it
    on the merits instead of the whole skill going pending."""
    from ouroboros.skill_review import _read_skill_file

    skills_root = tmp_path / "skills"
    skill_dir = skills_root / "bin1"
    skill_dir.mkdir(parents=True)
    payload = b"\xff\xfeBEGIN CERT leak-me-please\xff\xc0\xc1\xfe\xff"
    (skill_dir / "cert.dat").write_bytes(payload)

    text, digest, descriptor = _read_skill_file(
        skill_dir / "cert.dat", relpath="cert.dat",
    )
    assert text is None and digest
    assert descriptor is not None
    assert descriptor["path"] == "cert.dat"
    assert descriptor["size"] == len(payload)
    assert len(descriptor["sha256"]) == 64


def test_skill_review_blocks_loadable_native_binaries(tmp_path):
    """Phase 3 round 13 regression, re-based on CONTENT (#447 X4): loadable
    native code must hard-block review whatever its NAME says. The subprocess
    could otherwise ``ctypes.CDLL`` / import / require the blob and execute
    never-reviewed code even under a PASS verdict."""
    from ouroboros.skill_review import _read_skill_file, _SkillBinaryPayload

    skills_root = tmp_path / "skills"
    skill_dir = skills_root / "nativelink"
    skill_dir.mkdir(parents=True)
    target = skill_dir / "evil.so"
    target.write_bytes(b"\x7fELF" + b"\x00" * 128)
    with pytest.raises(_SkillBinaryPayload):
        _read_skill_file(target, relpath="evil.so")

    # A RENAMED ELF is still blocked: the magic bytes decide, not the suffix.
    renamed = skill_dir / "notes.txt"
    renamed.write_bytes(b"\x7fELF" + b"\x00" * 128)
    with pytest.raises(_SkillBinaryPayload):
        _read_skill_file(renamed, relpath="notes.txt")


def test_skill_review_pack_carries_descriptor_not_raw_bytes(tmp_path):
    """The assembled review pack must contain the JSON descriptor block for a
    non-UTF-8 file — never its raw bytes — while text files stay inlined."""
    import hashlib as _hashlib

    from ouroboros.skill_review import _build_skill_file_packs

    skill_dir = tmp_path / "skills" / "mixed"
    skill_dir.mkdir(parents=True)
    (skill_dir / "skill.json").write_text('{"name": "mixed"}', encoding="utf-8")
    payload = b"\x80\x81\x82opaque-data\xff"
    (skill_dir / "blob.dat").write_bytes(payload)

    packs = _build_skill_file_packs(skill_dir)
    joined = "\n".join(packs)
    assert '{"name": "mixed"}' in joined
    assert "descriptor only" in joined
    assert _hashlib.sha256(payload).hexdigest() in joined
    assert "opaque-data" not in joined  # raw bytes never inlined


def test_skill_review_blocks_executables_by_magic_not_filename(tmp_path):
    """X4/В21: loadable executables are hard-blocked by CONTENT magic bytes
    (ELF/PE/Mach-O/.pyc) regardless of filename; a disguised extension
    does not evade the block."""
    import importlib.util

    from ouroboros.skill_review import _read_skill_file, _SkillBinaryPayload

    skill_dir = tmp_path / "skills" / "nativelink2"
    skill_dir.mkdir(parents=True)
    samples = {
        "innocent.txt": b"\x7fELF" + b"\x00" * 128,          # ELF, disguised name
        "tool.dat": b"MZ\x90\x00" + b"\xff" * 64,            # PE (non-UTF-8 body)
        "lib.data": b"\xcf\xfa\xed\xfe" + b"\x00" * 32,      # Mach-O 64-bit LE
        "cache.dat": importlib.util.MAGIC_NUMBER + b"\x00" * 32,  # .pyc
    }
    for name, payload in samples.items():
        target = skill_dir / name
        target.write_bytes(payload)
        with pytest.raises(_SkillBinaryPayload):
            _read_skill_file(target, relpath=name)


def test_skill_review_admits_wasm_as_content_hash_bound_descriptor(tmp_path):
    """Q15=A: WebAssembly left the loader-magic hard-block list. The host never
    loads it natively (it runs only inside the browser's sandboxed widget frame),
    so a ``.wasm`` file takes the ordinary non-UTF-8 path: the review pack carries
    a typed {path,size,mime_from_name,sha256} descriptor — the reviewer does not
    read the WebAssembly code — while the payload content hash binds every byte.
    Native loader magics (the renamed ELF above) still hard-block."""
    import hashlib as _hashlib

    from ouroboros.skill_loader import compute_content_hash
    from ouroboros.skill_review import _build_skill_file_packs, _read_skill_file
    from ouroboros.skill_review_passes import executable_magic_kind

    skill_dir = tmp_path / "skills" / "wasmpack"
    skill_dir.mkdir(parents=True)
    (skill_dir / "skill.json").write_text('{"name": "wasmpack"}', encoding="utf-8")
    wasm = b"\x00asm\x01\x00\x00\x00" + b"\x01\x85\x80\x80\x80\x00\xff\xfe" * 4
    (skill_dir / "core.wasm").write_bytes(wasm)

    assert executable_magic_kind(wasm, is_utf8_text=False) == ""
    text, digest, descriptor = _read_skill_file(skill_dir / "core.wasm", relpath="core.wasm")
    assert text is None and digest == _hashlib.sha256(wasm).digest()
    assert descriptor is not None and descriptor.pop("mime_from_name")
    assert descriptor == {"path": "core.wasm", "size": len(wasm), "sha256": _hashlib.sha256(wasm).hexdigest()}
    joined = "\n".join(_build_skill_file_packs(skill_dir))
    assert "core.wasm (binary file — descriptor only" in joined
    assert _hashlib.sha256(wasm).hexdigest() in joined
    # Content-hash-bound: one changed byte is a different payload (the stored
    # review goes stale), exactly like every other payload file.
    before = compute_content_hash(skill_dir)
    (skill_dir / "core.wasm").write_bytes(wasm[:-1] + b"\x00")
    assert compute_content_hash(skill_dir) != before


def test_skill_review_routes_utf8_decodable_wasm_to_descriptor(tmp_path):
    """W5-7/W5-11: the descriptor route is chosen by the WebAssembly magic, not by a
    failed UTF-8 decode. The canonical 8-byte module and a functional module made
    only of ASCII bytes both decode as UTF-8, yet neither may be inlined as text —
    "the reviewer does not read the WebAssembly code" must hold for every module.
    A native loader magic renamed ``.wasm`` still hard-blocks by content."""
    import hashlib as _hashlib

    from ouroboros.skill_review import _build_skill_file_packs, _read_skill_file, _SkillBinaryPayload
    from ouroboros.skill_review_passes import WASM_MAGIC

    skill_dir = tmp_path / "skills" / "wasmtext"
    skill_dir.mkdir(parents=True)
    (skill_dir / "skill.json").write_text('{"name": "wasmtext"}', encoding="utf-8")
    canonical = WASM_MAGIC + b"\x01\x00\x00\x00"
    # (func (export "f") (result i32) i32.const 42) — type, function, export and code
    # sections; every byte < 0x80 (validated with WebAssembly.validate, f() == 42).
    functional = (
        canonical
        + b"\x01\x05\x01\x60\x00\x01\x7f"
        + b"\x03\x02\x01\x00"
        + b"\x07\x05\x01\x01f\x00\x00"
        + b"\x0a\x06\x01\x04\x00\x41\x2a\x0b"
    )
    for name, module in (("empty.wasm", canonical), ("answer.wasm", functional)):
        module.decode("utf-8")  # the precondition the decode-first branch mistook for text
        (skill_dir / name).write_bytes(module)
        text, digest, descriptor = _read_skill_file(skill_dir / name, relpath=name)
        assert text is None and digest == _hashlib.sha256(module).digest(), name
        assert descriptor is not None and descriptor["sha256"] == _hashlib.sha256(module).hexdigest()
        assert descriptor["path"] == name and descriptor["size"] == len(module)
    joined = "\n".join(_build_skill_file_packs(skill_dir))
    assert "empty.wasm (binary file — descriptor only" in joined
    assert "answer.wasm (binary file — descriptor only" in joined
    assert "\x00asm" not in joined  # module bytes never inlined, not even as "text"
    (skill_dir / "core.wasm").write_bytes(b"\x7fELF" + b"\x00" * 32)  # ELF disguised as wasm
    with pytest.raises(_SkillBinaryPayload):
        _read_skill_file(skill_dir / "core.wasm", relpath="core.wasm")


def test_skill_review_does_not_block_text_by_scary_filename(tmp_path):
    """Capability preservation: the block is content-judged, so a valid UTF-8
    text file survives review even with a formerly-blocking extension or an
    ambiguous 'MZ' text prefix."""
    from ouroboros.skill_review import _read_skill_file

    skill_dir = tmp_path / "skills" / "textish"
    skill_dir.mkdir(parents=True)
    (skill_dir / "notes.so").write_text("just text, not an ELF", encoding="utf-8")
    text, _digest, descriptor = _read_skill_file(
        skill_dir / "notes.so", relpath="notes.so"
    )
    assert text == "just text, not an ELF"
    assert descriptor is None

    # 'MZ' is ambiguous (2 bytes): only non-UTF-8 content is judged as PE.
    (skill_dir / "mz.md").write_text("MZ stands for initials", encoding="utf-8")
    text, _digest, descriptor = _read_skill_file(skill_dir / "mz.md", relpath="mz.md")
    assert text == "MZ stands for initials"
    assert descriptor is None


def test_skill_review_session_prompt_allows_binary_inspection():
    """X4/В21 leg 3: the agent_session reviewer's retrieval assignment tells it
    that descriptor-only binary files may be inspected with its own tools."""
    from ouroboros.skill_review_passes import _SESSION_RETRIEVAL

    assert "{path,size,mime_from_name,sha256}" in _SESSION_RETRIEVAL
    assert "your own read/search tools" in _SESSION_RETRIEVAL
    assert "judge by the descriptor" in _SESSION_RETRIEVAL



def test_review_skill_fails_closed_on_unreadable_payload(tmp_path, monkeypatch):
    """Phase 3 round 18 regression: an unreadable payload file must
    fail review CLOSED (pending + error) instead of letting the
    placeholder slip past the gate. Regression for the old behaviour
    where ``_read_skill_file``'s predecessor returned a string on OSError and
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
    assert "loadable executable" in outcome.error.lower()
    assert "hard-blocks" in outcome.error.lower()


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

    def fake_review(ctx_, *, content, prompt, models, stable_prefix_len=0, **delivery):
        captured["prompt"] = prompt
        captured["stable_prefix_len"] = stable_prefix_len
        captured["delivery"] = delivery
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
    session_task = captured["delivery"]["session_task"]
    assert prompt[captured["stable_prefix_len"]:] in session_task
    assert "## Governance context — docs/ARCHITECTURE.md" not in session_task
    assert "docs/CHECKLISTS.md" in session_task and "docs/CREATING_SKILLS.md" in session_task
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
