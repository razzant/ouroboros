"""Phase 3 regression tests for the ``skill_exec`` tool itself: what it refuses, and what it runs.

This module owns the gating a call must pass — unconfigured host, disabled skill, non-pass
or stale review, an extension skill in phase 3, an instruction skill, a runtime outside the
allowlist, a path outside the declared scripts, a missing manifest permission grant — and
the actual subprocess execution behind it, including the lifecycle event, light mode,
runaway output, the wall-clock timeout, a nonzero exit, an unreadable payload and the
environment denylist.

The registry surface, the preflight, ``toggle_skill``, the heal context and the review-job
lifecycle were split verbatim into ``tests/test_skill_exec_registry_surface.py``,
``tests/test_skill_preflight.py``, ``tests/test_skill_toggle.py``,
``tests/test_skill_heal_context.py`` and ``tests/test_skill_review_lifecycle.py``; the
skill builders, context factory and review-state helpers they share live in
``tests/_skill_exec_shared.py``.
"""

from __future__ import annotations

import json
import shutil
from unittest.mock import patch

import pytest

from ouroboros.skill_loader import (
    SkillPayloadUnreadable,
    SkillReviewState,
    compute_content_hash,
    save_enabled,
    save_review_state,
)
from ouroboros.tools import skill_exec as skill_exec_mod

from tests._skill_exec_shared import (
    _build_skill,
    _make_ctx,
    _mark_reviewed_and_enabled,
    _valid_script_manifest,
)


def test_skill_exec_refuses_when_unconfigured(tmp_path, monkeypatch):
    monkeypatch.delenv("OUROBOROS_SKILLS_REPO_PATH", raising=False)
    ctx = _make_ctx(tmp_path)
    result = skill_exec_mod._handle_skill_exec(ctx, skill="x", script="y")
    assert "SKILLS_UNAVAILABLE" in result


def test_skill_exec_refuses_disabled_skill(tmp_path, monkeypatch):
    skills_root = tmp_path / "skills"
    skill_dir = _build_skill(skills_root, "hello")
    ctx = _make_ctx(tmp_path)
    # Only mark review PASS; leave enabled=False.
    content_hash = compute_content_hash(skill_dir)
    save_review_state(
        ctx.drive_root,
        "hello",
        SkillReviewState(status="pass", content_hash=content_hash),
    )
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    result = skill_exec_mod._handle_skill_exec(
        ctx, skill="hello", script="scripts/hello.py"
    )
    assert "SKILL_EXEC_BLOCKED" in result
    assert "disabled" in result


def test_skill_exec_refuses_non_pass_review(tmp_path, monkeypatch):
    skills_root = tmp_path / "skills"
    skill_dir = _build_skill(skills_root, "hello")
    ctx = _make_ctx(tmp_path)
    content_hash = compute_content_hash(skill_dir)
    save_enabled(ctx.drive_root, "hello", True)
    save_review_state(
        ctx.drive_root,
        "hello",
        SkillReviewState(status="blockers", content_hash=content_hash),
    )
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    result = skill_exec_mod._handle_skill_exec(
        ctx, skill="hello", script="scripts/hello.py"
    )
    assert "SKILL_EXEC_BLOCKED" in result
    assert "'blockers'" in result


def test_skill_exec_refuses_stale_review(tmp_path, monkeypatch):
    skills_root = tmp_path / "skills"
    skill_dir = _build_skill(skills_root, "hello")
    ctx = _make_ctx(tmp_path)
    save_enabled(ctx.drive_root, "hello", True)
    # Save review keyed to an old hash, then edit the script.
    save_review_state(
        ctx.drive_root,
        "hello",
        SkillReviewState(status="pass", content_hash="OLD_HASH"),
    )
    (skill_dir / "scripts" / "hello.py").write_text("print('edited')\n", encoding="utf-8")
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    result = skill_exec_mod._handle_skill_exec(
        ctx, skill="hello", script="scripts/hello.py"
    )
    assert "SKILL_EXEC_BLOCKED" in result
    assert "edited since the last review" in result


def test_skill_exec_refuses_extension_skill_in_phase3(tmp_path, monkeypatch):
    skills_root = tmp_path / "skills"
    manifest = (
        "---\n"
        "name: ext1\n"
        "version: 0.1.0\n"
        "type: extension\n"
        "entry: plugin.py\n"
        "permissions: [widget]\n"
        "---\n"
        "body\n"
    )
    skill_dir = _build_skill(skills_root, "ext1", manifest=manifest)
    (skill_dir / "plugin.py").write_text("def register(api): pass\n", encoding="utf-8")
    ctx = _make_ctx(tmp_path)
    _mark_reviewed_and_enabled(ctx.drive_root, skill_dir, "ext1")
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    result = skill_exec_mod._handle_skill_exec(
        ctx, skill="ext1", script="plugin.py"
    )
    # Phase 4: extension skills no longer return SKILL_EXEC_DEFERRED —
    # they return SKILL_EXEC_EXTENSION pointing the caller at the
    # in-process PluginAPI surface (Phase 5 wires the dispatchers).
    assert "SKILL_EXEC_EXTENSION" in result
    assert "extension_loader" in result


def test_skill_exec_rejects_absolute_and_parent_paths(tmp_path, monkeypatch):
    skills_root = tmp_path / "skills"
    skill_dir = _build_skill(skills_root, "hello")
    ctx = _make_ctx(tmp_path)
    _mark_reviewed_and_enabled(ctx.drive_root, skill_dir, "hello")
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")

    for bad in ("/etc/passwd", "~/.ssh/id_rsa", "../../etc/passwd", ""):
        result = skill_exec_mod._handle_skill_exec(
            ctx, skill="hello", script=bad
        )
        if bad == "":
            assert "SKILL_EXEC_ERROR" in result
        else:
            assert "SKILL_EXEC_ERROR" in result


def test_skill_exec_rejects_file_outside_declared_scripts(tmp_path, monkeypatch):
    """Regression (Phase 3 round 4): skill_exec's executable surface must
    equal the manifest-declared ``scripts:`` list, not the broader
    reviewed-content set. Assets, SKILL.md, or stray in-repo files must
    not be runnable even if they live in the reviewed skill directory."""
    skills_root = tmp_path / "skills"
    skill_dir = _build_skill(skills_root, "hello")
    # Drop a stray file directly in skill_dir (not declared in manifest).
    (skill_dir / "unreviewed.py").write_text("print('unreviewed')\n", encoding="utf-8")
    (skill_dir / "assets").mkdir()
    (skill_dir / "assets" / "data.py").write_text("print('asset-code')\n", encoding="utf-8")
    ctx = _make_ctx(tmp_path)
    _mark_reviewed_and_enabled(ctx.drive_root, skill_dir, "hello")
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")

    for bad in ("unreviewed.py", "assets/data.py", "SKILL.md"):
        result = skill_exec_mod._handle_skill_exec(
            ctx, skill="hello", script=bad
        )
        assert "SKILL_EXEC_ERROR" in result, f"bad={bad!r}: {result}"
        assert "not a declared script" in result, f"bad={bad!r}: {result}"


def test_skill_exec_refuses_instruction_type_skill(tmp_path, monkeypatch):
    """Phase 3 only executes ``type: script`` skills. An ``instruction``
    skill that went through review PASS must still be blocked at
    execution (its manifest declares no scripts anyway, but we want
    belt-and-braces type gating)."""
    skills_root = tmp_path / "skills"
    skill_dir = skills_root / "guide"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: guide\n"
        "description: Pure markdown guide.\n"
        "version: 0.1.0\n"
        "type: instruction\n"
        "---\n"
        "# body\nread me.\n",
        encoding="utf-8",
    )
    # Drop a file just to see if skill_exec tries to run it.
    (skill_dir / "scripts").mkdir()
    (skill_dir / "scripts" / "boom.py").write_text("print('boom')\n", encoding="utf-8")
    ctx = _make_ctx(tmp_path)
    _mark_reviewed_and_enabled(ctx.drive_root, skill_dir, "guide")
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")

    result = skill_exec_mod._handle_skill_exec(
        ctx, skill="guide", script="scripts/boom.py"
    )
    assert "SKILL_EXEC_ERROR" in result
    assert "'instruction'" in result, result


def test_skill_exec_rejects_runtime_outside_allowlist(tmp_path, monkeypatch):
    skills_root = tmp_path / "skills"
    skill_dir = _build_skill(
        skills_root,
        "hello",
        manifest=_valid_script_manifest("hello", runtime="perl"),
    )
    ctx = _make_ctx(tmp_path)
    _mark_reviewed_and_enabled(ctx.drive_root, skill_dir, "hello")
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    result = skill_exec_mod._handle_skill_exec(
        ctx, skill="hello", script="scripts/hello.py"
    )
    assert "SKILL_EXEC_ERROR" in result
    assert "allowlist" in result


@pytest.mark.skipif(shutil.which("python3") is None, reason="python3 not on PATH")
def test_skill_exec_runs_reviewed_skill_successfully(tmp_path, monkeypatch):
    skills_root = tmp_path / "skills"
    # Use a script that prints env + cwd so we can verify environment scrubbing.
    skill_dir = _build_skill(
        skills_root,
        "hello",
        script_body=(
            "import json, os, sys\n"
            # ``has_home`` must be True when either the Unix ``HOME`` or the
            # Windows ``USERPROFILE`` is forwarded — the scrub layer copies
            # both (see ``_ALWAYS_FORWARDED_ENV``); checking only ``HOME``
            # would spuriously fail on Windows CI where the parent process
            # exports ``USERPROFILE`` instead.
            "print(json.dumps({'cwd': os.getcwd(), 'skill': os.environ.get('OUROBOROS_SKILL_NAME'), "
            "'argv': sys.argv[1:], "
            "'has_home': ('HOME' in os.environ) or ('USERPROFILE' in os.environ), "
            "'openrouter_leaked': 'OPENROUTER_API_KEY' in os.environ}))\n"
        ),
    )
    ctx = _make_ctx(tmp_path)
    _mark_reviewed_and_enabled(ctx.drive_root, skill_dir, "hello")

    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    # Deliberately set a secret that the scrubbed env must NOT forward.
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-must-not-leak")

    raw = skill_exec_mod._handle_skill_exec(
        ctx, skill="hello", script="scripts/hello.py", args=["alpha", "beta"]
    )
    payload = json.loads(raw)
    assert payload["skill"] == "hello"
    assert payload["script"] == "scripts/hello.py"
    assert payload["exit_code"] == 0
    stdout_line = payload["stdout"].strip().splitlines()[-1]
    stdout = json.loads(stdout_line)
    # cwd must be inside the skill directory, not the main repo.
    assert stdout["cwd"].startswith(str(skill_dir))
    assert stdout["skill"] == "hello"
    assert stdout["argv"] == ["alpha", "beta"]
    assert stdout["has_home"] is True
    # Secret key must not leak into the subprocess environment.
    assert stdout["openrouter_leaked"] is False
    events = [
        json.loads(line)
        for line in (ctx.drive_root / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert events[-1]["type"] == "skill_exec_finished"
    assert events[-1]["skill"] == "hello"
    assert events[-1]["exit_code"] == 0


def test_skill_exec_queues_lifecycle_event_for_supervisor(tmp_path, monkeypatch):
    skills_root = tmp_path / "skills"
    skill_dir = _build_skill(
        skills_root,
        "hello",
        script_body="print('ok')\n",
    )
    ctx = _make_ctx(tmp_path)
    queued = []

    class _Queue:
        def put_nowait(self, item):
            queued.append(item)

    ctx.event_queue = _Queue()
    _mark_reviewed_and_enabled(ctx.drive_root, skill_dir, "hello")
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")

    raw = skill_exec_mod._handle_skill_exec(ctx, skill="hello", script="scripts/hello.py")

    assert json.loads(raw)["exit_code"] == 0
    assert queued[-1]["type"] == "skill_exec_finished"
    assert queued[-1]["skill"] == "hello"
    assert queued[-1]["exit_code"] == 0


def test_skill_exec_runs_in_light_mode(tmp_path, monkeypatch):
    """v5.1.2 Frame A: ``light`` allows reviewed + enabled skills to
    execute. The privilege scope ``light`` controls is repo
    self-modification and the runtime_mode elevation ratchet, NOT
    owner-approved skills (skills already pass tri-model review +
    enabled.json toggle + content-hash freshness + sandboxed
    subprocess). This is the positive replacement for the deleted
    Frame-B regression ``test_skill_exec_blocked_in_light_mode``.
    """
    skills_root = tmp_path / "skills"
    skill_dir = _build_skill(
        skills_root,
        "hello",
        script_body="import json; print(json.dumps({'ok': True}))\n",
    )
    ctx = _make_ctx(tmp_path)
    _mark_reviewed_and_enabled(ctx.drive_root, skill_dir, "hello")

    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")

    raw = skill_exec_mod._handle_skill_exec(
        ctx, skill="hello", script="scripts/hello.py"
    )
    # Must NOT be the v5.0.0 Frame-B sentinel.
    assert "SKILL_EXEC_BLOCKED" not in raw
    payload = json.loads(raw)
    assert payload["skill"] == "hello"
    assert payload["exit_code"] == 0
    stdout_line = payload["stdout"].strip().splitlines()[-1]
    assert json.loads(stdout_line) == {"ok": True}


def test_skill_exec_allows_warnings_under_blocking(tmp_path, monkeypatch):
    skills_root = tmp_path / "skills"
    skill_dir = _build_skill(skills_root, "alpha")
    ctx = _make_ctx(tmp_path)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    save_enabled(ctx.drive_root, "alpha", True)
    save_review_state(
        ctx.drive_root,
        "alpha",
        SkillReviewState(status="warnings", content_hash=compute_content_hash(skill_dir)),
    )

    resp = skill_exec_mod._handle_skill_exec(ctx, skill="alpha", script="hello.py")

    payload = json.loads(resp)
    assert payload["exit_code"] == 0
    assert "hello from skill" in payload["stdout"]


def test_skill_exec_refuses_missing_manifest_permission_grant(tmp_path, monkeypatch):
    skills_root = tmp_path / "skills"
    manifest = (
        "---\n"
        "name: alpha\n"
        "description: Permission grant test.\n"
        "version: 0.1.0\n"
        "type: script\n"
        "runtime: python3\n"
        "permissions: [inject_chat]\n"
        "scripts:\n"
        "  - name: hello.py\n"
        "    description: Print hello.\n"
        "---\n"
        "# body\n"
    )
    skill_dir = _build_skill(skills_root, "alpha", manifest=manifest)
    ctx = _make_ctx(tmp_path)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    _mark_reviewed_and_enabled(ctx.drive_root, skill_dir, "alpha")

    resp = skill_exec_mod._handle_skill_exec(ctx, skill="alpha", script="hello.py")

    assert "SKILL_EXEC_GRANT_REQUIRED" in resp
    assert "inject_chat" in resp


def test_skill_exec_rejects_misserialized_args(tmp_path, monkeypatch):
    """Phase 3 round 16 regression: args as a scalar/string must be
    rejected explicitly, not exploded per-character into argv."""
    skills_root = tmp_path / "skills"
    skill_dir = _build_skill(skills_root, "hello")
    ctx = _make_ctx(tmp_path)
    _mark_reviewed_and_enabled(ctx.drive_root, skill_dir, "hello")
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")

    for bogus in ("alpha", 1, 2.5, True, False, {"k": "v"}):
        result = skill_exec_mod._handle_skill_exec(
            ctx, skill="hello", script="scripts/hello.py", args=bogus
        )
        assert "SKILL_EXEC_ERROR" in result, f"args={bogus!r}: {result}"


def test_skill_exec_kills_runaway_stdout_output(tmp_path, monkeypatch):
    """Phase 3 round 17 regression: stdout/stderr byte caps must be
    enforced at STREAMING time, not post-hoc. A malicious skill that
    writes >>cap bytes must be killed and surface SKILL_EXEC_OVERFLOW
    instead of buffering into Ouroboros memory."""
    skills_root = tmp_path / "skills"
    # Write far more than _MAX_STDOUT_BYTES (64 KB) — 4 MiB forces a
    # streamer that only post-hoc caps to buffer the whole thing.
    body = (
        "import sys\n"
        "chunk = 'x' * 4096\n"
        "for _ in range(1024):\n"
        "    sys.stdout.write(chunk)\n"
        "    sys.stdout.flush()\n"
    )
    skill_dir = _build_skill(skills_root, "flood", script_body=body)
    ctx = _make_ctx(tmp_path)
    _mark_reviewed_and_enabled(ctx.drive_root, skill_dir, "flood")
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    result = skill_exec_mod._handle_skill_exec(
        ctx, skill="flood", script="scripts/hello.py"
    )
    assert "SKILL_EXEC_OVERFLOW" in result, result[:500]
    # Output in the returned payload must be bounded by the cap.
    import json as _json
    json_start = result.find("{")
    payload = _json.loads(result[json_start:])
    # Streamed stdout buffer must be close to the cap, not megabytes.
    assert len(payload["stdout"]) <= skill_exec_mod._MAX_STDOUT_BYTES + 1024
    assert payload["output_overflow"] is True


def test_skill_exec_surfaces_wall_clock_timeout(tmp_path, monkeypatch):
    """Phase 3 round 23 regression: wall-clock timeout surfaces as
    ``SKILL_EXEC_TIMEOUT`` with captured partial output instead of
    silently hanging."""
    skills_root = tmp_path / "skills"
    # Manifest declares 1-second timeout; script sleeps 10s.
    manifest = (
        "---\n"
        "name: sleepy\n"
        "description: Sleeps too long.\n"
        "version: 0.1.0\n"
        "type: script\n"
        "runtime: python3\n"
        "timeout_sec: 1\n"
        "scripts:\n"
        "  - name: hello.py\n"
        "---\n"
        "body\n"
    )
    skill_dir = _build_skill(
        skills_root,
        "sleepy",
        manifest=manifest,
        script_body=(
            "import sys, time\n"
            "sys.stdout.write('hi\\n'); sys.stdout.flush()\n"
            "time.sleep(10)\n"
        ),
    )
    ctx = _make_ctx(tmp_path)
    _mark_reviewed_and_enabled(ctx.drive_root, skill_dir, "sleepy")
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    result = skill_exec_mod._handle_skill_exec(
        ctx, skill="sleepy", script="scripts/hello.py"
    )
    assert "SKILL_EXEC_TIMEOUT" in result, result[:400]
    assert "1s limit" in result
    # Partial stdout captured before the kill.
    assert "hi" in result


def test_skill_exec_surfaces_nonzero_exit_as_failure(tmp_path, monkeypatch):
    """Phase 3 round 16 regression: a crashing skill script must be
    reported as a failed tool outcome (with SKILL_EXEC_FAILED sentinel),
    not a normal structured response the model might skim past."""
    skills_root = tmp_path / "skills"
    skill_dir = _build_skill(
        skills_root,
        "crashy",
        script_body="import sys\nprint('before crash')\nsys.exit(7)\n",
    )
    ctx = _make_ctx(tmp_path)
    _mark_reviewed_and_enabled(ctx.drive_root, skill_dir, "crashy")
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")

    result = skill_exec_mod._handle_skill_exec(
        ctx, skill="crashy", script="scripts/hello.py"
    )
    assert "SKILL_EXEC_FAILED" in result
    assert "exit_code" in result
    assert "7" in result
    events = [
        json.loads(line)
        for line in (ctx.drive_root / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert events[-1]["type"] == "skill_exec_failed"
    assert events[-1]["skill"] == "crashy"
    assert events[-1]["exit_code"] == 7


def test_skill_exec_returns_controlled_error_when_payload_becomes_unreadable(
    tmp_path, monkeypatch
):
    skills_root = tmp_path / "skills"
    skill_dir = _build_skill(skills_root, "alpha")
    ctx = _make_ctx(tmp_path)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    save_enabled(ctx.drive_root, "alpha", True)
    save_review_state(
        ctx.drive_root,
        "alpha",
        SkillReviewState(
            status="pass",
            content_hash=compute_content_hash(skill_dir, manifest_entry="", manifest_scripts=[{"name": "run.py"}]),
        ),
    )
    with patch.object(
        skill_exec_mod,
        "compute_content_hash",
        side_effect=SkillPayloadUnreadable(
            "blocked.txt",
            PermissionError("permission denied"),
        ),
    ):
        result = skill_exec_mod._handle_skill_exec(ctx, skill="alpha", script="run.py")
    assert "SKILL_EXEC_ERROR" in result
    assert "payload became unreadable" in result


def test_skill_exec_bare_name_resolves_only_to_scripts_dir(tmp_path, monkeypatch):
    """Phase 3 round 8 regression: a bare manifest name (``hello.py``)
    must resolve ONLY to ``scripts/hello.py`` — never to a top-level
    shadow file of the same name. Otherwise a skill author could drop a
    hostile ``hello.py`` next to the real ``scripts/hello.py`` and
    skill_exec would pick the top-level one."""
    skills_root = tmp_path / "skills"
    skill_dir = _build_skill(
        skills_root,
        "hello",
        script_body="print('FROM_SCRIPTS_DIR')\n",
    )
    # Drop a shadow file at the top level — this must NOT run.
    (skill_dir / "hello.py").write_text("print('FROM_SHADOW_TOPLEVEL')\n", encoding="utf-8")
    ctx = _make_ctx(tmp_path)
    _mark_reviewed_and_enabled(ctx.drive_root, skill_dir, "hello")
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")

    raw = skill_exec_mod._handle_skill_exec(
        ctx, skill="hello", script="hello.py"
    )
    # Must succeed and run the scripts/hello.py file, not the shadow.
    import json as _json
    payload = _json.loads(raw)
    assert "FROM_SCRIPTS_DIR" in payload["stdout"]
    assert "FROM_SHADOW_TOPLEVEL" not in payload["stdout"]


def test_env_denylist_blocks_secret_forwarding(tmp_path, monkeypatch):
    """Core settings keys are withheld unless a content-bound grant exists.

    Patch target note (round 21 fix): ``skill_exec.py`` imports
    ``load_settings`` from ``ouroboros.config`` as a bound alias via
    ``from ouroboros.config import … load_settings``. Monkeypatching
    the original in ``ouroboros.config`` leaves the alias unaffected —
    we patch the alias on ``ouroboros.tools.skill_exec`` directly so
    the code under test actually sees the mocked payload."""
    from ouroboros.tools import skill_exec as se

    skill_state_dir_path = tmp_path / "state" / "skills" / "ok"
    skill_state_dir_path.mkdir(parents=True, exist_ok=True)

    with patch.object(
        se,
        "load_settings",
        return_value={
            "OPENROUTER_API_KEY": "sk-or-v1-LEAK-ME",
            "OUROBOROS_NETWORK_PASSWORD": "deadbeef",
            "GITHUB_TOKEN": "ghp_leak",
            "TIMEZONE": "UTC",
            "SOME_OK_KEY": "visible-value",
        },
    ):
        env = se._scrub_env(
            manifest_env_keys=[
                "OPENROUTER_API_KEY",
                "GITHUB_TOKEN",
                "OUROBOROS_NETWORK_PASSWORD",
                "SOME_OK_KEY",
            ],
            skill_state_dir_path=skill_state_dir_path,
            skill_name="ok",
        )
    # Core keys are dropped when no explicit owner grant exists.
    assert "OPENROUTER_API_KEY" not in env, (
        "Runtime must refuse to forward the OpenRouter key without a grant."
    )
    assert "GITHUB_TOKEN" not in env
    assert "OUROBOROS_NETWORK_PASSWORD" not in env
    # Non-protected manifest-requested keys still flow without grants; custom
    # secrets become grant-bound after the owner stores them in Settings.
    assert env["SOME_OK_KEY"] == "visible-value"

    with patch.object(se, "load_settings", return_value={"OPENROUTER_API_KEY": "sk-or-v1-GRANTED"}):
        granted_env = se._scrub_env(
            manifest_env_keys=["OPENROUTER_API_KEY"],
            skill_state_dir_path=skill_state_dir_path,
            skill_name="ok",
            granted_keys=["OPENROUTER_API_KEY"],
        )
    assert granted_env["OPENROUTER_API_KEY"] == "sk-or-v1-GRANTED"

    with patch.object(se, "load_settings", return_value={"SOME_OK_KEY": "visible-value"}):
        custom_env = se._scrub_env(
            manifest_env_keys=["SOME_OK_KEY"],
            skill_state_dir_path=skill_state_dir_path,
            skill_name="ok",
            granted_keys=["SOME_OK_KEY"],
        )
    assert custom_env["SOME_OK_KEY"] == "visible-value"


def test_skill_exec_uses_shared_settings_denylist():
    from ouroboros.contracts.plugin_api import FORBIDDEN_SKILL_SETTINGS

    assert skill_exec_mod._FORBIDDEN_ENV_FORWARD_KEYS == FORBIDDEN_SKILL_SETTINGS
