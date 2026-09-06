"""Behavioral tests for ``ouroboros.tools.shell._run_shell``.

Consolidated in v5.15.x from three previous files that all exercised the
same ``_run_shell`` entrypoint:

- ``test_shell_recovery.py``       — string/json/ast cmd recovery, malformed
                                    bracket refusal, env-ref policy, timeout
- ``test_shell_regex_hint.py``     — grep ``A\\|B`` argv-mode trap detection
                                    and auto-correct
- ``test_shell_no_match_semantics.py`` — grep/rg exit-1 without stderr is
                                         "no matches", not SHELL_EXIT_ERROR

The grep regex-hint matrix is collapsed into one parametrize table; the
recovery + env + timeout suite retains its scenarios (each tests a
distinct branch of the cascade).
"""
from __future__ import annotations

import pathlib
from subprocess import CompletedProcess
from types import SimpleNamespace

import pytest

from ouroboros.tools.shell import _resolve_effective_timeout, _run_script, _run_shell


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _ctx(tmp_path):
    """Minimal ctx used by all _run_shell tests."""
    import pathlib

    return SimpleNamespace(
        repo_dir=tmp_path,
        drive_root=tmp_path,
        drive_logs=lambda: pathlib.Path(str(tmp_path)),
    )


def test_run_shell_preserves_leading_stdout_whitespace(tmp_path, fake_subprocess):
    fake_subprocess(stdout="  indented\n")
    result = _run_shell(_ctx(tmp_path), ["printf", "x"])
    assert "STDOUT:\n  indented\n" in result


def test_run_shell_accepts_task_drive_label_as_cwd(tmp_path, fake_subprocess):
    repo = tmp_path / "repo"
    repo.mkdir()
    ctx = _ctx(repo)
    ctx.drive_root = tmp_path / "drive"
    ctx.task_id = "task1"
    calls = fake_subprocess(stdout="ok")
    result = _run_shell(ctx, ["pwd"], cwd="task_drive")
    assert "SHELL_CWD_BLOCKED" not in result
    assert pathlib.Path(calls[0]["kwargs"]["cwd"]).parts[-2:] == ("task_drives", "task1")


def test_run_shell_accepts_user_files_label_as_safe_deliverables_cwd(tmp_path, fake_subprocess, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    ctx = _ctx(repo)
    user_home = tmp_path / "user-home"
    deliverables = user_home / "Deliverables"
    ctx.drive_root = tmp_path / "drive"
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(user_home))
    monkeypatch.setenv("OUROBOROS_DELIVERABLES_ROOT", str(deliverables))
    calls = fake_subprocess(stdout="ok")
    result = _run_shell(ctx, ["pwd"], cwd="user_files")
    assert "SHELL_CWD_BLOCKED" not in result
    assert calls[0]["kwargs"]["cwd"] == str(deliverables.resolve())


@pytest.fixture
def fake_subprocess(monkeypatch):
    """Patch _tracked_subprocess_run with a closure that returns a queued result.

    Usage:
        def test_X(fake_subprocess):
            calls = fake_subprocess(stdout="ok", returncode=0)
            _run_shell(...)
            assert calls[0]["cmd"] == [...]
    """
    monkeypatch.setattr("ouroboros.tools.shell_process.load_settings", lambda: {})

    def _install(*, returncode: int = 0, stdout: str = "", stderr: str = ""):
        calls: list[dict] = []

        def fake_run(cmd, **kwargs):
            calls.append({"cmd": cmd, "kwargs": kwargs})
            return CompletedProcess(cmd, returncode, stdout, stderr)

        monkeypatch.setattr("ouroboros.tools.shell._tracked_subprocess_run", fake_run)
        return calls

    return _install


# ---------------------------------------------------------------------------
# T3 (v6.35.0): per-call timeout_sec override for run_command/run_script
# ---------------------------------------------------------------------------


class TestPerCallTimeout:
    """An explicit timeout_sec (or its `timeout` alias) overrides the default,
    still clamped to the remaining task deadline."""

    def _ctx_with_deadline(self, tmp_path, secs):
        import pathlib
        from datetime import datetime, timedelta, timezone

        deadline = (datetime.now(timezone.utc) + timedelta(seconds=secs)).isoformat()
        return SimpleNamespace(
            repo_dir=tmp_path,
            drive_logs=lambda: pathlib.Path(str(tmp_path)),
            task_metadata={"deadline_at": deadline},
        )

    def test_resolve_override_no_deadline_passthrough(self):
        assert _resolve_effective_timeout(360, None, override_sec=5) == 5

    def test_resolve_override_clamped_by_deadline(self, tmp_path):
        # remaining ~100s -> cap = max(60, min(1800, 50)) = 60 -> min(99999, 60)
        ctx = self._ctx_with_deadline(tmp_path, 100)
        assert _resolve_effective_timeout(360, ctx, override_sec=99999) == 60

    def test_resolve_override_zero_falls_through_to_default(self, monkeypatch):
        # override 0 -> falls through to the config SSOT default (OUROBOROS_TOOL_TIMEOUT_SEC=600),
        # NOT the in-code 360 (the prior `!= default_setting` skip wrongly returned 360).
        monkeypatch.setenv("OUROBOROS_TOOL_TIMEOUT_SEC", "600")
        assert _resolve_effective_timeout(360, None, override_sec=0) == 600

    def test_resolve_override_none_is_default(self, monkeypatch):
        monkeypatch.setenv("OUROBOROS_TOOL_TIMEOUT_SEC", "600")
        assert _resolve_effective_timeout(360, None, override_sec=None) == 600  # config SSOT, not in-code 360

    def test_run_shell_threads_timeout_sec(self, tmp_path, fake_subprocess):
        calls = fake_subprocess(stdout="ok")
        _run_shell(_ctx(tmp_path), ["echo", "hi"], timeout_sec=5)
        assert calls[0]["kwargs"]["timeout"] == 5

    def test_run_shell_accepts_timeout_alias(self, tmp_path, fake_subprocess):
        calls = fake_subprocess(stdout="ok")
        _run_shell(_ctx(tmp_path), ["echo", "hi"], timeout=7)
        assert calls[0]["kwargs"]["timeout"] == 7

    def test_run_shell_default_timeout_when_omitted(self, tmp_path, fake_subprocess, monkeypatch):
        monkeypatch.setenv("OUROBOROS_TOOL_TIMEOUT_SEC", "600")
        calls = fake_subprocess(stdout="ok")
        _run_shell(_ctx(tmp_path), ["echo", "hi"])
        assert calls[0]["kwargs"]["timeout"] == 600  # config SSOT default (was a buggy effective 360)

    def test_schema_exposes_timeout_sec_and_registry_owns_the_timeout_alias(self, tmp_path, fake_subprocess):
        """The alias moved from two duplicated per-tool schema rows to the one
        registry alias table: `timeout_sec` is the declared property, `timeout`
        is not, and the raw spelling still reaches the subprocess."""
        from ouroboros.tools.registry import ToolContext, ToolRegistry
        from ouroboros.tools.shell import get_tools

        entries = {e.name: e for e in get_tools()}
        for name in ("run_command", "run_script"):
            props = entries[name].schema["parameters"]["properties"]
            assert "timeout_sec" in props, f"{name} missing timeout_sec"
            assert "timeout" not in props, f"{name} still declares the alias as a property"

        calls = fake_subprocess(stdout="ok")
        registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
        registry.set_context(ToolContext(repo_dir=tmp_path, drive_root=tmp_path))
        result = registry.execute("run_command", {"cmd": ["echo", "hi"], "timeout": 7})
        assert "TOOL_ARG_ERROR" not in result, result
        assert calls[0]["kwargs"]["timeout"] == 7


# ---------------------------------------------------------------------------
# cmd recovery cascade (string → json → ast → shlex; bracket-prefix refusal)
# ---------------------------------------------------------------------------


class TestShellArgContract:
    """run_shell recovers string cmd via cascade, only errors on unrecoverable input."""

    def test_string_cmd_recovered_via_shlex(self, tmp_path, fake_subprocess):
        fake_subprocess(stdout="hello")
        result = _run_shell(_ctx(tmp_path), "echo hello")
        assert "SHELL_ARG_ERROR" not in result
        assert f"exit_code=0 (cwd={tmp_path.resolve()})" in result

    def test_json_array_string_recovered(self, tmp_path, fake_subprocess):
        fake_subprocess(stdout="ok")
        result = _run_shell(_ctx(tmp_path), '["echo", "hello"]')
        assert "SHELL_ARG_ERROR" not in result
        assert "exit_code=0" in result

    def test_python_literal_string_recovered(self, tmp_path, fake_subprocess):
        fake_subprocess(stdout="ok")
        result = _run_shell(_ctx(tmp_path), "['echo', 'hello']")
        assert "SHELL_ARG_ERROR" not in result
        assert "exit_code=0" in result

    def test_unrecoverable_string_returns_error(self, tmp_path):
        result = _run_shell(_ctx(tmp_path), "")
        assert "SHELL_ARG_ERROR" in result

    def test_string_cmd_still_discloses_env_refs(self, tmp_path, fake_subprocess):
        # #447 A5: literal shell syntax in direct argv is DATA — the command runs
        # and the literal pass-through is disclosed, not refused.
        fake_subprocess(stdout="ok")
        result = _run_shell(_ctx(tmp_path), 'curl -H "x-api-key: $SECRET"')
        assert "SHELL_LITERAL_ARGV_NOTE" in result
        assert "exit_code=0" in result

    # JSON-shape refusal — 2026-05-03 production bug. See module docstring
    # for the failure mode this guard prevents.

    def test_malformed_json_array_refused_not_shlex_split(self, tmp_path):
        result = _run_shell(_ctx(tmp_path), '["git", "log",')
        assert "SHELL_ARG_ERROR" in result
        assert "stringified array" in result.lower()
        assert "Errno" not in result

    def test_malformed_dict_literal_refused(self, tmp_path):
        result = _run_shell(_ctx(tmp_path), '{key: value, broken')
        assert "SHELL_ARG_ERROR" in result
        assert "Errno" not in result

    def test_valid_json_array_still_works_after_refusal_branch(self, tmp_path, fake_subprocess):
        fake_subprocess(stdout="ok")
        result = _run_shell(_ctx(tmp_path), '["echo", "ok"]')
        assert "SHELL_ARG_ERROR" not in result
        assert "exit_code=0" in result

    def test_legitimate_shell_string_still_recovers_via_shlex(self, tmp_path, fake_subprocess):
        fake_subprocess(stdout="hello")
        result = _run_shell(_ctx(tmp_path), "echo hello")
        assert "SHELL_ARG_ERROR" not in result
        assert "exit_code=0" in result

    def test_posix_bracket_test_command_still_recovers_via_shlex(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ouroboros.tools.shell_process.load_settings", lambda: {})

        def fake_run(cmd, **kwargs):
            assert cmd == ["[", "-f", "file.txt", "]"]
            return CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr("ouroboros.tools.shell._tracked_subprocess_run", fake_run)
        result = _run_shell(_ctx(tmp_path), "[ -f file.txt ]")
        assert "SHELL_ARG_ERROR" not in result
        assert "exit_code=0" in result

    def test_refusal_message_points_at_correct_usage(self, tmp_path):
        result = _run_shell(_ctx(tmp_path), '["git", "log",')
        assert 'run_command(cmd=["git"' in result

    def test_list_cmd_is_accepted(self, tmp_path, fake_subprocess):
        fake_subprocess(stdout="ok")
        result = _run_shell(_ctx(tmp_path), ["echo", "ok"])
        assert "SHELL_ARG_ERROR" not in result
        assert "exit_code=0" in result


# ---------------------------------------------------------------------------
# Env-ref + timeout + nonzero-exit behavior
# ---------------------------------------------------------------------------


def test_run_shell_discloses_literal_env_refs_in_argv(tmp_path, fake_subprocess):
    # #447 A5: the argv is executed as-is (no shell interprets "$..."), and the
    # unexpanded pass-through is disclosed in the result.
    calls = fake_subprocess(stdout="ok")
    result = _run_shell(_ctx(tmp_path), ["curl", "-H", "x-api-key: $ANTHROPIC_API_KEY"])
    assert "SHELL_LITERAL_ARGV_NOTE" in result
    assert "$ANTHROPIC_API_KEY" in result
    assert "exit_code=0" in result
    assert calls, "the command must actually run"


def test_run_shell_allows_shell_expansion_via_sh_c(tmp_path, fake_subprocess):
    fake_subprocess(stdout="ok")
    result = _run_shell(_ctx(tmp_path), ["sh", "-c", "printf '%s' \"$ANTHROPIC_API_KEY\""])
    assert "SHELL_LITERAL_ARGV_NOTE" not in result
    assert "exit_code=0" in result


def test_run_shell_nonzero_exit_is_reported_as_failure(tmp_path, fake_subprocess):
    fake_subprocess(returncode=3, stderr="permission denied")
    result = _run_shell(_ctx(tmp_path), ["npm", "install", "-g", "@anthropic-ai/claude-code"])

    assert result.startswith("⚠️ SHELL_EXIT_ERROR:")
    assert f"exit_code=3 (cwd={tmp_path.resolve()})" in result
    assert "permission denied" in result


def test_run_shell_timeout_uses_settings_timeout(tmp_path, monkeypatch):
    def fake_timeout(cmd, **kwargs):
        raise __import__("subprocess").TimeoutExpired(cmd=cmd, timeout=kwargs["timeout"])

    monkeypatch.setattr("ouroboros.tools.shell_process.load_settings", lambda: {"OUROBOROS_TOOL_TIMEOUT_SEC": 42})
    monkeypatch.delenv("OUROBOROS_TOOL_TIMEOUT_SEC", raising=False)
    monkeypatch.setattr("ouroboros.tools.shell._tracked_subprocess_run", fake_timeout)
    result = _run_shell(_ctx(tmp_path), ["sleep", "999"])

    assert "TOOL_TIMEOUT (run_command)" in result
    assert "42s" in result
    assert f"cwd={tmp_path.resolve()}" in result


def test_run_shell_deadline_derived_timeout_is_used_when_no_explicit_setting(monkeypatch):
    from datetime import datetime, timezone

    monkeypatch.setattr("ouroboros.tools.shell_process.load_settings", lambda: {"OUROBOROS_TOOL_TIMEOUT_SEC": 0})
    monkeypatch.delenv("OUROBOROS_TOOL_TIMEOUT_SEC", raising=False)
    monkeypatch.setattr("ouroboros.deadline_utils.utc_now", lambda: datetime(2026, 6, 10, 0, 0, tzinfo=timezone.utc))
    ctx = SimpleNamespace(task_metadata={"deadline_at": "2026-06-10T00:20:00Z"})

    assert _resolve_effective_timeout(360, ctx) == 600


def test_run_shell_deadline_caps_real_default_timeout(monkeypatch):
    from datetime import datetime, timezone

    monkeypatch.setattr("ouroboros.tools.shell_process.load_settings", lambda: {"OUROBOROS_TOOL_TIMEOUT_SEC": 600})
    monkeypatch.delenv("OUROBOROS_TOOL_TIMEOUT_SEC", raising=False)
    monkeypatch.setattr("ouroboros.deadline_utils.utc_now", lambda: datetime(2026, 6, 10, 0, 0, tzinfo=timezone.utc))
    ctx = SimpleNamespace(task_metadata={"deadline_at": "2026-06-10T00:10:00Z"})

    assert _resolve_effective_timeout(600, ctx) == 300


def test_run_shell_explicit_timeout_wins_over_deadline(monkeypatch):
    monkeypatch.setattr("ouroboros.tools.shell_process.load_settings", lambda: {"OUROBOROS_TOOL_TIMEOUT_SEC": 42})
    monkeypatch.delenv("OUROBOROS_TOOL_TIMEOUT_SEC", raising=False)
    ctx = SimpleNamespace(task_metadata={"deadline_at": "2026-06-10T00:20:00Z"})

    assert _resolve_effective_timeout(360, ctx) == 42


# ---------------------------------------------------------------------------
# grep/rg exit-1 without stderr semantics (no matches != shell error)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cmd", [
    ["grep", "-n", "missing", "file.py"],
    ["rg", "missing", "."],
])
def test_grep_or_rg_exit_one_without_stderr_is_no_match(cmd, tmp_path, fake_subprocess):
    fake_subprocess(returncode=1, stdout="", stderr="")
    result = _run_shell(_ctx(tmp_path), cmd)

    assert "SHELL_EXIT_ERROR" not in result
    assert "exit_code=1" in result
    assert f"cwd={tmp_path.resolve()}" in result
    assert "no matches" in result


def test_grep_exit_one_with_stderr_still_surfaces_shell_error(tmp_path, fake_subprocess):
    fake_subprocess(returncode=1, stderr="grep: file.py: No such file or directory\n")
    result = _run_shell(_ctx(tmp_path), ["grep", "missing", "file.py"])

    assert "SHELL_EXIT_ERROR" in result
    assert "No such file or directory" in result


# ---------------------------------------------------------------------------
# grep \| regex-escape hint / auto-correct (2026-05-04 hint class)
# ---------------------------------------------------------------------------


def test_user_file_output_audit_extracts_windows_absolute_paths():
    from ouroboros.tools import shell

    body = r"from pathlib import Path; Path('C:\\Users\\anton\\Desktop\\out.html').write_text('x')"
    redirect = r"echo x > C:\\Users\\anton\\Desktop\\out.html"

    assert shell._EMBEDDED_OUTPUT_PATH_RE.findall(body) == [r"C:\\Users\\anton\\Desktop\\out.html"]
    assert shell._USER_FILE_REDIRECT_RE.search(redirect).group("bare") == r"C:\\Users\\anton\\Desktop\\out.html"


class TestGrepRegexHint:
    """``grep "A\\|B" file`` in argv mode is BSD's literal two-char trap.

    The hint catches the class and rewrites to ``grep -E "A|B"`` so smaller
    models that learned bash idioms don't get stuck. Explicit -E/-G/-F flags,
    egrep/fgrep, and valid BRE patterns (``\\(...\\)``, ``\\+``) must pass
    through without the hint.
    """

    def test_grep_with_backslash_pipe_auto_corrects(self, tmp_path, fake_subprocess):
        calls = fake_subprocess(stdout="match\n")
        result = _run_shell(_ctx(tmp_path), ["grep", "-n", "A\\|B", "/tmp/x"])
        assert "SHELL_REGEX_AUTO_CORRECTED" in result
        assert "SHELL_REGEX_HINT" not in result
        assert calls[0]["cmd"] == ["grep", "-E", "-n", "A|B", "/tmp/x"]
        assert "match" in result

    def test_grep_with_path_basename_auto_corrected(self, tmp_path, fake_subprocess):
        calls = fake_subprocess()
        result = _run_shell(_ctx(tmp_path), ["/usr/bin/grep", "A\\|B", "/tmp/x"])
        assert "SHELL_REGEX_AUTO_CORRECTED" in result
        assert calls[0]["cmd"] == ["/usr/bin/grep", "-E", "A|B", "/tmp/x"]

    @pytest.mark.parametrize("argv,reason", [
        (["grep", "\\(foo\\)", "/tmp/x"], "POSIX BRE grouping, not the \\| trap"),
        (["grep", "ab\\+c", "/tmp/x"], "BRE extension, not the \\| trap"),
        (["grep", "-E", "A\\|B", "/tmp/x"], "explicit -E means user knows what they want"),
        (["grep", "-rnE", "A\\|B", "/tmp/x"], "clustered -rnE still explicit extended regex"),
        (["grep", "-G", "A\\|B", "/tmp/x"], "explicit -G is intentional GNU BRE"),
        (["grep", "-F", "A\\|B", "/tmp/x"], "-F means literal strings, two chars"),
        (["grep", "-n", "pattern", "/tmp/x"], "plain pattern without escapes"),
        (["egrep", "A\\|B", "/tmp/x"], "egrep already chose regex flavor"),
        (["fgrep", "A\\|B", "/tmp/x"], "fgrep already chose string flavor"),
        (["echo", "A\\|B"], "non-grep commands untouched"),
    ])
    def test_grep_regex_hint_skips(self, argv, reason, tmp_path, fake_subprocess):
        fake_subprocess()
        result = _run_shell(_ctx(tmp_path), argv)
        assert "SHELL_REGEX_HINT" not in result, reason


# ---------------------------------------------------------------------------
# SG-B (F17): a masked GREEN discloses the laundered exit code in the envelope
# ---------------------------------------------------------------------------


class TestMaskedGreenDisclosure:
    """`run_command` reads the same exit-masking sensor as `verify_and_record`
    and appends ONE advisory note to a green result. Result and trace only: no
    status, code, exit_code, gate, retry or receipt change — the reasons ride the
    published typed result's ``meta`` (``exit_masking_reasons``), read here the
    way the registry reads it: through the per-invocation result sidecar."""

    @staticmethod
    def _published(ctx, call):
        from ouroboros.tools.tool_result import (
            _install_tool_result_sidecar, _published_tool_result, _restore_tool_result_sidecar,
        )

        sentinel = object()
        token = _install_tool_result_sidecar(ctx, sentinel)
        try:
            text = call()
            published = _published_tool_result(ctx, sentinel)
        finally:
            _restore_tool_result_sidecar(token)
        assert published is not sentinel, "producer published no typed result"
        assert published.text == text
        return published

    @pytest.mark.parametrize("text, reason", [
        ("node t.js 2>&1 | tail -5", "pipeline_tail"),
        ("make test || true", "|| true"),
    ])
    def test_masked_green_run_command_carries_advisory_note(
        self, text, reason, tmp_path, fake_subprocess,
    ):
        fake_subprocess(stdout="ok", returncode=0)
        ctx = _ctx(tmp_path)
        published = self._published(ctx, lambda: _run_shell(ctx, ["sh", "-c", text]))
        assert "EXIT_MASKING_NOTE" in published.text
        assert reason in published.text
        assert published.status == "ok"
        assert published.meta["exit_code"] == 0
        assert reason in published.meta["exit_masking_reasons"]

    @pytest.mark.parametrize("cmd", [
        ["sh", "-c", "pytest -q"],
        ["sh", "-c", "true 2>/dev/null"],
        ["sh", "-c", "make test && true"],
        ["sh", "-c", "make test && exit 0"],
        ["go", "test", "./..."],
    ])
    def test_unmasked_green_carries_no_note(self, cmd, tmp_path, fake_subprocess):
        fake_subprocess(stdout="ok", returncode=0)
        ctx = _ctx(tmp_path)
        published = self._published(ctx, lambda: _run_shell(ctx, cmd))
        assert "EXIT_MASKING_NOTE" not in published.text
        assert "exit_masking_reasons" not in published.meta

    def test_masked_green_run_script_body_preserves_advisory_metadata(
        self, tmp_path, fake_subprocess,
    ):
        fake_subprocess(stdout="ok", returncode=0)
        ctx = _ctx(tmp_path)
        published = self._published(
            ctx, lambda: _run_script(ctx, "make test || true", interpreter="sh"),
        )
        assert "EXIT_MASKING_NOTE" in published.text
        assert "|| true" in published.text
        assert published.status == "ok"
        assert published.meta["exit_code"] == 0
        assert "|| true" in published.meta["exit_masking_reasons"]

    def test_masked_green_with_undeclared_output_still_discloses(self, tmp_path):
        """A green exit the shape laundered is disclosed even when the producer
        published a non-ok CODE for another reason (the undeclared-output nudge
        is a blocked code over an exit-0 process): the process fact decides."""
        from ouroboros.tools.shell_audit import _masked_green_disclosure
        from ouroboros.tools.tool_result import _publish_process_result

        ctx = _ctx(tmp_path)
        published = self._published(
            ctx,
            lambda: _masked_green_disclosure(
                ctx,
                _publish_process_result(ctx, "ARTIFACT_OUTPUT_UNDECLARED", "ok", exit_code=0),
                ["sh", "-c", "make test || true"],
            ),
        )
        assert published.code == "ARTIFACT_OUTPUT_UNDECLARED"
        assert published.status == "blocked"
        assert "EXIT_MASKING_NOTE" in published.text
        assert "|| true" in published.meta["exit_masking_reasons"]
        assert published.meta["exit_code"] == 0

    def test_masked_nonzero_exit_keeps_the_exit_error_envelope(self, tmp_path, fake_subprocess):
        fake_subprocess(stdout="", stderr="boom", returncode=1)
        ctx = _ctx(tmp_path)
        published = self._published(ctx, lambda: _run_shell(ctx, ["sh", "-c", "make test || true"]))
        assert "SHELL_EXIT_ERROR" in published.text
        assert "EXIT_MASKING_NOTE" not in published.text
        assert published.status != "ok"
        assert published.meta["exit_code"] == 1
        assert "exit_masking_reasons" not in published.meta

    def test_shell_and_verify_share_one_exit_masking_sensor(self):
        import ouroboros.tools.shell as shell_module
        import ouroboros.tools.verify as verify_module

        assert shell_module.check_exit_masking is verify_module._check_has_exit_masking
