"""Focused tests for the v6.54.3 runtime-reliability commit (Commit 1).

Covers: the file-API root-label hybrid (1.1), the safety parse-fix + owner-only
OUROBOROS_SAFETY_MODE guard set (1.2), the light-mode read-vs-write mention-scan
refinement + actionable path messages + attachment abs_path (1.3), the deadline
package (1.4), plan_task deadline scaling (1.5), and schedule_subagent slot
visibility (1.6).
"""

from __future__ import annotations

import json
import pathlib
import queue
from types import SimpleNamespace

import pytest

from ouroboros import config as config_mod
from ouroboros import safety as safety_mod
from ouroboros.tools.registry import ToolContext


# ---------------------------------------------------------------------------
# helpers


def _ctx(tmp_path: pathlib.Path, *, task_id: str = "t-v655", meta: dict | None = None) -> ToolContext:
    system = tmp_path / "system"
    data = tmp_path / "data"
    for p in (system, data):
        p.mkdir(exist_ok=True)
    return ToolContext(
        repo_dir=system,
        drive_root=data,
        task_id=task_id,
        task_metadata=dict(meta or {}),
        event_queue=queue.Queue(),
    )


# ---------------------------------------------------------------------------
# 1.2 — safety mode config: normalize / ratchet / owner-only merge-skip


def test_normalize_safety_mode_clamps_to_enum():
    assert config_mod.normalize_safety_mode(" LIGHT ") == "light"
    assert config_mod.normalize_safety_mode("off") == "off"
    assert config_mod.normalize_safety_mode("junk") == "full"
    assert config_mod.normalize_safety_mode(None) == "full"


def test_guard_safety_mode_lowering_refuses_downward_steps(tmp_path, monkeypatch):
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps({"OUROBOROS_SAFETY_MODE": "full"}), encoding="utf-8")
    monkeypatch.setattr(config_mod, "SETTINGS_PATH", settings_path)
    with pytest.raises(PermissionError):
        config_mod._guard_safety_mode_lowering({"OUROBOROS_SAFETY_MODE": "light"})
    with pytest.raises(PermissionError):
        config_mod._guard_safety_mode_lowering({"OUROBOROS_SAFETY_MODE": "off"})
    # Raising coverage and the explicit owner path are both permitted.
    config_mod._guard_safety_mode_lowering({"OUROBOROS_SAFETY_MODE": "full"})
    config_mod._guard_safety_mode_lowering(
        {"OUROBOROS_SAFETY_MODE": "off"}, allow_safety_lowering=True
    )
    settings_path.write_text(json.dumps({"OUROBOROS_SAFETY_MODE": "light"}), encoding="utf-8")
    config_mod._guard_safety_mode_lowering({"OUROBOROS_SAFETY_MODE": "full"})
    with pytest.raises(PermissionError):
        config_mod._guard_safety_mode_lowering({"OUROBOROS_SAFETY_MODE": "off"})


def test_generic_settings_merge_drops_safety_mode():
    from ouroboros.gateway.settings import _merge_settings_payload

    current = {"OUROBOROS_SAFETY_MODE": "full"}
    merged = _merge_settings_payload(current, {"OUROBOROS_SAFETY_MODE": "off"})
    assert merged.get("OUROBOROS_SAFETY_MODE") == "full"


def test_owner_safety_mode_endpoint_is_registered():
    from ouroboros.gateway.contracts import HTTP_ENDPOINTS

    assert "POST /api/owner/safety-mode" in HTTP_ENDPOINTS


# ---------------------------------------------------------------------------
# 1.2 — safety gate behavior per mode


def _run_gate(monkeypatch, tmp_path, *, mode: str, tool: str):
    monkeypatch.setenv("OUROBOROS_SAFETY_MODE", mode)
    calls: list[str] = []

    def _fake_llm_check(tool_name, arguments, messages, ctx, policy=None):
        calls.append(tool_name)
        return True, ""

    monkeypatch.setattr(safety_mod, "_run_llm_check", _fake_llm_check)
    ctx = _ctx(tmp_path)
    # A non-whitelisted command so CONDITIONAL does not take the deterministic
    # safe-subject bypass on its own.
    allowed, msg = safety_mod.check_safety(
        tool, {"cmd": "curl https://rcsb.org/x --output out.bin"}, ctx=ctx
    )
    return allowed, msg, calls, ctx


def test_safety_mode_light_skips_conditional_with_audit(tmp_path, monkeypatch):
    allowed, msg, calls, ctx = _run_gate(monkeypatch, tmp_path, mode="light", tool="run_command")
    assert allowed is True and msg == ""
    assert calls == []  # no LLM call
    # Durable-first (review round 9): the audit lands in events.jsonl at the
    # moment of the decision, not in a queue a worker death could lose.
    events_path = ctx.drive_logs() / "events.jsonl"
    rows = [json.loads(line) for line in events_path.read_text().splitlines()]
    skip_events = [e for e in rows if e.get("type") == "safety_mode_skip"]
    assert skip_events and skip_events[0]["safety_mode"] == "light"


def test_safety_mode_light_safe_subject_emits_no_skip_audit(tmp_path, monkeypatch):
    """Adversarial r1 #19: a whitelist-safe subject is allowed without any LLM
    call in FULL mode too, so light/off must not log a waved-through audit row
    for it — the skip event is reserved for real deltas vs full coverage."""
    monkeypatch.setenv("OUROBOROS_SAFETY_MODE", "light")
    calls: list[str] = []

    def _fake_llm_check(tool_name, arguments, messages, ctx, policy=None):
        calls.append(tool_name)
        return True, ""

    monkeypatch.setattr(safety_mod, "_run_llm_check", _fake_llm_check)
    ctx = _ctx(tmp_path)
    allowed, msg = safety_mod.check_safety(ctx=ctx, tool_name="run_command", arguments={"cmd": "ls -la"})
    assert allowed is True and msg == ""
    assert calls == []
    events_path = ctx.drive_logs() / "events.jsonl"
    rows = (
        [json.loads(line) for line in events_path.read_text().splitlines()]
        if events_path.exists()
        else []
    )
    assert not [e for e in rows if e.get("type") == "safety_mode_skip"]


def test_safety_mode_light_keeps_llm_for_policy_check_tools(tmp_path, monkeypatch):
    allowed, _msg, calls, _ctx2 = _run_gate(monkeypatch, tmp_path, mode="light", tool="skill_exec")
    assert allowed is True
    assert calls == ["skill_exec"]  # integration tools stay LLM-checked in light


def test_safety_mode_off_skips_all_llm_checks(tmp_path, monkeypatch):
    for tool in ("run_command", "skill_exec"):
        allowed, msg, calls, _c = _run_gate(monkeypatch, tmp_path, mode="off", tool=tool)
        assert allowed is True and msg == ""
        assert calls == []


def test_safety_mode_full_keeps_llm_path(tmp_path, monkeypatch):
    _allowed, _msg, calls, _c = _run_gate(monkeypatch, tmp_path, mode="full", tool="run_command")
    assert calls == ["run_command"]


# ---------------------------------------------------------------------------
# 1.2 — parse-failure classification


def test_classify_safety_parse_failure_classes():
    fn = safety_mod._classify_safety_parse_failure
    assert fn({"content": ""}, {"completion_tokens": 0}) == "empty"
    assert fn({"content": None}, None) == "empty"
    max_tokens = config_mod.get_safety_max_tokens()
    assert fn({"content": '{"status": "SA'}, {"completion_tokens": max_tokens}) == "truncated"
    assert fn({"content": "prose without json"}, {"completion_tokens": 10}) == "unparseable"


# ---------------------------------------------------------------------------
# 1.2 — self-lowering detectors (shell + browser JS)


def test_registry_detects_safety_mode_self_lowering():
    from ouroboros.tools.registry import _detect_safety_mode_self_lowering as det

    assert det("curl -x post http://127.0.0.1:8765/api/owner/safety-mode -d off")
    assert det("python -c \"...ouroboros_safety_mode...\" >> settings.json".lower())
    # Percent-encoded endpoint must not slip the scan (review round 6).
    assert det("curl -x post http://127.0.0.1:8765/api/owner/safety%2dmode -d off")
    assert not det("echo safety first")
    assert not det("grep ouroboros_safety_mode docs/architecture.md")


def test_browser_js_guard_blocks_safety_mode_change():
    from ouroboros.tools.browser import _blocks_safety_mode_self_lowering_js as js

    assert js("fetch('/api/owner/safety-mode', {method: 'POST'})")
    assert js("body: JSON.stringify({OUROBOROS_SAFETY_MODE: 'off'}) /api/settings")
    assert js("fetch('/api/owner/safety%2Dmode', {method: 'POST'})")
    assert not js("console.log('safety-mode docs')")


def test_safety_mode_owner_post_route_decodes_percent_encoding():
    from ouroboros.tools.browser import _is_safety_mode_owner_post

    req = SimpleNamespace(url="http://127.0.0.1:8765/api/owner/safety%2Dmode", method="POST")
    assert _is_safety_mode_owner_post(req) is True
    req2 = SimpleNamespace(url="http://127.0.0.1:8765/api/owner/auto-grant", method="POST")
    assert _is_safety_mode_owner_post(req2) is False


# ---------------------------------------------------------------------------
# 1.2 — response_format is droppable request intent in LLMClient


def test_response_format_in_droppable_params():
    from ouroboros.llm import _OPTIONAL_DROPPABLE_PARAMS, LLMClient

    assert "response_format" in _OPTIONAL_DROPPABLE_PARAMS
    assert "reasoning_effort" in _OPTIONAL_DROPPABLE_PARAMS  # round 6: rejected effort hint strips, not fails
    exc = RuntimeError("Error code: 400 - response_format is an unsupported parameter")
    assert LLMClient._parameter_rejection_error(exc) is True
    payload = {"model": "m", "messages": [], "response_format": {"type": "json_object"}}
    retry = LLMClient._retry_without_optional_sampling(payload, "prov/model-rf-test", exc)
    assert retry is not None and "response_format" not in retry
    # Remembered rejection strips it from the next payload for the same model.
    fresh = {"model": "m", "messages": [], "response_format": {"type": "json_object"}}
    LLMClient._apply_rejected_param_cache(fresh, "prov/model-rf-test")
    assert "response_format" not in fresh


def test_chat_signature_accepts_response_format():
    import inspect

    from ouroboros.llm import LLMClient

    assert "response_format" in inspect.signature(LLMClient.chat).parameters


# ---------------------------------------------------------------------------
# 1.3 — read-vs-write mention-scan refinement


def _guard(tmp_path, raw_cmd, *, writeish):
    from ouroboros.tools.shell_guards import runtime_data_guard_targets

    drive = tmp_path / "data"
    allowed = drive / "task_results" / "artifacts" / "t-v655"
    scratch = drive / "task_drives" / "t-v655"
    for p in (allowed, scratch):
        p.mkdir(parents=True, exist_ok=True)
    return runtime_data_guard_targets(
        raw_cmd,
        writeish=writeish,
        drive_root=drive,
        work_dir=scratch,
        allowed_roots=[scratch, allowed],
    )


def test_pure_read_python_mention_is_not_blocked(tmp_path):
    outside = tmp_path / "data" / "artifacts" / "attachments" / "clip.mp3"
    cmd = f"python3 -c \"import librosa; y, sr = librosa.load('{outside}')\""
    assert _guard(tmp_path, cmd, writeish=False) == []


def test_writeish_command_mentioning_outside_path_still_blocks(tmp_path):
    outside = tmp_path / "data" / "logs" / "events.jsonl"
    cmd = f"cp {outside} /tmp/x"
    blocked = _guard(tmp_path, cmd, writeish=True)
    assert blocked and str(outside) in blocked[0]


def test_python_literal_write_outside_blocks_but_inside_allowed(tmp_path):
    outside = tmp_path / "data" / "memory" / "notes.txt"
    inside = tmp_path / "data" / "task_results" / "artifacts" / "t-v655" / "out.txt"
    blocked = _guard(
        tmp_path,
        f"python3 -c \"open('{outside}', 'w').write('x')\"",
        writeish=False,
    )
    assert blocked and str(outside) in blocked[0]
    assert _guard(
        tmp_path,
        f"python3 -c \"open('{inside}', 'w').write('x')\"",
        writeish=False,
    ) == []


def test_open_based_pure_read_not_blocked_despite_coarse_writeish(tmp_path):
    """Review round 8: the coarse SHELL_WRITE_INDICATORS token `open(` marks a
    read-only open() as writeish — the guard must re-judge interpreter commands
    and let the pure read through (the original GAIA class)."""
    outside = tmp_path / "data" / "logs" / "events.jsonl"
    cmd = f"python3 -c \"print(open('{outside}').read())\""
    # Registry computes writeish=True for this command (the `open(` token).
    assert _guard(tmp_path, cmd, writeish=True) == []


def test_interpreter_shell_redirect_still_full_scans(tmp_path):
    """A SHELL-level write signal (redirect) on an interpreter command keeps the
    conservative full mention scan."""
    outside = tmp_path / "data" / "memory" / "notes.txt"
    cmd = f"python3 -c \"print('x')\" > {outside}"
    blocked = _guard(tmp_path, cmd, writeish=True)
    assert blocked and str(outside) in blocked[0]


def test_pure_read_of_secret_named_runtime_file_still_blocks(tmp_path):
    """Review round 2: the read-vs-write relaxation never opens secret/control
    files — a pure-read python mention of settings.json under the drive blocks."""
    secret = tmp_path / "data" / "settings.json"
    cmd = f"python3 -c \"print(open('{secret}').read())\""
    blocked = _guard(tmp_path, cmd, writeish=False)
    assert blocked and str(secret) in blocked[0]


def test_pure_read_of_cross_project_store_still_blocks(tmp_path):
    """Adversarial r1 #1: the read relaxation must mirror read_file(root=runtime_data)
    — a pure-read interpreter mention of another project's facts store
    (projects/<id>/...) blocks, matching the file API (no cross-project peeking)."""
    other = tmp_path / "data" / "projects" / "OTHER" / "knowledge" / "facts.md"
    cmd = f"python3 -c \"print(open('{other}').read())\""
    blocked = _guard(tmp_path, cmd, writeish=False)
    assert blocked and str(other) in blocked[0]


def test_rmdir_and_oslink_do_not_slip_the_write_guard(tmp_path):
    """Adversarial r1 #2: os.rmdir/Path.rmdir()/os.link on a runtime_data path
    outside the task roots must be caught by the write guard — the regex prefilter
    previously omitted rmdir(/os.link( so they were treated as pure reads."""
    outside_dir = tmp_path / "data" / "projects" / "OTHER" / "knowledge"
    for cmd in (
        f"python3 -c \"import os; os.rmdir('{outside_dir}')\"",
        f"python3 -c \"import pathlib; pathlib.Path('{outside_dir}').rmdir()\"",
        f"python3 -c \"import os; os.link('/etc/hosts', '{tmp_path / 'data' / 'memory' / 'hl'}')\"",
    ):
        blocked = _guard(tmp_path, cmd, writeish=False)
        assert blocked, f"not blocked: {cmd}"


def test_any_write_regex_matches_rmdir_and_oslink():
    from ouroboros.tools.shell_guards import _INTERPRETER_ANY_WRITE_RE

    assert _INTERPRETER_ANY_WRITE_RE.search("os.rmdir('/x')")
    assert _INTERPRETER_ANY_WRITE_RE.search("os.link('/a','/b')")


def test_opaque_write_primitives_do_not_slip_the_write_guard(tmp_path):
    """Adversarial r2 #1: opaque/unmodeled write-capable calls the write regex
    did not name — subprocess exec (rm/mv/dd), tarfile/zipfile extractall,
    shutil.unpack_archive, sqlite3.connect — must NOT be treated as pure reads.
    A write to a runtime_data path outside the task roots blocks (base behavior),
    while the AST models none of them (they fall to the conservative full scan)."""
    victim = tmp_path / "data" / "memory" / "identity.md"
    mem = tmp_path / "data" / "memory"
    db = tmp_path / "data" / "state" / "notes.db"
    for cmd in (
        f"python3 -c \"import subprocess; subprocess.run(['rm','-rf','{victim}'])\"",
        f"python3 -c \"import tarfile; tarfile.open('/tmp/x.tar').extractall('{mem}')\"",
        f"python3 -c \"import zipfile; zipfile.ZipFile('/tmp/x.zip').extractall('{mem}')\"",
        f"python3 -c \"import shutil; shutil.unpack_archive('/tmp/x.tar','{mem}')\"",
        f"python3 -c \"import os; os.system('rm -rf {victim}')\"",
        f"python3 -c \"import sqlite3; sqlite3.connect('{db}')\"",
    ):
        blocked = _guard(tmp_path, cmd, writeish=False)
        assert blocked, f"opaque write not blocked: {cmd}"


def test_opaque_exec_reading_own_root_file_still_allowed(tmp_path):
    """The r2 #1 fix must not over-block: an opaque exec (subprocess) whose only
    drive mention is the task's OWN staged attachment stays allowed — the
    conservative scan exempts the task roots."""
    own = tmp_path / "data" / "task_results" / "artifacts" / "t-v655" / "clip.mp3"
    cmd = f"python3 -c \"import subprocess; subprocess.run(['ffprobe','{own}'])\""
    assert _guard(tmp_path, cmd, writeish=False) == []


def test_secret_named_own_root_file_read_is_allowed(tmp_path):
    """Adversarial r2 #2: a staged attachment / own scratch file that merely
    NAME-matches the secret regex (secret_*, token_*) but lives under the task's
    OWN artifact_store/task_drive is the task's own content — reading it must not
    block (the GAIA own-file class). The owner's real settings.json at drive root
    (outside the task roots) still blocks (asserted separately above)."""
    own_secret = tmp_path / "data" / "task_results" / "artifacts" / "t-v655" / "secret_santa.docx"
    own_token = tmp_path / "data" / "task_drives" / "t-v655" / "token_usage.json"
    assert _guard(tmp_path, f"python3 -c \"print(open('{own_secret}','rb').read())\"", writeish=False) == []
    assert _guard(tmp_path, f"python3 -c \"print(open('{own_token}').read())\"", writeish=False) == []


def test_any_write_regex_has_no_midpattern_global_flags():
    """Review round 2: a second global (?is) mid-pattern is a hard re.error on
    Python 3.11+ (3.9/3.10 only warn, so CI is the first place it would crash)."""
    from ouroboros.tools.shell_guards import _INTERPRETER_ANY_WRITE_RE

    pattern = _INTERPRETER_ANY_WRITE_RE.pattern
    assert pattern.startswith("(?is)")
    assert "(?is)" not in pattern[5:]
    # The leading flags govern the appended alternation too.
    assert _INTERPRETER_ANY_WRITE_RE.search("OS.MAKEDIRS('/x')")


def test_python_dynamic_write_falls_back_to_full_mention_scan(tmp_path):
    outside = tmp_path / "data" / "state" / "x.json"
    cmd = (
        "python3 -c \"import sys; p=sys.argv[1]; open(p, 'w').write('x')\" "
        f"{outside}"
    )
    blocked = _guard(tmp_path, cmd, writeish=False)
    assert blocked and str(outside) in blocked[0]


def test_library_save_apis_do_not_slip_the_write_guard(tmp_path):
    """Fable-5 cumulative review F1: save-APIs carrying no base write-token
    (DataFrame.to_csv, plt.savefig, openpyxl Workbook.save, single-arg
    Path.open('w')) were classified as pure reads and skipped the runtime_data
    mention scan entirely."""
    outside = tmp_path / "data" / "logs" / "events.jsonl"
    for cmd in (
        f"python3 -c \"import pandas as pd; pd.DataFrame().to_csv('{outside}')\"",
        f"python3 -c \"import matplotlib.pyplot as plt; plt.savefig('{outside}')\"",
        f"python3 -c \"from openpyxl import Workbook; Workbook().save('{outside}')\"",
        f"python3 -c \"import pathlib; pathlib.Path('{outside}').open('w')\"",
    ):
        blocked = _guard(tmp_path, cmd, writeish=False)
        assert blocked, f"not blocked: {cmd}"


def test_any_write_regex_matches_library_save_apis_but_not_reads():
    from ouroboros.tools.shell_guards import _INTERPRETER_ANY_WRITE_RE as rx

    assert rx.search("df.to_csv('/x.csv')")
    assert rx.search("df.to_json('/x.json')")
    assert rx.search("plt.savefig('/x.png')")
    assert rx.search("wb.save('/x.xlsx')")
    assert rx.search("np.savez('/x.npz')")
    assert rx.search("cv2.imwrite('/x.png', img)")
    assert rx.search("json.dump(obj, fh)")
    assert rx.search("p.open('w')")
    assert rx.search("p.open(mode='ab')")
    assert rx.search("p.open('x+')")
    # Reads stay reads: no false positives on read modes, path-shaped args, or
    # the string-returning dumps().
    assert not rx.search("pd.read_csv('/x.csv')")
    assert not rx.search("wb = load_workbook('/x.xlsx')")
    assert not rx.search("p.open('r')")
    assert not rx.search("p.open('rb')")
    assert not rx.search("zf.open('warehouse.csv')")
    assert not rx.search("json.dumps(obj)")


def test_python_literal_path_windows_shape_uses_windows_semantics():
    """Windows CI full-test regression (v6.55.0): PurePosixPath('C:\\\\x\\\\y').parent
    is '.', so a windows-shaped literal's .parent collapsed to a cwd-shaped
    false-allow target. The flavor must follow the LITERAL's shape on every host."""
    import ast
    import pathlib

    from ouroboros.tools.shell_guards import _python_literal_path

    expr = ast.parse("p.parent").body[0].value
    win = "C:\\Users\\u\\AppData\\Temp\\home\\Ouroboros\\data\\uploads\\x.html"
    assert _python_literal_path(expr, {"p": win}) == str(pathlib.PureWindowsPath(win).parent)
    posix = "/home/u/data/uploads/x.html"
    assert _python_literal_path(expr, {"p": posix}) == "/home/u/data/uploads"
    # The / join follows the left operand's shape too.
    join = ast.parse("p / 'sub.txt'").body[0].value
    assert _python_literal_path(join, {"p": win}) == str(pathlib.PureWindowsPath(win) / "sub.txt")


def test_python_write_targets_windows_shape_and_degenerate_unknown():
    """The mkdir/touch modeling of a windows-shaped literal must resolve the REAL
    parent (deterministic on every host via PureWindowsPath); and a derivation
    that still collapses to '.'/'' is UNKNOWN → the caller keeps the conservative
    full mention scan instead of trusting a cwd-shaped target."""
    import pathlib

    from ouroboros.tools.shell_guards import _python_write_targets_and_unknown

    win = "C:\\Users\\u\\AppData\\Temp\\home\\Ouroboros\\data\\uploads\\touch-report.html"
    code = (
        "from pathlib import Path\n"
        f"p = Path({win!r})\n"
        "p.parent.mkdir(parents=True, exist_ok=True)\n"
        "p.touch()\n"
    )
    targets, unknown = _python_write_targets_and_unknown(code)
    assert str(pathlib.PureWindowsPath(win).parent) in targets

    degenerate = (
        "from pathlib import Path\n"
        "p = Path('relative.html')\n"
        "p.parent.mkdir(exist_ok=True)\n"
    )
    targets, unknown = _python_write_targets_and_unknown(degenerate)
    assert unknown and "." not in targets


# ---------------------------------------------------------------------------
# 1.3 — actionable messages carry REAL resolved paths


def test_shell_cwd_block_message_names_resolved_paths(tmp_path):
    from ouroboros.tool_access import shell_cwd_block_message

    ctx = _ctx(tmp_path)
    msg = shell_cwd_block_message(ctx, "/nonexistent/guessed", operation="shell")
    assert "SHELL_CWD_BLOCKED" in msg
    # Labels are rendered as label=<resolved path>, not bare labels.
    assert "task_drive=" in msg and str(tmp_path / "data") in msg


def test_stage_task_attachments_carries_abs_path(tmp_path):
    from ouroboros.artifacts import stage_task_attachments
    from ouroboros.gateway.tasks import _render_attachment_lines

    src = tmp_path / "table.xlsx"
    src.write_bytes(b"xlsx-bytes")
    drive = tmp_path / "data"
    drive.mkdir()
    manifest = stage_task_attachments(drive, "t-v655", [str(src)])
    assert manifest and manifest[0]["abs_path"].endswith("table.xlsx")
    staged = pathlib.Path(manifest[0]["abs_path"])
    assert staged.is_file()
    assert "task_results" in staged.parts and "attachments" in staged.parts
    rendered = _render_attachment_lines(manifest)
    assert "script/process path: " in rendered and manifest[0]["abs_path"] in rendered


# ---------------------------------------------------------------------------
# 1.1 — resolve_user_file_path absolute-outside-home rejection


def test_resolve_user_file_path_rejects_absolute_outside_home(tmp_path, monkeypatch):
    from ouroboros import tool_access as ta

    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(fake_home))
    ctx = _ctx(tmp_path)
    outside = tmp_path / "elsewhere" / "app" / "main.go"
    # Non-external ctx: the NEW actionable early rejection (not the later
    # opaque relative_to crash class).
    with pytest.raises(ValueError, match="outside the user_files home"):
        ta.resolve_user_file_path(ctx, str(outside))
    inside = fake_home / "Desktop" / "file.txt"
    assert ta.resolve_user_file_path(ctx, str(inside)) == inside.resolve()
    # Case-insensitive-platform parity (review round 7): a differently-cased
    # safe home path is not early-rejected where the casefold-aware authority
    # would accept it.
    cased = pathlib.Path(str(fake_home).upper()) / "Desktop" / "file2.txt"
    resolved_cased = ta.resolve_user_file_path(ctx, str(cased))
    assert resolved_cased.name == "file2.txt"
    # External-workspace ctx keeps its designed host-scratch reach (the
    # query_code external-target contract runs in this mode) — both with and
    # without the explicit opt-out flag.
    ext = ToolContext(
        repo_dir=tmp_path / "system",
        drive_root=tmp_path / "data",
        workspace_root=tmp_path / "elsewhere" / "app",
        workspace_mode="external",
        task_id="t-ext",
        task_metadata={},
    )
    resolved = ta.resolve_user_file_path(ext, str(outside), allow_outside_home=True)
    assert str(resolved).endswith("main.go")
    assert str(ta.resolve_user_file_path(ext, str(outside))).endswith("main.go")


# ---------------------------------------------------------------------------
# 1.1 — dispatch root-label hybrid: reads auto-route, writes redirect


def test_dispatch_auto_routes_user_files_read_under_workspace(tmp_path):
    from ouroboros.tools.registry import _normalize_dispatch_path_args

    ctx = _ctx(tmp_path)
    target = tmp_path / "system" / "src" / "x.py"
    args = {"root": "user_files", "path": str(target)}
    note = _normalize_dispatch_path_args(ctx, "read_file", args)
    assert note.startswith("⚠️ AUTO_ROUTED_TO_ACTIVE_WORKSPACE")
    assert args["root"] == "active_workspace"


def test_dispatch_redirects_user_files_write_under_workspace(tmp_path):
    from ouroboros.tools.registry import _normalize_dispatch_path_args

    ctx = _ctx(tmp_path)
    target = tmp_path / "system" / "src" / "x.py"
    args = {"root": "user_files", "path": str(target)}
    note = _normalize_dispatch_path_args(ctx, "write_file", args)
    assert note.startswith("⚠️ ROOT_REQUIRED_ACTIVE_WORKSPACE")
    assert args["root"] == "user_files"  # write args stay untouched


def test_dispatch_leaves_query_code_and_true_user_files_alone(tmp_path, monkeypatch):
    from ouroboros.tools.registry import _normalize_dispatch_path_args

    ctx = _ctx(tmp_path)
    target = tmp_path / "system" / "src" / "x.py"
    assert _normalize_dispatch_path_args(ctx, "query_code", {"root": "user_files", "path": str(target)}) == ""
    fake_home = tmp_path / "home"
    fake_home.mkdir(exist_ok=True)
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(fake_home))
    args = {"root": "user_files", "path": str(fake_home / "doc.txt")}
    assert _normalize_dispatch_path_args(ctx, "read_file", args) == ""
    assert args["root"] == "user_files"


# ---------------------------------------------------------------------------
# 1.4 — deadline-clamped outer timeouts for network/long tools


def _clamp(tools_ctx, name, base):
    from ouroboros.loop_tool_execution import _deadline_clamped_timeout

    return _deadline_clamped_timeout(SimpleNamespace(_ctx=tools_ctx), name, base)


def test_deadline_clamp_is_inert_without_deadline(tmp_path):
    ctx = _ctx(tmp_path)
    assert _clamp(ctx, "web_search", 540) == 540
    assert _clamp(ctx, "run_command", 540) == 540


def test_deadline_clamp_bounds_web_tools(tmp_path, monkeypatch):
    from datetime import datetime, timedelta, timezone

    monkeypatch.setenv("OUROBOROS_FINALIZATION_GRACE_SEC", "120")
    deadline = (datetime.now(timezone.utc) + timedelta(seconds=400)).isoformat()
    ctx = _ctx(tmp_path, meta={"deadline_at": deadline})
    clamped = _clamp(ctx, "web_search", 540)
    # window = remaining - reserve ≈ 280; the clamp never exceeds it.
    assert 1 <= clamped <= 281
    # Non-network tools are untouched.
    assert _clamp(ctx, "run_command", 540) == 540
    # Elapsed deadline: unclamped (forced finalization owns that path).
    past = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
    ctx2 = _ctx(tmp_path, meta={"deadline_at": past})
    assert _clamp(ctx2, "web_search", 540) == 540


def test_deadline_clamp_never_floors_past_reserve(tmp_path, monkeypatch):
    """Review round 1 regression: remaining INSIDE the finalization reserve must
    yield a near-immediate timeout, never a 30s floor that eats the reserve."""
    from datetime import datetime, timedelta, timezone

    monkeypatch.setenv("OUROBOROS_FINALIZATION_GRACE_SEC", "120")
    deadline = (datetime.now(timezone.utc) + timedelta(seconds=20)).isoformat()
    ctx = _ctx(tmp_path, meta={"deadline_at": deadline})
    assert _clamp(ctx, "web_search", 540) == 1


# ---------------------------------------------------------------------------
# 1.5 — plan_task deadline scaling


def test_plan_task_skips_under_tight_deadline(tmp_path):
    from datetime import datetime, timedelta, timezone

    from ouroboros.tools.plan_review import _handle_plan_task

    deadline = (datetime.now(timezone.utc) + timedelta(seconds=400)).isoformat()
    ctx = _ctx(tmp_path, meta={"deadline_at": deadline})
    out = _handle_plan_task(ctx, plan="do X then Y", goal="ship X")
    assert out.startswith("PLAN_TASK_SKIPPED_DEADLINE")
    events = []
    while not ctx.event_queue.empty():
        events.append(ctx.event_queue.get_nowait())
    assert any(e.get("type") == "plan_task_deadline_skip" for e in events)


def test_plan_task_no_deadline_does_not_skip(tmp_path, monkeypatch):
    from ouroboros.tools import plan_review as pr

    sentinel = {"called": False}

    def _fake_run(coro=None, *a, **k):
        sentinel["called"] = True
        if coro is not None and hasattr(coro, "close"):
            coro.close()
        return "ok"

    monkeypatch.setattr(pr.asyncio, "run", _fake_run)
    ctx = _ctx(tmp_path)
    out = pr._handle_plan_task(ctx, plan="p", goal="g")
    assert sentinel["called"] is True and out == "ok"


# ---------------------------------------------------------------------------
# 1.6 — schedule_subagent slot visibility


def test_subagent_slot_note_reads_snapshot(tmp_path):
    from ouroboros.tools.control import _subagent_slot_note

    ctx = _ctx(tmp_path)
    state = tmp_path / "data" / "state"
    state.mkdir(parents=True, exist_ok=True)
    snap = {
        "running": [
            {"id": "c1", "task": {"delegation_role": "subagent", "root_task_id": "root-1"}},
            {"id": "other", "task": {"delegation_role": "subagent", "root_task_id": "root-2"}},
        ],
        "pending": [
            {"id": "c2", "task": {"delegation_role": "subagent", "root_task_id": "root-1"}},
        ],
    }
    (state / "queue_snapshot.json").write_text(json.dumps(snap), encoding="utf-8")
    note = _subagent_slot_note(ctx, "root-1")
    assert "1/" in note and "1 queued" in note


def test_subagent_slot_note_fail_soft_without_snapshot(tmp_path):
    from ouroboros.tools.control import _subagent_slot_note

    assert _subagent_slot_note(_ctx(tmp_path), "root-1") == ""


# ---------------------------------------------------------------------------
# review round 1 regressions


def test_safety_mode_skip_falls_back_to_drive_logs(tmp_path, monkeypatch):
    """A context WITHOUT a live event_queue still leaves a durable audit row."""
    import json as _json

    ctx = ToolContext(
        repo_dir=tmp_path / "system",
        drive_root=tmp_path / "data",
        task_id="t-noq",
        task_metadata={},
    )
    (tmp_path / "data" / "logs").mkdir(parents=True, exist_ok=True)
    safety_mod._emit_safety_mode_skip(ctx, "run_command", "light", "check_conditional")
    rows = [
        _json.loads(line)
        for line in (tmp_path / "data" / "logs" / "events.jsonl").read_text().splitlines()
    ]
    assert any(r.get("type") == "safety_mode_skip" and r.get("safety_mode") == "light" for r in rows)


def test_list_files_hard_failure_is_first_class_error(tmp_path, monkeypatch):
    """Review round 3: an iterdir/permission failure inside a listing helper must
    surface as the first-class LIST_FILES_ERROR string, never ok-shaped JSON."""
    from ouroboros.tools import core as core_mod

    ctx = _ctx(tmp_path)
    boom = tmp_path / "data" / "task_drives" / "t-v655"
    boom.mkdir(parents=True, exist_ok=True)

    def _explode(*a, **k):
        raise PermissionError("simulated iterdir failure")

    monkeypatch.setattr(core_mod, "_list_dir", _explode)
    out = core_mod._list_files(ctx, path=".", root="task_drive")
    assert out.startswith("⚠️ LIST_FILES_ERROR (PermissionError)")


def test_safety_parse_failed_event_is_durable_without_queue(tmp_path):
    import json as _json

    ctx = ToolContext(
        repo_dir=tmp_path / "system",
        drive_root=tmp_path / "data",
        task_id="t-noq2",
        task_metadata={},
    )
    (tmp_path / "data" / "logs").mkdir(parents=True, exist_ok=True)
    safety_mod._emit_durable_safety_event(
        ctx, {"type": "safety_parse_failed", "tool": "run_command", "failure_class": "empty"}
    )
    rows = [
        _json.loads(line)
        for line in (tmp_path / "data" / "logs" / "events.jsonl").read_text().splitlines()
    ]
    assert any(
        r.get("type") == "safety_parse_failed" and r.get("failure_class") == "empty" for r in rows
    )


def test_route_note_trails_result_for_failure_classification():
    from ouroboros.tools.registry import _compose_execute_result

    out = _compose_execute_result(
        "⚠️ TOOL_ERROR: File not found: x.py",
        "⚠️ AUTO_ROUTED_TO_ACTIVE_WORKSPACE: ...",
        "",
    )
    assert out.splitlines()[0].startswith("⚠️ TOOL_ERROR")
    assert "AUTO_ROUTED_TO_ACTIVE_WORKSPACE" in out


def test_owner_safety_mode_response_in_frozen_contract():
    from ouroboros.gateway import contracts

    assert "OwnerSafetyModeResponse" in contracts.__all__
    assert set(contracts.OwnerSafetyModeResponse.__annotations__) == {"ok", "safety_mode"}
