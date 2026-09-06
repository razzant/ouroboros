"""Read-carve for the owner-control mention detectors (family-wide).

The scope-floor guard adjudicated the contract at v6.80.0: naming an owner
key/endpoint blocks UNLESS the whole command line is demonstrably read-only
inspection. The other six family members (runtime mode, context mode, safety
mode, skill attestation, mutative toggle, evolution controls) stayed
read-blind, deterministically blocking the product's own mandated inspection
flows — ``grep OUROBOROS_SAFETY_MODE data/settings.json``, the reuse-first
callers matrix (``rg ouroboros.config.save_settings``), route inspection
(``rg /api/owner/safety-mode``) — in every runtime mode. These tests pin the
family-wide carve: pure-read heads pass, every write shape and every
non-inspection head (interpreter, HTTP client) still blocks, and a caller
that cannot supply the write-shape fact stays fail-closed by default.
"""

from __future__ import annotations

import pathlib

import pytest

pytestmark = pytest.mark.serial

from ouroboros.tools.registry import (
    ToolContext,
    ToolRegistry,
    _detect_context_mode_self_lowering,
    _detect_evolution_owner_control_self_change,
    _detect_mutative_toggle_self_change,
    _detect_owner_skill_attest_self_call,
    _detect_runtime_mode_elevation,
    _detect_safety_mode_self_lowering,
)

from tests._typed_guard_shared import _shell_guard_text



@pytest.fixture(autouse=True)
def _home_outside_tmp(tmp_path, monkeypatch):
    fake_home = tmp_path / "_home"
    fake_home.mkdir(exist_ok=True)
    monkeypatch.setattr(pathlib.Path, "home", lambda: fake_home)


DETECTOR_READ_WRITE_CASES = [
    (
        _detect_runtime_mode_elevation,
        "grep -rn ouroboros.config.save_settings ouroboros/",
        "python -c \"from ouroboros.config import save_settings; save_settings({'OUROBOROS_RUNTIME_MODE': 'pro'})\"",
    ),
    (
        _detect_context_mode_self_lowering,
        "rg /api/owner/context-mode ouroboros/gateway",
        "curl -X POST http://127.0.0.1:8765/api/owner/context-mode -d '{\"mode\":\"low\"}'",
    ),
    (
        _detect_safety_mode_self_lowering,
        "grep OUROBOROS_SAFETY_MODE data/settings.json",
        "python -c \"import httpx; httpx.request('POST','http://127.0.0.1:8765/api/owner/safety-mode',json={'mode':'off'})\"",
    ),
    (
        _detect_owner_skill_attest_self_call,
        "rg '/api/owner/skills/.+/attest-review' ouroboros/gateway",
        "curl -X POST http://127.0.0.1:8765/api/owner/skills/foo/attest-review",
    ),
    (
        _detect_mutative_toggle_self_change,
        "grep OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS data/settings.json",
        "python -c \"import json,pathlib; p=pathlib.Path('data/settings.json'); d=json.loads(p.read_text()); d['OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS']='true'; p.write_text(json.dumps(d))\"",
    ),
    (
        _detect_evolution_owner_control_self_change,
        "grep OUROBOROS_POST_TASK_EVOLUTION data/settings.json",
        "sh -c \"ouroboros settings set OUROBOROS_POST_TASK_EVOLUTION true\"",
    ),
]


@pytest.mark.parametrize("detector, read_cmd, write_cmd", DETECTOR_READ_WRITE_CASES)
def test_pure_read_inspection_passes_and_mutation_blocks(detector, read_cmd, write_cmd):
    # Pure read-only inspection with no write shape passes.
    assert detector(read_cmd.lower(), writeish=False) is False
    # The same read shape carrying a write-shape fact stays fail-closed.
    assert detector(read_cmd.lower(), writeish=True) is True
    # A mutation shape blocks regardless of the writeish fact: interpreters and
    # HTTP clients are not inspection heads.
    assert detector(write_cmd.lower(), writeish=False) is True


@pytest.mark.parametrize("detector, read_cmd, _write_cmd", DETECTOR_READ_WRITE_CASES)
def test_default_stays_fail_closed_without_the_fact(detector, read_cmd, _write_cmd):
    # A caller that cannot supply the write-shape fact keeps the pre-carve
    # fail-closed behavior (writeish defaults True).
    assert detector(read_cmd.lower()) is True


def test_inplace_editor_on_allowlisted_head_is_not_pure_read():
    """sol review: yq is a read head, but `yq -i` edits the named file in place.
    A settings mutation through an in-place editor must NOT be exempted as
    inspection (jq has no in-place edit and stays a stdout-only read)."""
    from ouroboros.tools.registry import _is_pure_read_inspection
    assert _is_pure_read_inspection("yq -i '.ouroboros_safety_mode = \"off\"' /x/data/settings.json".lower()) is False
    assert _is_pure_read_inspection("yq --inplace '.ouroboros_context_mode = \"low\"' /x/data/settings.json".lower()) is False
    # yq WITHOUT -i and jq (no in-place) stay reads.
    assert _is_pure_read_inspection("yq '.ouroboros_safety_mode' /x/data/settings.json".lower()) is True
    assert _is_pure_read_inspection("jq '.ouroboros_safety_mode' /x/data/settings.json".lower()) is True
    # The owner-control detectors therefore block the in-place edit.
    assert _detect_safety_mode_self_lowering(
        "yq -i '.ouroboros_safety_mode = \"off\"' /x/data/settings.json".lower(), writeish=False
    ) is True
    assert _detect_context_mode_self_lowering(
        "yq -i '.ouroboros_context_mode = \"low\"' /x/data/settings.json".lower(), writeish=False
    ) is True


def test_registry_level_read_carve_end_to_end(tmp_path):
    system = tmp_path / "system"
    data = tmp_path / "data"
    for p in (system, data):
        p.mkdir()
    (data / "settings.json").write_text("{}", encoding="utf-8")
    reg = ToolRegistry(repo_dir=system, drive_root=data)
    reg.set_context(ToolContext(repo_dir=system, drive_root=data, task_id="carve-test"))

    # Route inspection of an owner endpoint in the repo source: allowed.
    out = _shell_guard_text(reg,
        {"cmd": ["rg", "/api/owner/safety-mode", str(system)], "cwd": str(system)}, "advanced"
    )
    assert out is None
    # An HTTP client naming the same endpoint: blocked, whatever verb it spells.
    out = _shell_guard_text(reg,
        {
            "cmd": [
                "python3",
                "-c",
                "import httpx; httpx.request('POST','http://127.0.0.1:8765/api/owner/safety-mode',json={'mode':'off'})",
            ],
            "cwd": str(system),
        },
        "advanced",
    )
    assert "SAFETY_MODE_SELF_LOWERING_BLOCKED" in (out or "")
