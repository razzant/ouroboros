"""Boundary tests for producer and host-owned ToolResult metadata."""

from __future__ import annotations

import json

import pytest

from ouroboros.tools.tool_result import (
    TOOL_CODE_SPECS,
    LegacyTextResultAdapter,
    ToolResult,
    _compose_execute_result_result,
    _replace_tool_result,
)


def test_composition_reserves_host_keys_beyond_32_producer_items() -> None:
    ordinary = {f"k{index}": index for index in range(32)}
    base = ToolResult(
        status="error",
        code="TOOL_ERROR",
        text="failed",
        meta=ordinary,
    )

    composed = _compose_execute_result_result("fixture", base, "route", "warning")

    assert composed.meta == {
        **ordinary,
        "route_note": True,
        "safety_warning": True,
    }
    with pytest.raises(ValueError, match="at most 32"):
        ToolResult(
            status="ok",
            code="OK",
            text="done",
            meta={f"k{index}": index for index in range(33)},
        )


def test_process_result_codes_and_immutable_replacement_are_exact(
    monkeypatch,
) -> None:
    expected = {
        "SHELL_NO_MATCH": ("ok", "ok", "info"),
        "OWNER_STATE_RESTORED": ("ok", "ok", "warning"),
        "LIGHT_MODE_REPO_WRITE_BLOCKED": (
            "blocked", "light_mode_blocked", "warning",
        ),
        "WORKSPACE_GIT_REF_CHANGED": (
            "blocked", "workspace_blocked", "warning",
        ),
    }
    for code, mapping in expected.items():
        spec = TOOL_CODE_SPECS[code]
        assert (spec.status, spec.outcome_bucket, spec.ui_severity) == mapping
        adapted = LegacyTextResultAdapter.from_text(
            "run_command", f"⚠️ {code}: fixture",
        )
        assert (adapted.status, adapted.code) == (mapping[0], code)

    base = ToolResult(
        status="ok",
        code="SHELL_NO_MATCH",
        text="base",
        meta={"exit_code": 1},
    )
    monkeypatch.setattr(
        LegacyTextResultAdapter,
        "from_text",
        lambda *_args, **_kwargs: pytest.fail("legacy adapter used"),
    )
    replaced = _replace_tool_result(
        base,
        text="wrapped",
        code="WORKSPACE_GIT_REF_CHANGED",
        meta_updates={"workspace_git_refs_changed": True},
    )
    assert replaced == ToolResult(
        status="blocked",
        code="WORKSPACE_GIT_REF_CHANGED",
        text="wrapped",
        meta={"exit_code": 1, "workspace_git_refs_changed": True},
    )


def test_composition_reserves_host_bytes_beyond_exact_producer_limit() -> None:
    empty_payload_size = len(json.dumps({"payload": ""}, separators=(",", ":")))
    exact_meta = {"payload": "x" * (8192 - empty_payload_size)}
    assert len(json.dumps(exact_meta, separators=(",", ":")).encode()) == 8192
    base = ToolResult(
        status="ok",
        code="GIT_ERROR",
        text="failed",
        meta=exact_meta,
    )

    composed = _compose_execute_result_result("fixture", base, "route", "warning")

    assert composed.meta == {
        **exact_meta,
        "route_note": True,
        "safety_warning": True,
    }
    postchecked = _replace_tool_result(
        composed,
        meta_updates={
            "owner_state_restored": True,
            "light_repo_changed": True,
            "workspace_git_refs_changed": True,
        },
    )
    assert postchecked.meta == {
        **exact_meta,
        "route_note": True,
        "safety_warning": True,
        "owner_state_restored": True,
        "light_repo_changed": True,
        "workspace_git_refs_changed": True,
    }
    empty_host_size = len(json.dumps({"route_note": ""}, separators=(",", ":")))
    existing_host_meta = {"route_note": "x" * (8192 - empty_host_size)}
    existing = ToolResult(status="ok", code="OK", text="done", meta=existing_host_meta)
    assert _compose_execute_result_result(
        "fixture",
        existing,
        "route",
        "",
    ).meta == {"route_note": True}
    with pytest.raises(ValueError, match="8192"):
        ToolResult(
            status="ok",
            code="OK",
            text="done",
            meta={"payload": exact_meta["payload"] + "x"},
        )
    with pytest.raises(ValueError, match="reserved overhead"):
        ToolResult(
            status="ok",
            code="OK",
            text="done",
            meta={"route_note": "x" * 9000},
        )
