"""A skill's own typed refusal reaches the result's meta and dates its unavailable permission.

The C10/C04 contract: a reported failure's top-level string ``reason`` is its code and an optional ``refusal``
object names the permission; a success may list ``permissions_available``. The host copies the validated words
into ``meta`` (the result code stays ``TOOL_REPORTED_FAILURE``) and keeps an open refusal per permission in the
skill's health record until a success reports that permission available.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from ouroboros import extension_health
from ouroboros.tools.extension_dispatch import _extension_completion, _with_extension_report
from ouroboros.tools.tool_result import ToolResult

_SKILL = {"skill": "desk"}
_DENIED = {"ok": False, "reason": "accessibility_untrusted", "refusal": {
    "capability_unavailable": True, "permission": "accessibility", "repair": "owner",
    "where": "System Settings > Privacy & Security >   Accessibility", "unknown_key": 1}}


def _completed(body: dict) -> ToolResult:
    return _extension_completion(json.dumps(body), "")


def _refusals(root) -> dict:
    return extension_health.open_permission_refusals(extension_health.read_extension_health(root, "desk"))


def test_a_reported_reason_reaches_meta_beside_the_hosts_own_facts_and_older_bodies_read_as_before(tmp_path):
    ctx = SimpleNamespace(drive_root=tmp_path, task_metadata={})
    result = _with_extension_report(ctx, _SKILL, _completed(_DENIED))
    assert (result.status, result.code) == ("error", "TOOL_REPORTED_FAILURE")
    assert result.meta["extension_reason"] == {
        "code": "accessibility_untrusted", "capability_unavailable": True, "permission": "accessibility",
        "repair": "owner", "where": "System Settings > Privacy & Security > Accessibility"}
    assert result.meta["physical_dispatch"] is True and json.loads(result.text) == _DENIED
    controlled = ToolResult(status="error", code="TOOL_REPORTED_FAILURE", text=json.dumps(_DENIED),
                            meta={"physical_dispatch": True, "control_reason": "owner_pause"})
    assert _with_extension_report(ctx, _SKILL, controlled).meta["control_reason"] == "owner_pause"
    for older in ({"ok": False, "reason": {"code": "x"}}, {"ok": False, "error": "boom"},
                  {"ok": False, "reason": "Permission denied by the user"}):
        assert "extension_reason" not in _with_extension_report(ctx, _SKILL, _completed(older)).meta, older
    success = _with_extension_report(ctx, _SKILL, _completed(
        {"ok": True, "permissions_available": ["post_events", "screen_recording", "Bad Id", 7]}))
    assert (success.code, success.meta["extension_permissions_available"]) == ("OK", ["post_events", "screen_recording"])
    assert "extension_permissions_available" not in _with_extension_report(ctx, _SKILL, _completed({"ok": True})).meta
    unsent = ToolResult(status="error", code="TOOL_REPORTED_FAILURE", text=json.dumps(_DENIED), meta={})
    assert "extension_reason" not in _with_extension_report(ctx, _SKILL, unsent).meta  # no physical dispatch


def test_an_unavailable_permission_opens_a_dated_refusal_a_repeat_keeps_it_and_a_success_closes_it(tmp_path):
    ctx = SimpleNamespace(drive_root=tmp_path, task_metadata={})
    extension_health.record_extension_health(tmp_path, "desk", status=extension_health.LIVE)
    _with_extension_report(ctx, _SKILL, _completed(_DENIED))
    opened = _refusals(tmp_path)
    assert list(opened) == ["accessibility"] and opened["accessibility"]["code"] == "accessibility_untrusted"
    since = opened["accessibility"]["since"]
    path = extension_health.health_path(tmp_path, "desk")
    stamp = path.stat().st_mtime_ns
    _with_extension_report(ctx, _SKILL, _completed(_DENIED))  # the same refusal again: nothing is written
    assert _refusals(tmp_path) == opened and path.stat().st_mtime_ns == stamp
    other = {**_DENIED, "refusal": {**_DENIED["refusal"], "capability_unavailable": False, "permission": "screen_recording"}}
    _with_extension_report(ctx, _SKILL, _completed(other))  # not a capability refusal: nothing opens
    extension_health.record_extension_health(tmp_path, "desk", status=extension_health.LIVE)  # a later observation
    assert _refusals(tmp_path) == opened and extension_health.read_extension_health(tmp_path, "desk")["status"] == "live"
    _with_extension_report(ctx, _SKILL, _completed({"ok": True, "permissions_available": ["post_events"]}))
    assert _refusals(tmp_path)["accessibility"]["since"] == since  # another permission says nothing of this one
    _with_extension_report(ctx, _SKILL, _completed({"ok": True, "permissions_available": ["accessibility"]}))
    assert _refusals(tmp_path) == {} and "permission_refusals" not in json.loads(path.read_text(encoding="utf-8"))


def test_a_refusal_without_a_health_record_writes_nothing(tmp_path):
    ctx = SimpleNamespace(drive_root=tmp_path, task_metadata={})
    result = _with_extension_report(ctx, _SKILL, _completed(_DENIED))
    assert result.meta["extension_reason"]["permission"] == "accessibility"
    assert extension_health.read_extension_health(tmp_path, "desk") is None
