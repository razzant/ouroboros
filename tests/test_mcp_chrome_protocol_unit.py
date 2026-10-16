"""Chrome wire facts that must refuse before any browser operation is sent."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from ouroboros import mcp_chrome_protocol as chrome

OWNER = ("task-one", "2")


@pytest.fixture(autouse=True)
def cyber_consumer(monkeypatch):
    monkeypatch.setattr("ouroboros.config.get_runtime_mode", lambda: "cyber_pro")


def state(*, connected=True, document_id="doc-1", url="https://example.test/"):
    return {
        "protocol": "ouroboros_chrome", "version": 1,
        "owner": {"task_id": OWNER[0], "attempt": OWNER[1]},
        "executor_generation": 4, "connected": connected,
        "lease": ({"lease_id": "lease-1", "generation": 7, "tab_id": "tab-1",
                   "document_id": document_id, "url": url, "guard_armed": True}
                  if connected else None),
        "closure": {"revoked": False, "detached": False},
    }


def result(payload):
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=json.dumps(payload))], isError=False)


class ChromeWireTests(unittest.TestCase):
    def test_state_requires_one_owned_json_result_and_all_lease_facts(self):
        self.assertEqual(chrome.parse_state(result(state()), OWNER)["lease"]["tab_id"], "tab-1")
        self.assertIsNone(chrome.parse_state(result(state(connected=False)), OWNER)["lease"])
        with self.assertRaisesRegex(chrome.ChromeStateError, "OWNER_CHANGED"):
            chrome.parse_state(result(state()), ("another", "2"))
        with self.assertRaisesRegex(chrome.ChromeStateError, "STATE_UNKNOWN"):
            chrome.parse_state(SimpleNamespace(content=[SimpleNamespace(type="text", text="{}"),
                                                         SimpleNamespace(type="text", text="{}")]), OWNER)
        broken = state()
        del broken["lease"]["document_id"]
        with self.assertRaisesRegex(chrome.ChromeStateError, "STATE_UNKNOWN"):
            chrome.parse_state(result(broken), OWNER)

    def test_revoked_and_detached_are_distinct(self):
        for fact, code in (("revoked", "SESSION_REVOKED"), ("detached", "SESSION_DETACHED")):
            broken = state()
            broken["closure"][fact] = True
            with self.assertRaisesRegex(chrome.ChromeStateError, code):
                chrome.parse_state(result(broken), OWNER)

    def test_primitive_binds_invocation_step_and_exact_lease(self):
        current = state()
        callback = {"kind": "primitive", "invocation_id": "inv-1", "step_id": 1,
                    "operation": "click", "args": {"selector": "#go"}, "state": current}
        invocation = ("inv-1", "browser_action", {"operation": "click", "args": {"selector": "#go"}})
        with patch.object(chrome, "action_reason", return_value=""):
            self.assertEqual(chrome.guard_reason(None, callback, current, invocation, False), "")
            changed = {**callback, "state": state(document_id="doc-2")}
            self.assertIn("LEASE_CHANGED", chrome.guard_reason(None, changed, current, invocation, False))
            wrong = {**callback, "invocation_id": "inv-2"}
            self.assertIn("invocation", chrome.guard_reason(None, wrong, current, invocation, False))
            no_step = {**callback, "step_id": 0}
            self.assertIn("step", chrome.guard_reason(None, no_step, current, invocation, False))
            self.assertIn("SESSION_REVOKED", chrome.guard_reason(None, callback, current, invocation, True))

    def test_network_requires_complete_body_and_request_identity(self):
        current = state()
        callback = {"kind": "request", "invocation_id": "inv-1", "step_id": 1,
                    "request_id": "req-1", "url": "https://example.test/api", "method": "POST",
                    "post_data": None, "complete": True, "state": current}
        invocation = ("inv-1", "browser_action", {"operation": "click", "args": {}})
        with patch.object(chrome, "request_reason", return_value="") as judge:
            self.assertEqual(chrome.guard_reason(None, callback, current, invocation, False), "")
            judge.assert_called_once()
            malformed = dict(callback)
            malformed.pop("complete")
            self.assertIn("incomplete", chrome.guard_reason(None, malformed, current, invocation, False))
        for key in ("request_id", "post_data"):
            malformed = dict(callback)
            malformed.pop(key)
            self.assertIn("UNAVAILABLE", chrome.request_reason(None, malformed))

    def test_navigation_only_rebinds_same_owner_executor_and_tab(self):
        prior, after = state(), state(document_id="doc-2", url="https://example.test/next")
        after["lease"]["lease_id"] = "lease-2"
        after["lease"]["generation"] = 8
        with patch.object(chrome, "target_reason", return_value="") as judge:
            self.assertEqual(chrome.navigation_transition(None, prior, after), "")
            judge.assert_called_with(None, "https://example.test/next")
        foreign = state(document_id="doc-2")
        foreign["lease"]["tab_id"] = "tab-2"
        self.assertIn("LEASE_CHANGED", chrome.navigation_transition(None, prior, foreign))

    def test_disconnected_controls_and_revocation(self):
        disconnected = state(connected=False)
        self.assertEqual(chrome.action_reason(None, "browser_state", {}, disconnected), "")
        self.assertEqual(chrome.action_reason(None, "browser_connect", {"tab_id": "tab-1"}, disconnected), "")
        with patch.object(chrome, "target_reason", return_value="") as target:
            self.assertEqual(chrome.action_reason(None, "browser_connect", {"url": "https://example.test/new"}, disconnected), "")
            target.assert_called_with(None, "https://example.test/new")
        for args in ({}, {"url": "https://example.test/new", "tab_id": "tab-1"}, {"tab_id": ""}):
            self.assertIn("exactly url or tab_id", chrome.action_reason(None, "browser_connect", args, disconnected))
        self.assertEqual(chrome.action_reason(None, "browser_resume", {}, disconnected), "")
        self.assertIn("DETACHED", chrome.action_reason(None, "browser_disconnect", {}, disconnected))
        disconnected["closure"]["revoked"] = True
        self.assertIn("REVOKED", chrome.action_reason(None, "browser_resume", {}, disconnected))

    def test_connect_url_callback_checks_requested_and_current_target(self):
        current = state()
        args = {"url": "https://example.test/new"}
        callback = {"kind": "primitive", "invocation_id": "inv-1", "step_id": 1,
                    "operation": "connect", "args": args, "state": current}
        invocation = ("inv-1", "browser_connect", args)
        with patch.object(chrome, "target_reason", side_effect=["", ""] ) as target:
            self.assertEqual(chrome.guard_reason(None, callback, current, invocation, False), "")
            self.assertEqual(target.call_count, 2)
        with patch.object(chrome, "target_reason", return_value="BROWSER_URL_BLOCKED"):
            self.assertEqual(chrome.guard_reason(None, callback, current, invocation, False), "BROWSER_URL_BLOCKED")

    def test_unqualified_effect_channels_require_cyber_but_reads_remain_available(self):
        with (patch("ouroboros.config.get_runtime_mode", return_value="advanced"),
              patch("ouroboros.tools.core.is_restricted_subagent_profile", return_value=False),
              patch.object(chrome, "target_reason", return_value="")):
            for operation in ("click", "fill", "press", "navigate", "submit"):
                args = {"url": "https://example.test/"} if operation == "navigate" else {}
                self.assertIn("GUARD_UNAVAILABLE", chrome.action_reason(
                    None, "browser_action", {"operation": operation, "args": args}, state()))
            self.assertIn("GUARD_UNAVAILABLE", chrome.action_reason(
                None, "browser_connect", {"url": "https://example.test/"}, state(connected=False)))
            self.assertEqual(chrome.action_reason(
                None, "browser_action", {"operation": "snapshot", "args": {}}, state()), "")

    def test_execute_requires_unrestricted_agency_typed_read_remains_available(self):
        current = state()
        with (patch("ouroboros.config.get_runtime_mode", return_value="cyber"),
              patch("ouroboros.tools.core.is_restricted_subagent_profile", return_value=True),
              patch("ouroboros.tool_access.active_tool_profile", return_value="local_readonly_subagent"),
              patch.object(chrome, "target_reason", return_value="")):
            self.assertIn("GUARD_BYPASS", chrome.action_reason(
                None, "browser_execute", {"code": "42"}, current))
            self.assertEqual(chrome.action_reason(
                None, "browser_action", {"operation": "snapshot", "args": {}}, current), "")
            self.assertIn("READONLY_BLOCKED", chrome.action_reason(
                None, "browser_action", {"operation": "click", "args": {}}, current))
        with (patch("ouroboros.config.get_runtime_mode", return_value="restricted"),
              patch("ouroboros.tools.core.is_restricted_subagent_profile", return_value=False),
              patch("ouroboros.runtime_mode_policy.mode_has_unrestricted_agency", return_value=False)):
            self.assertIn("GUARD_BYPASS", chrome.action_reason(
                None, "browser_execute", {"code": "42"}, current))


    def test_sequence_shape_and_readonly_authority_are_not_arbitrary_javascript(self):
        cached = {"operation": "cached_text", "args": {"contains": "ready"}}
        read = {"operation": "get_text", "args": {"selector": "#result"}}
        outer = {"operation": "sequence", "args": {"steps": [read, cached]}}
        with (patch.object(chrome, "target_reason", return_value=""),
              patch("ouroboros.tool_access.active_tool_profile", return_value="local_readonly_subagent"),
              patch("ouroboros.tools.core.is_restricted_subagent_profile", return_value=True)):
            self.assertEqual(chrome.action_reason(None, "browser_action", outer, state()), "")
            outer["args"]["steps"].append({"operation": "click", "args": {}})
            self.assertIn("READONLY_BLOCKED", chrome.action_reason(None, "browser_action", outer, state()))
        for invalid in ({"steps": []}, {"steps": [cached] * 33},
                        {"steps": [{"operation": "sequence", "args": {"steps": [read]}}]},
                        {"steps": [read], "code": "arbitrary"}):
            self.assertIn("1–32", chrome.action_reason(
                None, "browser_action", {"operation": "sequence", "args": invalid}, state()))


if __name__ == "__main__":
    unittest.main()
