"""The live stand's SK1 probe plugin presents the host token only to the loopback Host Service
(docs/CHECKLISTS.md skill item 12, host_token_handling): a base URL naming any other host or
scheme is refused before a request is built. Pinned after the rc.15 paid stand, where the skill
review blocked the stand's own plugin for reading HOST_SERVICE_URL unvalidated. The SK1 review criterion
is the product gate (owner decision 2026-09-06): executable review, with the clean/all-PASS state a fact."""
from __future__ import annotations

import types

import pytest

from devtools.e2e_live import scenarios


class _Token:
    def use_in_request(self) -> str:
        return "token"


class _Api:
    def __init__(self) -> None:
        self.tools: dict = {}

    def get_skill_token(self) -> _Token:
        return _Token()

    def register_tool(self, name, fn, **_kwargs) -> None:
        self.tools[name] = fn


def _echo_tool():
    module = types.ModuleType("sk1_plugin_probe")
    exec(compile(scenarios.SK1_PLUGIN, "plugin.py", "exec"), module.__dict__)
    api = _Api()
    module.register(api)
    return api.tools["echo"]


@pytest.mark.parametrize("base", ["http://evil.example.com:8767", "https://127.0.0.1:8767", "http://127.0.0.1:8767/proxy"])
def test_the_probe_refuses_a_non_loopback_host_service_base_before_any_request(monkeypatch, base):
    monkeypatch.setenv("HOST_SERVICE_URL", base)
    with pytest.raises(RuntimeError, match="loopback http URL"):
        _echo_tool()(None, "x")


def test_the_probe_accepts_the_loopback_base_and_only_then_reaches_the_transport(monkeypatch):
    monkeypatch.setenv("HOST_SERVICE_URL", "http://127.0.0.1:1")   # loopback, nothing listening
    with pytest.raises(Exception) as excinfo:
        _echo_tool()(None, "x")
    assert not isinstance(excinfo.value, RuntimeError) or "loopback" not in str(excinfo.value)


_FINDINGS = [{"item": "manifest_schema", "verdict": "PASS"}, {"item": "bug_hunting", "verdict": "FAIL"}]
_CLEAN = [{"item": "a", "verdict": "PASS"}]


@pytest.mark.parametrize("status,http,body_exec,executable,findings,expected", [
    ("clean", 200, True, True, _CLEAN, True),
    ("warnings", 200, True, True, _FINDINGS, True),        # rc.15 SK1_a2: warnings are executable
    ("blockers", 200, True, True, _FINDINGS, True),        # rc.15 SK1_a3: blockers executable under advisory
    ("blockers", 200, False, False, _FINDINGS, False),     # the same review under blocking enforcement
    ("pending", 200, False, True, _CLEAN, False),          # duplicate job: the call's own gate says no
    ("clean", 500, True, True, _CLEAN, False),             # the review call itself failed
    ("clean", 200, True, True, [], False),                 # no recorded findings: no review actually ran
    ("clean", 200, True, None, _CLEAN, False),             # the /api/extensions row carries no gate fact
])
def test_sk1_review_verdict_is_the_product_gate_and_records_the_clean_state_as_a_fact(status, http, body_exec,
                                                                                       executable, findings, expected):
    review = {"status": http, "body": {"status": status, "executable_review": body_exec}}
    entry = {"executable_review": executable,
             "review_gate": {"review_enforcement": "advisory", "blocking_reason": "x"}}
    ok, facts = scenarios.sk1_review_gate(review, entry, findings)
    assert ok is expected
    failed = [f["item"] for f in findings if f["verdict"] != "PASS"]
    assert facts == {"review_status": status, "review_executable": executable, "review_body_executable": body_exec,
                     "review_enforcement": "advisory", "review_blocking_reason": "x", "findings": len(findings),
                     "findings_failed": failed, "review_clean": bool(findings) and not failed}


def test_the_product_gate_the_stand_relies_on_executes_warnings_and_advisory_blockers_only():
    """The contract behind ``sk1_review_gate``: ``skill_review_gate`` (the same rule the ``/api/extensions``
    row and the review call project as ``executable_review``)."""
    from ouroboros.skill_review_status import skill_review_gate

    assert skill_review_gate("clean", enforcement="blocking")["executable_review"] is True
    assert skill_review_gate("warnings", enforcement="blocking")["executable_review"] is True
    assert skill_review_gate("blockers", enforcement="advisory")["executable_review"] is True
    assert skill_review_gate("blockers", enforcement="blocking")["executable_review"] is False
    assert skill_review_gate("pending", enforcement="advisory")["executable_review"] is False


def test_only_the_absorbing_scenario_promotes_post_task_evolution():
    """Under ``--self-mod`` SW1/SK1 would otherwise promote from their own roots and their one-shot cycle could
    re-exec the server inside the lifecycle under test; the lane overrides pin post-task evolution off for
    every scenario that commits nothing, and leave SM1 (``expects_absorb``) to the run-level setting."""
    for sid, row in scenarios.SCENARIOS.items():
        applied = row.overrides("stub/model").get("OUROBOROS_POST_TASK_EVOLUTION")
        assert applied == (None if row.expects_absorb else "false"), (sid, applied)
