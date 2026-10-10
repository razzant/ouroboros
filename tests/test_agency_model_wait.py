"""Cyber advisory resource failures preserve the task and the physical call ledger."""
import json
import sys

import pytest

from ouroboros import safety
from ouroboros.model_wait import ModelWaitInterrupted, current_model_wait
from ouroboros.tools.registry import ToolContext, ToolRegistry
from tests.test_llm_claudexor import MODEL, ledger
from tests.test_model_wait import _refusal, live_wait, setup  # noqa: F401 - pytest fixtures

pytestmark = pytest.mark.serial


@pytest.mark.parametrize("mode", ["advanced", "cyber_pro"])
@pytest.mark.parametrize("code", ["auth_required", "subscription_window_exhausted", "credential_pool_exhausted"])
def test_resource_refusal_does_not_suspend_cyber_execution(live_wait, tmp_path, monkeypatch, mode, code):  # noqa: F811 - pytest fixture
    root, transport, client, controller, _events, _decide = live_wait
    refusal = _refusal(code)
    if code == "credential_pool_exhausted":
        refusal["problem"]["context"]["poolCause"] = "mixed"
    transport.results, transport.dispatch = [refusal], ["not_started"]
    waits = []

    def observed_wait(*args, **kwargs):
        waits.append((args, kwargs))
        raise ModelWaitInterrupted("fixture_stop_after_wait")

    monkeypatch.setattr(controller, "wait", observed_wait)
    factory = lambda: client
    factory.supports_response_format = lambda *args, **kwargs: False
    monkeypatch.setattr(safety, "LLMClient", factory)
    monkeypatch.setattr(safety, "get_light_model", lambda: MODEL)
    monkeypatch.setattr(safety, "_SAFETY_STORM_UNTIL", 0)
    monkeypatch.setattr(safety, "get_runtime_mode", lambda: mode)
    monkeypatch.setattr(safety, "get_safety_mode", lambda: "full")
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", mode)
    monkeypatch.setenv("OUROBOROS_SAFETY_MODE", "full")
    repo, work = tmp_path / "repo", tmp_path / "work"
    repo.mkdir()
    work.mkdir()
    ctx = ToolContext(repo_dir=repo, system_repo_dir=repo, drive_root=root, workspace_root=work,
                      workspace_mode="external", task_id="task-one")
    registry = ToolRegistry(repo_dir=repo, drive_root=root)
    registry.set_context(ctx)
    command = {"cmd": [sys.executable, "-c",
        "from pathlib import Path; p=Path('once.txt'); p.write_text(p.read_text()+'x' if p.exists() else 'x')"],
        "cwd": str(work), "outputs": ["once.txt"]}
    if mode == "advanced":
        with pytest.raises(ModelWaitInterrupted, match="fixture_stop_after_wait"):
            registry.execute_result("run_command", command)
        assert len(waits) == 1 and not (work / "once.txt").exists()
    else:
        result = registry.execute_result("run_command", command)
        assert result.status == "ok" and "SAFETY_ADVICE" in result.text
        assert (work / "once.txt").read_text() == "x" and not waits
        events = [json.loads(line) for line in (root / "logs/events.jsonl").read_text().splitlines()]
        advisory = next(row for row in events if row["type"] == "safety_advisory")
        assert advisory["assessment_allowed"] is False and code in advisory["assessment"]
    assert len(transport.accepted_operations) == 1
    rows = ledger(root)
    # The refusal's private evidence adds one metadata revision, not another call.
    assert [(row["state"], row["revision"]) for row in rows] == [("released", 4)]
    assert rows[0]["physical_failure"]["stage"] == "raised_exception"
    assert rows[0]["physical_failure"]["evidence_ref"]
    assert current_model_wait() is controller and not controller.closed
    assert "wait_for_resources" not in json.dumps(transport.uploads[0][0])


def test_explicit_local_safety_keeps_precedence_over_subscription_name(monkeypatch):
    monkeypatch.setenv("USE_LOCAL_LIGHT", "true")
    monkeypatch.setattr(safety, "get_light_model", lambda: MODEL)
    assert safety._resolve_safety_routing() == (True, False, None)
