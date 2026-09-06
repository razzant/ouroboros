"""Model-facing subagent contracts distinguish patch application from shared writes."""

from ouroboros.tools.control import get_tools as control_tools
from ouroboros.tools.subagent_integration import get_tools as integration_tools
from supervisor.events_subagent_admission import _compose_subagent_text


def test_schedule_and_integration_describe_the_three_write_surfaces():
    schedule = next(entry.schema for entry in control_tools() if entry.name == "schedule_subagent")
    description = schedule["description"]
    assert "self_worktree is an isolated git worktree" in description
    assert "external_workspace write directly to the SHARED" in description
    assert "without reapplying" in description
    assert "Harness-delegated work uses a private snapshot" in description
    assert "the project directory IS the deliverable" in description
    prop = schedule["parameters"]["properties"]["write_surface"]
    assert "native children write shared files directly" in prop["description"]
    integration = next(entry.schema for entry in integration_tools() if entry.name == "integrate_subagent_patch")
    assert "WITHOUT reapplying" in integration["description"]
    assert "commit_reviewed" in integration["description"]
    assert "verify shared external_workspace files" in integration["parameters"]["properties"]["decision"]["description"]
    prompt = _compose_subagent_text("Build a feature", role="builder", expected_output="working files",
                                    constraints="", context="", task_constraint={"mode": "acting_subagent", "surface": "external_workspace"})
    assert "shared external_workspace files are verified without reapplying" in prompt
