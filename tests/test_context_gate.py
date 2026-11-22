"""The context PR exception stays narrow, keyless and uses the local gate's list."""
from fnmatch import fnmatchcase
from pathlib import Path

import pytest
import yaml

from scripts import context_gate as gate

ROOT = Path(__file__).resolve().parents[1]


def test_workflow_filters_cover_the_lane_and_request_builders():
    text = (ROOT / ".github/workflows/context-gate.yml").read_text()
    workflow = yaml.load(text, Loader=yaml.BaseLoader)
    assert set(workflow["on"]) == {"pull_request", "workflow_dispatch"}
    paths = workflow["on"]["pull_request"]["paths"]
    for path in (*(node.split("::", 1)[0] for node in gate.TESTS), "ouroboros/context.py", "ouroboros/llm_helper_window.py",
                 "ouroboros/tools/registry.py", "ouroboros/physical_attempt_context.py",
                 "ouroboros/reviewer_window.py", "ouroboros/review_native_episode.py",
                 "ouroboros/server_window.py", "ouroboros/_usage_response.py", "prompts/SYSTEM.md", "BIBLE.md",
                 "docs/architecture/06-agent-core.md", "docs/development/14-build-and-ci.md",
                 "tests/fixtures/size_refusals/manifest.json", "tests/_window_history.py",
                 "tests/_first_contact_server.py", "scripts/context_gate.py", "ouroboros/deployment_policy.py",
                 "ouroboros/gateway/settings.py", "ouroboros/pricing.py", "tests/fixtures/llm_golden/aux_routes.json",
                 "docs/inventories/FACADE_INVENTORY.md",
                 # the call start's seam (tests/test_live_call_start_fit.py): the agent hands nano_base to the fit
                 "ouroboros/agent.py", "ouroboros/live_conversation.py"):
        assert any(fnmatchcase(path, pattern) for pattern in paths), path
    assert not any(fnmatchcase("web/styles.css", pattern) for pattern in paths)
    assert workflow["permissions"] == {"contents": "read"}
    jobs = list(workflow["jobs"].values())
    assert len(jobs) == 1 and jobs[0]["runs-on"] == "ubuntu-latest"
    assert int(jobs[0]["timeout-minutes"]) == 12
    assert "secrets." not in text and "persist-credentials: false" in text
    assert "python scripts/context_gate.py" in text
    assert "--workers 2" in text
    general = yaml.load((ROOT / ".github/workflows/ci.yml").read_text(), Loader=yaml.BaseLoader)
    assert "pull_request" not in general["on"] and set(general["on"]["push"]) == {"tags"}


def test_local_gate_preserves_failure_exit_and_uses_existing_retry(tmp_path, monkeypatch):
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return 17
    monkeypatch.setattr(gate.subprocess, "call", run)
    assert gate.main(["--evidence-dir", str(tmp_path / "evidence")]) == 17
    command, options = calls[0]
    assert options["cwd"] == ROOT and str(ROOT / "scripts/ci_retry.py") in command
    assert [part for part in command if part.startswith("tests/")] == list(gate.TESTS)
    parallel = gate.command(ROOT, tmp_path, workers=2)
    assert parallel[parallel.index("-n") + 1] == "2"
    assert parallel[parallel.index("--dist") + 1] == "loadgroup"
    monkeypatch.setattr(gate, "TESTS", ("tests/test_context_gate.py::test_context_gate_has_recovery_homes",))
    assert gate.TESTS[0] in gate.command(ROOT, tmp_path)
    monkeypatch.setattr(gate, "TESTS", ())
    with pytest.raises(ValueError, match="empty or missing"):
        gate.command(ROOT, tmp_path)


def test_context_gate_has_recovery_homes():
    assert {
        "tests/test_window_projection.py", "tests/test_required_development_book.py",
        "tests/test_context_economy_window.py::test_a_chosen_read_on_an_unknown_window_stays_whole_after_an_unmeasured_engine_refusal",
        "tests/test_context_economy_window.py::test_an_account_reprepare_that_selects_nano_narrows_the_shown_low_view_and_keeps_its_provenance",
        "tests/test_context_economy_window.py::test_a_fallback_and_its_return_resend_the_shown_history_byte_for_byte_on_the_wire",
        "tests/test_context_refusal_passes.py::test_a_refused_first_send_retries_the_view_its_route_rebind_narrowed",
    }.issubset(gate.TESTS)


def test_cli_acquires_preimport_owner_before_running_gate(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(gate.os, "execv", lambda *args: calls.append(args))
    monkeypatch.setattr(gate.subprocess, "call", lambda command: calls.append((command[0], command)))
    args = ["--temp-parent", str(tmp_path), "--evidence-dir", str(tmp_path / "evidence")]
    gate.cli(args)
    executable, command = calls[0]
    assert command[:4] == [executable, "-I", "-S", str(ROOT / "scripts/safe_test.py")]
    assert command[4:8] == args
    bootstrap = command[command.index("-c") + 1]
    assert "run_path" in bootstrap and "['main']" in bootstrap
    assert command[-2:] == args[-2:]
