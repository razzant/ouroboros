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
                 "docs/inventories/UI_CONTROL_TEXT_INVENTORY.md", "docs/architecture/11-frozen-contracts-v1.md",
                 "ouroboros/code_intelligence_architecture.py", "scripts/regenerate_inventories.py",
                 # the call start's seam (tests/test_live_call_start_fit.py): the agent hands nano_base to the fit
                 "ouroboros/agent.py", "ouroboros/live_conversation.py",
                 "ouroboros/agent_dispatch.py", "ouroboros/send_clock.py", "ouroboros/cold_return.py",
                 "ouroboros/server_owner_routing.py", "supervisor/worker_chat_lane.py", "supervisor/budget_resume.py",
                 "ouroboros/budget_pause.py", "ouroboros/work_stop.py", "supervisor/queue_transitions.py",
                 "tests/_budget_pause_exact_helpers.py",
                 "ouroboros/nano_minimum.py", "ouroboros/nano_host_publication.py", "ouroboros/nano_activation.py",
                 "ouroboros/knowledge_publication.py", "ouroboros/chronicle_publication.py",
                 "ouroboros/gateway/owner_settings.py", "ouroboros/gateway/safety_settings.py",
                 "ouroboros/skill_catalogue.py", "ouroboros/skill_lifecycle_actions.py", "ouroboros/settings_integrity.py",
                 "ouroboros/skill_acceptance.py", "ouroboros/skill_review.py", "ouroboros/skill_review_runner.py",
                 "ouroboros/skill_source_install.py",
                 "ouroboros/llm_ollama.py", "ouroboros/llm_ollama_wire.py", "ouroboros/llm_ollama_policy.py",
                 "ouroboros/llm_ollama_guard.py", "tests/_ollama.py", "tests/test_local_tool_parsing.py"):
        assert any(fnmatchcase(path, pattern) for pattern in paths), path
    assert not any(fnmatchcase("web/styles.css", pattern) for pattern in paths)
    for overlay_only in ("tests/test_cold_return_window_park_browser.py", "tests/system_e2e/test_cold_return_window_park.py"):
        assert not any(fnmatchcase(overlay_only, pattern) for pattern in paths), overlay_only
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
        "tests/test_local_tool_parsing.py", "tests/test_llm_ollama.py", "tests/test_llm_ollama_wire.py",
        "tests/test_llm_ollama_policy.py", "tests/test_llm_ollama_refusal.py", "tests/test_llm_ollama_composition.py",
        "tests/test_llm_ollama_gateway.py", "tests/test_window_projection.py", "tests/test_required_development_book.py",
        "tests/test_context_economy_window.py::test_a_chosen_read_on_an_unknown_window_stays_whole_after_an_unmeasured_engine_refusal",
        "tests/test_context_economy_window.py::test_an_account_reprepare_that_selects_nano_narrows_the_shown_low_view_and_keeps_its_provenance",
        "tests/test_context_economy_window.py::test_a_fallback_and_its_return_resend_the_shown_history_byte_for_byte_on_the_wire",
        "tests/test_context_refusal_passes.py::test_a_refused_first_send_retries_the_view_its_route_rebind_narrowed",
        "tests/test_local_input_measurement.py::test_finalize_measures_the_provider_clean_candidate_not_the_caller_payload",
        "tests/test_local_input_measurement.py::test_launched_n_ctx_reads_this_process_launch_argument",
        "tests/test_nano_root_frame.py",
        "tests/test_nano_startup_exact.py", "tests/test_nano_web_main_entry.py", "tests/test_cold_return.py",
        "tests/test_send_clock.py::test_only_a_main_loop_opts_in_and_the_zone_is_the_validated_sender_zone",
        "tests/test_send_clock.py::test_the_local_lane_measures_and_sends_the_same_single_line",
        "tests/test_nano_activation_boundaries.py::test_worker_enforces_saturated_ordinary_root_and_retains_proposal",
        "tests/test_nano_activation_boundaries.py::test_never_confirmed_editing_and_previously_confirmed_outage_are_different",
        "tests/test_nano_activation_boundaries.py::test_instruction_update_checks_candidate_before_replacing_active_bytes",
        "tests/test_nano_activation_boundaries.py::test_pending_instruction_review_rechecks_before_projecting_its_result",
        "tests/test_nano_minimum.py::test_two_memory_writers_cannot_spend_the_same_remaining_room",
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
