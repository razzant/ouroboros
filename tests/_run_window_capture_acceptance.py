"""Run the opt-in disposable-Mac acceptance through scripts/safe_test.py.

Pass this command to scripts/safe_test.py on a disposable runner with a clean,
detached checkout of e758a2190298d9890b1f26e632f63775d0b33d55 at OLD_CORE::

    python -m tests._run_window_capture_acceptance --disposable-desktop --rollback-checkout OLD_CORE REPORT

REPORT must be a new file. Exit 0 requires every case, 1 indicates failure,
and 77 a blocked or incomplete native run with passing units and rollback.
"""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

UNITS = (
    "tests/test_unix_computer_use_window_capture.py",
    "tests/test_unix_computer_use_window_session.py",
    "tests/test_unix_computer_use_window_session_server.py",
    "tests/test_skill_mcp_contract_errors.py",
    "tests/test_skill_mcp_readiness.py",
)
RESEED = "tests/test_builtin_skill_source.py::test_call_skill_release_reseeds_the_old_payload_but_preserves_owner_edits"
SEED = "unix_computer_use-e471e380e6de1e7f728e0fffbb4457667422ebf4-0.6.0"
UNIT_CASES = (
    "tests/test_unix_computer_use_remote_backends.py::test_remote_screenshot_result_confined_and_transform",
    "tests/test_unix_computer_use_targeting.py::test_screenshot_transform_uses_display_bounds_and_the_target_display",
    "tests/test_unix_computer_use_skill.py::test_observation_transform_remaps_click_coordinates",
    "tests/test_skill_mcp.py::test_agent_python_binding_and_start_refuse_a_managed_domain_for_script_skill",
    "tests/test_skill_mcp.py::test_unknown_mcp_runtime_never_becomes_a_container_command",
    f"{RESEED}[False-{SEED}]", f"{RESEED}[True-{SEED}]",
)
SESSION_DECLARATION = ("tests/test_unix_computer_use_window_session_server.py"
                       "::test_the_manifest_declares_the_server_for_a_task_session")
ROLLBACK_MODULE = "tests/qualify_unix_computer_use_rollback.py"
ROLLBACK = {
    "rollback_never_connected": f"{ROLLBACK_MODULE}::test_native_skill_across_rollback[never-connected]",
    "rollback_saved_connection": f"{ROLLBACK_MODULE}::test_native_skill_across_rollback[saved-connection]",
}
SCENES = "tests/test_unix_computer_use_stream_native.py::"
MANAGED = "tests/system_e2e/test_window_capture_task_native.py::test_managed_window_task_lifetime"
NATIVE = {
    **{scene: SCENES + f"test_owned_occluded_window_through_connect_and_task_session[{scene}]"
       for scene in ("occluded", "independent_lifecycle")},
    "stdin_eof": SCENES + "test_active_capture_loses_stdin",
    **{ending: f"{MANAGED}[{ending}]" for ending in ("finish", "immediate_stop", "window_gone")},
}
IDENTITY = {"server_pid", "server_birth_s", "server_birth_us"}
COLOR = {"stream_target_pixels", "stream_occluder_pixels"}
DISCOVERY = {"discovery_processes", "discovery_tool_calls"}
IDLE_STOP = {"window_stop_confirmed", "idle_after_loss_ms", "terminal_roundtrip_ms", "stop_age_ms"}
SCENE_FACTS = IDENTITY | COLOR | DISCOVERY | IDLE_STOP | {
    "one_shot_target_pixels", "one_shot_occluder_pixels", "one_shot_overlap_pixels", "before_close_target_pixels",
    "before_close_occluder_pixels", "repaint_seq_delta", "overlap_points2",
    "transport_server_gone", "release_confirmed",
}
TASK_FACTS = IDENTITY | DISCOVERY | {
    "worker_pid", "worker_birth_s", "worker_birth_us", "model_target_pixels",
    "model_occluder_pixels", "frame_seq", "target_app_alive", "target_window_visible", "server_gone", "fresh_session_idle",
}
REQUIRED = {
    "occluded": SCENE_FACTS | {"display_target_pixels", "display_occluder_pixels", "display_overlap_pixels", "display_mapping_preserved"},
    "independent_lifecycle": SCENE_FACTS,
    "stdin_eof": IDENTITY | COLOR | {"eof_exit"},
    "finish": TASK_FACTS | {"task_completed"},
    "immediate_stop": TASK_FACTS | {"stop_http", "worker_gone", "task_cancelled"},
    "window_gone": TASK_FACTS | IDLE_STOP | {"task_completed"},
}


class Outcomes:
    """Retain phase outcomes and numeric fixture measurements only."""

    def __init__(self):
        self.collected = []
        self.phases = defaultdict(dict)
        self.facts = defaultdict(dict)
        self.denied = set()

    def pytest_collection_finish(self, session):
        self.collected = [item.nodeid for item in session.items]

    def pytest_runtest_setup(self, item):
        if self.denied and item.nodeid in NATIVE.values():
            import pytest

            pytest.skip("NOT_RUN_SCREEN_RECORDING")

    def pytest_runtest_logreport(self, report):
        node = report.nodeid
        self.phases[node][report.when] = "failed" if hasattr(report, "wasxfail") else report.outcome
        if report.skipped and isinstance(report.longrepr, tuple):
            if str(report.longrepr[-1]).endswith("NOT_RUN_SCREEN_RECORDING"):
                self.denied.add(node)
        for key, value in report.user_properties:
            if type(value) is int and 0 <= value < 10**16:
                self.facts[node][key] = value

    def status(self, node):
        phases = self.phases[node]
        if "failed" in phases.values():
            return "FAIL"
        if node in self.denied or "skipped" in phases.values() or not phases:
            return "NOT_RUN"
        return "PASS" if phases == dict.fromkeys(("setup", "call", "teardown"), "passed") else "INCOMPLETE"


def eligible_desktop(report):
    # This SAME interpreter will enter pytest. Each native fixture rechecks
    # before opening windows. Both production ScreenCaptureKit paths also
    # preflight inside their own process before asking for content.
    permission, ready = "UNAVAILABLE", False
    if sys.platform == "darwin":
        try:
            import Quartz

            granted = bool(Quartz.CGPreflightScreenCaptureAccess())
            permission = "GRANTED" if granted else "DENIED"
            if granted:
                from tests._window_capture_observer import desktop_available

                ready = desktop_available()
        except Exception:
            # Unreadable native capability blocks capture. The portable cases still run.
            ready = False
    print("permission", permission, f"desktop_ready={int(ready)}", file=report, flush=True)
    return ready


def run(report, rollback_checkout):
    eligible = eligible_desktop(report)
    os.environ["OUROBOROS_MACOS_STREAM_QA"] = "1" if eligible else "0"
    os.environ["OUROBOROS_E2E_DEEP"] = "mock"
    os.environ["OUROBOROS_ROLLBACK_CHECKOUT"] = str(rollback_checkout.resolve())
    import pytest

    observed = Outcomes()
    nodes = [*UNITS, *UNIT_CASES]
    # A missing composed rollback fixture must fail its two named cases while
    # still allowing the other independent checks to return their outcomes.
    if Path(ROLLBACK_MODULE).is_file():
        nodes.extend(ROLLBACK.values())
    if eligible:
        nodes.extend(NATIVE.values())
    code = pytest.main(["-o", "addopts=", "--tb=no", "--show-capture=no", "--disable-warnings",
                        "-q", "-p", "no:cacheprovider", *nodes], plugins=[observed])
    unit_nodes = [n for n in observed.collected if n.split("::", 1)[0] in UNITS or n in UNIT_CASES]
    passed = sum(observed.status(n) == "PASS" for n in unit_nodes)
    unit_ok = (passed == len(unit_nodes) and {*UNIT_CASES, SESSION_DECLARATION} <= set(unit_nodes)
               and set(UNITS) <= {n.split("::", 1)[0] for n in unit_nodes})
    print("unit", "PASS" if unit_ok else "FAIL", f"passed={passed}", f"collected={len(unit_nodes)}", file=report)
    rollback_passed = 0
    for label, node in ROLLBACK.items():
        count = observed.collected.count(node)
        ok = count == 1 and observed.status(node) == "PASS"
        skipped = "skipped" in observed.phases[node].values()
        print(label, "PASS" if ok else "FAIL", f"collected={count}", f"skipped={int(skipped)}", file=report)
        rollback_passed += int(ok)
    rollback_failed = len(ROLLBACK) - rollback_passed
    print("rollback", "FAIL" if rollback_failed else "PASS", f"passed={rollback_passed}",
          f"failed={rollback_failed}", f"expected={len(ROLLBACK)}", file=report)
    statuses = []
    for label, node in NATIVE.items():
        result = observed.status(node)
        facts = observed.facts[node]
        keys = REQUIRED[label]
        if result == "PASS" and (observed.collected.count(node) != 1 or not keys <= facts.keys()):
            result = "INCOMPLETE"
        print(label, result, *(f"{k}={facts[k]}" for k in sorted(keys & facts.keys())), file=report)
        statuses.append(result)
    native = ("FAIL" if "FAIL" in statuses else "PASS" if all(s == "PASS" for s in statuses)
              else "NOT_RUN" if all(s == "NOT_RUN" for s in statuses) else "INCOMPLETE")
    print("native", native, f"passed={statuses.count('PASS')}", f"expected={len(NATIVE)}",
          f"pytest_exit={int(code)}", file=report, flush=True)
    if not unit_ok or rollback_failed or code or native == "FAIL":
        return 1
    return 0 if native == "PASS" else 77


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disposable-desktop", action="store_true", required=True)
    parser.add_argument("--rollback-checkout", type=Path, required=True)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    with args.report.open("x", encoding="ascii") as report:
        # An SDK or a failing assertion can write a full frame, including via
        # a child's inherited descriptors. Only the separate report is public.
        with open(os.devnull, "wb") as sink:
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
        return run(report, args.rollback_checkout)


if __name__ == "__main__":
    raise SystemExit(main())
