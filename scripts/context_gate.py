#!/usr/bin/env python3
"""Keyless context PR gate; the same file list and retry policy run locally and in CI.

``context-gate.yml`` selects request/tool/prompt/book changes and their test owners.
No model calls or provider secrets are needed. TESTS accepts modules and node IDs:
provider refusals, serving windows, authority histories, whole-input and recovery
contracts, helper/reviewer budgets, first contact, and generated inventories.
The CLI acquires ``safe_test`` before importing pytest; ``ci_retry`` owns retries and
outcome evidence. CI uses ``--workers 2`` (xdist loadgroup) inside its 12-minute job;
local callers may select 0–6 workers and a fresh ``--evidence-dir`` outside Git.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Add a slice's test module here once; neither CI nor documentation duplicates this list.
TESTS = (
    "tests/test_llm_extraction.py",
    "tests/test_nano_local_exact.py",
    "tests/test_local_context_evidence.py",
    "tests/test_llm_provider_golden.py",
    "tests/test_size_refusal.py",
    "tests/test_capability_route.py",
    "tests/test_server_window_readers.py",
    "tests/test_window_authority.py",
    "tests/test_context_window.py",
    "tests/test_window_projection.py",
    "tests/test_required_development_book.py",
    "tests/test_context_economy_window.py::test_a_chosen_read_on_an_unknown_window_stays_whole_after_an_unmeasured_engine_refusal",
    "tests/test_context_economy_window.py::test_an_account_reprepare_that_selects_nano_narrows_the_shown_low_view_and_keeps_its_provenance",
    "tests/test_context_economy_window.py::test_a_fallback_and_its_return_resend_the_shown_history_byte_for_byte_on_the_wire",
    "tests/test_context_refusal_passes.py::test_a_refused_first_send_retries_the_view_its_route_rebind_narrowed",
    "tests/test_known_route_reply_budget.py",
    "tests/test_first_contact_local_server.py",
    "tests/test_owner_continue.py",
    "tests/test_nano_tool_policy.py",
    "tests/test_completion_hold_observation.py",
    "tests/test_finish_silent_wake.py",
    "tests/test_result_review_handback.py",
    "tests/test_llm_provider_routing.py",
    "tests/test_live_call_start_fit.py",
    "tests/test_context_gate.py",
    "tests/test_image_refusal_retry.py",
    "tests/test_cross_model_reasoning_strip.py",
    "tests/test_generated_inventories.py",
)


def command(root: Path, evidence: Path, workers: int = 0) -> list[str]:
    if not TESTS or any(not (root / path.split("::", 1)[0]).is_file() for path in TESTS):
        raise ValueError("context gate has an empty or missing test module")
    return [sys.executable, "-I", "-S", str(root / "scripts/ci_retry.py"),
            "--evidence-dir", str(evidence), "--", sys.executable, "-m", "pytest",
            "-o", "addopts=", "-p", "no:cacheprovider", "-q",
            *(["-n", str(workers), "--dist", "loadgroup"] if workers else []), *TESTS,
            "--ci-evidence-dir", str(evidence)]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, help="fresh directory outside the checkout")
    parser.add_argument("--workers", type=int, choices=range(7), default=0,
                        help="xdist workers (0 runs serially, at most 6)")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    evidence = (args.evidence_dir or Path(tempfile.mkdtemp(prefix="ouroboros-context-gate-"))).resolve()
    print(f"Context gate evidence: {evidence}", flush=True)
    return subprocess.call(command(root, evidence, args.workers), cwd=root)


def cli(argv=None):
    """Acquire the same pre-import filesystem/configuration owner as run_tests.py."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--temp-parent")
    parser.add_argument("--evidence-dir")
    args, remaining = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    root = Path(__file__).resolve().parents[1]
    options = (["--temp-parent", args.temp_parent] if args.temp_parent else [])
    evidence = ["--evidence-dir", args.evidence_dir] if args.evidence_dir else []
    bootstrap = "import runpy,sys; raise SystemExit(runpy.run_path(sys.argv[1])['main'](sys.argv[2:]))"
    owner = [sys.executable, "-I", "-S", str(root / "scripts/safe_test.py"), *options, *evidence, "--",
             sys.executable, "-c", bootstrap, str(Path(__file__).resolve()), *remaining, *evidence]
    if os.name == "nt":
        return subprocess.call(owner)
    os.execv(sys.executable, owner)


if __name__ == "__main__":
    sys.exit(cli())
