from __future__ import annotations

import json
import hashlib
import os
import subprocess
import zipfile
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.contributor_review_evidence import finalize_contributor_outcome
from scripts.run_external_review import (
    _REVIEW_SUBSTRATE_PATHS,
    _apply_contributor_landing_obligations,
    _apply_contributor_review_env,
    _assert_contributor_review_config,
    _classify_exit,
    _configured_openrouter_models,
    _contributor_execution_receipts,
    _contributor_proposal,
    _contributor_result,
    _freeze_contributor_slots,
    _openrouter_key_health,
    _openrouter_pool,
    _prepare_review_configuration,
    _record_actors,
    _record_outcome,
    _resolved_review_config,
    _review_evidence_and_cost,
    _select_healthy_openrouter_key,
    _write_contributor_packet,
)
from tests import _contributor_packet_shared as shared

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('delivery,refused', [('native', False), ('packet', True), ('', False)])
def test_contributor_config_roundtrip_keeps_direct_api_delivery(monkeypatch, delivery, refused):
    from scripts.run_external_review import _diff_size_refusal, _frozen_pool_catalog
    from ouroboros.reviewer_slot_config import review_pool_rows
    from tests.review_pool_rosters import pool_roster, pool_seat, set_review_pool

    set_review_pool(monkeypatch, pool_roster(pool_seat('t', 'openai/test', delivery=delivery or None)))
    resolved = _resolved_review_config()
    # A catalog api row's delivery is its own explicit fact; '' means native (F8).
    assert resolved['pool_slots'][0].get('delivery', '') == (delivery or 'native')
    monkeypatch.setenv('OUROBOROS_SUBAGENTS', _frozen_pool_catalog(resolved))
    assert review_pool_rows()[0].native_retrieval is (not refused)
    # I3-D1: the cap binds a packet seat on either lane; the operator lane is not refused on its own.
    for lane in (SimpleNamespace(contributor=True), SimpleNamespace(contributor=False)):
        assert _diff_size_refusal(lane, resolved, 101, 100) is refused
        assert _diff_size_refusal(lane, resolved, 100, 100) is False


def test_contributor_mixed_panel_keeps_packet_limit():
    from scripts.run_external_review import _diff_size_refusal

    rows = [{'route': {'kind': 'api_chat'}, 'delivery': 'native'},
            {'route': {'kind': 'agent_session'}},
            {'route': {'kind': 'api_chat'}, 'subagent_id': 'reader', 'delivery': 'native'}]  # F8: the fact, not the id
    args = SimpleNamespace(contributor=True)
    assert not _diff_size_refusal(args, {'pool_slots': rows}, 101, 100)
    assert _diff_size_refusal(args, {'pool_slots': rows + [{'route': {'kind': 'api_chat'}, 'delivery': 'packet'}]},
                              101, 100)


def test_contributor_trust_boundary_covers_functional_review_dependencies():
    from ouroboros.tools.review_helpers import CANONICAL_GOVERNANCE_DOCS

    assert set(CANONICAL_GOVERNANCE_DOCS) <= _REVIEW_SUBSTRATE_PATHS and "docs/DESIGN.md" in CANONICAL_GOVERNANCE_DOCS
    assert {
        "docs/ARCHITECTURE.md",
        "ouroboros/capability_evidence.py",
        "ouroboros/code_intelligence.py",
        "ouroboros/claudexor_daemon.py",
        "ouroboros/deadline_utils.py",
        "ouroboros/delegate_custody.py",
        "ouroboros/delegate_custody_usage.py",
        "ouroboros/delegate_output.py",
        "ouroboros/gateways/claudexor.py",
        "ouroboros/outcomes.py",
        "ouroboros/openrouter_attribution.py",
        "ouroboros/platform_layer.py",
        "ouroboros/pricing.py",
        # The v6.87.21 seam split moved route vocabulary, transport dispatch and
        # api_chat prompt rendering BELOW the substrate into review_execution.py;
        # a PR editing the route/executor seam there must still show in the
        # review_substrate_changed diagnostic, exactly as one editing
        # review_substrate.py does (XG-5R4.1).
        "ouroboros/review_execution.py",
        "ouroboros/review_actor_aggregation.py",
        "ouroboros/review_dispatch.py",
        "ouroboros/review_slot_cancel.py",
        "ouroboros/review_evidence.py",
        "ouroboros/reviewer_slot_config.py",
        "ouroboros/reviewer_window.py",
        "ouroboros/review_substrate.py",
        "ouroboros/review_state.py",
        "ouroboros/runtime_mode_policy.py",
        "ouroboros/usage_accounting.py",
        "ouroboros/utils.py",
        "ouroboros/tools/preflight_review.py",
        "ouroboros/tools/commit_gate.py",
        "ouroboros/tools/registry.py",
        "ouroboros/tools/release_sync.py",
        "ouroboros/tools/review_synthesis.py",
        "ouroboros/tools/review_binary_context.py",
        "ouroboros/tools/review_brief_coupling.py",
        "ouroboros/tools/scope_review_contract.py",
        "ouroboros/tools/scope_window.py",
        "ouroboros/subagents.py",
        "ouroboros/review_native_episode.py",
        "ouroboros/review_verdict_extraction.py",
        "ouroboros/review_execution_projection.py",
        "scripts/contributor_review_evidence.py",
        # The contributor lane's isolation leaf and the runtime leaves its pin,
        # default panel and run cap are read through.
        "ouroboros/review_model_routes.py",
        "ouroboros/review_run_isolation.py",
        "ouroboros/settings_integrity.py",
        "ouroboros/settings_setup_contract.py",
        "ouroboros/usage_admission.py",
    }.issubset(_REVIEW_SUBSTRATE_PATHS)


def test_external_review_script_is_a_wrapper_over_the_review_operation():
    import scripts.contributor_review_evidence as evidence
    import scripts.run_external_review as module

    source = Path("scripts/run_external_review.py").read_text(encoding="utf-8")
    assert "v6.10.0" not in source
    assert "Google Colab" not in source
    # Contributor lane: the review operation over the frozen base..head subject.
    assert "ouroboros.tools import review_change" in source
    assert 'root="system_repo", surface="change"' in source
    assert 'subject="base..head"' in source
    # Operator lane: the exact commit-gate dry-run, in the runtime's isolated
    # checkout of the staged index, with the named preflight's record and full
    # answer beside it (decision 3A; approval item 3).
    assert "_run_non_committing_review_cycle(" in source
    assert "preflight_reviewer=args.preflight_reviewer" in source
    assert '"preflight.json"' in source and '"preflight.txt"' in source
    assert "advisory.txt" not in source and "skip_advisory_review" not in source
    assert 'kind="index", surface="commit_gate"' in source
    assert "adaptive_quorum" not in source
    assert "aggregate_review_verdict" not in source
    # Neither lane materializes, replays or re-executes anything itself: the
    # runtime's isolated_checkout owns the worktree and the patch bytes.
    for retired in ("_run_on_trusted_base", '"worktree", "add"', "git apply", "sys.executable",
                    "operator_binding", "_handle_advisory_pre_review"):
        assert retired not in source, retired
    assert source.count("isolated_checkout(host_ctx, spec") == 2
    # The packet vocabulary has one owner; the wrapper's literals mirror it.
    assert module._CONTRIBUTOR_PROFILE == evidence.CONTRIBUTOR_PROFILE == "external_pr_readiness"
    assert module._EXIT_CLASS == evidence.EXIT_CLASS


def test_external_review_script_defaults_to_pro_mode():
    source = Path("scripts/run_external_review.py").read_text(encoding="utf-8")
    assert 'setdefault("OUROBOROS_RUNTIME_MODE", "pro")' in source


def test_the_operator_lane_sets_no_retired_diff_aware_knob():
    """The commit gate pays the suite on every diff (owner answer A, 2026-10-08), so nothing
    reads ``OUROBOROS_PREFLIGHT_DIFF_AWARE`` any more; the lane neither sets it nor leaves a
    mention for an operator to copy. The whole tree under test has no reader of the name."""
    name = "OUROBOROS_PREFLIGHT_" + "DIFF_AWARE"
    assert name not in Path("scripts/run_external_review.py").read_text(encoding="utf-8")
    readers = [
        path.relative_to(REPO_ROOT).as_posix()
        for folder in ("ouroboros", "supervisor", "scripts", "web", "docs", "prompts")
        for path in (REPO_ROOT / folder).rglob("*")
        if path.is_file() and path.suffix in {".py", ".js", ".md", ".json"}
        and name in path.read_text(encoding="utf-8", errors="ignore")
    ]
    assert readers == []


def test_external_review_script_resolves_models_and_efforts(monkeypatch):
    for key in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "CLOUDRU_FOUNDATION_MODELS_API_KEY",
        "GIGACHAT_CREDENTIALS",
        "GIGACHAT_USER",
        "GIGACHAT_PASSWORD",
        "OPENAI_BASE_URL",
        "OPENAI_COMPATIBLE_BASE_URL",
        "OUROBOROS_MODEL",
        "OUROBOROS_MODEL_LIGHT",
    ):
        monkeypatch.delenv(key, raising=False)
    from tests.review_pool_rosters import pool_roster, pool_seat, set_review_pool

    set_review_pool(monkeypatch, pool_roster(
        pool_seat("r1", "anthropic/claude-opus-4.8", effort="high"),
        pool_seat("r2", "google/gemini-3.5-flash", effort="high"),
        pool_seat("r3", "openai/gpt-5.5", delivery="native", effort="xhigh")))
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", "max")

    config = _resolved_review_config()

    assert config["pool_models"] == [
        "anthropic/claude-opus-4.8",
        "google/gemini-3.5-flash",
        "openai/gpt-5.5",
    ]
    assert config["pool_efforts"] == ["high", "high", "xhigh"]
    assert all(row["route"]["kind"] == "api_chat" for row in config["pool_slots"])
    assert [row["delivery"] for row in config["pool_slots"]] == ["packet", "packet", "native"]
    assert config["review_enforcement"] == "blocking"
    # v6.80.0: the scope-review floor key is gone; the operator line pins the context
    # mode instead, because that is now what decides scope-review applicability.
    assert config["context_mode"] == "max"


def _golden() -> dict:
    path = Path(__file__).parent / "fixtures" / "contributor_review_packet_golden.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _contributor_fakes(module, monkeypatch, repo: Path, drive: Path) -> None:
    """The golden capture's stand-ins: isolation, settings and panel are the
    fixture's, and no key is probed."""
    monkeypatch.setattr(module, "REPO", repo)
    monkeypatch.setattr(module, "isolate_review_data", lambda **_kwargs: shared.isolation_record(drive))
    monkeypatch.setattr(module, "_load_settings_into_env", lambda: None)
    monkeypatch.setattr(module, "_apply_contributor_review_env", lambda: None)
    monkeypatch.setattr(module, "_resolved_review_config",
                        lambda *, profile="production_commit_gate": json.loads(json.dumps(shared.GOLDEN_CONFIG)))
    monkeypatch.setattr(module, "_select_healthy_openrouter_key", lambda **_kwargs: None)
    # The gate's panel is the review pool: the golden's three seats as catalog rows.
    monkeypatch.setenv("OUROBOROS_SUBAGENTS", shared.golden_pool())


def _run_golden_contributor_review(tmp_path: Path, monkeypatch) -> SimpleNamespace:
    """The golden capture's invocation (see the golden's provenance) through the
    REAL review operation: only the paid seam (the review substrate) and the
    commit gate's hermetic test runner are stand-ins, answering as the golden's
    seats did."""
    import ouroboros.review_substrate as substrate
    import scripts.run_external_review as module
    from ouroboros.tools import review_change as operation_module
    from ouroboros.tools import review_helpers

    fixture = shared.init_installed_body(tmp_path)
    drive, output = (tmp_path / "drive").resolve(), (tmp_path / "out").resolve()
    host = tmp_path / "host-data"
    host.mkdir()
    (host / "settings.json").write_text('{"OUROBOROS_REVIEW_ENFORCEMENT": "advisory"}\n', encoding="utf-8")
    _contributor_fakes(module, monkeypatch, Path(fixture["repo"]), drive)
    monkeypatch.setattr(module, "DATA", host)
    monkeypatch.setenv("OUROBOROS_PRE_PUSH_TESTS", "1")
    briefs: list[dict] = []
    monkeypatch.setattr(substrate, "run_review_request", shared.golden_substrate(briefs))
    monkeypatch.setattr(review_helpers, "_run_review_preflight_tests", shared.passing_test_runner)
    calls: list[dict] = []
    operation, wrapper_running = operation_module.run_review_change, [False]

    def review_change(ctx, **arguments):
        wrapper_running[0] = False  # what the runtime operation spawns is the runtime's business
        try:
            result = operation(ctx, **arguments)
        finally:
            wrapper_running[0] = True
        calls.append({"arguments": dict(arguments), "repo_dir": str(ctx.repo_dir), "pid": os.getpid(),
                      "record_id": result.get("record_id"), "result": result})
        return result

    monkeypatch.setattr(module, "run_review_change", review_change)
    spawned: list[dict] = []
    popen = subprocess.Popen

    class RecordingPopen(popen):
        def __init__(self, args, *rest, **kwargs):
            if wrapper_running[0]:
                command = [str(part) for part in args] if isinstance(args, (list, tuple)) else [str(args)]
                env = kwargs.get("env")
                # Rewritten: an inherited variable changed, or a non-git variable added.
                spawned.append({"command": command, "env": env, "env_rewritten": env is not None and any(
                    os.environ[key] != value if key in os.environ else not key.startswith("GIT_")
                    for key, value in env.items())})
            super().__init__(args, *rest, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", RecordingPopen)
    monkeypatch.setattr(module.sys, "argv", [
        "run_external_review.py", "--contributor", "--base-ref=base", f"--head-ref={fixture['head_sha']}",
        f"--output={output}", f"--drive-root={drive}", "--run-cap-usd=5", "--attach-host-engine",
        "--", shared.PROPOSAL_TITLE,
    ])
    host_before = {path.name: path.read_bytes() for path in host.iterdir()}
    environ_before = dict(os.environ)
    wrapper_running[0] = True
    try:
        exit_code = module.main()
    finally:
        wrapper_running[0] = False
        for key in set(os.environ) - set(environ_before):
            del os.environ[key]
        os.environ.update(environ_before)
    return SimpleNamespace(fixture=fixture, output=output, calls=calls, spawned=spawned, exit_code=exit_code,
                           host=host, host_before=host_before, briefs=briefs, drive=drive)


def _installed_rules_source() -> dict:
    """R5: the record's rules source is the git blob id of the executing install's
    checklist bytes, whatever tree that install is checked out at."""
    blob = subprocess.run(["git", "hash-object", "docs/CHECKLISTS.md"], cwd=str(REPO_ROOT),
                          capture_output=True, text=True, check=True).stdout.strip()
    return {"path": "docs/CHECKLISTS.md", "sha": blob}


def test_contributor_packet_is_the_pre_move_packet(tmp_path, monkeypatch):
    """The same proposal yields the same public packet before and after the
    review moved into the runtime operation; only the declared fields differ."""
    run = _run_golden_contributor_review(tmp_path, monkeypatch)
    golden = _golden()
    evidence = json.loads((run.output / "review-evidence.json").read_text(encoding="utf-8"))
    new, old = shared.normalized(evidence, run.fixture), json.loads(json.dumps(golden["evidence"]))

    assert run.exit_code == golden["exit_code"] == 0
    assert (old.pop("schema_version"), new.pop("schema_version")) == (2, 3)
    old_outcome, new_outcome = old.pop("production_outcome"), new.pop("production_outcome")
    assert {key: new_outcome[key] for key in old_outcome} == old_outcome
    assert new_outcome["aggregate"] == "PASS" and new_outcome["record_id"] == run.calls[0]["record_id"]
    old_trust, new_trust = old.pop("trust"), new.pop("trust")
    assert old_trust.pop("trusted_base_execution").startswith("The review ran on the target base's own machinery")
    installed = new_trust.pop("installed_body_execution")
    assert new_trust == old_trust
    review_record = new.pop("review_record")
    # Declared: the panel is ONE review pool now (PR-3). The pre-move wrapper described
    # the same three seats as a triad lane plus a scope lane; the pool describes them
    # as pool rows (the scope seat retrieving natively) and pins them as catalog rows.
    old_config, new_config = old.pop("review_config"), new.pop("review_config")
    lane_rows = [*old_config.pop("triad_slots"), *old_config.pop("scope_slots")]
    assert new_config.pop("pool_slots") == [
        {**lane_rows[0], "delivery": "packet"}, lane_rows[1], {**lane_rows[2], "delivery": "native"}]
    assert new_config.pop("pool_models") == old_config.pop("triad_models") + old_config.pop("scope_models")
    assert new_config.pop("pool_efforts") == old_config.pop("triad_efforts") + old_config.pop("scope_efforts")
    assert (old_config.pop("execution_slot_config_source"), new_config.pop("execution_slot_config_source")) == (
        "frozen_structured", "frozen_review_pool")
    assert (old_config.pop("slot_config_source"), new_config.pop("slot_config_source")) == ("settings", "review_pool")
    assert len(old_config.pop("slot_plan_sha256")) == len(new_config.pop("slot_plan_sha256")) == 64
    assert new_config == old_config
    assert (old["review_completeness"].pop("contract"), new["review_completeness"].pop("contract")) == (
        "production_triad_quorum_plus_authoritative_scope", "production_pool_quorum_plus_coupling")
    # ... and every receipt names the one pool surface; a seat's configured row carries
    # its explicit delivery (the pre-move lane rows left it implicit).
    lane_surfaces = {"triad", "scope"}

    def one_pool(value):
        if isinstance(value, dict):
            out = {key: one_pool(item) for key, item in value.items()}
            if out.get("surface") in lane_surfaces:
                out["surface"] = "pool"
            if "configured" in out and isinstance(out["configured"], dict) and "delivery" not in out["configured"]:
                out["configured"]["delivery"] = "native" if out["configured"].get("slot_id") == "s1" else "packet"
            return out
        if isinstance(value, list):
            return [one_pool(item) for item in value]
        return value

    old = one_pool(old)
    for receipt in [*old["review_execution"]["receipts"], *new["review_execution"]["receipts"]]:
        if receipt["configured"].get("route", {}).get("kind") == "agent_session":
            receipt["configured"].pop("delivery", None)
    assert new == old

    # The review ran the installed body's rules, not the proposal's relaxed
    # checklist: the rules source is the blob id of the executing install's
    # checklist bytes (R5), never the fixture's or the proposal's.
    rules_source = _installed_rules_source()
    assert installed["rules_source"] == rules_source
    assert rules_source["sha"] != hashlib.sha256(shared.BASE_FILES["docs/CHECKLISTS.md"].encode()).hexdigest()
    assert installed["executing_checkout_head"] == "<base_sha>"
    assert "installed body's review flow and rules" in installed["statement"]
    assert review_record["record_id"] == run.calls[0]["record_id"]
    assert (review_record["surface"], review_record["state"], review_record["aggregate"]) == (
        "change", "settled", "PASS")
    assert review_record["subject"]["root"] == "$REPO"
    assert {key: review_record["subject"][key] for key in ("kind", "base", "head", "tree_sha")} == {
        "kind": "base..head", "base": "<base_sha>", "head": "<head_sha>", "tree_sha": "<head_tree_sha>"}
    assert review_record["checklist"]["rules_source"] == rules_source
    assert review_record["checklist"]["layer"] == "body"
    # R2: the proposal's hermetic test preflight ran on the reviewed tree and is
    # attached to the record as the gate's own candidate-bound fact.
    assert review_record["tests"] == {"policy": "run", "result": "passed", "proof": "candidate_bound",
                                      "tree_sha": "<head_tree_sha>"}
    assert review_record["path"].startswith("$REVIEW_DRIVE")
    assert review_record["path"].endswith(f"{review_record['record_id']}.json")
    # Every seat was briefed on the frozen base..head subject through the body
    # layer; the retrieving seats read the frozen checkout, never the installed repo.
    assert sorted(brief["slot_id"] for brief in run.briefs) == ["s1", "t1", "t2"]
    checkouts = run.drive / "state" / "review_checkouts"
    for brief in run.briefs:
        text = "\n".join(str(message.get("content") or "") for message in brief["messages"]) + brief["session_task"]
        assert "Ouroboros Body Layer" in text and shared.PROPOSAL_TITLE in text, brief["slot_id"]
        if brief["slot_id"] in {"t2", "s1"}:
            assert Path(brief["session_root"]).parent.parent == checkouts, brief["session_root"]
        else:
            assert brief["session_root"] == ""

    full_output = (run.output / "full-output.txt").read_text(encoding="utf-8")
    sections = shared.full_output_sections(full_output)
    assert json.loads(sections["CONTRIBUTOR REVIEW EVIDENCE"]) == evidence
    transcripts = json.loads(sections["AGENT SESSION TRANSCRIPTS (full, redacted)"])
    assert shared.normalized(transcripts, run.fixture) == [
        {**row, "surface": "pool"} for row in golden["session_transcripts"]]  # one pool surface
    assert sorted(slot for slot, answer in shared.ANSWERS.items()
                  if answer in full_output or json.dumps(answer)[1:-1] in full_output
                  ) == golden["answered_slots_in_full_output"]
    seats = json.loads(sections["REVIEW POOL SEAT RECORDS (ledger rows with retained answers, full, redacted)"])
    assert [(seat["seat_id"], seat["answer"]) for seat in seats] == [
        (slot, shared.ANSWERS[slot]) for slot in ("t1", "t2", "s1")]
    with zipfile.ZipFile(run.output / "review-packet.zip") as archive:
        assert set(archive.namelist()) == {"review-evidence.json", "outcome.json", "full-output.txt"}


def test_contributor_review_runs_in_this_process_on_the_installed_body(tmp_path, monkeypatch):
    """D31: the installed body's review flow and rules run, in this process.

    The wrapper hands the review operation its own checkout (the target base
    here) with the frozen subject; it never re-executes a copy of itself, never
    hands a child process a rewritten environment, and never writes the host's
    data root."""
    run = _run_golden_contributor_review(tmp_path, monkeypatch)

    assert run.exit_code == 0
    [call] = run.calls
    assert call["pid"] == os.getpid()
    assert Path(call["repo_dir"]) == Path(run.fixture["repo"])
    assert {key: call["arguments"][key] for key in ("root", "surface", "subject", "base", "head")} == {
        "root": "system_repo", "surface": "change", "subject": "base..head",
        "base": run.fixture["base_sha"], "head": run.fixture["head_sha"]}
    assert f"External PR title: {shared.PROPOSAL_TITLE}" in call["arguments"]["goal"]
    assert run.spawned and all(item["command"][0] == "git" for item in run.spawned)
    # The runtime's git helpers may drop a variable (GIT_DIFF_OPTS); nothing is rewritten.
    assert not any(item["env_rewritten"] for item in run.spawned)
    assert {path.name: path.read_bytes() for path in run.host.iterdir()} == run.host_before
    repo = Path(run.fixture["repo"])
    assert shared.git(repo, "rev-parse", "HEAD") == run.fixture["base_sha"]
    assert shared.git(repo, "status", "--porcelain") == ""
    assert shared.git(repo, "worktree", "list", "--porcelain").count("worktree ") == 1


def test_contributor_lane_refuses_before_review_when_the_proposal_tests_fail(tmp_path, monkeypatch):
    """R2: the proposal's hermetic suite runs on an isolated checkout of the frozen
    base..head subject BEFORE any reviewer is paid; a failed suite is a typed
    refusal (exit 3, $0, no record, no checkout left behind)."""
    import ouroboros.review_substrate as substrate
    import scripts.run_external_review as module
    from ouroboros.tools import review_helpers

    fixture = shared.init_installed_body(tmp_path)
    repo, drive, output = Path(fixture["repo"]), (tmp_path / "drive").resolve(), (tmp_path / "out").resolve()
    _contributor_fakes(module, monkeypatch, repo, drive)
    monkeypatch.setenv("OUROBOROS_PRE_PUSH_TESTS", "1")
    monkeypatch.setattr(substrate, "run_review_request",
                        lambda *_args, **_kwargs: pytest.fail("no reviewer is dispatched after a failed preflight"))
    monkeypatch.setattr(module, "run_review_change",
                        lambda *_args, **_kwargs: pytest.fail("the review operation runs only after the tests pass"))
    tested: list[dict] = []

    def failing_test_runner(ctx, **_kwargs):
        tested.append({"repo_dir": Path(ctx.repo_dir), "tree": shared.git(Path(ctx.repo_dir), "write-tree")})
        ctx._preflight_tests_passed = False
        return "FAILED tests/test_change.py::test_proposal - assert 1 == 2"

    monkeypatch.setattr(review_helpers, "_run_review_preflight_tests", failing_test_runner)
    monkeypatch.setattr(module.sys, "argv", [
        "run_external_review.py", "--contributor", "--base-ref=base", f"--head-ref={fixture['head_sha']}",
        f"--output={output}", f"--drive-root={drive}", "--run-cap-usd=5", "--attach-host-engine",
        "--", shared.PROPOSAL_TITLE,
    ])

    exit_code = module.main()

    assert exit_code == 3
    [run] = tested
    assert run["tree"] == fixture["head_tree_sha"] and run["repo_dir"] != repo
    assert run["repo_dir"].parent.parent == drive / "state" / "review_checkouts"
    outcome = json.loads((output / "outcome.json").read_text(encoding="utf-8"))
    assert outcome["exit_code"] == 3 and outcome["outcome"]["block_reason"] == "tests_preflight_blocked"
    assert "FAILED tests/test_change.py::test_proposal" in outcome["outcome"]["message"]
    assert outcome["outcome"]["tested_tree_sha"] == fixture["head_tree_sha"]
    evidence = json.loads((output / "review-evidence.json").read_text(encoding="utf-8"))
    assert evidence["result"] == "INCOMPLETE"
    assert evidence["review_record"] == {"record_id": None, "available": False}
    assert evidence["review_execution"]["receipts"] == [] and evidence["review_execution"]["mismatches"] == []
    assert evidence["production_outcome"]["block_reason"] == "tests_preflight_blocked"
    assert evidence["cost_report"]["reported_actor_cost_usd"] == 0
    assert evidence["cost_report"]["reported_cost_slots"] == [] and evidence["raw_evidence_refs"] == []
    assert not list(drive.rglob("rl-*.json")), "no review record: nothing was dispatched"
    assert not (drive / "state" / "review_checkouts").exists() or not any(
        (drive / "state" / "review_checkouts").iterdir())
    assert shared.git(repo, "worktree", "list", "--porcelain").count("worktree ") == 1
    assert shared.git(repo, "rev-parse", "HEAD") == fixture["base_sha"]


def test_the_wrapper_run_from_the_proposal_refuses_before_any_review(tmp_path, monkeypatch, capsys):
    import scripts.run_external_review as module

    fixture = shared.init_installed_body(tmp_path)
    repo = Path(fixture["repo"])
    shared.git(repo, "checkout", "-q", "proposal")
    _contributor_fakes(module, monkeypatch, repo, (tmp_path / "drive").resolve())
    monkeypatch.setattr(module, "run_review_change",
                        lambda *_args, **_kwargs: pytest.fail("the proposal's own checkout must not review it"))
    monkeypatch.setattr(module.sys, "argv", [
        "run_external_review.py", "--contributor", "--base-ref=base", "--head-ref=proposal",
        f"--output={tmp_path / 'out'}", "--run-cap-usd=5", "--attach-host-engine", "--", "PR title",
    ])

    assert module.main() == 3
    assert "already contains proposal" in capsys.readouterr().err


def _record(aggregate: str, *, state: str = "settled", critical=(), degraded=(), subject=None) -> dict:
    return {"record_id": "r1", "state": state, "subject": subject or {"base": "b", "head": "h", "tree_sha": "t"},
            "verdict": {"aggregate": aggregate, "per_question": {"change": "PASS"},
                        "critical_findings": list(critical), "degraded_reasons": list(degraded)}}


def test_the_review_record_decides_the_typed_outcome_and_exit():
    bound = {"base": "b", "head": "h", "tree_sha": "t"}
    passed = _record_outcome(_record("PASS"), "", subject=bound)
    assert (passed["status"], passed["record_id"], _classify_exit(passed)) == ("passed", "r1", 0)
    finding = {"item": "code_quality", "severity": "critical"}
    failed = _record_outcome(_record("FAIL", critical=[finding]), "", subject=bound)
    assert (failed["block_reason"], failed["combined_findings"], _classify_exit(failed)) == (
        "critical_findings", [finding], 1)
    open_seat = _record_outcome(_record("NOT_PERFORMED", state="pending"), "", subject=bound)
    assert (open_seat["block_reason"], _classify_exit(open_seat)) == ("review_custody_unresolved", 3)
    for aggregate in ("QUORUM_FAILED", "NOT_PERFORMED", "NOT_DISPATCHED"):
        outcome = _record_outcome(_record(aggregate, degraded=["t2=timeout"]), "", subject=bound)
        assert (outcome["block_reason"], outcome["degraded_reasons"], _classify_exit(outcome)) == (
            f"review_{aggregate.lower()}", ["t2=timeout"], 3)
    missing = _record_outcome(None, "review ledger record r1 is absent")
    assert (missing["block_reason"], missing["message"], _classify_exit(missing)) == (
        "review_record_unavailable", "review ledger record r1 is absent", 3)
    # A record of another subject is never this proposal's verdict, whatever it says.
    drifted = _record_outcome(_record("PASS", subject={"base": "b", "head": "other", "tree_sha": "t"}),
                              "", subject=bound)
    assert (drifted["block_reason"], drifted["subject_mismatches"], _classify_exit(drifted)) == (
        "reviewed_subject_mismatch", ["head:h->other"], 3)
    # The operator lane binds no subject: the record's verdict alone decides.
    assert _record_outcome(_record("PASS", subject={"kind": "index"}), "")["status"] == "passed"


def _write_target_config(repo: Path) -> None:
    package = repo / "ouroboros"
    package.mkdir(exist_ok=True)
    (package / "config.py").write_text(
        "SETTINGS_DEFAULTS = {\n"
        "    'OUROBOROS_REVIEW_MODELS': 'anthropic/fable,openai/sol,google/flash',\n"
        "    'OUROBOROS_SCOPE_REVIEW_MODELS': 'anthropic/fable',\n"
        "    'OUROBOROS_EFFORT_REVIEW': 'high',\n"
        "    'OUROBOROS_EFFORT_SCOPE_REVIEW': 'high',\n"
        "}\n",
        encoding="utf-8",
    )


def _init_contributor_repo(tmp_path: Path, monkeypatch) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "core.autocrlf", "false")
    (repo / ".gitignore").write_text("__pycache__/\n*.pyc\n", encoding="utf-8")
    (repo / "scripts").mkdir()
    (repo / "scripts" / "run_external_review.py").write_text("# installed wrapper\n", encoding="utf-8")
    _write_target_config(repo)
    (repo / "ouroboros" / "review_substrate.py").write_text(
        "# installed review substrate\n", encoding="utf-8"
    )
    (repo / "ouroboros" / "utils.py").write_text(
        "# installed review utilities\n", encoding="utf-8"
    )
    (repo / "ouroboros" / "tools").mkdir()
    (repo / "ouroboros" / "tools" / "registry.py").write_text(
        "# installed review context\n", encoding="utf-8"
    )
    (repo / "pyproject.toml").write_text(
        '[project]\nname = "test-project"\nversion = "1.2.3"\n',
        encoding="utf-8",
    )
    (repo / "uv.lock").write_text(
        'version = 1\n\n[[package]]\nname = "ouroboros"\nversion = "1.2.3"\n'
        'source = { editable = "." }\n',
        encoding="utf-8",
    )
    (repo / "web" / "modules").mkdir(parents=True)
    (repo / "web" / "package.json").write_text(
        '{"name":"ouroboros-ui","version":"1.2.3"}\n', encoding="utf-8"
    )
    (repo / "web" / "modules" / "api_types.js").write_text(
        'export const GATEWAY_CONTRACT_VERSION = "1.2.3";\n', encoding="utf-8"
    )
    (repo / "README.md").write_text(
        "[![Version 1.2.3](https://example.test/version.svg)](#)\n\n"
        "[download-macos-arm64]: https://example.test/v1.2.3/Ouroboros-1.2.3.dmg\n\n"
        "## Version History\n\n| 1.2.3 | Current |\n",
        encoding="utf-8",
    )
    (repo / "docs").mkdir()
    (repo / "docs" / "ARCHITECTURE.md").write_text(
        "# Ouroboros v1.2.3\n", encoding="utf-8"
    )
    for root in (repo / "site" / "install", repo / "docs" / "install"):
        root.mkdir(parents=True)
        (root / "index.html").write_text(
            '<a href="https://example.test/v1.2.3/Ouroboros-1.2.3.dmg" '
            'data-release-download="macos-arm64">Download</a>\n'
            '<a data-release-download="macos-arm64" '
            'href="https://example.test/v1.2.3/Ouroboros-1.2.3.dmg">'
            'Quick start</a>\n',
            encoding="utf-8",
        )
    (repo / "VERSION").write_text("1.2.3\n", encoding="utf-8")
    (repo / "a.txt").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "base")
    _git(repo, "branch", "base")
    (repo / "a.txt").write_text("proposal\n", encoding="utf-8")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-m", "proposal")

    import scripts.run_external_review as module

    monkeypatch.setattr(module, "REPO", repo)
    return repo


def _proposal_from_installed_base(repo: Path) -> dict:
    """The committed proposal (HEAD) read from a clean checkout of its base."""
    head = _git(repo, "rev-parse", "HEAD").strip()
    _git(repo, "checkout", "-q", "--detach", "base")
    return _contributor_proposal("base", head)


def test_a_checkout_that_already_contains_the_proposal_is_refused(tmp_path, monkeypatch):
    """D31: the review runs this checkout's review flow and rules, so a checkout
    holding the proposal, or anything built on it, would review the proposal
    with its own copy. That is refused before anything is spent; the base
    checkout the contributor is pointed to passes."""
    repo = _init_contributor_repo(tmp_path, monkeypatch)
    head = _git(repo, "rev-parse", "HEAD").strip()
    with pytest.raises(RuntimeError, match="already contains HEAD"):
        _contributor_proposal("base", "HEAD")
    (repo / "later.txt").write_text("built on the proposal\n", encoding="utf-8")
    _git(repo, "add", "later.txt")
    _git(repo, "commit", "-m", "descendant")
    with pytest.raises(RuntimeError, match="git worktree add --detach <dir> base"):
        _contributor_proposal("base", head)

    _git(repo, "checkout", "-q", "--detach", "base")
    proposal = _contributor_proposal("base", head)
    assert proposal["installed_head_sha"] == proposal["base_sha"] != proposal["head_sha"] == head


def test_contributor_result_is_decided_by_the_exit_code_alone():
    """The retired D31 classifier is not a gate anywhere in the outcome path.

    A proposal rewriting the review machinery gets the same result vocabulary as
    any other, because nothing but the review's exit code reaches this decision.
    """
    assert _contributor_result(0) == "READY_FOR_INTEGRATION"
    assert _contributor_result(1) == "BLOCKED"
    assert _contributor_result(3) == "INCOMPLETE"


def test_contributor_policy_preserves_configured_routes(monkeypatch):
    from tests.review_pool_rosters import pool_roster, pool_seat, set_review_pool

    raw = pool_roster(
        pool_seat("session", "codex=gpt-5.6-sol", kind="agent_session", profile_id="account-a", effort="high"),
        pool_seat("direct", "anthropic::claude-fable-5", effort="xhigh"),
        pool_seat("reader", "openai/gpt-5.6-sol", delivery="native", effort="high"))
    set_review_pool(monkeypatch, raw)
    for key in ("OUROBOROS_REVIEW_ENFORCEMENT", "OUROBOROS_CONTEXT_MODE",
                "OUROBOROS_OBSERVABILITY_KEEP_RAW", "OUROBOROS_PRE_PUSH_TESTS"):
        monkeypatch.setenv(key, "")

    _apply_contributor_review_env()
    config = _resolved_review_config(profile="external_pr_readiness")

    assert os.environ["OUROBOROS_SUBAGENTS"] == raw
    assert os.environ["OUROBOROS_REVIEW_ENFORCEMENT"] == "blocking"
    assert os.environ["OUROBOROS_OBSERVABILITY_KEEP_RAW"] == "0"
    # The review operation runs no tests (its record says tests NOT_RUN), so the
    # wrapper no longer pins the commit gate's test-preflight knob.
    assert os.environ["OUROBOROS_PRE_PUSH_TESTS"] == ""
    assert [row["route"]["kind"] for row in config["pool_slots"]] == [
        "agent_session", "api_chat", "api_chat",
    ]
    assert config["pool_slots"][0]["route"]["profile_id"] == "account-a"
    assert _configured_openrouter_models(config) == ["openai/gpt-5.6-sol"]
    assert _configured_openrouter_models({
        "pool_slots": [{"route": {
            "kind": "api_chat", "target_id": "openrouter::openai/gpt-5.6-sol",
        }}],
    }) == ["openai/gpt-5.6-sol"]
    _assert_contributor_review_config(config)
    frozen = _freeze_contributor_slots(config)
    assert frozen["execution_slot_config_source"] == "frozen_review_pool"
    assert len(frozen["slot_plan_sha256"]) == 64
    pinned = json.loads(os.environ["OUROBOROS_SUBAGENTS"])["items"]
    assert [row["subagent_id"] for row in pinned] == ["session", "direct", "reader"]
    assert pinned[0]["route"] == {"kind": "agent_session", "target_id": "codex=gpt-5.6-sol",
                                  "credential_profile_id": "account-a"}
    assert all(row["review_eligible"] and row["enabled"] for row in pinned)


def test_agent_session_only_preflight_needs_no_api_budget_or_key(monkeypatch):
    import scripts.run_external_review as module

    config = {
        "pool_slots": [{"slot_id": "t1", "route": {
            "kind": "agent_session", "target_id": "codex=gpt-5.6-sol"},
            "effort": "high"}, {"slot_id": "s1", "route": {
            "kind": "agent_session", "target_id": "cursor=claude-fable-5"},
            "effort": "high"}],
        "review_enforcement": "blocking", "context_mode": "max",
    }
    monkeypatch.delenv("TOTAL_BUDGET", raising=False)
    isolation = {"run_cap_usd": 3.0, "review_data_root": "/isolated/drive"}
    monkeypatch.setattr(module, "isolate_review_data", lambda **_kwargs: isolation)
    monkeypatch.setattr(module, "_load_settings_into_env", lambda: None)
    monkeypatch.setattr(module, "_contributor_proposal", lambda *_args: {"base_sha": "a" * 40})
    monkeypatch.setattr(module, "_apply_contributor_review_env", lambda: None)
    monkeypatch.setattr(module, "_resolved_review_config", lambda **_kwargs: config)
    monkeypatch.setattr(module, "_freeze_contributor_slots", lambda value: value)
    monkeypatch.setattr(
        module, "_select_healthy_openrouter_key",
        lambda **_kwargs: pytest.fail("agent-only review must not probe OpenRouter"),
    )
    args = SimpleNamespace(contributor=True, base_ref="", head_ref="proposal",
                           drive_root="", run_cap_usd="3", attach_host_engine=False)

    # An isolated review never starts its own engine: session rows need the
    # explicit attach to the host's running one.
    with pytest.raises(RuntimeError, match="--attach-host-engine"):
        _prepare_review_configuration(args)
    args.attach_host_engine = True
    proposal, resolved = _prepare_review_configuration(args)

    assert proposal == {"base_sha": "a" * 40}
    assert resolved == config and resolved["data_isolation"] is isolation
    assert args.drive_root == "/isolated/drive"  # the review drive IS the isolated data root


def test_contributor_run_cap_must_be_explicit_positive_and_finite(monkeypatch):
    """The cap is a CLI fact decided before settings load; no saved budget stands in."""
    import scripts.run_external_review as module
    from ouroboros.review_run_isolation import parse_run_cap

    for invalid in ("", "0", "-1", "inf", "nan", "not-a-number"):
        with pytest.raises(ValueError, match="positive finite"):
            parse_run_cap(invalid)
    assert parse_run_cap("125.50") == 125.5
    monkeypatch.setenv("TOTAL_BUDGET", "500")  # an inherited budget is not a run cap
    monkeypatch.setattr(module.sys, "argv", [
        "run_external_review.py", "--contributor", "--head-ref", "proposal"])
    with pytest.raises(SystemExit) as refused:
        module._parse_args()
    assert refused.value.code == 2
    monkeypatch.setattr(module.sys, "argv", ["run_external_review.py", "--run-cap-usd", "5"])
    with pytest.raises(SystemExit):
        module._parse_args()  # the cap and the attach selection belong to --contributor
    monkeypatch.setattr(module.sys, "argv", [
        "run_external_review.py", "--contributor", "--head-ref", "proposal", "--run-cap-usd", "5",
        "--attach-host-engine"])
    args = module._parse_args()
    assert (args.run_cap_usd, args.attach_host_engine, args.head_ref) == ("5", True, "proposal")


def test_contributor_lane_names_its_proposal_and_the_operator_lane_none(monkeypatch, capsys):
    """The executing checkout is the installed body, never the proposal, so the
    proposal has no default: it is named with --head-ref."""
    import scripts.run_external_review as module

    for argv, message in (
        (["--contributor", "--run-cap-usd", "5"], "--contributor requires --head-ref"),
        (["--head-ref", "proposal"], "--base-ref/--head-ref require --contributor"),
        (["--contributor", "--head-ref", "proposal", "--run-cap-usd", "5", "--no-isolated-checkout"],
         "--contributor requires the frozen isolated checkout"),
    ):
        monkeypatch.setattr(module.sys, "argv", ["run_external_review.py", *argv])
        with pytest.raises(SystemExit) as refused:
            module._parse_args()
        assert refused.value.code == 2
        assert message in capsys.readouterr().err


def test_contributor_proposal_binds_clean_base_head_and_tree(tmp_path, monkeypatch):
    repo = _init_contributor_repo(tmp_path, monkeypatch)
    head_tree = _git(repo, "rev-parse", "HEAD^{tree}").strip()

    proposal = _proposal_from_installed_base(repo)

    assert proposal["base_sha"] == proposal["merge_base_sha"]
    assert proposal["target_version"] == "1.2.3"
    assert proposal["head_tree_sha"] == head_tree
    assert proposal["changed_paths"] == ["a.txt"]
    assert proposal["review_substrate_changed"] == []
    assert proposal["diff_sha256"]
    assert proposal["base_script_sha256"] == proposal["head_script_sha256"]

    (repo / "dirty.txt").write_text("not committed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="not clean"):
        _contributor_proposal("base", proposal["head_sha"])


@pytest.mark.parametrize(
    ("relative_path", "before", "after", "carrier"),
    [
        ("VERSION", "1.2.3", "1.2.4", "VERSION"),
        ("pyproject.toml", 'version = "1.2.3"', 'version = "1.2.4"',
         "pyproject.project.version"),
        ("uv.lock", 'version = "1.2.3"', 'version = "1.2.4"',
         "uv.editable_root.version"),
        ("web/package.json", '"version":"1.2.3"', '"version":"1.2.4"',
         "web.package.version"),
        ("web/modules/api_types.js", 'VERSION = "1.2.3"', 'VERSION = "1.2.4"',
         "gateway.contract.version"),
        ("README.md", "Version 1.2.3", "Version 1.2.4",
         "readme.badge.version"),
        ("README.md", "| 1.2.3 | Current |", "| 1.2.4 | Current |",
         "readme.latest_history_row"),
        ("README.md", "v1.2.3/Ouroboros-1.2.3.dmg",
         "v1.2.4/Ouroboros-1.2.4.dmg", "readme.download.macos-arm64.0"),
        ("site/install/index.html", "v1.2.3/Ouroboros-1.2.3.dmg",
         "v1.2.4/Ouroboros-1.2.4.dmg", "site.install.download.macos-arm64.0"),
        ("docs/install/index.html", "v1.2.3/Ouroboros-1.2.3.dmg",
         "v1.2.4/Ouroboros-1.2.4.dmg", "docs.install.download.macos-arm64.0"),
        ("docs/ARCHITECTURE.md", "Ouroboros v1.2.3", "Ouroboros v1.2.4",
         "architecture.header.version"),
    ],
)
def test_contributor_proposal_rejects_every_version_carrier(
    tmp_path, monkeypatch, relative_path, before, after, carrier
):
    repo = _init_contributor_repo(tmp_path, monkeypatch)
    path = repo / relative_path
    path.write_text(path.read_text(encoding="utf-8").replace(before, after), encoding="utf-8")
    _git(repo, "add", relative_path)
    _git(repo, "commit", "-m", "bad contributor bump")

    with pytest.raises(RuntimeError, match=carrier):
        _proposal_from_installed_base(repo)


def test_contributor_proposal_checks_each_duplicate_installer_link(
    tmp_path, monkeypatch
):
    repo = _init_contributor_repo(tmp_path, monkeypatch)
    path = repo / "site" / "install" / "index.html"
    current = "https://example.test/v1.2.3/Ouroboros-1.2.3.dmg"
    stale = "https://example.test/v1.2.4/Ouroboros-1.2.4.dmg"
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace(current, stale, 1), encoding="utf-8")
    _git(repo, "add", str(path.relative_to(repo)))
    _git(repo, "commit", "-m", "change one duplicate installer link")

    with pytest.raises(RuntimeError, match="site.install.download.macos-arm64.0"):
        _proposal_from_installed_base(repo)


@pytest.mark.parametrize(
    "relative_path",
    [
        "ouroboros/review_substrate.py",
        "ouroboros/review_execution.py",
        "ouroboros/utils.py",
        "ouroboros/tools/registry.py",
        "scripts/run_external_review.py",
    ],
)
def test_contributor_proposal_flags_transitive_review_substrate_changes(
    tmp_path, monkeypatch, relative_path
):
    repo = _init_contributor_repo(tmp_path, monkeypatch)
    path = repo / relative_path
    path.write_text("# proposal changes the review substrate\n", encoding="utf-8")
    _git(repo, "add", str(path.relative_to(repo)))
    _git(repo, "commit", "-m", "change review substrate")

    proposal = _proposal_from_installed_base(repo)

    assert proposal["review_substrate_changed"] == [relative_path]
    assert proposal["review_substrate_matches_base"] is False
    assert (proposal["base_script_sha256"] != proposal["head_script_sha256"]) is (
        relative_path == "scripts/run_external_review.py")


def test_contributor_landing_obligations_are_exact_typed_items_only():
    version_only = {
        "status": "blocked",
        "block_reason": "critical_findings",
        "combined_findings": [
            {"item": "version_bump", "severity": "critical"},
            {"item": "changelog_and_badge", "severity": "critical"},
        ],
    }
    deferred = _apply_contributor_landing_obligations(version_only)
    assert deferred["status"] == "passed"
    assert {item["item"] for item in deferred["landing_obligations"]} == {
        "version_bump",
        "changelog_and_badge",
    }

    real_defect = {
        **version_only,
        "combined_findings": [
            *version_only["combined_findings"],
            {"item": "self_consistency", "severity": "critical"},
        ],
    }
    assert _apply_contributor_landing_obligations(real_defect) == real_defect
    scope_failure = {
        **version_only,
        "block_reason": "scope_blocked",
    }
    assert _apply_contributor_landing_obligations(scope_failure) == scope_failure
    assert _apply_contributor_landing_obligations(
        version_only,
        release_sensitive=True,
    ) == version_only


def test_contributor_packet_is_redacted_and_shareable(tmp_path):
    output = tmp_path / "packet"
    output.mkdir()
    local_root = "/Users/example/private/repo"
    bearer = " ".join(("Bearer", "secret-token-value"))
    packet = _write_contributor_packet(
        output_dir=output,
        snapshot={
            "base_sha": "a" * 40,
            "head_sha": "b" * 40,
            "review_substrate_changed": [],
            "patch": "diff --git a/a.txt b/a.txt\n",
            "installed_head_sha": "a" * 40,
        },
        resolved_config={"pool_models": ["anthropic/fable"]},
        outcome={"status": "passed", "path": local_root, "api_key": "test-secret-value"},
        exit_code=0,
        evidence_refs=[],
        cost_report={"reported_actor_cost_usd": 1.0},
        elapsed_sec=1.5,
        seats=[
            {"seat_id": "slot_1", "parts": ["change"], "answer": f"saw {bearer} under {local_root}"},
            {"seat_id": "scope_1", "parts": ["coupling"], "answer": "scope answer EOF_SCOPE"},
        ],
        review_record={"record_id": "r1", "path": f"{local_root}/state/review_ledger/r1.json",
                       "checklist": {"rules_source": {"path": "docs/CHECKLISTS.md", "sha": "c" * 64}}},
        execution_receipts=[{
            "surface": "pool", "slot_id": "slot_1",
            "observed": {"route_kind": "agent_session"},
            "model_verification": "observed_display_label",
        }],
        execution_mismatches=[],
        session_transcripts=[{
            "surface": "pool", "slot_id": "slot_1", "sha256": "a" * 64,
            "chars": 18, "transcript": "transcript EOF_MARK",
        }],
        degraded_reasons=["reviewer-3=parse_failure (quorum still met)"],
        replacements=[(local_root, "$REPO")],
    )

    evidence_text = (output / "review-evidence.json").read_text(encoding="utf-8")
    full_text = (output / "full-output.txt").read_text(encoding="utf-8")
    public_evidence = json.loads(evidence_text)
    assert "test-secret-value" not in evidence_text
    assert "secret-token-value" not in full_text
    assert local_root not in evidence_text + full_text
    assert "$REPO" in evidence_text + full_text
    assert "production_pool_quorum_plus_coupling" in evidence_text
    assert '"execution_receipts_consistent": true' in evidence_text
    assert "pool:slot_1:observed_model_is_display_label" in evidence_text
    assert "quorum still met" in evidence_text
    assert "transcript EOF_MARK" in full_text
    assert "scope answer EOF_SCOPE" in full_text
    assert "patch" not in public_evidence["snapshot"]
    assert "installed_head_sha" not in public_evidence["snapshot"]
    assert public_evidence["trust"]["installed_body_execution"]["executing_checkout_head"] == "a" * 40
    assert public_evidence["trust"]["installed_body_execution"]["rules_source"]["sha"] == "c" * 64
    assert public_evidence["review_record"]["path"] == "$REPO/state/review_ledger/r1.json"
    transcript_meta = public_evidence["review_execution"]["session_transcript_artifacts"][0]
    assert transcript_meta["chars"] == len("transcript EOF_MARK")
    assert transcript_meta["sha256"] == hashlib.sha256(
        b"transcript EOF_MARK"
    ).hexdigest()
    with zipfile.ZipFile(packet) as archive:
        assert set(archive.namelist()) == {
            "review-evidence.json",
            "outcome.json",
            "full-output.txt",
        }


def _actors(triad: list[dict], scope: list[dict] | None = None) -> list[tuple[str, dict]]:
    """The seats as a review record carries them: ledger rows of the raw actors
    of ONE wave — a seat configured under the scope role is a coupling-only seat
    (``parts=["coupling"]``) beside the triad seats."""
    from ouroboros.review_ledger import build_commit_gate_record

    coupling_only = [{**row, "parts": ["coupling"]} for row in (scope or [])]
    record = build_commit_gate_record({"triad_raw": [*triad, *coupling_only]})
    return _record_actors(asdict(record))


def test_external_review_cost_report_never_turns_unknown_into_zero():
    triad = [
        {
            "slot_id": f"slot_{idx}",
            "model_id": f"reviewer-{idx}",
            "status": "responded",
            "tokens_in": 100,
            "cost_usd": 0.01,
            "prompt_ref": {"manifest_ref": f"prompt-{idx}"},
            "response_ref": {"manifest_ref": f"response-{idx}"},
        }
        for idx in range(1, 4)
    ]
    scope_actor = {
        "slot_id": "scope_slot_1",
        "model_id": "scope-reviewer",
        "status": "responded",
        "tokens_in": 200,
        "cost_usd": 0.0,
        "prompt_ref": {"manifest_ref": "scope-prompt"},
        "response_ref": {"manifest_ref": "scope-response"},
    }
    actors = _actors(triad, [scope_actor])
    evidence, report = _review_evidence_and_cost(actors)
    assert len(evidence) == 4
    assert [(surface, actor["slot_id"]) for surface, actor in actors][-1] == ("pool", "scope_slot_1")
    assert evidence[0]["prompt_ref"] == {"manifest_ref": "prompt-1"}
    assert report["reported_actor_cost_usd"] == 0.03
    assert report["unreported_or_unknown_cost_slots"] == ["scope_slot_1"]
    assert "not treated as $0" in report["note"]
    # An open seat has no cost yet (unknown, never $0); a seat never dispatched is no actor.
    pending = _actors([{**triad[0], "status": "pending"}])
    assert _review_evidence_and_cost(pending)[1]["unreported_or_unknown_cost_slots"] == ["slot_1"]
    assert _record_actors({"rows": [{"seat_id": "x", "status": "not_dispatched"}]}) == []


def test_exit_classification_separates_infra_from_genuine_blocks():
    assert _classify_exit({"status": "passed"}) == 0
    assert _classify_exit({"status": "blocked", "block_reason": "critical_findings"}) == 1
    # A scope CRITICAL with concrete findings is a genuine reviewer verdict...
    assert _classify_exit({
        "status": "blocked",
        "block_reason": "scope_blocked",
        "combined_findings": [{"severity": "CRITICAL", "text": "real defect"}],
    }) == 1
    # ...while a findings-less scope block is fail-closed infrastructure.
    assert _classify_exit({"status": "blocked", "block_reason": "scope_blocked"}) == 3
    for infra_reason in (
        "tests_preflight_blocked",
        "core_protection_blocked",
        "no_advisory",
        "review_quorum",
        "fingerprint_unavailable",
        "",
    ):
        assert _classify_exit({"status": "blocked", "block_reason": infra_reason}) == 3, infra_reason


def test_contributor_outcome_fails_closed_on_receipt_drift_only():
    exit_code, outcome = finalize_contributor_outcome(
        outcome={"status": "passed"}, exit_code=0,
        mismatches=["provider_mismatch:pool:t1"],
    )
    assert exit_code == 3
    assert outcome["block_reason"] == "execution_receipt_mismatch"

    # Nothing about the proposal's contents downgrades a clean run any more: the
    # review flow and rules that produced it were the installed body's either way.
    assert finalize_contributor_outcome(
        outcome={"status": "passed"}, exit_code=0, mismatches=[],
    ) == (0, {"status": "passed"})


def test_openrouter_pool_orders_hope_keys_last(monkeypatch, tmp_path):
    keys = tmp_path / "file1.txt"
    keys.write_text(
        "hope_new_key_openrouter: sk-or-hope-000\n"
        "openrouter_kuznetsov3: sk-or-kuz-111\n"
        "backup_hope_openrouter: sk-or-hope-bak-444\n"
        "openai: sk-oa-222\n"
        "anton_openrouter_main: sk-or-anton-333\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OUROBOROS_KEYS_FILE", str(keys))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    pool = _openrouter_pool()

    names = [name for name, _ in pool]
    # Any hope-bucket key sinks to the tail, prefix or not.
    assert names == [
        "openrouter_kuznetsov3",
        "anton_openrouter_main",
        "hope_new_key_openrouter",
        "backup_hope_openrouter",
    ]


def test_contributor_openrouter_preflight_fails_closed(monkeypatch):
    import scripts.run_external_review as module

    monkeypatch.setattr(module, "_openrouter_pool", lambda: [])
    with pytest.raises(RuntimeError, match="no OpenRouter key"):
        _select_healthy_openrouter_key(required=True)

    monkeypatch.setattr(module, "_openrouter_pool", lambda: [("key", "secret")])
    monkeypatch.setattr(
        module,
        "_openrouter_key_health",
        lambda _token, **_kwargs: (False, "model_probe_http_403"),
    )
    with pytest.raises(RuntimeError, match="no healthy OpenRouter key"):
        _select_healthy_openrouter_key(required=True)


def _persist_review_prompt(tmp_path, *, call_id: str, slot: dict):
    from ouroboros.observability import persist_call

    return persist_call(
        tmp_path, task_id="review", call_id=call_id, call_type="prompt",
        payload={"request": {"surface": "review"}, "slot": slot},
    )


def _persist_review_response(
    tmp_path, *, call_id: str, usage: dict, transcript: str = "",
):
    from ouroboros.observability import persist_call

    message = {"content": "[]"}
    if transcript:
        message["session_transcript"] = transcript
        usage = {**usage, "verdict_provenance": {
            "raw_transcript_chars": len(transcript),
            "raw_transcript_sha256": hashlib.sha256(
                transcript.encode("utf-8", "replace")
            ).hexdigest(),
        }}
    return persist_call(
        tmp_path, task_id="review", call_id=call_id, call_type="response",
        payload={"message": message, "usage": usage},
    )


def test_contributor_receipts_bind_session_and_api_execution(tmp_path):
    config = {
        "pool_slots": [{"slot_id": "t1", "route": {
            "kind": "agent_session", "target_id": "codex=gpt-5.6-sol",
            "profile_id": "pinned"}, "effort": "high"}, {"slot_id": "s1", "route": {
            "kind": "api_chat", "target_id": "openai/gpt-5.6-sol"},
            "effort": "xhigh"}],
        "review_enforcement": "blocking",
        "context_mode": "max",
    }
    _assert_contributor_review_config(config)
    session_prompt = _persist_review_prompt(tmp_path, call_id="session_prompt", slot={
        "slot_id": "t1", "model": "codex=gpt-5.6-sol", "effort": "high",
        "route": "agent_session", "session_target": "codex=gpt-5.6-sol",
        "session_profile": "pinned",
    })
    transcript = "full session transcript\nEOF_SENTINEL"
    session_ref = _persist_review_response(
        tmp_path, call_id="session", transcript=transcript, usage={
            "provider": "claudexor", "delegated_route": "codex",
            "resolved_model": "gpt-5.6-sol", "applied_profile": "pinned",
            "applied_access": "readonly", "delegated_run_id": "run-1",
            "custody_durable": True, "output_conformance": "passed",
            "verdict_method": "schema", "settlement": {
                "settled": True, "ledger_recorded": True,
                "project_retired": True,
            }},
    )
    api_prompt = _persist_review_prompt(tmp_path, call_id="api_prompt", slot={
        "slot_id": "s1", "model": "openai/gpt-5.6-sol", "effort": "xhigh",
        "route": "api_chat", "session_target": "", "session_profile": "",
    })
    api_ref = _persist_review_response(
        tmp_path, call_id="api", usage={
            "provider": "openrouter", "resolved_model": "openai/gpt-5.6-sol"},
    )
    actors = _actors([{
        "slot_id": "t1", "model_id": "gpt-5.6-sol", "status": "responded",
        "prompt_ref": session_prompt, "response_ref": session_ref,
    }], [{
        "slot_id": "s1", "model_id": "openai/gpt-5.6-sol",
        "status": "responded", "prompt_ref": api_prompt, "response_ref": api_ref,
    }])

    unbound_receipts, unbound_mismatches, _ = _contributor_execution_receipts(
        actors, config, tmp_path
    )
    assert "session_custody_settlement_absent:pool:t1:run-1" in unbound_mismatches
    assert unbound_receipts[0]["observed"]["settlement"] is None
    assert unbound_receipts[1]["observed"]["route_kind"] == "api_chat"

    from ouroboros import delegate_custody as custody

    custody.record_started(tmp_path, custody.RunCustody(
        run_id="run-1", task_id="review", project_id="review-project",
        project_owned=True, ledger_root=str(tmp_path),
    ))
    custody.emit(tmp_path, custody.LEDGER_RECORDED, {"run_id": "run-1"})
    custody.emit(tmp_path, custody.SETTLED, {"run_id": "run-1"})
    custody.emit(tmp_path, custody.PROJECT_RETIRED, {"run_id": "run-1"})
    custody._CUSTODY.clear()

    receipts, mismatches, transcripts = _contributor_execution_receipts(
        actors, config, tmp_path
    )

    assert mismatches == []
    assert receipts[0]["observed"] == {
        "route_kind": "agent_session", "provider": "claudexor",
        "harness": "codex", "model": "gpt-5.6-sol",
        "profile_id": "pinned", "access": "readonly", "effort": None,
        "delegated_run_id": "run-1", "custody_durable": True,
        "settlement": {
            "settled": True, "ledger_recorded": True,
            "project_retired": True, "project_persistent": False,
            "bound_at": "panel_complete_custody_replay",
        },
        "output_conformance": "passed", "verdict_method": "schema",
    }
    assert receipts[0]["dispatched"]["effort"] == "high"
    assert receipts[0]["model_verification"] == "exact"
    assert transcripts[0]["transcript"].endswith("EOF_SENTINEL")
    drifted = json.loads(json.dumps(config))
    drifted["pool_slots"][0]["route"]["target_id"] = "cursor=gpt-5.6-sol"
    _, mismatches, _ = _contributor_execution_receipts(actors, drifted, tmp_path)
    assert any(item.startswith("harness_mismatch:pool:t1") for item in mismatches)


def test_contributor_receipts_fail_closed_on_blob_provider_model_and_status_drift(tmp_path):
    config = {
        "pool_slots": [{"slot_id": "t1", "route": {
            "kind": "api_chat", "target_id": "anthropic::claude-fable-5"},
            "effort": "high"}, {"slot_id": "s1", "route": {
            "kind": "agent_session", "target_id": "codex=gpt-5.6-sol"},
            "effort": "high"}],
        "review_enforcement": "blocking", "context_mode": "max",
    }
    api_prompt = _persist_review_prompt(tmp_path, call_id="api_prompt_bad", slot={
        "slot_id": "t1", "model": "anthropic::claude-fable-5", "effort": "high",
        "route": "api_chat", "session_target": "", "session_profile": "",
    })
    api_response = _persist_review_response(
        tmp_path, call_id="api_bad", usage={
            "provider": "openrouter", "resolved_model": "openai/gpt-5.5"},
    )
    session_prompt = _persist_review_prompt(tmp_path, call_id="session_prompt_bad", slot={
        "slot_id": "s1", "model": "codex=gpt-5.6-sol", "effort": "high",
        "route": "agent_session", "session_target": "codex=gpt-5.6-sol",
        "session_profile": "",
    })
    session_response = _persist_review_response(
        tmp_path, call_id="session_bad", transcript="raw\nEOF", usage={
            "provider": "claudexor", "delegated_route": "codex",
            "resolved_model": "GPT-5.6 Terra 300K High",
            "applied_profile": "auto-profile",
            "applied_access": "readonly", "custody_durable": True,
            "capability_delta": [{"reason": "session_ran_off_pinned_route"}],
        },
    )
    triad_actor = {
        "slot_id": "t1", "status": "parse_failure",
        "prompt_ref": api_prompt, "response_ref": api_response,
    }
    scope_actors = [{
        "slot_id": "s1", "status": "responded",
        "prompt_ref": session_prompt, "response_ref": session_response,
    }]

    _, mismatches, _ = _contributor_execution_receipts(
        _actors([triad_actor], scope_actors), config, tmp_path)

    assert any(item.startswith("provider_mismatch:pool:t1") for item in mismatches)
    assert any(item.startswith("model_mismatch:pool:t1") for item in mismatches)
    assert any(item.startswith("model_identity_unverified:pool:s1")
               for item in mismatches)
    assert "delegated_run_id_absent:pool:s1" in mismatches
    assert any(item.startswith("session_settlement_unproven:pool:s1")
               for item in mismatches)
    assert "capability_delta:pool:s1:session_ran_off_pinned_route" in mismatches

    missing_response = json.loads(json.dumps(triad_actor))
    missing_response["response_ref"] = {}
    _, missing_response_mismatches, _ = _contributor_execution_receipts(
        _actors([missing_response], scope_actors), config, tmp_path
    )
    assert "response_receipt_absent:pool:t1" in missing_response_mismatches

    tampered = json.loads(json.dumps(triad_actor))
    tampered["response_ref"]["redacted_projection_ref"]["sha256"] = "0" * 64
    _, tampered_mismatches, _ = _contributor_execution_receipts(
        _actors([tampered], scope_actors), config, tmp_path
    )
    assert any(item.startswith("unreadable_response_receipt:pool:t1")
               for item in tampered_mismatches)


def test_contributor_receipts_require_settlement_but_keep_advisory_delta(tmp_path):
    config = {
        "pool_slots": [{"slot_id": "t1", "route": {
            "kind": "agent_session", "target_id": "codex=gpt-5.6-sol"},
            "effort": "high"}], "review_enforcement": "blocking", "context_mode": "max",
    }
    prompt_ref = _persist_review_prompt(tmp_path, call_id="session_prompt_terminal", slot={
        "slot_id": "t1", "model": "codex=gpt-5.6-sol", "effort": "high",
        "route": "agent_session", "session_target": "codex=gpt-5.6-sol",
        "session_profile": "",
    })
    response_ref = _persist_review_response(
        tmp_path, call_id="session_terminal", transcript="raw\nEOF", usage={
            "provider": "claudexor", "delegated_route": "codex",
            "resolved_model": "gpt-5.6-sol", "applied_profile": "auto-profile",
            "applied_access": "readonly", "custody_durable": True,
            "delegated_run_id": "run-1", "settlement": {
                "settled": True, "ledger_recorded": False,
                "project_retired": True,
            },
            "capability_delta": [{"reason": "schema_unavailable_on_effective_route"}],
        },
    )
    actors = _actors([{
        "slot_id": "t1", "status": "responded",
        "prompt_ref": prompt_ref, "response_ref": response_ref,
    }])

    from ouroboros import delegate_custody as custody

    custody.record_started(tmp_path, custody.RunCustody(
        run_id="run-1", task_id="review", project_id="review-project",
        project_owned=True, ledger_root=str(tmp_path),
    ))
    custody.emit(tmp_path, custody.PROJECT_RETIRED, {"run_id": "run-1"})
    custody._CUSTODY.clear()

    _, mismatches, _ = _contributor_execution_receipts(actors, config, tmp_path)

    assert "session_settlement_unproven:pool:t1:settled,ledger_recorded" in mismatches
    assert not any(item.startswith("capability_delta:") for item in mismatches)


def test_contributor_receipts_accept_present_usage_less_error_payload(tmp_path):
    from ouroboros.observability import persist_call
    config = {
        "pool_slots": [{"slot_id": "t1", "route": {
            "kind": "api_chat", "target_id": "openai/gpt-5.6-sol"},
            "effort": "high"}], "review_enforcement": "blocking", "context_mode": "max",
    }
    prompt_ref = _persist_review_prompt(tmp_path, call_id="error_prompt", slot={
        "slot_id": "t1", "model": "openai/gpt-5.6-sol", "effort": "high",
        "route": "api_chat", "session_target": "", "session_profile": "",
    })
    response_ref = persist_call(
        tmp_path, task_id="review", call_id="error_response", call_type="response",
        payload={"error": "Timeout after 300s"},
    )
    actors = _actors([{
        "slot_id": "t1", "status": "error",
        "prompt_ref": prompt_ref, "response_ref": response_ref,
    }])
    receipts, mismatches, _ = _contributor_execution_receipts(actors, config, tmp_path)
    assert mismatches == []
    assert receipts[0]["observed"]["route_kind"] is None


def test_normal_key_probe_stays_single_model_but_contributor_probes_all(monkeypatch):
    import scripts.run_external_review as module

    calls: list[str] = []
    monkeypatch.setattr(module, "_review_probe_models", lambda: ["one", "two", "three"])
    monkeypatch.setattr(
        module,
        "_probe_model_for_key",
        lambda _token, model: (calls.append(model) is None, f"ok:{model}"),
    )
    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {"data": {"limit": None}}

    import httpx

    monkeypatch.setattr(httpx, "get", lambda *_args, **_kwargs: Response())

    assert _openrouter_key_health("secret")[0] is True
    assert calls == ["one"]
    calls.clear()
    assert _openrouter_key_health("secret", probe_all_models=True)[0] is True
    assert calls == ["one", "two", "three"]


def _git(repo: Path, *args: str) -> str:
    patch = args[:1] == ("diff",) and "--binary" in args
    proc = subprocess.run(
        ["git", *args], cwd=str(repo), capture_output=True, text=not patch, check=True,
    )
    return proc.stdout.decode("utf-8", errors="surrogateescape") if patch else proc.stdout
