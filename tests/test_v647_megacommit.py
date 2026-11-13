"""v6.47.0 mega-commit immune tests (TIER-5): verify-before-done (FR3), cooperative
subagents (FR2), workspace-aware code-intel (R1/R2/R5), skill-publish SSOT (FR1),
and the M2/M6 reliability invariants. Pure-logic where possible; a few use a tmp
git tree / user_files root."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import types
from pathlib import Path



# ── R2: PYTHONPATH repo-scrub env isolation ───────────────────────────────────
def test_scrub_repo_from_pythonpath_drops_only_repo_entry():
    from ouroboros.platform_layer import scrub_repo_from_pythonpath

    repo = "/obo/repo"
    sep = os.pathsep
    env = {"PYTHONPATH": sep.join([repo, "/app", "/usr/lib/py"]), "X": "1"}
    out = scrub_repo_from_pythonpath(env, repo)
    assert out["PYTHONPATH"] == sep.join(["/app", "/usr/lib/py"])
    assert out["X"] == "1"
    assert env["PYTHONPATH"].startswith(repo)  # original not mutated
    # only the repo entry -> PYTHONPATH removed entirely
    assert "PYTHONPATH" not in scrub_repo_from_pythonpath({"PYTHONPATH": repo}, repo)
    # no PYTHONPATH / no repo_dir -> no-op
    assert scrub_repo_from_pythonpath({"A": "b"}, repo) == {"A": "b"}
    assert scrub_repo_from_pythonpath({"PYTHONPATH": repo}, None) == {"PYTHONPATH": repo}
    # trailing-slash equivalence
    assert "PYTHONPATH" not in scrub_repo_from_pythonpath({"PYTHONPATH": repo + "/"}, repo)


# ── R5: effect-based artifact-audit gate ──────────────────────────────────────
def test_user_files_run_effect_gate():
    from ouroboros.tools.shell import _shallow_listing, _user_files_run_had_effect

    d = Path(tempfile.mkdtemp())
    (d / "a.txt").write_text("1")
    sig = _shallow_listing(d)
    assert _user_files_run_had_effect([], [], sig, d) is False  # read-only
    (d / "b.txt").write_text("2")
    assert _user_files_run_had_effect([], [], sig, d) is True   # new file
    assert _user_files_run_had_effect(["x"], ["x", "y"], None, d) is True  # git delta
    assert _user_files_run_had_effect(["x"], ["x"], None, d) is False


# ── R1: query_code root=user_files guards ─────────────────────────────────────
def test_query_code_user_files_empty_path_hard_error():
    from ouroboros.tools.query_code import _query_code

    ctx = types.SimpleNamespace(
        drive_root=tempfile.mkdtemp(), repo_dir=tempfile.mkdtemp(),
        workspace_root="", workspace_mode="", task_constraint=None,
    )
    out = _query_code(ctx, "symbols", root="user_files", path="")
    assert "requires an explicit path" in out


def test_a_turn_without_effects_carries_no_tool_count_warning():
    """The retired M2 check warned when a declared deliverable came with no tool calls and no
    FINAL ANSWER marker; plain prose can itself be the deliverable (#1748), so how many tools
    an answer used is no judgment of it. A written result nothing examined is still flagged."""
    from ouroboros import outcomes as O

    for final_answer in ("", "42"):
        clean = {"outcome_axes": {"execution": {"status": "ok"}, "objective": {"status": "not_evaluated"}},
                 "final_answer": final_answer}
        O.apply_receipt_absent_flag(clean, {"tool_calls": []}, tempfile.mkdtemp(), "m1")
        assert clean["outcome_axes"]["objective"] == {"status": "not_evaluated"}
    effects = {"outcome_axes": {"execution": {"status": "ok"}, "objective": {"status": "not_evaluated"}}, "final_answer": ""}
    written = {"tool": "write_file", "status": "ok", "args": {"path": "report.md"}}
    O.apply_receipt_absent_flag(effects, {"tool_calls": [written]}, tempfile.mkdtemp(), "m2")
    assert effects["outcome_axes"]["objective"]["warnings"] == ["receipt_absent"]


def test_masked_pass_reconciliation_cannot_use_the_check_text_identity():
    """The masked path must NOT inherit the red path's check-text identity. A masked
    receipt's only text identity is its own MASKED command, and the sensor is a pure
    function of argv, so the byte-identical re-run is re-flagged masked and could never
    be the clean reconciler — text equality would make the flag unclearable by the very
    remediation the host prescribes ("drop the masking pipe"). Rule: equal criterion_id
    when the MASKED receipt carries one — a later clean receipt that omits its id does
    not clear it — else ANY later clean grounding."""
    from ouroboros import outcomes as O
    from ouroboros.tools.verify import _check_has_exit_masking

    # WHY text identity cannot bind here: remediation necessarily changes the text.
    assert _check_has_exit_masking(["sh", "-c", "make test | tail"])[0] is True
    assert _check_has_exit_masking(["sh", "-c", "make test"])[0] is False

    masked = {"status": "pass", "check": "sh -c make test | tail", "check_exit_masking": True}
    assert O.latest_unreconciled_masked_pass([masked]) is not None
    # the PRESCRIBED remediation (same run, masking pipe dropped) clears it
    assert O.latest_unreconciled_masked_pass(
        [masked, {"status": "pass", "check": "sh -c make test"}]) is None
    # so does any other later clean grounding, as before v6.78.0
    assert O.latest_unreconciled_masked_pass(
        [masked, {"status": "pass", "check": "make lint"}]) is None
    # a later receipt that is ITSELF masked never reconciles
    assert O.latest_unreconciled_masked_pass(
        [masked, {"status": "pass", "check": "make lint | tail", "check_exit_masking": True}]) is not None
    # criterion_id still binds when BOTH receipts carry one
    assert O.latest_unreconciled_masked_pass([
        dict(masked, criterion_id="c1"),
        {"status": "pass", "criterion_id": "c1", "check": "sh -c make test"},
    ]) is None
    assert O.latest_unreconciled_masked_pass([
        dict(masked, criterion_id="c1"),
        {"status": "pass", "criterion_id": "c2", "check": "sh -c make test"},
    ]) is not None


def test_artifact_observation_class_reconciles_on_its_observed_path_set():
    """Adversarial review (minor, accepted): the artifact-observation class runs NO
    command, so it has neither `check` nor (usually) `criterion_id` — under strict
    single-identity equality a red "report.md missing" could never be cleared by the
    byte-identical green observation after the file was written, leaving a permanent
    unreconciled red and a nudge on a task that DID verify itself. The observed path
    SET is that class's real identity."""
    from ouroboros import _outcome_receipts as R
    from ouroboros import outcomes as O

    red = {"status": "fail", "contract_kind": "artifact_observation", "paths": ["report.md"]}
    green = {"status": "observed", "contract_kind": "artifact_observation", "paths": ["report.md"]}
    other = {"status": "observed", "contract_kind": "artifact_observation", "paths": ["other.md"]}

    # The key VALUE is the injective serialization of the set, not a line-join: a path
    # may legally contain a newline, so `["a\nb"]` and `["a", "b"]` must not share a key.
    assert R.receipt_identity(red) == ("artifact_paths", '["report.md"]')
    assert R.receipt_identity({"paths": ["a\nb"]}) != R.receipt_identity({"paths": ["a", "b"]})
    # same path set -> reconciled (order-insensitive; the SET is the identity)
    assert O.latest_unreconciled_failed_receipt([red, green]) is None
    assert O.latest_unreconciled_failed_receipt([
        {"status": "fail", "paths": ["b.md", "a.md"]},
        {"status": "observed", "paths": ["a.md", "b.md"]},
    ]) is None
    # a DIFFERENT path does not reconcile it
    assert O.latest_unreconciled_failed_receipt([red, other]) is not None
    # a green COMMAND check does not reconcile an observation (different identity)
    assert O.latest_unreconciled_failed_receipt(
        [red, {"status": "pass", "check": "ls report.md"}]) is not None
    # criterion_id still wins when present
    assert O.latest_unreconciled_failed_receipt([
        {"status": "fail", "criterion_id": "c1", "paths": ["report.md"]},
        {"status": "observed", "criterion_id": "c1", "paths": ["elsewhere.md"]},
    ]) is None
    # an identity-LESS red keeps the pre-v6.78.0 rule (any later green clears it): it has
    # no identity to protect, and verify.py writes exactly this shape for a malformed
    # `artifact_observation` call with no artifact_paths — under the narrowing that one
    # mistake poisoned `unreconciled_red` for the rest of the task.
    assert R.receipt_identity({}) == ("none", "")
    assert R.receipt_identity({"paths": []}) == ("none", "")
    assert O.latest_unreconciled_failed_receipt([{"status": "fail"}, {"status": "observed"}]) is None
    malformed = {"status": "fail", "contract_kind": "artifact_observation",
                 "paths": [], "summary": "no artifact_paths given"}
    assert O.latest_unreconciled_failed_receipt(
        [malformed, {"status": "pass", "check": "pytest -q"}]) is None
    assert O.latest_unreconciled_failed_receipt([malformed]) is not None
    # a `declared` escape hatch still does not reconcile it
    assert O.latest_unreconciled_failed_receipt(
        [malformed, {"status": "declared", "check": "c"}]) is not None


def test_receipt_identity_and_disclosed_whitespace_flag():
    """The identity rule and its DISCLOSED flag (Q28=B): criterion_id wins; without
    one the whitespace-normalized check text IS the verification's identity, and both
    reviewer-facing consumers say so."""
    from ouroboros import _outcome_receipts as R
    from ouroboros.outcomes import build_verification_ledger
    from ouroboros.review_evidence import _accept_verification_summary

    # The check key pairs the canonical text with the RENDERING that produced the stored
    # string (round 8) — `receipt_identity_parts` below still exposes the plain text.
    go_test = ("check", json.dumps(["unversioned", "go test"]))
    assert R.receipt_identity({"criterion_id": " c1 ", "check": "x"}) == ("criterion_id", "c1")
    assert R.receipt_identity({"check": "go   test"}) == go_test
    assert R.receipt_identity({"paths": ["a.md"], "check": "go test"}) == go_test
    # The three components are independent; `receipt_identity` selects the ONE typed key
    # sameness is decided by, and `receipt_identity_parts` discloses all three.
    assert R.receipt_identity_parts(
        {"criterion_id": " c1 ", "check": "go   test", "paths": ["b.md", "a.md"]}
    ) == ("c1", "go test", "a.md\nb.md")
    assert R.receipt_identity_parts({}) == ("", "", "")
    assert R.receipt_identity_parts({"paths": "not-a-list"}) == ("", "", "")
    # The flag is TRUE for the `check` kind and false for every other kind — see
    # test_the_whitespace_flag_is_derived_once_for_every_identity_kind.
    assert R.receipt_expected_whitespace_normalized({"check": "go test"}) is True
    assert R.receipt_expected_whitespace_normalized({"paths": ["a.md"]}) is False
    assert R.receipt_expected_whitespace_normalized({"criterion_id": "c1", "check": "go test"}) is False
    assert R.receipt_expected_whitespace_normalized({}) is False

    # Consumer 1: the result reviewer's verification summary (review_result._verification).
    summary = _accept_verification_summary([{"status": "fail", "check": "pytest tests/x.py"}])
    assert summary["expected_whitespace_normalized"] is True
    assert summary["reconciliation_identity_kinds"] == ["check"]
    assert summary["latest_identity"]["paths"] == []  # a command check observes no path set
    id_summary = _accept_verification_summary([
        {"status": "pass", "criterion_id": "c1", "check": "pytest"},
        {"status": "observed", "paths": ["report.md"]},
    ])
    # Neither row is governed by command text: the `criterion_id` row is named, and the
    # observation row's path SET normalizes nothing (round 7).
    assert id_summary["expected_whitespace_normalized"] is False
    assert id_summary["reconciliation_identity_kinds"] == ["artifact_paths", "criterion_id"]
    # the observation class runs no command, so its identity reaches the reviewer as the
    # observed path set (latest_check is empty for it) — never silently dropped
    assert id_summary["latest_check"] == ""
    assert id_summary["latest_identity"]["paths"] == ["report.md"]
    assert _accept_verification_summary(
        [{"status": "pass", "criterion_id": "c1", "check": "pytest"}]
    )["expected_whitespace_normalized"] is False

    # Consumer 2: the FIXED verification-ledger receipt projection.
    ledger = build_verification_ledger(
        task={"id": "t1"},
        loop_outcome={"outcome_axes": {"execution": {"status": "ok"}}},
        llm_trace={"verification_receipts": [
            {"status": "fail", "check": "pytest tests/x.py"},
            {"status": "observed", "contract_kind": "artifact_observation", "paths": ["report.md"]},
        ]},
        artifact_bundle={"status": "not_applicable", "artifacts": [], "errors": []},
    )
    rows = [r for r in ledger["entries"] if r.get("kind") == "verification_receipt"]
    assert rows[0]["expected_whitespace_normalized"] is True
    assert rows[0]["reconciliation_identity"] == "check"
    assert rows[0]["paths"] == []
    assert rows[1]["reconciliation_identity"] == "artifact_paths"
    assert rows[1]["expected_whitespace_normalized"] is False
    # the FIXED projection carries the identity the reconciliation actually used
    assert rows[1]["paths"] == ["report.md"]


# ── FR2: cooperative subagent shared tree + depth reservation ─────────────────
def test_depth_reservation_admits():
    from supervisor.events import _depth_reservation_admits

    def t(tid, parent, rt="R"):
        return {"id": tid, "parent_task_id": parent, "root_task_id": rt, "delegation_role": "subagent"}

    running = {"w1": {"task": t("P", "ROOT")}}
    pending = [t(f"c{i}", "ROOT") for i in range(6)]  # tree at cap=6
    # P is a running subagent with no direct child -> reservation admits ONE
    assert _depth_reservation_admits("R", "P", pending, running, 6) is True
    # once P has a direct child, no further reservation for P
    assert _depth_reservation_admits("R", "P", pending + [t("pc", "P")], running, 6) is False
    # parent not a running subagent -> no reservation
    assert _depth_reservation_admits("R", "ROOT", pending, running, 6) is False
    # hard ceiling (2*cap) bounds it
    assert _depth_reservation_admits("R", "P", [t(f"d{i}", "ROOT") for i in range(12)], running, 6) is False


def test_ensure_cooperative_shared_root_mints_git_tree(monkeypatch):
    from ouroboros.tools import control_delegation as CD

    projects_root = Path(tempfile.mkdtemp()) / "coop_projects"
    monkeypatch.setenv("OUROBOROS_SUBAGENT_PROJECTS_ROOT", str(projects_root))
    CD._COOP_SHARED_ROOTS.clear()
    ctx = types.SimpleNamespace(repo_dir=str(Path(tempfile.mkdtemp())), task_id="root-1")
    path = CD.ensure_cooperative_shared_root(ctx, "root-1")
    assert not path.startswith("⚠️"), path
    assert (Path(path) / ".git").exists()  # a real git tree was minted
    head = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert head  # has the seed commit
    # cached: a second call returns the SAME tree (one shared tree per task-tree)
    assert CD.ensure_cooperative_shared_root(ctx, "root-1") == path


def test_resolve_cooperative_write_root_routes_flat_parent(monkeypatch):
    from ouroboros.tools import control_delegation as CD

    monkeypatch.setenv("OUROBOROS_SUBAGENT_PROJECTS_ROOT", str(Path(tempfile.mkdtemp()) / "p"))
    monkeypatch.setenv("OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS", "true")
    CD._COOP_SHARED_ROOTS.clear()
    ctx = types.SimpleNamespace(repo_dir=str(Path(tempfile.mkdtemp())), task_id="r2", task_constraint=None,
                                is_workspace_mode=lambda: False, is_direct_chat=False)
    eff, profile, err = CD.resolve_cooperative_write_root(ctx, "external_workspace", "", "", {"root_task_id": "r2"})
    assert err == "" and eff and (Path(eff) / ".git").exists()
    # an explicit write_root is passed through untouched
    eff2, _p, err2 = CD.resolve_cooperative_write_root(ctx, "external_workspace", "/some/dir", "", {})
    assert err2 == "" and eff2 == "/some/dir"


def test_v651_stringified_argv_recovery_and_check_normalization():
    """v6.51.0 idea-1: the SSOT stringified-argv recovery + check normalization."""
    from ouroboros.shell_parse import normalize_check_argv, recover_stringified_argv

    # recovery: JSON list, Python list literal, plain string, malformed
    assert recover_stringified_argv('["go","test"]') == ["go", "test"]
    assert recover_stringified_argv("['go','test']") == ["go", "test"]
    assert recover_stringified_argv("go test") is None      # plain string is NOT shell-split here
    assert recover_stringified_argv("[broken") is None
    assert recover_stringified_argv("[]") == []             # no-drift edge: matches the old inline _run_shell
    # normalization: stringified-argv recovers to argv; plain string -> NON-login sh -c (PATH parity)
    assert normalize_check_argv('["go","test"]') == ["go", "test"]
    assert normalize_check_argv("go test") == ["sh", "-c", "go test"]
    assert normalize_check_argv(["go", "test"]) == ["go", "test"]
    assert normalize_check_argv("   ") is None
    # issue #2906: preserve empty argv elements in check list/tuple (e.g. pytest -m "")
    assert normalize_check_argv(["pytest", "-q", "-m", "", "tests/test_foo.py"]) == [
        "pytest",
        "-q",
        "-m",
        "",
        "tests/test_foo.py",
    ]
    assert normalize_check_argv([""]) is None
    assert normalize_check_argv(["   "]) is None
    assert normalize_check_argv([]) is None
    # reject blank argv[0] without shifting argument positions
    assert normalize_check_argv(["", "pytest", "tests/test_foo.py"]) is None
    assert normalize_check_argv(["   ", "pytest"]) is None
    assert normalize_check_argv('["", "pytest"]') is None


def test_v651_orchestrate_timeout_reaps_group_and_container(monkeypatch):
    """v6.51.0 review round-4: a host-timed-out run_pro is killed via platform_layer (cross-
    platform, NOT a raw os.killpg) and the worker's leaked obopro-w{N}-* container is removed."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "orchp_t", "devtools/benchmarks/swe_bench_pro/e1v2/orchestrate_probe.py")
    orchp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(orchp)

    killed = {}
    monkeypatch.setattr(orchp, "kill_process_tree", lambda p: killed.setdefault("tree", p))
    calls = []

    class _R:
        stdout = "abc123\n"

    monkeypatch.setattr(orchp.subprocess, "run", lambda cmd, **kw: (calls.append(cmd), _R())[1])

    class _Proc:
        def wait(self, timeout=None):
            return 0

    orchp.reap_timed_out_runpro(_Proc(), 2, {})
    assert killed.get("tree") is not None                                    # process TREE killed (cross-platform)
    assert ["docker", "ps", "-q", "--filter", "name=obopro-w2-"] in calls    # worker-scoped reap filter
    assert ["docker", "rm", "-f", "abc123"] in calls                         # leaked container removed


def _verify_ctx(tmp_path, *, task_id="vhandler"):
    from ouroboros.tools.registry import ToolContext

    (tmp_path / "repo").mkdir(parents=True, exist_ok=True)
    (tmp_path / "drive").mkdir(parents=True, exist_ok=True)
    ctx = ToolContext(repo_dir=str(tmp_path / "repo"), drive_root=str(tmp_path / "drive"))
    ctx.task_id = task_id
    return ctx


def test_verify_and_record_handler_fail_closed(tmp_path):
    from ouroboros.tools.verify import _verify_and_record

    ctx = _verify_ctx(tmp_path)
    unknown = _verify_and_record(ctx, contract_kind="some_future_kind")
    assert "TOOL_ARG_ERROR" in unknown and "contract_kind='some_future_kind' must be one of" in unknown
    assert "expected_match='fuzzy' must be one of" in _verify_and_record(
        ctx, contract_kind="explicit_command", check=["true"], expected_match="fuzzy")
    # A run kind without a command names what arrived, a ready call and the kinds that need none.
    absent = _verify_and_record(ctx, contract_kind="explicit_command", expected="OK", cwd=".")
    assert "requires `check`, and this call sent none" in absent
    assert 'check=["pytest","-q"]' in absent and "no_visible_machine_contract" in absent
    assert absent.endswith("Nothing ran; no receipt was written.")
    for blank in ("   ", [], [""], {"cmd": "pytest"}):
        refused = _verify_and_record(ctx, contract_kind="explicit_command", check=blank)
        assert f"check={blank!r} holds no command" in refused, refused


def test_verify_and_record_handler_artifact_and_declared(tmp_path):
    from ouroboros.outcomes import read_verification_receipts
    from ouroboros.tools.verify import _verify_and_record

    ctx = _verify_ctx(tmp_path)
    # a relative artifact path resolves under the active workspace (the repo dir here)
    (Path(ctx.repo_dir) / "deliv.txt").write_text("x")
    assert "OBSERVED" in _verify_and_record(ctx, contract_kind="artifact_observation", artifact_paths=["deliv.txt"])
    assert "FAIL" in _verify_and_record(ctx, contract_kind="artifact_observation", artifact_paths=["missing.txt"])
    # triad #A: a RELATIVE path that escapes the workspace cannot probe arbitrary host
    # files (no `../../../etc/passwd` existence oracle) — confined post-resolution. The
    # security invariant is that the traversal target is NEVER OBSERVED. v6.57.0: an
    # out-of-scope path is now a POLICY refusal (REFUSED_OUT_OF_SCOPE), not a FAIL — it
    # still never yields an existence oracle.
    # Deep enough to escape to the FILESYSTEM ROOT on every OS (Windows runners
    # nest tmp ~7 levels under the user home — six `..` landed INSIDE home there,
    # where the deliberate user_files read lane made the probe an honest miss
    # instead of a refusal).
    escaped = _verify_and_record(
        ctx, contract_kind="artifact_observation",
        artifact_paths=["../" * 15 + "etc/passwd"],
    )
    assert "OBSERVED" not in escaped and "REFUSED_OUT_OF_SCOPE" in escaped
    # no_visible_machine_contract -> honest declared receipt (a declaration, not an examination)
    assert "DECLARED" in _verify_and_record(ctx, contract_kind="no_visible_machine_contract", check="manual UI review")
    assert read_verification_receipts(ctx.drive_root, "vhandler")[-1]["status"] == "declared"


def test_configured_actor_can_record_typed_zero_run_only_before_leaf(tmp_path):
    from ouroboros.outcomes import apply_receipt_absent_flag, read_verification_receipts
    from ouroboros.tools.verify import _verify_and_record

    ctx = _verify_ctx(tmp_path)
    ctx._configured_actor_bootstrap = {
        "route_id": "session-a",
        "work_order_fingerprint": "a" * 64,
        "physical_started": False,
    }
    # A decision outside the list is refused by naming what arrived, and records nothing.
    refused = _verify_and_record(ctx, contract_kind="delegation_zero_run", zero_run_decision="Maybe", zero_run_basis="x")
    assert "zero_run_decision='maybe' must be one of" in refused
    assert "zero_run_receipt_recorded" not in ctx._configured_actor_bootstrap
    result = _verify_and_record(
        ctx,
        contract_kind="delegation_zero_run",
        zero_run_decision="incomplete",
        zero_run_basis="The selected route was unavailable and no host child produced the requested artifact.",
    )
    assert "INCOMPLETE" in result
    receipt = read_verification_receipts(ctx.drive_root, ctx.task_id)[-1]
    assert receipt["contract_kind"] == "delegation_zero_run"
    assert receipt["status"] == "declared"
    assert receipt["zero_run_decision"] == "incomplete"
    assert receipt["physical_run_started"] is False
    assert ctx._configured_actor_bootstrap["zero_run_receipt_recorded"] is True

    # A no-leaf lifecycle decision is not proof that the requested deliverable
    # is correct.  It remains visible in the receipt packet without examining the
    # result: the record still says nothing examined it (receipt_absent).
    trace = {"tool_calls": [{"tool": "integrate_subagent_patch", "status": "ok", "args": {}}]}
    loop_outcome = {
        "outcome_axes": {
            "execution": {"status": "ok"},
            "objective": {"status": "satisfied"},
        },
        "final_answer": "host-side result",
    }
    apply_receipt_absent_flag(loop_outcome, trace, ctx.drive_root, ctx.task_id)
    assert "receipt_absent" in loop_outcome["outcome_axes"]["objective"]["warnings"]

    ctx._configured_actor_bootstrap["physical_started"] = True
    refused = _verify_and_record(
        ctx,
        contract_kind="delegation_zero_run",
        zero_run_decision="unknown",
        zero_run_basis="late decision",
    )
    assert "TOOL_ARG_ERROR" in refused

    duplicate = _verify_and_record(
        ctx,
        contract_kind="delegation_zero_run",
        zero_run_decision="incomplete",
        zero_run_basis="second terminal decision",
    )
    assert "TOOL_ARG_ERROR" in duplicate


def test_zero_run_receipt_write_failure_does_not_claim_terminal_truth(tmp_path, monkeypatch):
    from ouroboros.tools import verify as verify_module
    from ouroboros.tools.verify import _verify_and_record

    ctx = _verify_ctx(tmp_path)
    ctx._configured_actor_bootstrap = {
        "route_id": "session-a",
        "work_order_fingerprint": "a" * 64,
        "physical_started": False,
    }
    monkeypatch.setattr(verify_module, "append_verification_receipt", lambda *_a, **_k: False)
    result = _verify_and_record(
        ctx,
        contract_kind="delegation_zero_run",
        zero_run_decision="unknown",
        zero_run_basis="receipt sink unavailable",
    )
    assert "could not be durably written" in result
    assert ctx._configured_actor_bootstrap.get("zero_run_receipt_recorded") is not True


def test_recorded_zero_run_incomplete_or_unknown_stays_degraded():
    from ouroboros.outcomes import derive_loop_outcome
    from ouroboros.subagent_bootstrap import actor_first_terminal_projection

    for decision in ("incomplete", "unknown"):
        ctx = types.SimpleNamespace(
            task_id=f"zero-{decision}",
            _configured_actor_bootstrap={
                "zero_run_receipt_recorded": True,
                "zero_run_decision": decision,
                "zero_run_basis": "typed route evidence",
            },
        )
        fact, usage, trace = actor_first_terminal_projection(
            ctx, {"id": ctx.task_id}, {}, {}, None,
        )
        assert fact["status"] == decision
        outcome = derive_loop_outcome("typed zero-run", usage, trace)
        assert outcome["reason_code"] == f"configured_actor_{decision}"
        assert outcome["outcome_axes"]["execution"]["status"] == "degraded"


def test_zero_run_requires_actor_marker_and_basis(tmp_path):
    from ouroboros.tools.verify import _verify_and_record

    ctx = _verify_ctx(tmp_path)
    assert "TOOL_ARG_ERROR" in _verify_and_record(
        ctx,
        contract_kind="delegation_zero_run",
        zero_run_decision="unknown",
        zero_run_basis="not an actor",
    )
    ctx._configured_actor_bootstrap = {"physical_started": False}
    assert "TOOL_ARG_ERROR" in _verify_and_record(
        ctx,
        contract_kind="delegation_zero_run",
        zero_run_decision="unknown",
    )


def test_finalize_schedule_emission_surfaces_coop_tree_path():
    """The host-minted shared coop tree is a SCHEDULE-TIME fact: the parent's
    tool result names the tree path + the root=subagent_projects read recipe,
    so continuation waves stop rediscovering their own tree by trial and error
    (submarine waves). The request-only doctrine (v6.87.28) is untouched: no
    lane/executor resolution rides along."""
    from ouroboros.tools.control import _finalize_schedule_emission

    ctx = types.SimpleNamespace(task_id="p1", drive_root=Path(tempfile.mkdtemp()))
    out = _finalize_schedule_emission(ctx, {
        "task_ids": ["c1"],
        "requested_model_lane": "light",
        "objective": "build",
        "role": "builder",
        "depth": 1,
        "parent_task_id": "p1",
        "root_task_id": "p1",
        "emitted_modes": ["live"],
        "write_surface": "external_workspace",
        "write_root": ("/tmp/projects/coop_abc123", "host-minted coop tree"),
    })
    # Windows renders the tree path with os separators — pin label + tree name.
    assert "write root: " in out and "coop_abc123 (host-minted coop tree)" in out
    assert "root=subagent_projects" in out
    assert "coop_abc123" in out
    assert "effective_lane=" not in out  # request-only doctrine intact
    # A caller's own root is named without the read recipe; a read-only child has none.
    caller = _finalize_schedule_emission(ctx, {
        "task_ids": ["c3"], "objective": "build", "write_surface": "external_workspace",
        "write_root": ("/tmp/projects/mine", "caller")})
    assert "/tmp/projects/mine (caller)" in caller and "root=subagent_projects" not in caller
    out2 = _finalize_schedule_emission(ctx, {
        "task_ids": ["c2"], "requested_model_lane": "light", "objective": "probe",
        "role": "scout", "depth": 1, "parent_task_id": "p1", "root_task_id": "p1",
        "emitted_modes": ["live"],
    })
    assert "write root: none (read-only child)" in out2 and "coop" not in out2


def test_zero_run_complete_is_no_longer_writable(tmp_path):
    # Owner N3=A (2026-08-28): a zero-run "complete" is unverifiable self-report
    # and stopped being writable; the write enum is incomplete|unknown.
    from ouroboros.tools.verify import _verify_and_record

    ctx = _verify_ctx(tmp_path)
    ctx._configured_actor_bootstrap = {
        "route_id": "session-a",
        "work_order_fingerprint": "a" * 64,
        "physical_started": False,
    }
    refused = _verify_and_record(
        ctx,
        contract_kind="delegation_zero_run",
        zero_run_decision="complete",
        zero_run_basis="claims to be done without a run",
    )
    assert "TOOL_ARG_ERROR" in refused
    assert "incomplete, unknown" in refused
    from ouroboros.outcomes import read_verification_receipts

    assert read_verification_receipts(tmp_path, ctx.task_id) == []


def test_historical_zero_run_complete_projects_unknown_with_disclosure():
    # Owner R4=A (2026-08-28): a historical "complete" receipt still fences a
    # second start on read, but the terminal projection degrades it to UNKNOWN
    # with disclosure — never a clean terminal.
    from ouroboros.subagent_bootstrap import actor_first_terminal_projection

    ctx = types.SimpleNamespace(
        task_id="actor-historical",
        _configured_actor_bootstrap={
            "zero_run_receipt_recorded": True,
            "zero_run_decision": "complete",
            "zero_run_basis": "pre-charter self-report",
            "route_available": True,
            "physical_started": False,
        },
    )
    fact, usage, trace = actor_first_terminal_projection(
        ctx, {"id": "actor-historical"}, {}, {}, None,
    )
    assert fact is not None
    assert fact["status"] == "unknown"
    assert fact["reason"] == "historical_zero_run_complete"
    assert fact["zero_run_decision"] == "complete"
    assert usage["actor_first_terminal"]["reason"] == "historical_zero_run_complete"
    assert trace["actor_first_terminal"]["status"] == "unknown"
