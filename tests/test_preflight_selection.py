"""Selected execution, honest inventory and the full-only consumer boundaries."""
import json
from types import SimpleNamespace

import pytest

from ouroboros import commit_admission as admission, preflight_runner as pr, test_selection as selection
from ouroboros.preflight_qualification import adoption_test_evidence, full_proof_error
from tests.test_preflight_runner import _make_repo, _commit_all
from tests.test_preflight_receipts import receipt_python  # noqa: F401 -- dependency-only lane fixture

pytestmark = pytest.mark.serial  # finite nested pytest controllers; one at a time


@pytest.fixture
def candidate(tmp_path, monkeypatch, receipt_python):  # noqa: F811 -- pytest fixture injection
    monkeypatch.setenv("OUROBOROS_AGENT_PYTHON", receipt_python)
    monkeypatch.setenv("OUROBOROS_PREFLIGHT_TEST_WORKERS", "2")
    monkeypatch.setenv("OURO_TEST_WORKER_BUDGET", "2")  # nested slots: the selected gate must really run two workers
    monkeypatch.delenv("OUROBOROS_PREFLIGHT_SERIAL", raising=False)
    monkeypatch.delenv("OUROBOROS_PREFLIGHT_TIMEOUT_SEC", raising=False)
    monkeypatch.setattr(selection, "FLOOR_TESTS", {"tests/test_floor.py": "floor witness"})
    repo = _make_repo(tmp_path, {
        ".gitignore": "__pycache__/\n.pytest_cache/\n",
        "tests/test_floor.py": "def test_floor(): assert True\n",
        "leaf.py": "value = 1\n",
        "tests/test_leaf.py": "from leaf import value\ndef test_leaf(): assert value == 2\n",
        "tests/test_unrelated.py": "raise AssertionError('unrelated collection must be skipped')\n",
    })
    (repo / "leaf.py").write_text("value = 2\n", encoding="utf-8")
    return SimpleNamespace(repo_dir=repo, drive_root=tmp_path / "data", task_id="selected-proof",
                           task_metadata={}, current_task_type="task", emit_progress_fn=lambda *a: None)


def test_real_selected_run_records_inventory_and_cannot_qualify_managed_update(candidate, monkeypatch):
    from ouroboros import preflight_receipts
    from ouroboros.tools import git

    ctx = candidate
    assert pr.run_hermetic_pytest(ctx.repo_dir, ctx=ctx, selected=True, phase=pr.PRE_COMMIT_PHASE) is None
    proof = ctx._preflight_test_proof
    assert proof and proof.tier == "selected"
    assert not admission.preflight_test_proof_matches(ctx, ctx.repo_dir)
    assert admission.preflight_test_proof_matches(ctx, ctx.repo_dir, require_full=False)
    assert not admission.preflight_test_workload_unchanged(proof, ctx.repo_dir, timeout=1800, pytest_args=None)
    assert full_proof_error(ctx, ctx.repo_dir)
    monkeypatch.setattr(git, "_authorized_managed_update_resolver", lambda c: True)
    assert git._managed_candidate_needs_proof(ctx)
    row = ctx._preflight_test_disclosure
    assert row["tier"] == "selected" and row["candidate"] == proof.tree
    assert row["base"] == proof.head and row["policy"] == proof.selection.policy
    assert row["expected_files"] == ["tests/test_floor.py", "tests/test_leaf.py"]
    assert row["omitted_files"] == 1
    assert row["receipt"]["status"] == "declined"
    assert "selected" in row["receipt"]["reason"]
    assert not list(ctx.drive_root.glob("state/preflight_receipts/*.json"))
    inventory = row["inventory"]  # durable records keep counts, a digest and only the ids worth reading
    assert inventory["parallel"]["expected"] == inventory["parallel"]["executed"] == 2
    assert inventory["parallel"]["missing"] == inventory["parallel"]["unexpected"] == []
    assert len(inventory["parallel"]["digest"]) == 64 and inventory["serial"]["expected"] == 0
    exhibit = preflight_receipts.execution_exhibit(ctx)
    assert exhibit["tier"] == "selected" and exhibit["inventory"] == inventory
    assert "selected" in preflight_receipts.execution_section(exhibit)
    monkeypatch.setattr(pr, "_execute_pytest_pass", lambda *a: pytest.fail("unchanged selected proof should reuse"))
    assert pr.run_hermetic_pytest(ctx.repo_dir, ctx=ctx, selected=True, phase=pr.PRE_COMMIT_PHASE) is None
    assert ctx._preflight_test_disclosure["action"] == "reused"
    assert ctx._preflight_test_disclosure["inventory"] == inventory


def test_selected_widening_executes_the_named_consumer(candidate):
    error = pr.run_hermetic_pytest(candidate.repo_dir, ctx=candidate, selected=True, phase=pr.PRE_COMMIT_PHASE,
                                  extra_tests=["tests/test_unrelated.py"], test_reason="dynamic dispatch")
    assert error and "unrelated collection" in error
    assert candidate._preflight_test_proof is None


def test_apparatus_change_executes_full_and_reports_actual_failure(candidate):
    with (candidate.repo_dir / "pytest.ini").open("a", encoding="utf-8") as stream:
        stream.write("\n# verifier changed\n")
    error = pr.run_hermetic_pytest(candidate.repo_dir, ctx=candidate, selected=True, phase=pr.PRE_COMMIT_PHASE)
    assert error and "unrelated collection" in error
    assert candidate._preflight_test_proof is None


@pytest.mark.parametrize("mode", ["missing", "empty"])
def test_required_floor_cannot_silently_disappear(candidate, mode):
    (candidate.repo_dir / "tests/test_unrelated.py").write_text("def test_other(): pass\n", encoding="utf-8")
    path = candidate.repo_dir / "tests/test_floor.py"
    path.unlink() if mode == "missing" else path.write_text("# no witness\n", encoding="utf-8")
    error = pr.run_hermetic_pytest(candidate.repo_dir, ctx=candidate, selected=True, phase=pr.PRE_COMMIT_PHASE)
    assert error and "floor" in error
    assert candidate._preflight_test_proof is None


def test_a_selected_run_of_uncommitted_bytes_is_refused_durability_before_installation_probe(candidate, monkeypatch,
                                                                                               tmp_path):
    from ouroboros.preflight_receipts import GateReceipt
    plan = selection.select_tests(candidate.repo_dir)
    proof = admission.capture_preflight_test_subject(candidate.repo_dir, selection=plan)
    passes = pr._preflight_pass_specs(probe_module="nonce")
    monkeypatch.setattr("ouroboros.preflight_receipts._installation_identity", lambda *a: pytest.fail("selected proof"))
    receipt = GateReceipt.prepare(candidate, proof, passes, "nonce", "python", tmp_path, candidate.repo_dir)
    assert not receipt.covered() and receipt.path is None
    assert receipt.reason == "a selected run is durable only as an exact commit"


def test_undisclosed_execution_keeps_the_proofs_actual_tier(tmp_path, monkeypatch):
    from ouroboros.preflight_receipts import execution_exhibit, execution_section
    from ouroboros.preflight_inventory import scope_record

    workload = ((("custom", ("tests/test_leaf.py",), False),), 1800, "env", (), (), "probe")
    proof = admission.PreflightTestProof("tree", "index", workload, "head")
    ctx = SimpleNamespace(repo_dir=tmp_path, _preflight_test_proof=proof, _preflight_tests_passed=True)
    monkeypatch.setattr(admission, "preflight_test_proof_matches", lambda *a, **kw: True)
    exhibit = execution_exhibit(ctx)
    assert exhibit["tier"] == "selected" and exhibit["evidence"] == "undisclosed"
    assert "The selected hermetic test gate" in execution_section(exhibit)
    assert scope_record(proof)["rule"] == "custom pytest selection"


def test_adoption_accepts_a_full_receipt_of_exactly_the_named_commit(candidate):
    from ouroboros.preflight_receipts import capture_executing_source_baseline

    capture_executing_source_baseline()  # receipts bind the loaded code generation, as server.py does at import
    repo = candidate.repo_dir
    (repo / "tests/test_unrelated.py").write_text("def test_other(): assert True\n", encoding="utf-8")
    _commit_all(repo)
    sha = pr._run_git(repo, ["rev-parse", "HEAD"]).stdout.strip()
    assert pr.run_hermetic_pytest(repo, ctx=candidate) is None  # a full green run mints the durable receipt
    [record] = [json.loads(path.read_text(encoding="utf-8")) for path in candidate.drive_root.glob(
        "state/preflight_receipts/*.json")]
    # A full record keeps exactly its historical shape, so receipts written before selected tiers still match.
    assert record["schema"] == 1 and "selection" not in record
    assert set(record["identity"]) == {"schema", "head", "tree", "index_tree", "workload", "installation", "source"}
    assert adoption_test_evidence(candidate.drive_root, repo, sha, ctx=candidate)["source"] == "proof"
    (repo / "leaf.py").write_text("value = 99\n", encoding="utf-8")  # later dirty bytes do not matter
    evidence = adoption_test_evidence(candidate.drive_root, repo, sha)
    assert evidence["tier"] == "full" and evidence["source"] == "receipt" and evidence["head"] == sha
    _commit_all(repo)
    later = pr._run_git(repo, ["rev-parse", "HEAD"]).stdout.strip()
    with pytest.raises(ValueError, match="full_tests=true"):  # another commit has no full run on record
        adoption_test_evidence(candidate.drive_root, repo, later)


def test_a_green_run_of_uncommitted_bytes_never_qualifies_the_commit_beneath_them(candidate):
    from ouroboros.preflight_receipts import capture_executing_source_baseline

    capture_executing_source_baseline()
    repo = candidate.repo_dir
    (repo / "tests/test_unrelated.py").write_text("def test_other(): assert True\n", encoding="utf-8")
    (repo / "leaf.py").write_text("value = 3\n", encoding="utf-8")  # the committed body fails test_leaf
    _commit_all(repo)
    sha = pr._run_git(repo, ["rev-parse", "HEAD"]).stdout.strip()
    (repo / "leaf.py").write_text("value = 2\n", encoding="utf-8")  # an uncommitted repair turns the run green
    assert pr.run_hermetic_pytest(repo, ctx=candidate) is None and candidate._preflight_test_proof.head == sha
    with pytest.raises(ValueError, match="full_tests=true"):  # neither the proof nor its receipt is that commit
        adoption_test_evidence(candidate.drive_root, repo, sha, ctx=candidate)


def test_a_run_admitted_one_worker_still_leaves_adoption_its_receipt(candidate, monkeypatch):
    from ouroboros.preflight_receipts import capture_executing_source_baseline

    capture_executing_source_baseline()
    monkeypatch.setenv("OURO_TEST_WORKER_BUDGET", "1")  # a small machine's budget admits one worker
    repo = candidate.repo_dir
    (repo / "tests/test_unrelated.py").write_text("def test_other(): assert True\n", encoding="utf-8")
    _commit_all(repo)
    sha = pr._run_git(repo, ["rev-parse", "HEAD"]).stdout.strip()
    assert pr.run_hermetic_pytest(repo, ctx=candidate) is None
    assert candidate._preflight_test_disclosure["receipt"]["status"] == "published"
    evidence = adoption_test_evidence(candidate.drive_root, repo, sha)  # the supervisor path has no task ctx
    assert evidence["source"] == "receipt" and evidence["head"] == sha


def test_evolution_publication_refuses_selected_but_accepts_full(candidate, monkeypatch):
    from ouroboros.tools import git
    from ouroboros.tools.git_evolution import _evolution_test_publication_stop

    ctx = candidate
    ctx.current_task_type = "evolution"
    monkeypatch.setattr(git, "_evolution_commit_authority", lambda *a, **k: ({}, {"ok": True}))
    assert pr.run_hermetic_pytest(ctx.repo_dir, ctx=ctx, selected=True, phase=pr.PRE_COMMIT_PHASE) is None
    assert "EVOLUTION_PUBLICATION_STOPPED" in _evolution_test_publication_stop(ctx, ctx._preflight_test_proof.head, "")
    (ctx.repo_dir / "tests/test_unrelated.py").write_text("def test_other(): assert True\n", encoding="utf-8")
    _commit_all(ctx.repo_dir)  # evolution's forced full run follows its commit, on exactly that tree
    assert pr.run_hermetic_pytest(ctx.repo_dir, ctx=ctx) is None
    assert _evolution_test_publication_stop(ctx, ctx._preflight_test_proof.head, "") == ""


def test_adoption_without_full_evidence_is_refused_before_writing_handoff(tmp_path, monkeypatch):
    from ouroboros import body_adoption, preflight_qualification
    from tests.test_body_adoption import Scene

    real = preflight_qualification.adoption_test_evidence
    scene = Scene(tmp_path, monkeypatch)  # stubs the evidence for its miniature bodies
    scene.commit()
    monkeypatch.setattr(preflight_qualification, "adoption_test_evidence", real)  # nothing is on record here
    with pytest.raises(body_adoption.AdoptionRefused, match="full_tests=true") as caught:
        scene.authorize()
    assert caught.value.code == "ADOPTION_TESTS_REQUIRED"
    assert not body_adoption.read(scene.data)


def test_durable_records_carry_counts_not_the_whole_inventory():
    from ouroboros.preflight_inventory import compact_inventory

    ids = [f"tests/test_x.py::test_{i}" for i in range(500)]
    row = {"expected": ids, "executed": ids + [ids[0]], "skipped": [], "failed": [ids[2]], "exitcode": 1}
    compact = compact_inventory({"parallel": row})["parallel"]
    assert compact["expected"] == 500 and compact["executed"] == 501
    assert compact["unexpected"] == [ids[0]] and compact["missing"] == [] and compact["failed"] == [ids[2]]
    assert len(json.dumps(compact)) < 2000  # repeated in events, exhibits, ledgers and handoffs


@pytest.mark.parametrize("selected", [("tests/deep/test_b.py", "tests/test_a.py"), ("tests/node_modules/test_d.py",)],
                         ids=["reached", "unreachable"])
def test_a_selected_pass_walks_tests_once_and_collects_exactly_its_modules(tmp_path, selected):
    """A selected pass points pytest at ``tests/`` once (a module named as an argument costs a walk of the
    tree of its own) and keeps the selected modules before any is imported: it collects exactly their
    nodes. A selected module the walk cannot reach fails the pass instead of vanishing from it."""
    import os
    import sys

    from ouroboros import preflight_inventory as inventory
    from tests.group_run import run_in_group

    repo = tmp_path / "repo"
    for path, text in {"pytest.ini": "[pytest]\n", "tests/test_a.py": "def test_one(): pass\ndef test_two(): pass\n",
                       "tests/deep/test_b.py": "def test_three(): pass\n",
                       "tests/node_modules/test_d.py": "def test_four(): pass\n",
                       "tests/test_c.py": "raise AssertionError('an unselected module is never imported')\n"}.items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(text, encoding="utf-8")
    scratch = tmp_path / "scratch"
    first = inventory.selection_passes(SimpleNamespace(tier="selected", files=selected), temp_root=scratch)[0]
    scope = first.args[:first.args.index("tests/") + 1]
    assert scope == [f"--select-test={path}" for path in selected] + ["tests/"]
    done = run_in_group([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", *scope,
                         "-p", first.args[-1]], timeout=120, cwd=repo,
                        env={**os.environ, "PYTHONPATH": str(pr._probe_dir(scratch))})
    expected = inventory.read_inventory(scratch, "parallel")[0].get("expected")
    if selected[0].startswith("tests/node_modules"):
        assert done.returncode != 0 and "never reached" in done.stdout + done.stderr, done.stdout + done.stderr
        assert not expected
    else:
        assert done.returncode == 0, done.stdout + done.stderr
        assert expected == ["tests/deep/test_b.py::test_three", "tests/test_a.py::test_one", "tests/test_a.py::test_two"]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unfinished"])
def test_inventory_rejects_incomplete_or_duplicate_execution(tmp_path, mutation):
    from ouroboros.preflight_inventory import read_inventory
    rows = [{"event": "start"}, {"event": "collection", "ids": ["tests/test_x.py::test_x"]},
            {"event": "report", "nodeid": "tests/test_x.py::test_x", "when": "teardown", "outcome": "passed"},
            {"event": "finish", "exitcode": 0}]
    if mutation == "missing":
        rows.pop(2)
    elif mutation == "duplicate":
        rows.insert(2, rows[2])
    else:
        rows.pop()
    (tmp_path / "inventory-parallel.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    assert read_inventory(tmp_path, "parallel")[1]


def _coherent(schema=1, **identity):
    full = {"schema": schema, "head": "h", "tree": "t", "index_tree": "t", "workload": "w", "installation": "i",
            "source": "s", **identity}
    return {"schema": schema, "identity": full, "result": "passed", "authenticity": "unsigned",
            "lanes": {"node": "absent", "parallel_workers": 2, "admitted_workers": 2, "serial": "passed"},
            "pass_seconds": {"parallel": 1.0, "serial": 1.0}, "created_at": "now", "task_id": "t", "phase": "p"}


SCOPE = {"base": "b", "policy": "p", "changed": ["x.py"], "extra_tests": [], "full_tests": False, "test_reason": ""}


@pytest.mark.parametrize("document,problem", [
    (_coherent(), ""),
    (_coherent(2, tier="selected", scope=SCOPE), ""),
    (_coherent(1, tier="selected", scope=SCOPE), "schema does not match"),  # a selected record relabelled full
    (_coherent(2, scope=SCOPE), "schema does not match"),
    (_coherent(2, tier="selected", scope={**SCOPE, "full_tests": True}), "the selection scope is incomplete"),
    (_coherent(2, tier="selected", scope={k: v for k, v in SCOPE.items() if k != "base"}),
     "the selection scope is incomplete"),
    (_coherent(3, tier="selected", scope=SCOPE), "schema does not match"),
])
def test_record_coherence_keeps_the_tiers_apart(document, problem):
    from ouroboros.preflight_receipts import _record_problem

    assert _record_problem(document, document["identity"], False) == problem


def test_a_selected_scope_qualifies_only_its_exact_base_policy_paths_and_declared_widening(tmp_path):
    from ouroboros.preflight_qualification import _scope_gap

    repo = _make_repo(tmp_path, {"tests/test_kept.py": "def test_k(): pass\n"})
    sha = pr._run_git(repo, ["rev-parse", "HEAD"]).stdout.strip()
    scope = {**SCOPE, "base": "serving", "policy": selection.policy_digest(), "changed": ["a.py", "b.py"],
             "extra_tests": ["tests/test_kept.py"]}
    local = {"onto": "serving", "paths": ["a.py"], "widening": {"extra_tests": ["tests/test_kept.py"]}}
    assert _scope_gap(repo, sha, scope, local) == ""
    assert "not the serving HEAD" in _scope_gap(repo, sha, {**scope, "base": "older"}, local)
    assert "another selection policy" in _scope_gap(repo, sha, {**scope, "policy": "other"}, local)
    assert "did not cover c.py" in _scope_gap(repo, sha, scope, {**local, "paths": ["a.py", "c.py"]})
    assert "declared broad impact" in _scope_gap(repo, sha, scope, {**local, "widening": {"full_tests": True}})
    assert "declared indirect consumers tests/test_kept.py" in _scope_gap(repo, sha, {**scope, "extra_tests": []}, local)
    # A declared consumer the commit no longer carries has nothing left to run.
    gone = {**local, "widening": {"extra_tests": ["tests/test_gone.py"]}}
    assert _scope_gap(repo, sha, scope, gone) == ""
