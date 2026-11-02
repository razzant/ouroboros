"""Local adoption on exact selected qualification (owner choice A, #2110).

A miniature body carrying the real package-init hook gains a tiny suite. Its candidate
commits run the REAL post-commit gate (``tools.git._run_pre_push_tests``: nested pytest on a
dependency-only interpreter), and the REAL ``body_adoption`` reads what that run left: the
task's own proof or the durable selected record. No lane outcome is stubbed.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

from ouroboros import body_adoption, body_candidate, preflight_qualification, preflight_receipts, test_selection
from ouroboros import preflight_runner as pr
from ouroboros.tools import git as git_tools
from tests.body_candidate_support import REPO, candidate_commit, git, isolate, make_ctx, make_serving
from tests.test_preflight_receipts import receipt_python  # noqa: F401 -- dependency-only lane fixture

pytestmark = pytest.mark.serial  # finite nested pytest controllers; one at a time

REAL_EVIDENCE = preflight_qualification.adoption_test_evidence  # isolate() stubs it for switching tests
FLOOR = {"tests/test_floor.py": "floor witness"}
SUITE = {
    "pytest.ini": "[pytest]\nmarkers =\n    serial: serial lane\n",
    "pyproject.toml": '[project]\nname = "toy"\nversion = "1.0.0"\ndependencies = []\n',
    "tests/test_floor.py": "def test_floor():\n    assert True\n",
    "tests/test_mod_a.py": "from ouroboros import mod_a\n\n\ndef test_a():\n    assert mod_a.GEN\n",
    "tests/test_mod_b.py": "from ouroboros import mod_b\n\n\ndef test_b():\n    assert mod_b.GEN\n",
    # Reads mod_a by path, which no import graph shows: only authored widening runs it.
    "tests/test_consumer.py": ("import pathlib\n\n\ndef test_consumer():\n    text = (pathlib.Path(__file__)"
                               ".parents[1] / 'ouroboros' / 'mod_a.py').read_text()\n    assert 'BROKEN' not in text\n"),
}
A_GOOD, A_BROKEN, B_GOOD = ({"ouroboros/mod_a.py": "GEN = 'GEN_CAND'\n"}, {"ouroboros/mod_a.py": "GEN = 'BROKEN'\n"},
                            {"ouroboros/mod_b.py": "GEN = 'GEN_CAND'\n"})
DEFAULTS = {"extra_tests": (), "full_tests": False, "test_reason": ""}


class Scene:
    def __init__(self, tmp_path, monkeypatch, python):
        self.serving = make_serving(tmp_path)
        for relative, text in SUITE.items():
            target = self.serving / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        git(self.serving, "add", "-A")
        git(self.serving, "commit", "-q", "-m", "suite")
        self.data = isolate(monkeypatch, tmp_path)
        monkeypatch.setattr(preflight_qualification, "adoption_test_evidence", REAL_EVIDENCE)
        for key, value in {"OUROBOROS_RUNTIME_MODE": "pro", "OUROBOROS_AGENT_PYTHON": python,
                           "OUROBOROS_PRE_PUSH_TESTS": "1", "OUROBOROS_PREFLIGHT_TEST_WORKERS": "2",
                           "OURO_TEST_WORKER_BUDGET": "2"}.items():
            monkeypatch.setenv(key, value)
        for key in ("OUROBOROS_PREFLIGHT_SERIAL", "OUROBOROS_PREFLIGHT_TIMEOUT_SEC"):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setattr(test_selection, "FLOOR_TESTS", dict(FLOOR))
        preflight_receipts.capture_executing_source_baseline()  # receipts bind the loaded code, as server.py does
        self.ctx = make_ctx(self.serving, self.data, "root-qualify")
        body_candidate.prepare(self.ctx)
        self.candidate = pathlib.Path(self.ctx.repo_dir)
        self.old = git(self.serving, "rev-parse", "HEAD")

    def commit(self, files, *, force=False, **request):
        """One reviewed candidate commit and its REAL post-commit gate, in the commit tool's order."""
        sha = candidate_commit(self.candidate, files=files)
        self.ctx._commit_test_request = {**DEFAULTS, **request, "extra_tests": tuple(request.get("extra_tests", ()))}
        error = git_tools._run_pre_push_tests(self.ctx, force=force)
        body_candidate.record_reviewed_commit(self.ctx, sha)
        return sha, error

    def records(self, schema=None):
        root = self.data / "state" / preflight_receipts.RECEIPTS_DIR
        documents = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(root.glob("*.json"))]
        return [document for document in documents if schema in (None, document.get("schema"))]

    def forget_records(self):
        for path in (self.data / "state" / preflight_receipts.RECEIPTS_DIR).glob("*.json"):
            path.unlink()

    def refused(self, sha, match):
        with pytest.raises(body_adoption.AdoptionRefused, match=match) as caught:
            body_adoption.authorize(self.ctx, sha, reason="adopt")
        assert caught.value.code == "ADOPTION_TESTS_REQUIRED" and not body_adoption.read(self.data)
        return caught.value.text


@pytest.fixture
def scene(tmp_path, monkeypatch, receipt_python):  # noqa: F811 -- pytest fixture injection
    return Scene(tmp_path, monkeypatch, receipt_python)


_NEW_PROCESS = r'''
import json, sys
repo, data, candidate, commit, local, floor = sys.argv[1:7]
sys.path.insert(0, repo)
from ouroboros import preflight_qualification, test_selection
test_selection.FLOOR_TESTS = json.loads(floor)  # the producer's floor: the same loaded selection policy
print(json.dumps({"local": preflight_qualification.adoption_test_evidence(data, candidate, commit, local=json.loads(local)),
                  "full_only": preflight_qualification.receipt_for_commit(data, candidate, commit)}))
'''


def test_a_selected_commit_qualifies_its_adoption_from_its_proof_and_its_durable_record(scene, tmp_path):
    sha, error = scene.commit(A_GOOD)

    assert error is None
    proof = scene.ctx._preflight_test_proof
    assert (proof.tier, proof.head, proof.selection.base) == ("selected", sha, scene.old)
    assert set(proof.selection.files) == {"tests/test_floor.py", "tests/test_mod_a.py"}
    [record] = scene.records()
    assert record["schema"] == preflight_receipts.SELECTED_SCHEMA and record["identity"]["tier"] == "selected"
    assert record["identity"]["scope"] == proof.selection.scope() and record["selection"]["file_count"] == 2
    # Full-only readers skip it: no full receipt exists and none is minted from it.
    assert preflight_qualification.receipt_for_commit(scene.data, scene.candidate, sha) == {}

    tests = body_adoption.authorize(scene.ctx, sha, reason="adopt")["tests"]
    assert (tests["tier"], tests["source"], tests["base"], tests["changed"]) == ("selected", "proof", scene.old, 1)
    assert tests["policy"] == test_selection.policy_digest() and tests["file_count"] == 2
    body_adoption.abandon(scene.data, "test")

    # The supervisor's context-free path has no proof: the durable record qualifies the same commit.
    assert body_adoption.authorize_for_task(scene.data, "root-qualify", sha, "evo") is True
    assert body_adoption.read(scene.data)["tests"]["source"] == "receipt"
    body_adoption.abandon(scene.data, "test")

    script = tmp_path / "reader.py"
    script.write_text(_NEW_PROCESS, encoding="utf-8")
    local = {"onto": scene.old, "paths": ["ouroboros/mod_a.py"], "widening": {"extra_tests": [], "full_tests": False}}
    read = subprocess.run([sys.executable, "-I", str(script), str(REPO), str(scene.data), str(scene.candidate), sha,
                           json.dumps(local), json.dumps(FLOOR)], capture_output=True, text=True, timeout=120,
                          env={**os.environ, "OUROBOROS_DATA_DIR": str(scene.data)})
    assert read.returncode == 0, read.stderr[-3000:]
    later = json.loads(read.stdout.strip().splitlines()[-1])
    assert (later["local"]["tier"], later["local"]["source"], later["local"]["head"]) == ("selected", "receipt", sha)
    assert later["full_only"] == {} and scene.records(preflight_receipts.RECEIPT_SCHEMA) == []


def test_a_later_commit_is_qualified_by_the_whole_candidate_delta_never_the_last_commit(scene):
    first, _ = scene.commit(A_GOOD)
    second, error = scene.commit(B_GOOD)

    assert error is None
    selection = scene.ctx._preflight_test_proof.selection
    assert selection.base == scene.old and {"ouroboros/mod_a.py", "ouroboros/mod_b.py"} <= set(selection.changed)
    assert {"tests/test_mod_a.py", "tests/test_mod_b.py"} <= set(selection.files)
    assert body_adoption.authorize(scene.ctx, second, reason="adopt")["tests"]["changed"] == 2
    body_adoption.abandon(scene.data, "test")

    # The scope the post-commit gate used before (HEAD~1: the last commit alone) cannot qualify the switch.
    scene.forget_records()
    assert pr.run_hermetic_pytest(scene.candidate, ctx=scene.ctx, selected=True) is None
    assert scene.ctx._preflight_test_proof.selection.base == first
    assert f"selected against {first[:12]}, not the serving HEAD {scene.old[:12]}" in scene.refused(second, "")


def test_an_earlier_commits_declared_consumer_keeps_running_and_catches_a_later_default_commit(scene):
    first, error = scene.commit(A_GOOD, extra_tests=["tests/test_consumer.py"], test_reason="reads mod_a by path")
    assert error is None and "tests/test_consumer.py" in scene.ctx._preflight_test_proof.selection.files

    # A fresh commit_reviewed call with the defaults: the declared consumer still runs, and B breaks it.
    second, error = scene.commit(A_BROKEN)

    assert error and "test_consumer" in error and scene.ctx._preflight_test_proof is None
    assert scene.records(preflight_receipts.SELECTED_SCHEMA)[-1]["identity"]["head"] == first
    scene.refused(second, "no qualifying test run")


def test_adoption_refuses_a_selected_run_that_left_out_a_declared_consumer(scene):
    scene.commit(A_GOOD, extra_tests=["tests/test_consumer.py"], test_reason="reads mod_a by path")
    second = candidate_commit(scene.candidate, files=B_GOOD)
    scene.ctx._commit_test_request = dict(DEFAULTS)
    body_candidate.record_reviewed_commit(scene.ctx, second)
    # Exactly B against the serving HEAD, but without the widening its delta declared (an older
    # producer, or a history unreadable at run time): the gate passes, adoption re-checks it.
    assert pr.run_hermetic_pytest(scene.candidate, ctx=scene.ctx, selected=True, base=scene.old) is None
    assert "did not run the declared indirect consumers tests/test_consumer.py" in scene.refused(second, "")


@pytest.mark.parametrize("source", ["legacy", "recovery"])
def test_missing_historical_scope_keeps_full_instead_of_inventing_defaults(scene, source):
    if source == "legacy":
        first, error = scene.commit(A_GOOD, full_tests=True, test_reason="broad indirect impact")
        assert error is None
        # Pre-upgrade rows carried reviewed provenance, not the new scope map.
        body_candidate._update_row(body_candidate.descriptor(scene.ctx), lambda row: row.pop("test_requests", None))
    else:
        first = candidate_commit(scene.candidate, files=A_GOOD)
        body_candidate.record_reviewed_commit(None, first, row=body_candidate.descriptor(scene.ctx))
    second, error = scene.commit(B_GOOD)
    assert error is None and scene.ctx._preflight_test_proof.tier == "full"
    assert body_adoption.authorize(scene.ctx, second, reason="old scope unknown")["tests"]["tier"] == "full"
    body_adoption.abandon(scene.data, "test")
    scene.forget_records()
    # Neither an independently green selection nor an intent recovery makes the old scope known.
    assert pr.run_hermetic_pytest(scene.candidate, ctx=scene.ctx, selected=True, base=scene.old) is None
    body_candidate.record_reviewed_commit(None, second, row=body_candidate.descriptor(scene.ctx))
    assert "prior reviewed test scope is unknown" in scene.refused(second, "")


def test_an_earlier_full_declaration_keeps_the_delta_full_until_that_commit_is_adopted(scene):
    first, error = scene.commit(A_GOOD, full_tests=True, test_reason="broad impact")
    assert error is None and scene.ctx._preflight_test_proof.tier == "full"

    second, error = scene.commit(B_GOOD)

    proof = scene.ctx._preflight_test_proof
    assert error is None and proof.tier == "full" and proof.selection.test_reason == "broad impact"
    assert body_adoption.authorize(scene.ctx, second, reason="adopt")["tests"]["tier"] == "full"
    body_adoption.abandon(scene.data, "test")
    scene.forget_records()
    assert pr.run_hermetic_pytest(scene.candidate, ctx=scene.ctx, selected=True, base=scene.old) is None
    assert "declared broad impact (full_tests): broad impact" in scene.refused(second, "")


def test_a_selection_against_another_serving_head_cannot_qualify_even_when_its_paths_cover_the_switch(scene):
    # S1 changes a dependency in a release carrier; the candidate merges it and bumps only the version.
    s1 = candidate_commit(scene.serving, files={"pyproject.toml": SUITE["pyproject.toml"].replace(
        "[]", '["requests"]')})
    git(scene.candidate, "merge", "-q", "--no-ff", "--no-edit", "ouroboros")
    merge = git(scene.candidate, "rev-parse", "HEAD")
    scene.ctx._commit_test_request = dict(DEFAULTS)
    body_candidate.record_reviewed_commit(scene.ctx, merge)
    bumped = (scene.candidate / "pyproject.toml").read_text(encoding="utf-8").replace('"1.0.0"', '"1.0.1"')
    sha, error = scene.commit({**A_GOOD, "pyproject.toml": bumped})
    assert error is None and scene.ctx._preflight_test_proof.selection.base == s1
    # Against S1 the carrier only moved its version; against S0 it changes a dependency.
    assert test_selection.select_tests(scene.candidate, base=s1).tier == "selected"
    assert test_selection.select_tests(scene.candidate, base=scene.old).tier == "full"

    git(scene.serving, "reset", "-q", "--hard", scene.old)  # the owner restores the serving line to S0
    switch = {row["path"] for row in body_adoption._switch_rows(scene.serving, scene.old, sha)}
    assert switch <= set(scene.ctx._preflight_test_proof.selection.changed)  # paths alone would accept it
    assert f"selected against {s1[:12]}, not the serving HEAD {scene.old[:12]}" in scene.refused(sha, "")


def test_red_dirty_unrun_and_other_policy_evidence_never_qualifies(scene, monkeypatch):
    broken, error = scene.commit({"ouroboros/mod_a.py": "GEN = ''\n"})  # test_mod_a fails
    assert error and scene.ctx._preflight_test_proof is None and scene.records() == []
    scene.refused(broken, "no qualifying test run")

    (scene.candidate / "ouroboros/mod_a.py").write_text("GEN = 'FIXED'\n", encoding="utf-8")
    assert git_tools._run_pre_push_tests(scene.ctx) is None  # an uncommitted repair turns the run green
    assert scene.ctx._preflight_test_proof.head == broken and scene.records() == []  # not an exact commit
    assert scene.ctx._preflight_test_disclosure["receipt"]["reason"] == "a selected run is durable only as an exact commit"
    scene.refused(broken, "no qualifying test run")

    repaired = candidate_commit(scene.candidate, files={"ouroboros/mod_a.py": "GEN = 'FIXED'\n"})
    body_candidate.record_reviewed_commit(scene.ctx, repaired)
    assert git_tools._post_commit_result(scene.ctx, "repair", True, [""]) is None  # skip_tests: no run at all
    scene.refused(repaired, "no qualifying test run")

    assert git_tools._run_pre_push_tests(scene.ctx) is None and scene.records()
    monkeypatch.setattr(test_selection, "FLOOR_TESTS", {"tests/test_floor.py": "another invariant"})
    assert "it ran under another selection policy" in scene.refused(repaired, "")


def test_a_later_context_reuses_an_identical_selected_record_and_another_workload_runs_again(scene, monkeypatch):
    sha, _ = scene.commit(A_GOOD)
    fresh = make_ctx(scene.serving, scene.data, "root-qualify")
    body_candidate.prepare(fresh)
    fresh._commit_test_request = dict(DEFAULTS)
    executed = pr._execute_pytest_pass
    monkeypatch.setattr(pr, "_execute_pytest_pass", lambda *a: pytest.fail("an identical selected record is reused"))

    assert git_tools._run_pre_push_tests(fresh) is None
    assert fresh._preflight_test_disclosure["receipt"]["status"] == "reused"
    assert fresh._preflight_test_proof.head == sha and fresh._preflight_test_proof.tier == "selected"

    monkeypatch.setattr(pr, "_execute_pytest_pass", executed)
    monkeypatch.setenv("OUROBOROS_PREFLIGHT_TIMEOUT_SEC", "1234")  # another workload: another identity
    assert git_tools._run_pre_push_tests(fresh) is None
    assert fresh._preflight_test_disclosure["receipt"]["status"] == "published"
    assert len(scene.records(preflight_receipts.SELECTED_SCHEMA)) == 2


def test_bound_evolution_runs_its_forced_gate_selected_and_waits_on_no_publication(scene, monkeypatch):
    from ouroboros.tools.git_evolution import _evolution_test_publication_stop

    scene.ctx.current_task_type = "evolution"
    monkeypatch.setenv("OUROBOROS_PRE_PUSH_TESTS", "0")  # evolution's run is forced past the switch
    sha, error = scene.commit(B_GOOD, force=True)

    proof = scene.ctx._preflight_test_proof
    assert error is None and (proof.tier, proof.selection.base) == ("selected", scene.old)
    assert _evolution_test_publication_stop(scene.ctx, sha, "") == ""
    assert body_adoption.authorize(scene.ctx, sha, reason="evo")["tests"]["tier"] == "selected"


def test_cyber_pro_unreviewed_commits_keep_the_full_route(scene, monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "cyber_pro")
    sha = candidate_commit(scene.candidate, files=A_GOOD)  # never recorded as reviewed: Cyber Pro's exception
    scene.ctx._commit_test_request = dict(DEFAULTS)
    assert git_tools._run_pre_push_tests(scene.ctx) is None and scene.ctx._preflight_test_proof.tier == "selected"
    scene.refused(sha, "no qualifying test run")

    assert pr.run_hermetic_pytest(scene.candidate, ctx=scene.ctx) is None  # the full battery of exactly that commit
    handoff = body_adoption.authorize(scene.ctx, sha, reason="adopt")
    assert handoff["tests"]["tier"] == "full" and handoff["unreviewed_commits"] == [sha]


def test_a_selected_adoption_switches_and_settles_but_publishes_neither_branch_nor_tag(scene, monkeypatch, tmp_path):
    from supervisor import git_ops, git_ops_reset, message_bus, state
    from tests.test_body_adoption_consumers import _switch

    remote = tmp_path / "origin.git"
    git(tmp_path, "init", "-q", "--bare", str(remote))
    git(scene.serving, "remote", "add", "origin", str(remote))
    scene.cand, error = scene.commit(A_GOOD)
    assert error is None
    git(scene.serving, "tag", "-a", "v1.0.1", scene.cand, "-m", "v1.0.1: reachable once adopted")
    monkeypatch.setattr(git_ops, "REPO_DIR", scene.serving)
    monkeypatch.setattr(git_ops, "BRANCH_DEV", "ouroboros")
    monkeypatch.setattr(git_ops_reset, "_record_checkout_facts", lambda value: None)
    monkeypatch.setattr(state, "load_state", lambda: {})
    monkeypatch.setattr(message_bus, "send_with_budget", lambda *a, **kw: pytest.fail("no owner chat is configured"))

    _switch(scene)  # the real authorization (selected), bound restart, arming, switch helper and attestation
    settled = body_adoption.settle_on_boot(scene.data, scene.serving, supervisor_ready=True)

    assert settled["outcome"] == "adopted" and git(scene.serving, "rev-parse", "HEAD") == scene.cand
    assert settled["tests"]["tier"] == "selected" and settled["publication"]["status"] == "withheld"
    assert git(remote, "for-each-ref") == ""
