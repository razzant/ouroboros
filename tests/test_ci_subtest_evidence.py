"""Real pytest/xdist reports: substeps are evidence, not repeated parent executions."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest

from tests import ci_evidence
from tests.test_ci_evidence import RUN_FACTS, SECRET, _events, _producer, _public_text, _results



def _proof_tree(tmp_path, facts):
    nodes = list(dict.fromkeys(row["nodeid"] for row in facts["reports"]))
    manifest, shards = tmp_path / "manifest", tmp_path / "shards"
    manifest.mkdir()
    shards.mkdir()
    common = {"format": 1, "github": facts["github"], "collection_failures": []}
    (manifest / "results.json").write_text(json.dumps({
        **common, "session_exit_code": 0, "reports": [], "ui_browser": {"full": nodes},
    }), encoding="utf-8")
    proof = {**facts, "ui_browser": {"full": nodes, "assigned": nodes, "shard": [1, 1]}}
    (shards / "results.json").write_text(json.dumps(proof), encoding="utf-8")
    return manifest, shards, proof


def _reconcile(manifest, shards):
    return ci_evidence.reconcile_shards(manifest, shards, count=1,
                                       sha=RUN_FACTS["GITHUB_SHA"], run_id=RUN_FACTS["GITHUB_RUN_ID"])


@pytest.mark.skipif(not ci_evidence.SubtestReport, reason="pytest has no built-in subtest reports")
@pytest.mark.parametrize(("fails", "workers"), [(False, 0), (True, 2)])
def test_real_subtests_keep_failures_without_duplicate_parent_execution(tmp_path, workers, fails):
    result, public, _ = _producer(tmp_path, f'''
        import unittest
        import pytest
        class TestSteps(unittest.TestCase):
            def test_steps(self):
                for value in range(4):
                    with self.subTest(value=value, private_label={SECRET!r}):
                        assert value != {2 if fails else -1}
        def test_ordinary():
            pass
        @pytest.mark.skip(reason="synthetic setup skip")
        def test_skipped():
            raise AssertionError("must not run")
    ''', args=("-n", "2") if workers else ())
    assert result.returncode == int(fails), result.stdout + result.stderr
    facts = _results(public)
    substeps = [row for row in facts["reports"] if row["is_subtest"]]
    calls = [row for row in facts["reports"] if row["phase"] == "call" and not row["is_subtest"]]
    assert len(substeps) == 4 and len(calls) == 2
    assert all(row["phase"] == "call" for row in substeps)
    assert sum(row["outcome"] == "failed" for row in substeps) == int(fails)
    assert len([row for row in _events(public) if row.get("is_subtest")]) == 4
    assert SECRET not in _public_text(public)
    assert "private_label" not in _public_text(public), "only kind, never subtest context, is exported"
    outcomes = ci_evidence._case_outcomes(facts["reports"])
    assert outcomes[substeps[0]["nodeid"]] == ("failed" if fails else "passed")
    assert set(outcomes.values()) == ({"failed", "passed", "skipped"} if fails else {"passed", "skipped"})
    manifest, shards, proof = _proof_tree(tmp_path, facts)
    gaps, _ = _reconcile(manifest, shards)
    assert not any("duplicate" in gap for gap in gaps), gaps
    assert bool(gaps) is fails, gaps  # A failed subtest still fails its entire session proof.
    if fails:
        assert any("exit status 1" in gap for gap in gaps), gaps
    # Inject a second *ordinary* execution receipt. Do not collapse equal node IDs wholesale.
    proof["reports"].append(copy.deepcopy(calls[0]))
    (shards / "results.json").write_text(json.dumps(proof), encoding="utf-8")
    gaps, _ = _reconcile(manifest, shards)
    assert any("duplicate executions" in gap for gap in gaps), gaps
    # Legacy projections do not carry this distinction. Never infer it from zero duration.
    proof["reports"] = [{key: value for key, value in row.items() if key != "is_subtest"}
                        for row in facts["reports"]]
    (shards / "results.json").write_text(json.dumps(proof), encoding="utf-8")
    gaps, _ = _reconcile(manifest, shards)
    assert any("duplicate executions" in gap for gap in gaps), gaps


@pytest.mark.parametrize("flag", [None, 0, {}])
def test_subtest_kind_must_be_a_boolean_when_present(tmp_path, flag):
    row = {"nodeid": "test_sample.py::test_sample", "phase": "call", "outcome": "passed", "is_subtest": flag}
    (tmp_path / "results.json").write_text(json.dumps({"format": 1, "reports": [row]}), encoding="utf-8")
    assert ci_evidence._final(tmp_path)[1] == "invalid result projection"


def test_failed_substep_is_not_erased_by_a_passed_parent_report():
    node = "test_sample.py::test_sample"
    assert ci_evidence._case_outcomes([
        {"nodeid": node, "phase": "call", "outcome": "failed", "is_subtest": True},
        {"nodeid": node, "phase": "call", "outcome": "passed", "is_subtest": False},
    ]) == {node: "failed"}


@pytest.mark.parametrize("missing", ["_pytest.subtests", "unrelated_dependency"])
def test_reporter_import_keeps_supported_older_pytest_without_hiding_other_errors(monkeypatch, missing):
    import builtins

    original = builtins.__import__

    def import_without_subtests(name, *args, **kwargs):
        if name == "_pytest.subtests":
            raise ModuleNotFoundError(name=missing)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_subtests)
    path = Path(ci_evidence.__file__)
    if missing != "_pytest.subtests":
        with pytest.raises(ModuleNotFoundError) as failure:
            runpy.run_path(str(path), run_name="ci_import_probe")
        assert failure.value.name == missing
        return
    loaded = runpy.run_path(str(path), run_name="ci_import_probe")
    receiver = SimpleNamespace(reports=[], _event=lambda *args, **kwargs: None)
    loaded["_Results"].pytest_runtest_logreport(receiver, SimpleNamespace(
        nodeid="test_sample.py::test_sample", when="call", outcome="passed", duration=0.1))
    assert receiver.reports[0]["is_subtest"] is False
