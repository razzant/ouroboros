"""Provenance: the seed, the attestation record and the outcome the ledger points at.

Split verbatim out of ``tests/test_osworld_cu_bridge.py`` by theme. This module owns the
seed-gate refusals that short-circuit the preflight, the attestation record every entry
point persists, the campaign-pin check that runs before the VM boots, the module
grandfather matcher, and the durability rules that keep an obtained score and its ledger
row consistent.

These exercise the pure helpers only — no OSWorld VM, no Ouroboros server.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


from devtools.benchmarks.osworld import run_cu_bridge_agent as rcb

from tests._osworld_cu_bridge_shared import (
    _attempt_dirs,
    _cu_bridge_argv,
    _cu_bridge_stubs,
)


def test_cu_bridge_refuses_before_the_claim_when_attestation_fails(tmp_path, monkeypatch, capsys):
    """Owner Q9/Q10: the bridge attests the running server before its first paid POST. The
    helper fails CLOSED, so the launcher must turn that into a typed `blocked` row — and must
    not park a claim lock on a run that never starts."""
    import sys as _sys

    osworld = tmp_path / "OSWorld"
    (osworld / "evaluation_examples" / "examples" / "chrome").mkdir(parents=True)
    task = osworld / "evaluation_examples" / "examples" / "chrome" / "abc.json"
    task.write_text(json.dumps({"id": "abc", "instruction": "no-op"}), encoding="utf-8")
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "VERSION").write_text("6.76.0\n", encoding="utf-8")
    results = tmp_path / "results"
    claims = tmp_path / "claims"
    monkeypatch.setattr(_sys, "argv", [
        "run_cu_bridge_agent.py",
        "--osworld-root", str(osworld),
        "--provider_name", "docker",
        "--path_to_vm", "/vm/Ubuntu.qcow2",
        "--task", str(task),
        "--result_dir", str(results),
        "--repo-dir", str(repo_dir),
        "--data-dir", str(tmp_path / "data"),
        "--settings-path", str(tmp_path / "settings.json"),
        "--ouroboros-url", "http://127.0.0.1:9",   # nothing listens: attestation fails closed
        "--target-file", str(tmp_path / "target.txt"),
        "--claim-dir", str(claims),
        "--allow-dirty-seed",                       # provenance is not what this test pins
    ])

    assert rcb.main() == 2
    outcome = json.loads(capsys.readouterr().out)
    assert outcome["status"] == "blocked"
    # The EXACT typed reason, not the generic string: nothing listens on the URL, so no live
    # runtime identity was established at all.
    assert outcome["reason_code"] == "runtime_unreachable"
    # The refusal precedes the claim, so no lock/marker is left for another lane to trip over.
    assert not claims.exists() or not any(claims.iterdir())

def test_step_agent_seed_gate_refusal_is_typed_records_not_a_traceback(tmp_path, monkeypatch, capsys):
    """Owner Q19 fails the seed gate CLOSED. Nothing is spent at that point, so the launcher
    must report its own `blocked/seed_gate_failed` records (ledger row included) instead of a
    bare traceback. `repo_dir` here is a non-git directory, so the verdict does not depend on
    the ambient checkout being clean or dirty."""
    import sys as _sys

    from devtools.benchmarks.osworld import run_step_agent

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "VERSION").write_text("6.76.0\n", encoding="utf-8")
    task = tmp_path / "OSWorld" / "evaluation_examples" / "examples" / "chrome" / "abc.json"
    task.parent.mkdir(parents=True)
    task.write_text(json.dumps({"id": "abc", "instruction": "no-op"}), encoding="utf-8")
    results = tmp_path / "results"
    monkeypatch.setattr(_sys, "argv", [
        "run_step_agent.py",
        "--osworld-root", str(tmp_path / "OSWorld"),
        "--task", str(task),
        "--result_dir", str(results),
        "--repo-dir", str(repo_dir),
        "--data-dir", str(tmp_path / "data"),
        "--settings-path", str(tmp_path / "settings.json"),
        "--ouroboros-url", "http://127.0.0.1:9",
        "--provider_name", "docker",
    ])

    assert run_step_agent.main() == 2
    outcome = json.loads(capsys.readouterr().out)
    assert outcome["status"] == "blocked" and outcome["reason_code"] == "seed_gate_failed"
    assert "seed_identity_unavailable" in outcome["error"]
    rows = [json.loads(line) for line
            in (results / "result_index.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["reason_code"] == "seed_gate_failed"

def test_osworld_skeleton_seed_gate_refusal_short_circuits_the_preflight(tmp_path, monkeypatch, capsys):
    """Same gate, non-spending entry point: fold the refusal into the existing typed refusal
    (return 2 with a `seed_gate_error`) and still report the other preflight failures, so the
    gate cannot MASK an isolation refusal the operator also needs to see."""
    import sys as _sys

    from devtools.benchmarks.osworld import osworld_adapter_skeleton as skeleton

    repo_root = tmp_path / "repo"  # deliberately NOT a git checkout: verdict is ambient-free
    osworld = tmp_path / "OSWorld"
    payload = tmp_path / "unix_computer_use"
    output_root = tmp_path / "runs" / "osworld"
    for path in (repo_root, osworld, payload):
        path.mkdir(parents=True)
    (osworld / "evaluation_examples").mkdir()
    monkeypatch.setattr(skeleton, "DEFAULT_REPO_ROOT", repo_root)
    monkeypatch.setattr(skeleton, "DEFAULT_DATA_ROOT", tmp_path / "live-data")
    monkeypatch.setattr(_sys, "argv", [
        "osworld_adapter_skeleton.py",
        "--osworld-root", str(osworld),
        "--ouroboros-url", "http://127.0.0.1:9",
        "--osworld-server-url", "http://127.0.0.1:9",
        "--unix-computer-use-payload", str(payload),
        "--output-root", str(output_root),
    ])

    assert skeleton.main() == 2
    result = json.loads(capsys.readouterr().out)
    assert result["ok"] is False
    assert "seed_identity_unavailable" in result["details"]["seed_gate_error"]
    assert any("seed gate refused" in failure for failure in result["failures"])
    # SHORT-CIRCUIT (v6.76.0): the preflight does NOT run after a refused admission. It probes
    # the filesystem and reaches two servers over the network, and the documented contract says
    # an unidentifiable seed stops the run BEFORE the preflight — so no other finding is
    # reported here, deliberately, and none is spent on.
    assert result["details"]["skipped"] == "preflight not run: admission refused"
    assert not any("not reachable" in failure for failure in result["failures"])
    # v6.76.0: a refused seed now leaves a DURABLE record of what was refused. Writing
    # nothing (the previous behaviour) meant the one path where provenance was refused was
    # also the one path that left no evidence of the refusal. It still leaves no LEDGER row:
    # the run never started, so it owns no denominator entry.
    manifest = json.loads(
        (output_root / "osworld_preflight.run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["outcome"] == "refused"
    assert manifest["extra"]["exit_code"] == 2                    # == the process status
    assert manifest["extra"]["refusal"]["stage"] == "seed_gate"
    assert manifest["seed_gate"]["ok"] is False
    assert not (output_root / "osworld_preflight.ledger.jsonl").exists()

def _attempt_manifests(run_dir):
    return [json.loads((d / "task_run_manifest.json").read_text(encoding="utf-8"))
            for d in _attempt_dirs(run_dir)]

def _refused_attestation_record():
    """The record `runtime_attestation()` builds before refusing a version skew."""
    return {
        "ok": False,
        "reason": "runtime_skew",
        "runtime_version": "6.75.0",
        "repo_head": "a" * 40,
        "repo_version": "6.76.0",
        "url": "http://127.0.0.1:9/",
        "overridden": False,
        "override_set": False,
    }

def test_cu_bridge_persists_the_attestation_record_it_was_handed(tmp_path, monkeypatch, capsys):
    """`RuntimeAttestationRefused` CARRIES the record it built — the exact typed reason plus
    `runtime_version`, `repo_head` and `repo_version`. Catching a generic `RuntimeError` and
    keeping only the string `runtime_attestation_failed` threw that evidence away at the moment
    it matters most, and `docs/ARCHITECTURE.md` promises it is preserved. Same defect phase P1
    fixed for ProgramBench in its round 4."""
    from devtools.benchmarks.common.manifests import RuntimeAttestationRefused

    claims = tmp_path / "claims"
    rcb, _env = _cu_bridge_stubs(monkeypatch, tmp_path)
    argv, results = _cu_bridge_argv(tmp_path, claims)
    monkeypatch.setattr(sys, "argv", argv)
    record = _refused_attestation_record()

    def _refuse(url, repo):
        raise RuntimeAttestationRefused("runtime attestation failed reason=runtime_skew", record)

    monkeypatch.setattr(rcb, "runtime_attestation", _refuse)

    assert rcb.main() == 2
    outcome = json.loads(capsys.readouterr().out)
    assert outcome["reason_code"] == "runtime_skew"
    assert outcome["runtime_attestation"] == record
    # The attestation refusal happens BEFORE the claim, so this attempt never owned the task and
    # its record lives in its own attempt directory. Writing it to the shared canonical manifest
    # is exactly the clobber that made two overlapping lanes overwrite each other.
    manifest = _attempt_manifests(results / "chrome" / "abc")[-1]
    assert manifest["extra"]["runtime_attestation"] == record
    assert manifest["extra"]["refusal"] == {"stage": "runtime_attestation",
                                            "reason": "runtime_skew", "exit_code": 2}
    assert manifest["extra"]["outcome"] == "blocked" and manifest["extra"]["exit_code"] == 2
    assert manifest["extra"]["claim_owner"] is False
    assert not (results / "chrome" / "abc" / "task_run_manifest.json").exists()
    # A refusal that carries NO record still refuses, with the generic reason as the fallback.
    monkeypatch.setattr(rcb, "runtime_attestation",
                        lambda url, repo: (_ for _ in ()).throw(RuntimeError("no record")))
    assert rcb.main() == 2
    attempts = _attempt_manifests(results / "chrome" / "abc")
    # ...into a SECOND, independent attempt record: the first is not overwritten.
    assert len(attempts) == 2
    assert attempts[0]["extra"]["refusal"]["reason"] == "runtime_skew"
    assert attempts[-1]["extra"]["refusal"]["reason"] == "runtime_attestation_failed"

def test_step_agent_preflight_persists_the_attestation_record_it_was_handed(
        tmp_path, monkeypatch, capsys):
    """Same defect on the step loop: the preflight kept only the message, and the manifest is
    amended FROM the preflight details, so the loss propagated into the run's own record."""
    from devtools.benchmarks.common.manifests import RuntimeAttestationRefused
    from devtools.benchmarks.osworld import run_step_agent

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "VERSION").write_text("6.76.0\n", encoding="utf-8")
    task = tmp_path / "OSWorld" / "evaluation_examples" / "examples" / "chrome" / "abc.json"
    task.parent.mkdir(parents=True)
    task.write_text(json.dumps({"id": "abc", "instruction": "no-op"}), encoding="utf-8")
    results = tmp_path / "results"
    record = _refused_attestation_record()

    def _refuse(url, repo):
        raise RuntimeAttestationRefused("runtime attestation failed reason=runtime_skew", record)

    monkeypatch.setattr(run_step_agent, "runtime_attestation", _refuse)
    monkeypatch.setattr(sys, "argv", [
        "run_step_agent.py", "--osworld-root", str(tmp_path / "OSWorld"), "--task", str(task),
        "--result_dir", str(results), "--repo-dir", str(repo_dir),
        "--data-dir", str(tmp_path / "data"), "--settings-path", str(tmp_path / "settings.json"),
        "--ouroboros-url", "http://127.0.0.1:9", "--provider_name", "docker", "--model", "m",
        "--allow-dirty-seed",            # provenance is not what this test pins
    ])

    assert run_step_agent.main() == 2
    outcome = json.loads(capsys.readouterr().out)
    assert outcome["reason_code"] == "preflight_failed"
    assert any("reason=runtime_skew" in failure
               for failure in outcome["preflight"]["failures"])
    assert outcome["preflight"]["details"]["runtime_attestation"] == record
    run_dir = results / "pyautogui" / "screenshot_a11y_tree" / "m" / "chrome" / "abc"
    manifest = json.loads((run_dir / "task_run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["runtime_attestation"] == record
    assert manifest["extra"]["exit_code"] == 2
    # ...and the typed refusal NAMES the attestation reason. `preflight_failed` alone conflates
    # "the runtime disagrees with its checkout" with "the task JSON is missing" — different
    # operator actions — and the documented contract is the specific one.
    assert manifest["extra"]["refusal"] == {"stage": "runtime_attestation",
                                            "reason": "runtime_skew", "exit_code": 2}

def test_osworld_skeleton_persists_the_attestation_record_it_was_handed(
        tmp_path, monkeypatch, capsys):
    """Same defect on the non-spending entry point, whose whole job is to REPORT evidence."""
    from devtools.benchmarks.common.manifests import RuntimeAttestationRefused
    from devtools.benchmarks.osworld import osworld_adapter_skeleton as skeleton

    repo_root = tmp_path / "repo"
    osworld = tmp_path / "OSWorld"
    payload = tmp_path / "unix_computer_use"
    output_root = tmp_path / "runs" / "osworld"
    for path in (repo_root, osworld, payload):
        path.mkdir(parents=True)
    (osworld / "evaluation_examples").mkdir()
    record = _refused_attestation_record()

    def _refuse(url, repo):
        raise RuntimeAttestationRefused("runtime attestation failed reason=runtime_skew", record)

    monkeypatch.setattr(skeleton, "runtime_attestation", _refuse)
    monkeypatch.setattr(skeleton, "DEFAULT_REPO_ROOT", repo_root)
    monkeypatch.setattr(skeleton, "DEFAULT_DATA_ROOT", tmp_path / "live-data")
    monkeypatch.setattr(sys, "argv", [
        "osworld_adapter_skeleton.py", "--osworld-root", str(osworld),
        "--ouroboros-url", "http://127.0.0.1:9", "--osworld-server-url", "http://127.0.0.1:9",
        "--unix-computer-use-payload", str(payload), "--output-root", str(output_root),
        "--allow-dirty-seed",            # output isolation/attestation is what this pins
    ])

    assert skeleton.main() == 2
    result = json.loads(capsys.readouterr().out)
    assert result["details"]["runtime_attestation"] == record
    assert any("reason=runtime_skew" in failure for failure in result["failures"])
    manifest = json.loads((output_root / "osworld_preflight.run_manifest.json")
                          .read_text(encoding="utf-8"))
    assert manifest["extra"]["preflight"]["details"]["runtime_attestation"] == record
    # The contract is ONE place to read the carried record from, across all three launchers —
    # burying it under `extra.preflight.details` made this the site that did not honour it.
    assert manifest["extra"]["runtime_attestation"] == record
    assert manifest["extra"]["refusal"] == {"stage": "runtime_attestation",
                                            "reason": "runtime_skew", "exit_code": 2}

def test_osworld_operator_patch_raises_provider_lock_timeout_and_is_documented():
    root = Path(__file__).resolve().parent.parent / "devtools" / "benchmarks" / "osworld"
    patch = (root / "operator_patches" / "osworld_docker_lock_timeout.v6760.patch").read_text(encoding="utf-8")
    assert "desktop_env/providers/docker/provider.py" in patch
    assert "-LOCK_TIMEOUT = 10" in patch and "+LOCK_TIMEOUT = 60" in patch
    readme = (root / "operator_patches" / "README.md").read_text(encoding="utf-8")
    assert "osworld_docker_lock_timeout.v6760.patch" in readme
    assert "construct_desktop_env" in readme  # both halves of the fix are disclosed

def test_osworld_methodology_preregisters_the_dedup_rule_and_defers_the_lane_generator():
    text = (Path(__file__).resolve().parent.parent / "devtools" / "benchmarks" / "osworld"
            / "METHODOLOGY.md").read_text(encoding="utf-8")
    assert "FIRST SCORED ATTEMPT WINS" in text
    # Multiple lanes ARE supported and the smoke exercises them, so the disclosure must say so;
    # what is extracted is the lane-script GENERATOR, and the disclosure must not describe a
    # convenience the tree does not have either.
    assert "MULTIPLE LANES ARE SUPPORTED" in text
    assert "NO MULTI-LANE LAUNCHER GENERATOR IN\n     THIS RELEASE" in text
    assert "gen_lanes.py" in text and "lanes.json" in text
    # The rule is enforced by code that EXISTS, and the record layout that makes overlapping
    # attempts safe is disclosed rather than implied.
    assert "attempts/<attempt_id>/task_run_manifest.json" in text
    assert "claim_owner" in text
    # The residual-window disclosure must match the fix: the interrupt path is closed with a
    # durable marker; only SIGKILL remains open.
    assert "THE INTERRUPT WINDOW IS CLOSED; THE `SIGKILL` WINDOW IS NOT" in text
    assert "construct_desktop_env" in text
    assert "LOCK_TIMEOUT" in text
    assert "--allow-dirty-seed" in text

def test_module_grandfather_matcher_uses_exact_repo_relative_paths(monkeypatch):
    import ouroboros.review as review_mod
    from ouroboros.review import (
        GIANT_PATHS,
        _exact_repo_relative_path,
        module_is_grandfathered,
    )
    # Exact runtime helpers accept only actual repo-relative paths. Compatibility
    # section-prefix decoding belongs solely to compute_complexity_metrics.
    # The v7 size campaign paid the whole registry down (GIANT_PATHS is empty),
    # so the live-derived sample the anti-vacuity guard demanded is now pinned
    # the other way round: the EMPTINESS itself is the campaign outcome, and the
    # exact-path mechanism is exercised through a synthetic registry entry, the
    # same way the JS gate test survived chat.js paying its debt.
    assert GIANT_PATHS == frozenset()
    monkeypatch.setattr(
        review_mod, "GIANT_PATHS",
        frozenset({"skills/synthetic_fixture/plugin.py", "synthetic_root_fixture.py"}),
    )
    nested = sorted(path for path in review_mod.GIANT_PATHS if "/" in path)
    assert nested
    for path in nested:
        assert module_is_grandfathered(path), path
        # A repo/-prefixed variant is a DIFFERENT path and is not exempted.
        assert not module_is_grandfathered("repo/" + path), path
        # A same-basename module in another directory is not exempted either.
        assert not module_is_grandfathered("other_dir/" + path.rsplit("/", 1)[1]), path
    # A ROOT-level manifest path is an exact key too; a nested same-basename is
    # not. Live-derived for the same reason the nested loop is.
    for path in sorted(path for path in review_mod.GIANT_PATHS if "/" not in path):
        assert module_is_grandfathered(path), path
        assert not module_is_grandfathered("repo/" + path), path
        assert not module_is_grandfathered("ouroboros/" + path), path
        assert not module_is_grandfathered("repo/ouroboros/" + path), path
    # The four spellings of a root module remain four DIFFERENT keys even while
    # no root module is in debt, so the contract does not retire itself when that
    # loop runs empty. server.py was the sample until its composition split paid
    # it down out of the giant layer.
    root_spellings = (
        "server.py", "repo/server.py", "ouroboros/server.py", "repo/ouroboros/server.py",
    )
    assert len({_exact_repo_relative_path(name) for name in root_spellings}) == 4
    # Two modules that share a basename stay two different keys even when one of
    # them is in debt, so a debt path can never leak to its namesake. This was
    # pinned as "ouroboros/tools/control.py is grandfathered, gateway/control.py
    # is not" until the control catalog split paid the first one out of the giant
    # layer — a membership claim, not the contract, and the same vacuity trap the
    # live-derived loops above avoid.
    assert len({
        _exact_repo_relative_path(name)
        for name in ("ouroboros/tools/control.py", "ouroboros/gateway/control.py")
    }) == 2
    assert not module_is_grandfathered("ouroboros/gateway/control.py")

def test_cu_bridge_publication_failure_never_erases_an_obtained_score(tmp_path, monkeypatch):
    """An outcome that already carries an official score is never overwritten by a generic error.

    By the time publication runs, `mark_task_scored` has made `.scored` durable, so no later
    attempt may retry this task. Reporting `reward=None`/`not_run` from the broad handler
    therefore destroyed a score that EXISTS, permanently: the protection became the lock.
    """
    rcb, _env = _cu_bridge_stubs(monkeypatch, tmp_path, reward=1.0)
    argv, results = _cu_bridge_argv(tmp_path, tmp_path / "claims")
    monkeypatch.setattr(sys, "argv", argv)
    run_dir = results / "chrome" / "abc"
    (run_dir / "result.txt").mkdir(parents=True)     # fails the first artefact after the marker

    assert rcb.main() == 1
    outcome = json.loads((run_dir / "task_outcome.json").read_text(encoding="utf-8"))
    assert outcome["reward"] == 1.0                  # the obtained score survived the failure
    assert outcome["reason_code"] == "publication_failed_after_scoring"
    row = json.loads((results / "result_index.jsonl").read_text(
        encoding="utf-8").splitlines()[-1])
    assert row["official_eval_status"] == "completed"    # it WAS evaluated, not `not_run`

def test_cu_bridge_keeps_the_ledger_row_when_the_canonical_outcome_cannot_be_written(
        tmp_path, monkeypatch):
    """The score survives a failure INSIDE the writer, at the canonical outcome stage.

    The sibling of the `result.txt` case: there the failure happened BEFORE `_write_outcome`
    ran, so the broad handler could still publish. Here the writer itself dies partway, and the
    handler used to call the SAME aggregate writer again — reproducing the failure and escaping
    with no ledger row at all, while the durable `.scored` marker forbids any retry. Every
    destination is attempted independently, so the still-writable ledger records the truth.
    """
    rcb, _env = _cu_bridge_stubs(monkeypatch, tmp_path, reward=1.0)
    argv, results = _cu_bridge_argv(tmp_path, tmp_path / "claims")
    monkeypatch.setattr(sys, "argv", argv)
    run_dir = results / "chrome" / "abc"
    (run_dir / "task_outcome.json").mkdir(parents=True)   # canonical publication stage fails

    assert rcb.main() == 1
    row = json.loads((results / "result_index.jsonl").read_text(
        encoding="utf-8").splitlines()[-1])
    assert row["official_eval_status"] == "completed"     # it WAS evaluated, not `not_run`
    assert row["details"]["reward"] == 1.0                # the obtained score reached the ledger
    attempts = sorted((run_dir / "attempts").glob("*/task_outcome.json"))
    assert attempts, "the attempt's own record must still exist"
    assert json.loads(attempts[-1].read_text(encoding="utf-8"))["reward"] == 1.0

def test_cu_bridge_keeps_the_outcome_files_when_the_ledger_cannot_be_appended(
        tmp_path, monkeypatch):
    """The mirror case: the ledger is the dead destination, the outcome records must survive.

    A failure at the LAST publication stage must not roll back or re-run the ones that already
    succeeded, and must not escape as a traceback: the run reports a disclosed publication
    failure while the reward stays on every record that could still be written.
    """
    rcb, _env = _cu_bridge_stubs(monkeypatch, tmp_path, reward=1.0)
    argv, results = _cu_bridge_argv(tmp_path, tmp_path / "claims")
    monkeypatch.setattr(sys, "argv", argv)
    (results / "result_index.jsonl").mkdir(parents=True)  # ledger publication stage fails

    assert rcb.main() == 1
    run_dir = results / "chrome" / "abc"
    canonical = json.loads((run_dir / "task_outcome.json").read_text(encoding="utf-8"))
    assert canonical["reward"] == 1.0                     # written before the ledger, kept
    assert any("result_index" in e for e in canonical.get("publication_errors", [])), \
        "the dead destination must be disclosed, not swallowed"
    attempts = sorted((run_dir / "attempts").glob("*/task_outcome.json"))
    assert json.loads(attempts[-1].read_text(encoding="utf-8"))["reward"] == 1.0

def test_cu_bridge_ledger_row_never_points_at_an_outcome_that_was_not_written(
        tmp_path, monkeypatch):
    """The ledger row must describe the publication that HAPPENED, not the one intended.

    Independent destinations stopped one dead record from erasing an obtained score — but
    independence cuts both ways: the row is now written even when the artefact it points at
    is not. Emitting `output_paths.task_outcome` unconditionally, with the pre-failure status
    and without the collected `publication_errors`, makes the index assert a completed,
    readable outcome file that does not exist. An operator must be able to tell "scored,
    fully published" from "scored, partially published" from the row alone.
    """
    rcb_mod, _env = _cu_bridge_stubs(monkeypatch, tmp_path, reward=1.0)
    argv, results = _cu_bridge_argv(tmp_path, tmp_path / "claims")
    monkeypatch.setattr(sys, "argv", argv)
    real_write_json = rcb_mod.write_json

    def _dead_attempt_outcome(path, payload):
        target = Path(path)
        if target.name == "task_outcome.json" and "attempts" in target.parts:
            raise OSError("attempt outcome destination is dead")
        return real_write_json(path, payload)

    monkeypatch.setattr(rcb_mod, "write_json", _dead_attempt_outcome)

    assert rcb_mod.main() == 1
    row = json.loads((results / "result_index.jsonl").read_text(
        encoding="utf-8").splitlines()[-1])
    # No pointer to a destination that failed: the file genuinely is not there.
    assert not list((results / "chrome" / "abc" / "attempts").glob("*/task_outcome.json"))
    assert "task_outcome" not in row["output_paths"], \
        "the row must not point at an artefact whose write failed"
    # The status publication never achieved must not be reported as if it had been.
    assert row["status"] != "completed"
    # ...while everything the run DID achieve still reaches the ledger.
    assert row["official_eval_status"] == "completed"
    assert row["details"]["reward"] == 1.0
    assert any("attempt_outcome" in e for e in row["details"]["publication_errors"]), \
        "the row must carry the collected publication errors"
    # BOTH SIDES of the same rule. The previous round fixed the ledger row and left the
    # manifest lying: `_amend_manifest` still added `output_paths.task_outcome`
    # unconditionally, so the finalized attempt manifest kept naming the missing file. A
    # pointer is a pointer wherever it is written.
    attempt_manifests = sorted(
        (results / "chrome" / "abc" / "attempts").glob("*/task_run_manifest.json"))
    assert attempt_manifests, "the attempt manifest must still be finalized"
    manifest = json.loads(attempt_manifests[-1].read_text(encoding="utf-8"))
    assert "task_outcome" not in (manifest.get("output_paths") or {}), \
        "the manifest must not point at an artefact whose write failed either"
    assert (manifest.get("output_paths") or {}).get("attempt_dir"), \
        "...while the pointer that IS valid survives"

def test_a_checkout_other_than_the_campaign_pin_is_refused_before_the_vm_boots():
    """The graded-spec pin decides both the instruction the agent receives and the
    evaluator that scores it. Recording a mismatch in the manifest is a report:
    on 2026-07-29 a 75-task probe graded 21 tasks against a three-week-older
    checkout while every manifest faithfully recorded it and nobody read it."""
    import pytest

    rcb._refuse_wrong_dataset_commit("", {"git_commit": "whatever"})  # opt-in: no claim, no gate
    rcb._refuse_wrong_dataset_commit("091f5ef1d5544bc", {"git_commit": "091f5ef1d5544bc74953c"})
    with pytest.raises(SystemExit, match="graded against"):
        rcb._refuse_wrong_dataset_commit("091f5ef1", {"git_commit": "7a17d3abc86d5"})
    with pytest.raises(SystemExit, match="no readable git identity"):
        rcb._refuse_wrong_dataset_commit("091f5ef1", {"git_commit": ""})
