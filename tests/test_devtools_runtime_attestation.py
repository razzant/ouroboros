"""The runtime attestation a benchmark run has to pass before it attaches a URL.

Split verbatim out of ``tests/test_devtools_benchmarks.py`` by theme. This module owns the
two facts the attestation records, the fail-closed default, the override that waives only
the evolved-runtime reason, the lineage that admits descendants only, and the contracted
runtime version field it requires.
"""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
import urllib.error
import urllib.request

import pytest


from tests._devtools_benchmarks_shared import (
    REPO_ROOT,
    _git_commit_all,
    _git_repo,
)
from tests._devtools_benchmarks_shared import _isolate_bench_runs_root as __isolate_bench_runs_root

# Fixtures are requested by name as test parameters, so they are re-bound through a
# module attribute: a direct import of a name that reappears as a parameter is an F811
# redefinition under the CI ruff gate.
_isolate_bench_runs_root = __isolate_bench_runs_root


def test_runtime_attestation_records_both_facts_and_fails_closed(tmp_path, monkeypatch):
    """Owner Q7=B / Q8: record the HTTP runtime_version AND the local commit, and hard-stop on
    a skew unless the named override is set (the override is itself recorded)."""
    from devtools.benchmarks.common import manifests

    repo = tmp_path / "repo"
    _git_repo(repo)
    (repo / "VERSION").write_text("6.75.0\n", encoding="utf-8")
    _git_commit_all(repo)

    served = {"runtime_version": "6.75.0"}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def read(self):
            return json.dumps(served).encode("utf-8")

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Resp())
    monkeypatch.delenv(manifests.ALLOW_EVOLVED_VOLUME_ENV, raising=False)

    ok = manifests.runtime_attestation("http://127.0.0.1:9/", repo)
    assert ok["ok"] is True and ok["reason"] == ""
    assert ok["runtime_version"] == "6.75.0"
    assert ok["repo_version"] == "6.75.0"
    assert len(ok["repo_head"]) == 40
    assert ok["overridden"] is False

    served["runtime_version"] = "6.74.5"
    with pytest.raises(RuntimeError, match="reason=runtime_skew"):
        manifests.runtime_attestation("http://127.0.0.1:9/", repo)

    monkeypatch.setenv(manifests.ALLOW_EVOLVED_VOLUME_ENV, "1")
    overridden = manifests.runtime_attestation("http://127.0.0.1:9/", repo)
    assert overridden["reason"] == "runtime_skew" and overridden["overridden"] is True
    assert overridden["ok"] is False

def test_runtime_attestation_override_waives_only_the_evolved_runtime_reason(tmp_path, monkeypatch):
    """`OBO_ALLOW_EVOLVED_VOLUME` authorises a deliberately evolved / version-skewed runtime and
    NOTHING else. It used to be applied to every failure reason, so with the override exported
    ProgramBench admission continued after an unreachable `/api/health` — the attestation gate
    fail-open the phase exists to remove. Per reason, with the override SET: `runtime_skew`
    proceeds and is recorded; `runtime_unreachable` (no live identity at all) and
    `commit_unavailable` (no commit to attribute the numbers to) still raise."""
    from devtools.benchmarks.common import manifests

    repo = tmp_path / "repo"
    _git_repo(repo)
    (repo / "VERSION").write_text("6.75.0\n", encoding="utf-8")
    _git_commit_all(repo)

    served: dict = {"runtime_version": "6.74.5"}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def read(self):
            return json.dumps(served).encode("utf-8")

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Resp())
    monkeypatch.setenv(manifests.ALLOW_EVOLVED_VOLUME_ENV, "1")

    assert manifests.OVERRIDABLE_ATTESTATION_REASONS == ("runtime_skew",)

    skewed = manifests.runtime_attestation("http://127.0.0.1:9/", repo)
    assert skewed["reason"] == "runtime_skew"
    assert skewed["overridden"] is True and skewed["override_set"] is True
    assert skewed["override_waives"] == ["runtime_skew"]
    assert skewed["ok"] is False

    # (a) transport/parse failure -> no live runtime identity was established AT ALL.
    def _boom(*_a, **_k):
        raise OSError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    with pytest.raises(RuntimeError, match="reason=runtime_unreachable") as unreachable:
        manifests.runtime_attestation("http://127.0.0.1:9/", repo)
    assert "does NOT waive" in str(unreachable.value)
    assert "override_set=True" in str(unreachable.value)

    # ... including a 200 whose body is not the health contract (parse failure, same class).
    class _Garbage(_Resp):
        def read(self):
            return b"<html>not json</html>"

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Garbage())
    with pytest.raises(RuntimeError, match="reason=runtime_unreachable"):
        manifests.runtime_attestation("http://127.0.0.1:9/", repo)

    # (b) no local commit -> nothing to attribute the numbers to. `repo_dir` outside git makes
    # `repo_head` empty, and the version pin removes the skew reason so the missing commit is the
    # one under test (no dependence on the AMBIENT checkout: this is a fresh tmp dir).
    served["runtime_version"] = "6.75.0"
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Resp())
    bare = tmp_path / "not-a-repo"
    bare.mkdir()
    (bare / "VERSION").write_text("6.75.0\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="reason=commit_unavailable") as no_commit:
        manifests.runtime_attestation("http://127.0.0.1:9/", bare, expected_version="6.75.0")
    assert "does NOT waive" in str(no_commit.value)

def test_runtime_attestation_lineage_allows_descendants_only(tmp_path):
    """Evolution legitimately moves HEAD forward, so provenance compares a LINE OF DESCENT
    (`merge-base --is-ancestor`), never equality — and an unknown commit is False, not
    'probably fine'."""
    from devtools.benchmarks.common.manifests import commit_lineage_ok

    repo = tmp_path / "repo"
    _git_repo(repo)
    seed = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    (repo / "evolved.py").write_text("print('evolved')\n", encoding="utf-8")
    _git_commit_all(repo)
    evolved = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                             capture_output=True, text=True).stdout.strip()

    assert commit_lineage_ok(seed, seed, repo) is True
    assert commit_lineage_ok(seed, evolved, repo) is True
    assert commit_lineage_ok(evolved, seed, repo) is False
    assert commit_lineage_ok(seed, "", repo) is False
    assert commit_lineage_ok("0" * 40, evolved, repo) is False

def test_runtime_attestation_is_wired_into_url_attaching_readiness_paths():
    """Owner Q9=A+B: the shared helper exists AND every launcher that attaches to a live server
    URL calls it from its own readiness/admission path. This meta-test names the CONCRETE entry
    points, with their ARITY, so a call that would TypeError cannot pass as "wired". CLB's
    host-engine path is covered through IsolatedServer; the CLB-docker stand-in never calls
    `_wait_ready`, so its attestation arrives via the tracked operator patch and is asserted in
    `tests/test_continual_learning_launcher.py`. TB and GAIA are structurally immune (owner
    Q10) and deliberately have no lines here."""
    bench = REPO_ROOT / "devtools" / "benchmarks"
    wired = {
        # shared readiness seam: every IsolatedServer driver (evolve_smoke + CLB host engine)
        bench / "common" / "server_runner.py": "runtime_attestation(self.base_url, self.clone)",
        bench / "programbench" / "run_programbench_e2e.py": "runtime_attestation(str(args.ouroboros_url), repo_dir)",
        # OSWorld: the step loop attests inside `_preflight`, the cu_bridge before its first
        # POST /api/tasks, and the preflight-only skeleton alongside its reachability probes.
        bench / "osworld" / "run_step_agent.py": "runtime_attestation(config.ouroboros_url, config.repo_dir)",
        bench / "osworld" / "run_cu_bridge_agent.py": "runtime_attestation(args.ouroboros_url, repo_dir)",
        bench / "osworld" / "osworld_adapter_skeleton.py": "runtime_attestation(ouroboros_url, repo_root)",
    }
    for path, call in wired.items():
        assert call in path.read_text(encoding="utf-8"), f"{path.name} lost its attestation call"

    # SWE-Pro attests inside the container (it has no host-side URL): one-shot, after readiness
    # and before the paid solve.
    entrypoint = (bench / "swe_bench_pro" / "e1v2" / "entrypoint_pro.sh").read_text(encoding="utf-8")
    assert "/api/health" in entrypoint and "runtime_skew" in entrypoint

    # Every wired call above must actually BIND against the shared helper's signature: a
    # name-only check would pass a call missing the required `repo_dir` positional (which is
    # how the commit half of owner Q7=B is reported) and only fail at run time.
    import ast

    from devtools.benchmarks.common.manifests import runtime_attestation
    signature = inspect.signature(runtime_attestation)
    for call in wired.values():
        node = ast.parse(call, mode="eval").body
        signature.bind(*node.args, **{kw.arg: kw.value for kw in node.keywords})

def test_runtime_attestation_decides_commit_availability_before_skew(tmp_path, monkeypatch):
    """Reason ORDER is part of the fail-closed contract. A checkout with no readable commit that
    ALSO disagrees on the version was labelled `runtime_skew` — an OVERRIDABLE reason — so
    `OBO_ALLOW_EVOLVED_VOLUME=1` waived a run with no commit to attribute its numbers to."""
    from devtools.benchmarks.common import manifests

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def read(self):
            return b'{"runtime_version": "6.75.0"}'

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Resp())
    monkeypatch.setenv(manifests.ALLOW_EVOLVED_VOLUME_ENV, "1")

    bare = tmp_path / "not-a-repo"
    bare.mkdir()
    (bare / "VERSION").write_text("6.74.5\n", encoding="utf-8")   # skew AND no commit
    with pytest.raises(RuntimeError, match="reason=commit_unavailable") as refused:
        manifests.runtime_attestation("http://127.0.0.1:9/", bare)
    assert "does NOT waive" in str(refused.value)

    # With a real commit the same version disagreement IS the waivable skew.
    repo = tmp_path / "repo"
    _git_repo(repo)
    (repo / "VERSION").write_text("6.74.5\n", encoding="utf-8")
    _git_commit_all(repo)
    skewed = manifests.runtime_attestation("http://127.0.0.1:9/", repo)
    assert skewed["reason"] == "runtime_skew" and skewed["overridden"] is True

def test_programbench_e2e_persists_the_manifest_when_attestation_refuses(tmp_path, monkeypatch, capsys):
    """A runtime-attestation refusal must leave the seed-admission manifest ON DISK.

    Attestation used to be evaluated inside `admit_benchmark_run(...)`'s argument list, and Python
    evaluates arguments before entering the callee — so `runtime_unreachable` /
    `commit_unavailable` / `runtime_skew` raised with no `run_manifest.json` written at all,
    defeating the durable-refusal contract by evaluation order alone.
    """
    from devtools.benchmarks.programbench import run_programbench_e2e as e2e

    out_root = tmp_path / "pb-attest"
    settings = tmp_path / "settings.json"
    settings.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(e2e, "_load_instances",
                        lambda **_k: [{"instance_id": "inst-a", "image_name": "img-a"}])
    monkeypatch.setattr(e2e, "run_root", lambda *_a, **_k: out_root)

    from devtools.benchmarks.common.manifests import RuntimeAttestationRefused

    record = {"schema": "ouroboros.benchmark.runtime_attestation.v1",
              "reason": "runtime_unreachable", "ok": False, "runtime_version": "",
              "repo_head": "a" * 40, "repo_version": "6.75.0", "override_set": False,
              "http_error": "OSError: connection refused"}

    def _refuse(url, repo):
        raise RuntimeAttestationRefused(
            "runtime attestation failed reason=runtime_unreachable", record)

    monkeypatch.setattr(e2e, "runtime_attestation", _refuse)
    # An instance stand-in that must NEVER be reached: the refusal precedes all spend.
    monkeypatch.setattr(e2e, "_process_instance",
                        lambda instance, cfg: pytest.fail("an instance ran after the refusal"))
    monkeypatch.setattr(
        sys, "argv",
        ["run_programbench_e2e.py", "--allow-dirty-seed", "--settings-path", str(settings),
         "--ouroboros-url", "http://127.0.0.1:9"],
    )
    # RETURNS the recorded code. It used to re-raise, which exits the process with status 1 while
    # the manifest said 3 — the record and reality disagreeing (see
    # test_migrated_launcher_exit_status_matches_the_recorded_exit_code).
    assert e2e.main() == 3
    assert "reason=runtime_unreachable" in capsys.readouterr().err

    manifest = json.loads((out_root / "run_manifest.json").read_text(encoding="utf-8"))
    # The seed gate's SHAPE is on disk (never its verdict: `ok` mirrors the ambient checkout).
    assert set(manifest["seed_gate"]) >= {"ok", "reason", "require_clean", "allow_dirty_seed"}
    assert manifest["seed_gate"]["require_clean"] is False
    assert manifest["seed_gate"]["ok"] is (not manifest["seed_gate"]["reason"])
    extra = manifest["extra"]
    assert extra["outcome"] == "refused"
    assert extra["exit_code"] == 3
    # The EXACT typed reason, not a generic message: the helper builds the record and the launcher
    # persists it, so the manifest keeps the facts the provenance contract exists to preserve.
    assert extra["refusal"] == {"stage": "runtime_attestation", "exit_code": 3,
                                "reason": "runtime_unreachable"}
    assert extra["runtime_attestation"]["reason"] == "runtime_unreachable"
    assert extra["runtime_attestation"]["runtime_version"] == ""
    assert extra["runtime_attestation"]["repo_head"] == "a" * 40
    assert extra["runtime_attestation"]["repo_version"] == "6.75.0"
    # No `error` key: nothing escaped, because the refusal is RETURNED. The record is the report.
    assert "error" not in extra

    # A refusal that carries NO record still refuses and still records a durable manifest, with the
    # generic reason as the documented fallback.
    def _bare(url, repo):
        raise RuntimeError("attestation blew up with no record")

    monkeypatch.setattr(e2e, "runtime_attestation", _bare)
    assert e2e.main() == 3
    assert "no record" in capsys.readouterr().err
    extra = json.loads((out_root / "run_manifest.json").read_text(encoding="utf-8"))["extra"]
    assert extra["refusal"]["reason"] == "runtime_attestation_failed"
    assert extra["runtime_attestation"] == {"pending": "not_attested_yet"}

def test_runtime_attestation_requires_the_contracted_runtime_version_field(tmp_path, monkeypatch):
    """Only the CONTRACTED field counts as a runtime identity.

    `runtime_version` is part of the frozen `HealthResponse` (`ouroboros/gateway/contracts.py`).
    The helper used to fall back to a generic `version` key, so ANY unrelated HTTP server that
    answered `{"version": "6.75.0"}` attested successfully and ProgramBench's default admission
    path would bless a server that is not Ouroboros at all. Its absence is now the distinct,
    NON-overridable reason `runtime_version_absent` — the endpoint answered, but not with the
    health contract, so no live runtime identity was established.
    """
    from devtools.benchmarks.common import manifests

    repo = tmp_path / "repo"
    _git_repo(repo)
    (repo / "VERSION").write_text("6.75.0\n", encoding="utf-8")
    _git_commit_all(repo)

    served: dict = {"version": "6.75.0"}          # a stranger's field, not the contract's

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def read(self):
            return json.dumps(served).encode("utf-8")

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Resp())
    monkeypatch.delenv(manifests.ALLOW_EVOLVED_VOLUME_ENV, raising=False)

    with pytest.raises(RuntimeError, match="reason=runtime_version_absent"):
        manifests.runtime_attestation("http://127.0.0.1:9/", repo)

    # ... and the override does NOT rescue it: it waives a deliberate skew only.
    monkeypatch.setenv(manifests.ALLOW_EVOLVED_VOLUME_ENV, "1")
    with pytest.raises(RuntimeError, match="reason=runtime_version_absent") as refused:
        manifests.runtime_attestation("http://127.0.0.1:9/", repo)
    assert "does NOT waive" in str(refused.value)
    assert "runtime_version_absent" not in manifests.OVERRIDABLE_ATTESTATION_REASONS

    # The contracted field attests, with the same payload otherwise unchanged.
    served.clear()
    served["runtime_version"] = "6.75.0"
    attested = manifests.runtime_attestation("http://127.0.0.1:9/", repo)
    assert attested["ok"] is True and attested["reason"] == ""
    assert attested["runtime_version"] == "6.75.0"

def test_gaia_and_tb_launchers_add_no_runtime_attestation(tmp_path):
    """Owner Q10: TB and GAIA are structurally immune (each sample/trial starts its own server
    from the checkout under test), so they get the seed gate and NOT attestation lines."""
    tb_dir = REPO_ROOT / "devtools" / "benchmarks" / "terminal_bench"
    gaia_dir = REPO_ROOT / "devtools" / "benchmarks" / "gaia"
    for path in (tb_dir / "run_tb.py", tb_dir / "run_harbor_smoke.py", gaia_dir / "run_gaia.py",
                 gaia_dir / "run_harness.py"):
        src = path.read_text(encoding="utf-8")
        assert "runtime_attestation" not in src, f"{path.name} must not attest a live runtime"
    for path in (tb_dir / "run_tb.py", tb_dir / "run_harbor_smoke.py", gaia_dir / "run_gaia.py"):
        assert "require_clean=not " in path.read_text(encoding="utf-8"), f"{path.name} lost its seed gate"
