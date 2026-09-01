"""Structural coverage for the one pre-admission ownership primitive."""

from devtools.benchmarks.common import launcher_audit


def test_pre_admission_lock_exemption_is_exact_and_shape_checked():
    approved = '''
from devtools.benchmarks.common.manifests import admit_benchmark_run, finalize_run_manifest
from devtools.benchmarks.cybergym.cybergym_result_index import acquire_campaign_execution_lock

def main(out):
    lock = acquire_campaign_execution_lock(out, blocking=False)
    manifest = admit_benchmark_run(out / "run_manifest.json")
    with finalize_run_manifest(out / "run_manifest.json", manifest):
        return 0
'''
    assert launcher_audit.audit_source(approved, name="approved_lock.py") == []

    same_name_wrong_body = '''
from devtools.benchmarks.common.manifests import admit_benchmark_run, finalize_run_manifest

def acquire_campaign_execution_lock(path, blocking=False):
    return open(path / "inside-candidate.lock", "a")

def main(out):
    lock = acquire_campaign_execution_lock(out, blocking=False)
    manifest = admit_benchmark_run(out / "run_manifest.json")
    with finalize_run_manifest(out / "run_manifest.json", manifest):
        return 0
'''
    violations = launcher_audit.audit_source(
        same_name_wrong_body, name="unapproved_lock.py",
    )
    assert any("acquire_campaign_execution_lock -> open" in item for item in violations)
