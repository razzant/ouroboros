"""Structural coverage for the one pre-admission ownership primitive."""

import ast

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

    shadowed = approved.replace(
        "\ndef main(out):",
        '\ndef acquire_campaign_execution_lock(path, blocking=False):\n'
        '    return open(path / "shadow.lock", "a")\n'
        "\ndef main(out):",
    )
    violations = launcher_audit.audit_source(shadowed, name="shadowed_lock.py")
    assert any("acquire_campaign_execution_lock -> open" in item for item in violations)

    import_shadowed = approved.replace(
        "def main(out):",
        "def main(out):\n"
        "    import pathlib as acquire_campaign_execution_lock",
    )
    assert launcher_audit.audit_source(
        import_shadowed, name="import_shadowed_lock.py",
    )

    except_shadowed = approved.replace(
        "def main(out):",
        "def main(out):\n"
        "    try:\n"
        "        pass\n"
        "    except Exception as acquire_campaign_execution_lock:\n"
        "        pass",
    )
    assert launcher_audit.audit_source(
        except_shadowed, name="except_shadowed_lock.py",
    )


def test_pre_admission_lock_shape_rejects_a_second_open():
    source = '''
def acquire_campaign_execution_lock(run_root, blocking=True):
    root = pathlib.Path(run_root).resolve()
    root_digest = hashlib.sha256(str(root).encode("utf-8")).hexdigest()
    handle = (pathlib.Path(tempfile.gettempdir()) / f"lock-{root_digest}").open("a+")
    forbidden = (root / "inside.lock").open("a+")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    return handle
'''
    unit = launcher_audit._Unit(ast.parse(source), "synthetic_lock")
    target = unit.functions["acquire_campaign_execution_lock"]
    assert launcher_audit._safe_pre_admission_lock_helper(target, unit) is False


def test_pre_admission_lock_shape_rejects_temp_path_decoy():
    source = '''
def acquire_campaign_execution_lock(run_root, blocking=True):
    root = pathlib.Path(run_root).resolve()
    root_digest = hashlib.sha256(str(root).encode("utf-8")).hexdigest()
    receiver = pathlib.Path(tempfile.gettempdir()) and (root / f"{root_digest}.lock")
    handle = receiver.open("a+")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    return handle
'''
    unit = launcher_audit._Unit(ast.parse(source), "synthetic_lock")
    target = unit.functions["acquire_campaign_execution_lock"]
    assert launcher_audit._safe_pre_admission_lock_helper(target, unit) is False
