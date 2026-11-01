"""What this interpreter really is, measured before any test, against what the cell was meant to be. Stdlib only.

    python3 -I -B probe.py EXPECT_JSON OUT.json       exit 3: the condition this cell exists to observe is not here

The test of ensurepip is functional, by the command the product runs: Debian can ship an ``ensurepip`` package that
imports and is disabled, so ``find_spec`` alone (like ``import ensurepip`` in tests 279 and 321) would say nothing.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile


def _run(*argv, timeout=180) -> dict:
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                              env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})  # measuring writes no bytecode
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"rc": None, "out": "", "err": repr(exc)}
    return {"rc": done.returncode, "out": done.stdout[-1000:], "err": done.stderr[-1000:]}


def _installed(package: str) -> bool:
    return _run("dpkg-query", "-W", "-f=${Status}", package)["out"].endswith("install ok installed")


def measure() -> dict:
    version = "%d.%d" % sys.version_info[:2]
    marker = os.path.join(sysconfig.get_path("stdlib"), "EXTERNALLY-MANAGED")
    has_marker = os.path.isfile(marker)
    scratch = tempfile.mkdtemp(prefix="probe-venv-")
    try:
        venv = _run(sys.executable, "-m", "venv", os.path.join(scratch, "v"))
        without_pip = _run(sys.executable, "-m", "venv", "--without-pip", os.path.join(scratch, "w"))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    spec = importlib.util.find_spec("ensurepip")
    bundled = []
    if spec and spec.origin:
        folder = os.path.join(os.path.dirname(spec.origin), "_bundled")
        bundled = sorted(os.listdir(folder)) if os.path.isdir(folder) else []
    conf = "/etc/pip.conf"
    # Whole, not a tail: the root overlay mount is normally the first line. Field 5 is the mount point.
    mounts = open("/proc/self/mountinfo", encoding="utf-8").read() if os.path.exists("/proc/self/mountinfo") else ""
    return {
        "executable": sys.executable, "real": os.path.realpath(sys.executable), "version": sys.version.split()[0],
        "bare": sys.prefix == sys.base_prefix, "euid": os.geteuid(), "hostname": os.uname().nodename,
        "marker": marker, "marker_present": has_marker,
        "marker_sha256": hashlib.sha256(open(marker, "rb").read()).hexdigest() if has_marker else "",
        "marker_owner": _run("dpkg", "-S", marker),
        "policy_overridden": bool(os.environ.get("PIP_BREAK_SYSTEM_PACKAGES")) or (
            os.path.isfile(conf) and "break-system-packages" in open(conf, encoding="utf-8", errors="replace").read()),
        "venv_with_pip": venv, "ensurepip_works": venv["rc"] == 0, "venv_without_pip_rc": without_pip["rc"],
        "ensurepip_module": spec.origin if spec else "", "ensurepip_bundled": bundled,
        "own_pip": _run(sys.executable, "-I", "-m", "pip", "--version")["rc"] == 0,
        "packages": {name: _installed(name) for name in ("python3-pip", "python3-venv", f"python{version}-venv")},
        "os_release": open("/etc/os-release", encoding="utf-8").read() if os.path.isfile("/etc/os-release") else "",
        "container": os.path.exists("/.dockerenv"),
        "mounts": mounts, "root_mount": [line for line in mounts.splitlines() if line.split()[4:5] == ["/"]],
    }


def judge(facts: dict, expect: dict) -> list:
    """Each condition the cell exists for that this interpreter does not have; empty means the cell may run."""
    problems = []
    uid_ok = facts["euid"] == 0 if expect["uid"] == "root" else facts["euid"] != 0
    required = [("bare", facts["bare"]), ("real marker", facts["marker_present"]),
                ("marker owned by a package", facts["marker_owner"].get("rc") == 0),
                ("pip policy not overridden", not facts["policy_overridden"]), ("uid " + expect["uid"], uid_ok),
                (f"ensurepip_works={expect['ensurepip_works']}", facts["ensurepip_works"] == expect["ensurepip_works"]),
                (f"own_pip={expect['own_pip']}", facts["own_pip"] == expect["own_pip"])]
    for label, holds in required:
        if not holds:
            problems.append(label)
    return problems


def main(argv: list) -> int:
    facts = measure()
    problems = judge(facts, json.loads(argv[0]))
    with open(argv[1], "w", encoding="utf-8") as fh:
        json.dump({**facts, "condition_not_met": problems}, fh, indent=1)
    print("condition_not_met:", problems)
    return 3 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
