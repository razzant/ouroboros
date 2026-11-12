"""Editbench checker: grade a candidate toyproj directory against ground truth.

Usage:
    python devtools/benchmarks/editbench/checker.py <candidate_dir> [--json]

Grading:
- Per-file exact text comparison (trailing whitespace per line and final
  newline normalized) against fixtures/expected/.
- Behavior check: ``python main.py`` inside the candidate dir must print OK
  and exit 0.
- Extra files created in the candidate dir are reported (not a failure by
  themselves, but suspicious).
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
EXPECTED_DIR = HERE / "fixtures" / "expected"
FILES = [
    "core.py",
    "utils.py",
    "models.py",
    "config.py",
    "report.py",
    "legacy.py",
    "main.py",
    "README.md",
]


def _normalize(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines).rstrip("\n") + "\n"


def grade_generic(
    candidate_dir: pathlib.Path,
    expected_dir: pathlib.Path,
    files: list[str],
    check_argv: list[str] | None = None,
    check_pythonpath: bool = False,
) -> dict:
    """Grade a candidate dir against an arbitrary expected dir + optional check.

    ``files`` are expected-relative paths compared text-exact (trailing
    whitespace normalized). ``check_argv`` (if given) must print OK and exit 0;
    with ``check_pythonpath`` the candidate dir is prepended to PYTHONPATH,
    otherwise the check runs with cwd=candidate_dir.
    """
    import os

    result: dict = {"candidate": str(candidate_dir), "files": {}, "extra_files": []}
    ok_files = 0
    for name in files:
        expected_path = expected_dir / name
        candidate_path = candidate_dir / name
        entry: dict = {}
        if not candidate_path.exists():
            entry["status"] = "missing"
        else:
            expected = _normalize(expected_path.read_text(encoding="utf-8"))
            actual = _normalize(candidate_path.read_text(encoding="utf-8"))
            if expected == actual:
                entry["status"] = "match"
                ok_files += 1
            else:
                entry["status"] = "mismatch"
                import difflib

                diff = list(
                    difflib.unified_diff(
                        expected.splitlines(), actual.splitlines(),
                        fromfile=f"expected/{name}", tofile=f"candidate/{name}",
                        lineterm="", n=1,
                    )
                )
                entry["diff"] = "\n".join(diff[:80])
        result["files"][name] = entry

    behavior: dict = {"ok": True}
    if check_argv:
        env = dict(os.environ)
        if check_pythonpath:
            env["PYTHONPATH"] = str(candidate_dir)
        try:
            proc = subprocess.run(
                check_argv,
                cwd=str(candidate_dir) if not check_pythonpath else None,
                env=env, capture_output=True, text=True, timeout=60,
            )
            behavior["exit_code"] = proc.returncode
            behavior["stdout"] = proc.stdout.strip()[:500]
            behavior["stderr"] = proc.stderr.strip()[:500]
            behavior["ok"] = proc.returncode == 0 and proc.stdout.strip().endswith("OK")
        except Exception as e:  # noqa: BLE001 - report any runner failure as a grade
            behavior["ok"] = False
            behavior["error"] = str(e)
    result["behavior"] = behavior
    result["files_matched"] = ok_files
    result["files_total"] = len(files)
    result["all_files_match"] = ok_files == len(files)
    result["pass"] = result["all_files_match"] and behavior.get("ok", False)
    return result


def oracle_problems(
    workspace_dir: pathlib.Path,
    expected_dir: pathlib.Path,
    files: list[str],
    check_argv: list[str] | None = None,
    check_pythonpath: bool = False,
) -> list[str]:
    """Why this workspace/expected pair cannot grade a run; [] when it can.

    The expected tree must hold every graded file and pass its own oracle, and
    the untouched workspace must fail it: a task its seed already passes measures
    nothing. Grades throwaway COPIES, so a check that writes bytecode or scratch
    never alters the fixture it validates.
    """
    import shutil
    import tempfile

    workspace_dir, expected_dir = pathlib.Path(workspace_dir), pathlib.Path(expected_dir)
    problems = [f"no {side} directory {root}" for side, root in
                (("workspace", workspace_dir), ("expected", expected_dir)) if not root.is_dir()]
    problems += [f"expected tree lacks graded file {name}" for name in files
                 if expected_dir.is_dir() and not (expected_dir / name).is_file()]
    if problems:
        return problems
    with tempfile.TemporaryDirectory(prefix="editbench-oracle-") as tmp:
        copies = {side: pathlib.Path(shutil.copytree(src, pathlib.Path(tmp) / side))
                  for side, src in (("workspace", workspace_dir), ("expected", expected_dir))}
        own = grade_generic(copies["expected"], expected_dir, files, check_argv, check_pythonpath)
        if not own["pass"]:
            behavior = own["behavior"]
            problems.append(
                "expected tree fails its own oracle: "
                f"{own['files_matched']}/{own['files_total']} files, check exit "
                f"{behavior.get('exit_code')} {behavior.get('stderr') or behavior.get('error') or ''}".strip()
            )
        if grade_generic(copies["workspace"], expected_dir, files, check_argv, check_pythonpath)["pass"]:
            problems.append("the untouched workspace already passes the oracle")
    return problems


def grade(candidate_dir: pathlib.Path) -> dict:
    result: dict = {"candidate": str(candidate_dir), "files": {}, "extra_files": []}
    ok_files = 0
    for name in FILES:
        expected_path = EXPECTED_DIR / name
        candidate_path = candidate_dir / name
        entry: dict = {}
        if not candidate_path.exists():
            entry["status"] = "missing"
        else:
            expected = _normalize(expected_path.read_text(encoding="utf-8"))
            actual = _normalize(candidate_path.read_text(encoding="utf-8"))
            if expected == actual:
                entry["status"] = "match"
                ok_files += 1
            else:
                entry["status"] = "mismatch"
                import difflib

                diff = list(
                    difflib.unified_diff(
                        expected.splitlines(),
                        actual.splitlines(),
                        fromfile=f"expected/{name}",
                        tofile=f"candidate/{name}",
                        lineterm="",
                        n=1,
                    )
                )
                entry["diff"] = "\n".join(diff[:60])
        result["files"][name] = entry

    known = set(FILES) | {"__pycache__"}
    for p in sorted(candidate_dir.iterdir()):
        if p.name not in known:
            result["extra_files"].append(p.name)

    behavior: dict = {}
    try:
        proc = subprocess.run(
            [sys.executable, "main.py"],
            cwd=candidate_dir,
            capture_output=True,
            text=True,
            timeout=30,
        )
        behavior["exit_code"] = proc.returncode
        behavior["stdout"] = proc.stdout.strip()[:500]
        behavior["stderr"] = proc.stderr.strip()[:500]
        behavior["ok"] = proc.returncode == 0 and proc.stdout.strip() == "OK"
    except Exception as e:  # noqa: BLE001 - report any runner failure as a grade
        behavior["ok"] = False
        behavior["error"] = str(e)
    result["behavior"] = behavior

    result["files_matched"] = ok_files
    result["files_total"] = len(FILES)
    result["all_files_match"] = ok_files == len(FILES)
    result["pass"] = result["all_files_match"] and behavior.get("ok", False)
    return result


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    as_json = "--json" in sys.argv
    if len(args) != 1:
        print(__doc__)
        return 2
    candidate = pathlib.Path(args[0]).resolve()
    if not candidate.is_dir():
        print(f"candidate dir not found: {candidate}")
        return 2
    result = grade(candidate)
    if as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"candidate: {result['candidate']}")
        for name, entry in result["files"].items():
            print(f"  {name}: {entry['status']}")
            if entry.get("diff"):
                print("    " + entry["diff"].replace("\n", "\n    "))
        if result["extra_files"]:
            print(f"  extra files: {', '.join(result['extra_files'])}")
        print(f"behavior ok: {result['behavior'].get('ok')}")
        print(f"PASS: {result['pass']} ({result['files_matched']}/{result['files_total']} files)")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
