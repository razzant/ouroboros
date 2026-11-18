"""Project private runner reports to named public test targets and bounded failure kinds."""
import json
import os
import re
import sys
from pathlib import Path
import runpy

# The project's classifier distinguishes a real expected assertion from a skip or broken fixture.
verdicts = runpy.run_path(str(Path(__file__).resolve().parents[2] / 'tests' / 'ci_verdicts.py'))

LANES = ("main-chromium", "main-webkit", "find-chromium", "find-webkit")
RAW = Path(os.environ["QA_RAW"])


def python_report(file, paths):
    facts, error = verdicts['final'](file.parent)
    if error or not isinstance(facts.get('selected_nodeids'), list):
        return None, [{'kind': 'missing_or_invalid_final_projection'}]
    rows = facts['reports']
    xfails = verdicts['expected_failures'](rows)
    selected = set(facts['selected_nodeids'])
    counts = dict(tests=len(selected), failed=0, skipped=0, xfailed=0, passed=0)
    by_node = {}
    for row in rows:
        by_node.setdefault(row['nodeid'], []).append(row)
    issues = []
    # No parameter values, assertion text, stdout or fixture data in the published receipt.
    def note(node, kind):
        path, *parts = str(node).split('[', 1)[0].split('::')
        detail = {'file': path} if path in paths else {}
        if detail and parts and re.fullmatch(r'test_[A-Za-z0-9_]{1,160}', parts[-1]):
            detail['test'] = parts[-1]
        issues.append({'kind': kind, **detail})
    for node in sorted(selected | set(by_node)):
        reports = by_node.get(node, [])
        phases = [r['phase'] for r in reports if not r.get('is_subtest')]
        if node not in selected or sorted(phases) != ['call', 'setup', 'teardown']:
            counts['failed'] += 1
            note(node, 'unexpected_or_incomplete_test')
        elif node in xfails:
            counts['xfailed'] += 1
            note(node, 'expected_failure')
        else:
            bad = next((r for r in reports if r['outcome'] == 'failed' or r.get('wasxfail') is not None), None)
            if bad is not None:
                counts['failed'] += 1
                note(node, bad['phase'] + '_failure_or_xpass')
            elif any(r['outcome'] == 'skipped' for r in reports):
                counts['skipped'] += 1
                note(node, 'skipped')
            else:
                counts['passed'] += 1
    for tag in ('collection_failures', 'collection_skips'):
        for item in facts.get(tag, []):
            counts['failed'] += 1
            note(item.get('nodeid') if isinstance(item, dict) else item, tag)
    modules = {node.split('::', 1)[0] for node in selected}
    for path in sorted(set(paths) - modules):
        counts['failed'] += 1
        note(path, 'target_not_executed')
    return counts, issues


def node_report(file, paths):
    text = file.read_text(encoding="utf-8", errors="replace")
    footer = dict(re.findall(r"^# (tests|fail|skipped|cancelled) (\d+)\s*$", text, re.M))
    counts = None
    if {"tests", "fail", "skipped", "cancelled"} <= footer.keys():
        counts = {"tests": int(footer["tests"]), "failed": int(footer["fail"]),
                  "skipped": int(footer["skipped"]) + int(footer["cancelled"])}
    issues = []
    known = {"testTimeoutFailure": "timeout", "hookFailed": "hook_error", "subtestsFailed": "subtests_failed",
             "uncaughtException": "uncaught_exception", "unhandledRejection": "unhandled_rejection"}
    # TAP's diagnostic section is private. Read only its recognized enum and
    # allowlisted checkout-relative file/line; never its free-form title or stack.
    for case in re.finditer(r"(?m)^\s*not ok \d+ - ([^\n]*)\n[ \t]+---\n(.*?)(?m:^[ \t]+\.\.\.$)", text, re.S):
        title, diagnostic = case.groups()
        reason = re.search(r"(?m)^\s*failureType: ['\"]?([A-Za-z]+)", diagnostic)
        row = {"kind": known.get(reason[1], "test_failure") if reason else "test_failure"}
        location = re.search(r"(?m)^\s*location: ([^\n]+)", diagnostic)
        for path in paths:
            at = re.search(r"(?:^|/)" + re.escape(path) + r":([0-9]{1,8})(?=[:'\" ]|$)", location[1]) if location else None
            if at or title.strip() == path:
                row["file"] = path
                if at:
                    row["line"] = int(at[1])
                break
        issues.append(row)
    # De-duplicate file-level TAP wrappers without publishing their raw names.
    return counts, list({tuple(row.items()): row for row in issues}.values())


def record(lane, kind, code, expected, targets):
    if lane not in LANES or kind not in {"pytest", "tap"}:
        raise SystemExit(2)
    pattern = r"tests/(?:[a-zA-Z0-9_]+/)?[a-zA-Z0-9_]+\.py" if kind == "pytest" else r"web/tests/[a-zA-Z0-9_.-]+\.test\.js"
    paths = [path for path in targets if re.fullmatch(pattern, path)]
    file = RAW / lane / 'results.json' if kind == 'pytest' else RAW / f'{lane}.log'
    counts, issues = None, []
    try:
        if file.stat().st_size <= 16 * 1024 * 1024:
            counts, issues = (python_report if kind == "pytest" else node_report)(file, paths)
    except (OSError, ValueError, KeyError, TypeError):
        pass
    complete = (counts is not None and counts["tests"] > 0 and not counts["failed"] and not counts["skipped"]
                and (not expected or counts["tests"] == expected))
    status = "passed" if not code and complete else "failed" if code else "incomplete"
    reason = None if status == "passed" else "timeout" if code == 124 else "interrupted" if code >= 128 else (
        "nonzero_exit" if code else "missing_or_incomplete_test_report")
    result = {"status": status, "reason": reason, "exit_code": code, "counts": counts,
              "observations": issues[:40], "observations_omitted": max(0, len(issues) - 40)}
    (RAW / f"{lane}.json").write_text(json.dumps(result), encoding="utf-8")
    print(f"{lane}: {status} (exit {code}, counts {json.dumps(counts)})")
    return 0 if status == "passed" else 1


def summarize(output):
    def read(lane):
        path = RAW / f"{lane}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"status": "not_run"}

    candidate = os.environ["GITHUB_SHA"]
    if not re.fullmatch(r"[0-9a-f]{40}", candidate):
        raise SystemExit(2)
    outcomes = ("success", "failure", "skipped", "cancelled")
    stages = {k.lower(): v if v in outcomes else "unknown" for k in ("DEPS_SETUP", "NODE_SETUP", "BROWSER_SETUP", "MAIN_BASE",
              "MAIN_PACK", "FIND_BASE") if (v := os.environ.get("QA_" + k)) is not None}
    rows = {"scope": "focused diagnostic", "candidate_sha": candidate,
            "main_baseline": "e758a2190298d9890b1f26e632f63775d0b33d55",
            "find_baseline": "54f57a5b5baf9c6448bc60a3e7788cf7290dd9be",
            "stages": stages, "lanes": {lane: read(lane) for lane in LANES}}
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a", encoding="utf-8") as summary:
        print("## Main and Find preview\n\n| Lane | Outcome |\n| --- | --- |", file=summary)
        for lane, observation in rows["lanes"].items():
            print(f"| {lane} | {observation['status']} |", file=summary)


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "record":
        _, _, lane, kind, code, expected, *targets = sys.argv
        raise SystemExit(record(lane, kind, int(code), int(expected), targets))
    if command == "summarize":
        summarize(Path(sys.argv[2]))
    else:
        raise SystemExit(2)
