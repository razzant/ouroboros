#!/usr/bin/env python3
"""CI-only (not shipped): the one list of what this one-off check selects, and the verdict over what ran.

    verify_skills_candidate.py args <group>   # the selected test ids / `node --test` arguments, one per line
    verify_skills_candidate.py gate <dir>     # <dir>: status/<group> exit codes, <group>.junit.xml, node.tap, frames/

Runs and installs nothing; stdlib only (`python -I -S`). PASS needs, for every group, exit status 0 and every
selected case present exactly once and passed. The public verdict holds names and outcomes only: no messages,
output or tracebacks. The frame is uploaded by a separate workflow step, only when this gate passed.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

T = "tests/test_"
GROUPS = {
    "builtin-update": [  # an ordinary built-in update
        f"{T}builtin_skill_source.py::test_reseed_replaces_payload_keeps_state_and_owner_choice",
        f"{T}builtin_skill_source.py::test_builtin_grant_predecessor_classifies_only_bound_release_metadata[verified-prior-release]",
        f"{T}plugin_api_admission.py::test_builtin_reseed_with_unchanged_requested_sets_carries_grants",
        f"{T}skill_grant_follow.py::test_an_installed_builtin_rebinds_once_and_reinstalling_exact_bytes_keeps_the_grant_file",
    ],
    "repair-twice": [  # one click for an earlier grant, then repairs keep the explicit field
        f"{T}review_skill.py::test_an_earlier_grant_takes_one_click_then_repairs_keep_the_explicit_key[direct]",
        f"{T}review_skill.py::test_an_earlier_grant_takes_one_click_then_repairs_keep_the_explicit_key[edits]",
        f"{T}review_skill.py::test_an_earlier_grant_takes_one_click_then_repairs_keep_the_explicit_key[batch]",
        f"{T}review_skill.py::test_owner_repair_continues_only_explicit_local_keys_after_review[continuous]",
    ],
    "prior-off": [  # Off stays Off through a Repair, with the On control beside it
        f"{T}review_skill.py::test_owner_repair_continues_only_explicit_local_keys_after_review[explicit_off]",
        f"{T}skill_source_lifecycle.py::test_hub_repair_keeps_only_the_explicit_field_at_the_final_reviewed_digest[keeps-on]",
        f"{T}skill_source_lifecycle.py::test_hub_repair_keeps_only_the_explicit_field_at_the_final_reviewed_digest[keeps-off]",
    ],
    "saved-not-granted": [  # a saved field that is not granted: a note, not a blocker
        f"{T}installed_skills_status.py::test_a_saved_key_is_a_non_blocking_note_in_both_context_forms_and_list_skills",
        f"{T}skill_grant_follow.py::test_a_loaded_extension_reports_the_key_saved_but_not_granted",
    ],
    "browser": [f"{T}ci_skills_grant_cancel.py::test_grant_access_dialog_names_the_skill_and_field_and_cancel_changes_nothing"],
}
NODE_FILES = ["web/tests/skill_card_renderer.test.js", "web/tests/secret_hitl.test.js"]
NODE_NAMES = [
    "Enable asks first only for the access the host says stops the skill",
    "access the host says stops nothing stays a voluntary grant: Connect tools stays, no Needs access chip",
    "readers keep enabled and grant facts apart",
]
FRAME = "frames/skills-grant-dialog-1100x750.png"


def pytest_cases(path: Path, group: str) -> dict[str, str]:
    seen: Counter = Counter()
    result: dict[str, str] = {}
    if path.is_file():
        try:
            for case in ET.parse(path).getroot().iter("testcase"):
                key = (case.get("classname", ""), case.get("name", ""))
                seen[key] += 1
                bad = [child.tag for child in case if child.tag in {"failure", "error", "skipped"}]
                result[key] = bad[0] if bad else "passed"
        except ET.ParseError:
            return {nodeid: "unreadable" for nodeid in GROUPS[group]}
    out = {}
    for nodeid in GROUPS[group]:
        file, _, name = nodeid.partition("::")
        key = (file.removesuffix(".py").replace("/", "."), name)
        out[nodeid] = "missing" if key not in result else "duplicate" if seen[key] > 1 else result[key]
    return out


def node_cases(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    out = {}
    for name in NODE_NAMES:
        ok = len(re.findall(rf"^ok \d+ - {re.escape(name)}$", text, re.M))
        bad = len(re.findall(rf"^not ok \d+ - {re.escape(name)}$", text, re.M))
        out[name] = "passed" if (ok, bad) == (1, 0) else "failed" if bad else "duplicate" if ok > 1 else "missing"
    return out


NOTE = "DERIVED by verify_skills_candidate.py from the runner's original JUnit/TAP: selected names and statuses only; originals stay in the runner's temp area"


def derive(root: Path, groups: dict) -> None:
    """Signed, reconstructed lists: only selected names, allowlisted statuses and each group's exit, never raw fields or output."""
    suites = ET.Element("testsuites", {"derived": "true"})
    results = []  # node: (name, error) in TAP order; the last entry is the process exit
    for name, group in groups.items():
        code = group["exit"] if group["exit"].isdigit() else "not run"
        rows = [(case, "" if value == "passed" else value) for case, value in group["cases"].items()]
        rows.append(("process exit", "" if code == "0" else f"exit {code}"))
        suite = ET.SubElement(suites, "testsuite", {"name": name, "tests": str(len(rows)),
                                                    "failures": str(sum(1 for _, bad in rows if bad)), "exit": code})
        ET.SubElement(ET.SubElement(suite, "properties"), "property", {"name": "derived", "value": NOTE})
        for case, bad in rows:
            node = ET.SubElement(suite, "testcase", {"name": case})
            if bad:
                ET.SubElement(node, "failure", {"type": bad})
        if name == "node":
            results = rows
    ET.ElementTree(suites).write(root / "derived.junit.xml", encoding="utf-8", xml_declaration=True)
    tap = ["TAP version 13", f"# {NOTE}", f"1..{len(results)}"]
    tap += [f"{'not ok' if bad else 'ok'} {i} - {case}" + (f" # {bad}" if bad else "") for i, (case, bad) in enumerate(results, 1)]
    (root / "derived.tap").write_text("\n".join(tap) + "\n", encoding="utf-8")


def gate(root: Path) -> int:
    groups = {}
    for group in [*GROUPS, "node"]:
        status = root / "status" / group
        code = status.read_text().strip() if status.is_file() else "not run"
        cases = node_cases(root / "node.tap") if group == "node" else pytest_cases(root / f"{group}.junit.xml", group)
        groups[group] = {"exit": code, "cases": cases,
                         "ok": code == "0" and all(value == "passed" for value in cases.values())}
    frame = root / FRAME
    data = frame.read_bytes() if frame.is_file() else b""
    frame_ok = data[:8] == b"\x89PNG\r\n\x1a\n" and struct.unpack(">II", data[16:24]) == (1100, 750)
    ok = all(group["ok"] for group in groups.values()) and frame_ok
    verdict = {"verdict": "PASS" if ok else "FAIL", "groups": groups,
               "frame_sha256": hashlib.sha256(data).hexdigest() if ok else None}
    (root / "verdict.json").write_text(json.dumps(verdict, indent=1) + "\n", encoding="utf-8")
    derive(root, groups)
    lines = [f"## Skills candidate check: {verdict['verdict']}"]
    lines += [f"- {name}: {'PASS' if group['ok'] else 'FAIL'} (exit {group['exit']})" for name, group in groups.items()]
    lines += [f"  - {case}: {value}" for group in groups.values() for case, value in group["cases"].items() if value != "passed"]
    lines += [] if frame_ok else ["- frame: missing or not 1100x750"]
    print("\n".join(lines))
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
    return 0 if ok else 1


def main(argv: list[str]) -> int:
    if len(argv) == 3 and argv[1] == "args" and argv[2] in GROUPS:
        print("\n".join(GROUPS[argv[2]]))
    elif len(argv) == 3 and argv[1] == "args" and argv[2] == "node":
        assert not any(re.search(r"[\\^$.*+?()\[\]{}|/]", name) for name in NODE_NAMES), "a name must be a literal pattern"
        print("\n".join([*(f"--test-name-pattern=^{name}$" for name in NODE_NAMES), *NODE_FILES]))
    elif len(argv) == 3 and argv[1] == "gate":
        return gate(Path(argv[2]))
    else:
        print(__doc__, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
