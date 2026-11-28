#!/usr/bin/env python3
"""One-shot Ubuntu reference and sender evidence on an exact detached Nano product."""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from urllib.request import urlopen


SENDER = tuple(
    f"tests/test_nano_local_exact.py::test_exact_local_send_reprices_from_the_callers_ceiling[{case}]"
    for case in ("restore-room", "shrink-to-fit")
)
PHASES = {"setup", "call", "teardown"}
QA_REF = "refs/heads/qa/nano-guarded-memory-80k"


def reference_pins():
    tree = ast.parse(Path("scripts/nano_reference_capacity.py").read_bytes())
    wanted = {"MODELS", "HUB_REVISION", "ANTI_SLOP", "SKILL_PIN", "ANTI_SLOP_BODY_SHA256", "REFERENCE_DEPS"}
    return {stmt.targets[0].id: ast.literal_eval(stmt.value) for stmt in tree.body
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name) and stmt.targets[0].id in wanted}


def stage_reference(cache):
    """Fetch only the five public data pins, without importing the candidate."""
    if not sys.platform.startswith("linux") or os.environ.get("GITHUB_ACTIONS") != "true":
        raise SystemExit("Reference staging is for the approved ephemeral Ubuntu job")
    values = reference_pins()
    models = values["MODELS"]
    if set(models) != {"qwen35", "gemma4"}:
        raise SystemExit("The public pin set changed")
    pins = []
    for label, (repo, rev, _bos, assets) in models.items():
        if (not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo)
                or not re.fullmatch(r"[0-9a-f]{40}", rev)
                or set(assets) != {"tokenizer.json", "chat_template.jinja"}):
            raise SystemExit("Unexpected model pin")
        pins.extend((cache / label / name, f"https://huggingface.co/{repo}/resolve/{rev}/{name}", pin)
                    for name, pin in assets.items())
    hub, name = values["HUB_REVISION"], values["ANTI_SLOP"]
    if not re.fullmatch(r"[0-9a-f]{40}", hub) or name != "anti-slop.SKILL.md":
        raise SystemExit("Unexpected instruction pin")
    pins.append((cache / name, f"https://raw.githubusercontent.com/razzant/OuroborosHub/{hub}/skills/anti-slop/SKILL.md",
                 values["SKILL_PIN"]))
    cache.mkdir()
    for dst, url, (length, digest) in pins:
        if not 0 < length <= 35_000_000 or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise SystemExit("Invalid public file pin")
        with urlopen(url, timeout=45) as response:
            data = response.read(length + 1)
        if len(data) != length or hashlib.sha256(data).hexdigest() != digest:
            raise SystemExit(f"Pin integrity failure: {dst.name}")
        dst.parent.mkdir(exist_ok=True)
        with dst.open("xb") as handle:
            handle.write(data)


def node_proof(record, wanted, exit_code, identity):
    """Project only known node ids and terminal phase facts from the stock pytest reporter."""
    selected = record.get("selected_nodeids")
    report_rows = record.get("reports")
    expected = set(wanted)
    collected = (isinstance(selected, list) and all(isinstance(node, str) for node in selected)
                 and len(selected) == len(expected) and set(selected) == expected)
    observed = defaultdict(lambda: defaultdict(list))
    foreign_or_invalid = not isinstance(report_rows, list)
    for row in report_rows if isinstance(report_rows, list) else ():
        if not isinstance(row, dict):
            foreign_or_invalid = True
            continue
        node, phase, outcome = (row.get(key) for key in ("nodeid", "phase", "outcome"))
        if (not isinstance(node, str) or node not in expected or not isinstance(phase, str)
                or phase not in PHASES or outcome not in ("passed", "failed", "skipped")
                or row.get("wasxfail") is not None):
            foreign_or_invalid = True
            continue
        if row.get("is_subtest") is True:
            # A passing subtest is no substitute for call; only the ordinary node can qualify.
            foreign_or_invalid |= outcome != "passed"
            continue
        observed[node][phase].append(outcome)
    complete = {node for node, readings in observed.items()
                if set(readings) == PHASES and all(events == ["passed"] for events in readings.values())}
    environment = record.get("environment")
    bound = (record.get("format") == 1 and record.get("github") == identity
             and type(record.get("session_exit_code")) is int
             and isinstance(environment, dict) and environment.get("platform") == "Linux")
    accepted = (bool(wanted) and collected and not foreign_or_invalid and bound and exit_code == 0
                and record["session_exit_code"] == 0 and complete == expected
                and record.get("collection_failures") == record.get("collection_skips") == [])
    return {"accepted": accepted, "pytest_exit": exit_code, "session_bound": bound,
            "selection_matches": collected, "invalid_or_outside_reports": foreign_or_invalid,
            "selected_count": len(selected) if isinstance(selected, list) else None,
            "missing_from_selection": sorted(expected - {x for x in selected if isinstance(x, str)})
            if isinstance(selected, list) else sorted(expected),
            "foreign_selection_count": sum(not isinstance(x, str) or x not in expected for x in selected)
            if isinstance(selected, list) else None,
            "collection_skip_count": len(record["collection_skips"]) if isinstance(record.get("collection_skips"), list) else None,
            "collection_failure_count": len(record["collection_failures"]) if isinstance(record.get("collection_failures"), list) else None,
            "passed_nodes": sorted(complete), "incomplete_nodes": sorted(expected - complete),
            "observed_phases": {node: dict(sorted(observed.get(node, {}).items())) for node in sorted(expected)}}


def check_guarded_publication(row, digest, limits):
    """Require independent writer and fresh-start evidence; a count of seeded files is insufficient."""
    memory_limit, work_floor, reply = limits
    proof = row["guarded_publication"]
    sources = {"identity", "scratchpad", "overview", "knowledge_index", "mark", "rules"}
    active_sources = {"identity", "scratchpad", "scratchpad_blocks", "overview", "knowledge_index", "marks", "rules", "publication"}
    active, seeds, last, refused = (proof[key] for key in ("active_sha256", "seeds", "last_accept", "refusal"))
    if (set(seeds) != sources or set(active) != active_sources or not all(digest(value) for value in active.values())
            or proof["fresh_start_marker_sha256"].keys() != seeds.keys()
            or not all(digest(value) for value in proof["fresh_start_marker_sha256"].values())
            or last["revision_sha256"] != active["publication"] or refused["active_sha256"] != active
            or refused["outcome"] != "over_capacity" or type(refused["candidate_memory"]) is not int
            or refused["candidate_memory"] <= memory_limit or type(refused["work_shortfall"]) is not int
            or refused["work_shortfall"] < 0 or type(proof["accepted_rows"]) is not int
            or type(proof["refused_rows"]) is not int or not 32 < proof["accepted_rows"] < proof["refused_rows"] <= 640
            or proof["accepted_rows"] + 1 != proof["refused_rows"]):
        raise SystemExit("Guarded mature publication does not prove six edits and the unchanged refusal")
    revisions = [entry["revision_sha256"] for entry in (*seeds.values(), last)]
    if len(set(revisions)) != 7 or not all(digest(value) for value in revisions):
        raise SystemExit("The six seed writes and the final growth did not each advance the active revision")
    observations = [entry["counted_proposal"] for entry in (*seeds.values(), last, refused)]
    if (any(not digest(entry["sha256"]) or type(entry["input_tokens"]) is not int or entry["input_tokens"] <= 0
            for entry in observations) or len({entry["sha256"] for entry in observations}) != len(observations)):
        raise SystemExit("Each proposal must have its own exact count before activation")
    samples = [proof[key] for key in ("initial", "seeded", "boundary")]
    for sample in samples:
        if (not digest(sample["request_sha256"]) or not all(type(sample[key]) is int
                for key in ("background", "reply", "work_headroom", "memory")) or sample["reply"] != reply
                or not 0 <= sample["memory"] <= memory_limit or sample["work_headroom"] < work_floor
                or not 0 < sample["background"] <= 80_000 - memory_limit - work_floor - reply
                or sample["background"] + sample["memory"] + sample["work_headroom"] + reply != 80_000):
            raise SystemExit("A sampled writer envelope is not an exact, fully reserved 80k count")
    initial, seeded, boundary = (sample["memory"] for sample in samples)
    if not 0 <= initial < seeded < boundary <= memory_limit or memory_limit - boundary > 256:
        raise SystemExit("The accepted writer sequence did not reach the measured boundary")


def check_reference(path):
    """Validate the meaning of the positive public report, not just its exit code."""
    report = json.loads(Path(path).read_bytes())
    limits = report["memory_work_reply_reserved"]
    if (not isinstance(limits, list) or limits != [20_000, 15_000, 8_192]
            or any(type(n) is not int for n in limits) or report["frame_limit"] != 80_000 - sum(limits)):
        raise SystemExit("Reference report does not measure the 80k/20k/15k contract and one reply reserve")
    memory_limit, work_floor, reply = limits

    def digest(value):
        return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))

    # Match the already-bound script's public pins without importing it on the staging or proof path.
    pins = reference_pins()
    models = {name: {"repo": repo, "revision": rev, "asset_sha256": {path: pin[1] for path, pin in assets.items()}}
              for name, (repo, rev, _bos, assets) in pins["MODELS"].items()}
    if (report["models"] != models or report["reference_dependencies"] != pins["REFERENCE_DEPS"]
            or report["anti_slop_hub_revision"] != pins["HUB_REVISION"]
            or report["anti_slop_file_sha256"] != pins["SKILL_PIN"][1]):
        raise SystemExit("Reference model, dependency or instruction source drifted")
    if report["start_reference_kind"] != (
        "exact startup selector over pinned HF counts; synthetic queued Web Main, no serving formatter or request sent"
    ):
        raise SystemExit("Reference report does not claim the exact synthetic startup surface")
    if report["framing"] != {
        "add_generation_prompt": True, "enable_thinking": False, "eos_token": "",
        "bos_token_by_model": {name: pin[2] for name, pin in pins["MODELS"].items()},
        "add_special_tokens": False, "truncation": False, "padding": False,
        "clock": "one fixed trailing user row", "reply": "reserved, not tokenized",
        "work_occurrences_checked_in": ["prepared messages", "rendered template"],
    }:
        raise SystemExit("Reference framing changed")
    profiles = report["profiles"]
    if set(profiles) != {"fresh_bare", "fresh_recommended", "mature_recommended", "preseeded_overflow"}:
        raise SystemExit("Reference report profile selection changed")
    clock_rows = set()
    for name, paired in profiles.items():
        if set(paired) != {"qwen35", "gemma4"}:
            raise SystemExit(f"{name}: both tokenizers are required")
        qwen, gemma = paired["qwen35"], paired["gemma4"]
        for key in (() if name == "mature_recommended" else ("capture_sha256", "tool_schema_sha256", "source_sha256", "full_index_sha256",
                    "installed_anti_slop_catalogue_row_sha256", "extension_reader_sha256")):
            if not qwen[key] or qwen[key] != gemma[key]:
                raise SystemExit(f"{name}: the two tokenizers did not use the same captured source ({key})")
        common = ("work_sha256",) if name == "mature_recommended" else ("fixed_fields_sha256", "work_sha256")
        if any(qwen["work_pair"][key] != gemma["work_pair"][key] for key in common):
            raise SystemExit(f"{name}: the two tokenizers did not use the same fixed fields and work")
        for model, row in paired.items():
            if name == "preseeded_overflow":
                if ("guarded_publication" in row or not row["joint_witness"]["errors"]
                        or row["F"] - row["B"] <= memory_limit or row["F"] + reply <= 80_000
                        or row["F"] - row["F0"] < work_floor or row["F_kind"] != "whole_knowledge_index"
                        or row["forms_F"]["knowledge_index_form"] != "whole"
                        or row["work_pair"]["startup_probe_matches_W"] is not True):
                    raise SystemExit(f"{name}/{model}: unchanged positive-looking preseed did not fail full F")
                clock_rows.add(row["work_pair"]["clock_row_sha256"])
                continue
            if name == "mature_recommended":
                check_guarded_publication(row, digest, limits)
            elif "guarded_publication" in row:
                raise SystemExit(f"{name}/{model}: fresh fixture cannot certify an authored history")
            pair = row["work_pair"]
            joint = row["joint_witness"]
            if not all(type(n) is int and n > 0 for n in (
                row["F"], row["F0"], row["B"], row["B0_frame"], row["WB0_frame"],
                *(pair[k] for k in ("W", "W0", "WB", "prepared_reply_cap_tokens", "caller_reply_cap_tokens", "allocated_reply_cap_tokens")),
            )):
                raise SystemExit(f"{name}/{model}: an exact count or reply cap is missing")
            memory = pair["W"] - pair["WB"]
            full_memory = row["F"] - row["B"]
            minimum_memory = memory_limit * 3 // 4 if name.startswith("mature_") else 1
            frame_left = report["frame_limit"] - row["B0_frame"] - row["unspent_document_reserve_tokens"]
            remaining = 80_000 - pair["W"]
            if (row["errors"] != [] or row["negative_control"] is not False
                    or row["installed_anti_slop_catalogue_row_observed"] is not (name != "fresh_bare")
                    or row["anti_slop_body_sha256"] != (None if name == "fresh_bare" else pins["ANTI_SLOP_BODY_SHA256"])
                    or row["extension_catalogue_observed"] is not True
                    or not all(digest(row[k]) for k in ("capture_sha256", "full_index_sha256", "extension_reader_sha256",
                                                        "installed_anti_slop_catalogue_row_sha256", "tool_schema_sha256"))
                    or pair["W0_kind"] != "diagnostic_ablation_only" or pair["WB_kind"] != "diagnostic_ablation_only"
                    or row["F_kind"] != "whole_knowledge_index" or pair["startup_probe_matches_W"] is not True
                    or not minimum_memory <= memory == row["memory_W_minus_WB"] <= memory_limit
                    or not minimum_memory <= full_memory == row["whole_index_memory_minus_B"] <= memory_limit
                    or [pair[f"occurrences_in_{k}"] for k in ("W", "W0", "WB")] != [1, 0, 1]
                    or not all(digest(pair[k]) for k in ("W_sha256", "W0_sha256", "WB_sha256", "clock_row_sha256",
                                                         "fixed_fields_sha256", "work_sha256"))
                    or len({pair[k] for k in ("W_sha256", "W0_sha256", "WB_sha256")}) != 3
                    or set(pair["rendered_prompt_sha256"]) != {"W", "W0", "WB"}
                    or not all(digest(d) for d in pair["rendered_prompt_sha256"].values())
                    or len(set(pair["rendered_prompt_sha256"].values())) != 3
                    or pair["work_sha256"] != report["work_sha256"]
                    or not re.fullmatch(r"messages\[[1-9]\d*\]\.content", pair["changed_field"])
                    or pair["memory_changed_field"] != "messages[0].content"
                    or not work_floor <= pair["W"] - pair["W0"] == row["work_W_minus_W0"]
                    or not 0 <= row["reply_floor_remaining_after_W"] == remaining - reply
                    or pair["short_reply"] is not False
                     or not reply <= pair["allocated_reply_cap_tokens"] == min(pair["caller_reply_cap_tokens"], remaining)
                    or not 0 <= frame_left == row["uncommitted_frame_tokens"]):
                raise SystemExit(f"{name}/{model}: the actual start and ablations did not qualify")
            work = row["F"] - row["F0"]
            free = 80_000 - row["F"] - reply
            rendered = joint["rendered_prompt_sha256"]
            if (joint["errors"] != [] or work < work_floor or free < 0
                    or work != joint["work_F_minus_F0"] or free != joint["reply_floor_remaining_after_F"]
                    or joint["work_occurrences_F_B_F0"] != [1, 1, 0]
                    or joint["clock_row_sha256"] != pair["clock_row_sha256"]
                    or joint["F0_kind"] != "diagnostic_ablation_only"
                    or not all(digest(joint[k]) for k in ("F_sha256", "B_sha256", "F0_sha256", "clock_row_sha256"))
                    or len({joint[k] for k in ("F_sha256", "F0_sha256", "B_sha256")}) != 3
                    or set(rendered) != {"F", "F0", "B"} or not all(digest(d) for d in rendered.values())
                    or len(set(rendered.values())) != 3
                    or row["W_is_shorter_than_F_in_tokens"] is not (pair["W"] < row["F"])):
                raise SystemExit(f"{name}/{model}: diagnostic whole-index witness did not qualify")
            same_bytes = pair["W_sha256"] == joint["F_sha256"]
            if (row["forms_W"]["knowledge_index_form"] != "whole" or row["forms_F"]["knowledge_index_form"] != "whole"
                    or row["forms_W"]["world_form"] not in {"whole", "reader"}
                    or row["forms_W"]["extension_catalogue_form"] not in {"whole", "reader"}
                    or row["W_matches_F_bytes"] is not same_bytes
                    or (same_bytes and (pair["W0_sha256"] != joint["F0_sha256"]
                                        or pair["W"] != row["F"] or pair["W0"] != row["F0"]))):
                raise SystemExit(f"{name}/{model}: selected W and measured F are mislabeled")
            clock_rows.add(pair["clock_row_sha256"])
    if len(clock_rows) != 1:
        raise SystemExit("Reference profiles did not use one clock row")


def check_reference_controls(instruction_path, pins_path):
    """Keep the actual first unavailable stage separate from a numeric rejection."""
    instruction, pins = (json.loads(Path(path).read_bytes()) for path in (instruction_path, pins_path))
    if (instruction["state"] != "exception" or instruction["stage"] != "instruction_view"
            or instruction["exception"] != "ValueError"):
        raise SystemExit("The missing instruction did not fail at the body/view check")
    if (pins["state"] != "exception" or pins["stage"] != "read_pin"
            or pins["exception"] != "FileNotFoundError" or pins["completed_profiles"] != {}):
        raise SystemExit("Missing public pins did not fail before measuring a profile")


def main(argv):
    lane, runner_dir, base, base_tree, product, product_tree, overlay, run_id, attempt = argv
    runner = Path(runner_dir).resolve(strict=True)
    owned = Path(os.environ["OUROBOROS_TEST_TEMP_ROOT"]).resolve(strict=True)
    if any(not Path(os.environ[key]).resolve().is_relative_to(owned) for key in ("HOME", "OUROBOROS_DATA_DIR", "TMPDIR")):
        raise SystemExit("No isolated test environment")
    identity = {"sha": overlay, "run_id": run_id, "run_attempt": attempt}
    if (lane != "sender" or os.environ.get("GITHUB_ACTIONS") != "true"
            or os.environ.get("GITHUB_EVENT_NAME") != "push" or os.environ.get("GITHUB_REF") != QA_REF
            or any(os.environ.get("GITHUB_" + key.upper()) != value for key, value in identity.items())
            or any(not re.fullmatch(r"[0-9a-f]{40}", x) for x in (base, base_tree, product, product_tree, overlay))
            or not re.fullmatch(r"[0-9]+", run_id) or attempt != "1" or not sys.platform.startswith("linux")):
        raise SystemExit("Unbound Ubuntu sender subject")
    git_env = dict(os.environ, GIT_OPTIONAL_LOCKS="0", GIT_NO_LAZY_FETCH="1",
                   GIT_ALLOW_PROTOCOL="file", GIT_NO_REPLACE_OBJECTS="1")

    def git(*args):
        return subprocess.check_output(["git", *args], text=True, env=git_env).strip()

    def product_is_clean():
        return (git("rev-parse", "HEAD") == product and git("rev-parse", "HEAD^{tree}") == product_tree
                and not git("status", "--porcelain", "--untracked-files=all"))

    if (not product_is_clean() or git("show", "-s", "--format=%P", overlay) != product
            or git("show", "-s", "--format=%P", product) != base or git("rev-parse", f"{base}^{{tree}}") != base_tree):
        raise SystemExit("The actual checkout is not the clean, detached product")
    evidence, upload = owned / "nano-sender", runner / "nano-proof"
    evidence.mkdir()
    upload.mkdir(exist_ok=True)
    code = subprocess.call([sys.executable, "-m", "pytest", "-o", "addopts=", "-p", "no:cacheprovider", "-q",
                            "--ci-evidence-dir", str(evidence), *SENDER])
    try:
        record = json.loads((evidence / "results.json").read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            record = {}
    except (OSError, ValueError):
        record = {}
    proof = node_proof(record, SENDER, code, identity)
    unchanged = product_is_clean()
    report = {"schema": 3, "kind": "synthetic_only", "lane": lane, "public_base": base, "base_tree": base_tree,
              "product": product, "product_tree": product_tree, "workflow_source": identity,
              "full_battery": "not_run", "native_formatter_and_generation": "not_run", "physical_qwen_0_8b": "not_run",
              "phase_success_requires": "exact selected_nodeids and one setup/call/teardown passed per ordinary node; no xfails, skips or missing reports",
              "checkout_unchanged": unchanged, "phases": {"sender": proof}, "accepted": unchanged and proof["accepted"]}
    (upload / "proof.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "stage-reference":
        stage_reference(Path(sys.argv[2]))
    elif len(sys.argv) == 3 and sys.argv[1] == "check-reference":
        check_reference(sys.argv[2])
    elif len(sys.argv) == 4 and sys.argv[1] == "check-reference-controls":
        check_reference_controls(sys.argv[2], sys.argv[3])
    else:
        sys.exit(main(sys.argv[1:]))
