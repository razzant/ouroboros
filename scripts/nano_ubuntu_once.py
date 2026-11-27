#!/usr/bin/env python3
"""One-shot Ubuntu evidence for a detached Nano product; never exports raw reports."""
from __future__ import annotations

import ast
import hashlib
import io
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from urllib.request import urlopen


def nodes(file, *names):
    return tuple(f"tests/{file}.py::{name}" for name in names)


FINAL_FOCUS = nodes(
    "test_context_start_bound", "test_a_frame_just_under_the_window_reaches_the_ordinary_fit_unchanged",
) + nodes(
    "test_nano_startup_exact", *(f"test_start_count_waits_on_a_busy_slot_and_hands_its_control_to_the_loop[{case}]"
    for case in ("release", "pause", "stop", "deadline", "wrap_up")),
) + nodes(
    "test_nano_web_main_entry", *(f"test_a_control_found_during_prestart_contention_still_parks_the_routed_send[{case}]"
    for case in ("last_probe", "owner_pause_recheck")),
    *(f"test_a_contended_start_count_slot_keeps_the_routed_send_unstarted[{case}]"
      for case in ("pause", "release")),
) + nodes(
    "test_conversation_lane_saved_turn_restore", *(f"test_the_first_main_count_park_returns_unsent_after_boot[{case}]"
    for case in ("still_paused", "quit_before_launch")),
    *(f"test_a_pre_body_return_proves_nothing_once_a_launch_or_body_may_have_followed[{evidence}-{timing}]"
      for timing in ("before_body", "during_count") for evidence in ("launch_mark", "body_start")),
) + nodes(
    "test_nano_reference_capacity", "test_formatter_probe_distinguishes_reordered_schema_keys",
    "test_reference_uses_the_exact_start_and_keeps_a_separate_catalogue[False]",
    "test_reference_uses_the_exact_start_and_keeps_a_separate_catalogue[True]",
)


SIZE = nodes(
    "test_smoke", "test_no_oversized_modules", "test_size_ratchet_manifest_matches_live_tree",
    "test_size_ratchet_transition_against_explicit_base", "test_no_extremely_oversized_functions",
) + nodes(
    "test_repo_health_smoke", "test_checked_in_manifest_generator_check_is_exact",
) + nodes(
    "test_reference_book_budgets", "test_a_change_does_not_grow_a_reference_book",
    "test_a_change_does_not_grow_bible_or_the_prompts",
)
VIEWPORTS = ((1087, 740), (320, 640))
THEMES = ("dark", "light")
SETTINGS = nodes("test_settings_tools_models_browser", *(
    f"test_nano_context_projects_passive_facts_in_models[{theme}-{w}-{h}]"
    for theme in THEMES for w, h in VIEWPORTS
))
SETTINGS_FRAMES = {f"nano-context-{state}-{w}-{theme}.png": (w, h)
                   for state in ("summary", "measurement", "unconfirmed") for w, h in VIEWPORTS for theme in THEMES}
# Verify these declarations against the real-card fixture on the final product before releasing the overlay.
RESUME = nodes("test_cold_return_window_park_browser", "test_saved_work_card_uses_the_current_overview_then_resumes")
RESUME_SCENE = "cold-return-window-park"
RESUME_NAMES = (
    "01-held-does-not-fit.png", "02-dialog-three-steps.png", "03-choice-sent.png",
    "04-current-overview-held-after-reload.png", "05-dialog-can-continue.png", "06-continued.png",
)
# Must match the permanent fixture included in the exact tested product.
RESUME_VIEWPORT = (1440, 900)
RESUME_FRAMES = {name: RESUME_VIEWPORT for name in RESUME_NAMES}

PHASES = {"setup", "call", "teardown"}

TASK_CONTROL_CASES = (
    "a window hold offers its own steps, and each is one request naming the hold it was shown for",
    "a window hold the task's own read does not show sends nothing and says so",
    "the card's window hold is the accepted snapshot's: an older body answering late neither hides it nor sends a plain Resume",
)
MODEL_ROLES_CASES = ("Nano Settings distinguishes new memory checks from a dated count",)


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


def frame_proof(directory, expected):
    from PIL import Image

    if not expected or directory.is_symlink() or not directory.is_dir():
        return {}, False
    frames = {}
    actual = {p.name for p in directory.iterdir()}
    for name, size in sorted(expected.items()):
        if name not in actual or not re.fullmatch(r"[a-zA-Z0-9_.-]+\.png", name):
            continue
        path = directory / name
        if path.is_symlink() or not path.is_file():
            continue
        try:
            body = path.read_bytes()
            with Image.open(io.BytesIO(body), formats=["PNG"]) as image:
                if image.size != size:
                    continue
                image.verify()
            with Image.open(io.BytesIO(body), formats=["PNG"]) as image:
                image.load()
        except (OSError, SyntaxError, ValueError):
            continue
        frames[name] = (body, {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest(), "size": size})
    return frames, set(frames) == actual == set(expected)


def exact_selectors(wanted, families):
    """Expand no IDs locally: run each named function and reconcile its exact full matrix."""
    if not wanted or len(set(wanted)) != len(wanted):
        raise ValueError("Empty or duplicate exact case inventory")
    expected = set(wanted)
    extra = set()
    for selector, cases in families.items():
        if ("[" in selector or not cases or len(set(cases)) != len(cases)
                or any(not case.startswith(selector + "[") or not case.endswith("]") for case in cases)
                or selector in expected or set(cases) != {node for node in expected if node.startswith(selector + "[")}
                or extra.intersection(cases)):
            raise ValueError(f"The required parameter matrix is unbound: {selector}")
        extra.update(cases)
    return [node for node in wanted if node not in extra] + list(families)


def test_phase(name, wanted, scratch, upload, identity, base, runner, frames=None):
    if not wanted or frames is not None and not frames:
        return {"accepted": False, "state": "unbound_resume_card_not_run"}
    env = dict(os.environ)
    args = []
    shots = scratch / (name + "-frames")
    if name == "size":
        # Run the explicit local pair; a QA push is not an official-line push. No fake ref/event.
        env.pop("GITHUB_EVENT_NAME", None)
        env.update(OURO_SIZE_RATCHET_BASE_REF=base, OURO_BOOK_GROWTH_MODE="block")
        args = ["-m", "size_ratchet"]
    elif name in ("s47", "s48", "s49", "s50"):
        env["OUROBOROS_E2E_DEEP"] = "mock"
        args = ["-m", "integration"]
    elif name == "web-composer":
        env.pop("OUROBOROS_UI_BROWSER_EXECUTABLE", None)
        env.update(OUROBOROS_E2E_DEEP="mock", PLAYWRIGHT_BROWSERS_PATH=str(runner / "nano-chromium"))
        args = ["-m", "browser"]
    elif frames is not None:
        shots.mkdir()
        env.pop("OUROBOROS_UI_BROWSER_EXECUTABLE", None)
        env["PLAYWRIGHT_BROWSERS_PATH"] = str(runner / "nano-chromium")
        if name == "resume-card":
            env.update(OUROBOROS_E2E_DEEP="mock", OUROBOROS_BROWSER_EVIDENCE_OUT=str(shots))
            shots = shots / RESUME_SCENE
            args = ["-m", "browser"]
        else:
            env.update(OUROBOROS_RUN_UI_SMOKE="1", OUROBOROS_UI_BROWSER_ENGINE="chromium",
                       OUROBOROS_UI_EVIDENCE_DIR=str(shots))
            args = ["-m", "ui_browser"]
    families = {}
    argv_selectors = exact_selectors(wanted, families)
    evidence = scratch / name
    evidence.mkdir()
    code = subprocess.call([sys.executable, "-m", "pytest", "-o", "addopts=", "-p", "no:cacheprovider", "-q",
                            "--ci-evidence-dir", str(evidence), *args, *argv_selectors], env=env)
    try:
        record = json.loads((evidence / "results.json").read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            record = {}
    except (OSError, ValueError):
        record = {}
    proof = node_proof(record, wanted, code, identity)
    if families:
        proof["required_function_matrices"] = families
    if frames is not None:
        checked, complete = frame_proof(shots, frames)
        proof["accepted"] &= complete
        proof["expected_frames"] = sorted(frames)
        proof["frame_inventory_complete"] = complete
        proof["missing_or_invalid_frames"] = sorted(set(frames) - checked.keys())
        proof["frames"] = {file: meta for file, (_, meta) in checked.items()}
        if checked:
            target = upload / name
            target.mkdir()
            for file, (blob, _) in checked.items():
                (target / file).write_bytes(blob)
    return proof


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
    if set(profiles) != {"fresh_bare", "fresh_recommended", "mature_recommended"}:
        raise SystemExit("Reference report profile selection changed")
    clock_rows = set()
    for name, paired in profiles.items():
        if set(paired) != {"qwen35", "gemma4"}:
            raise SystemExit(f"{name}: both tokenizers are required")
        qwen, gemma = paired["qwen35"], paired["gemma4"]
        for key in ("capture_sha256", "tool_schema_sha256", "source_sha256", "full_index_sha256",
                    "installed_anti_slop_catalogue_row_sha256", "extension_reader_sha256"):
            if not qwen[key] or qwen[key] != gemma[key]:
                raise SystemExit(f"{name}: the two tokenizers did not use the same captured source ({key})")
        if any(qwen["work_pair"][key] != gemma["work_pair"][key] for key in ("fixed_fields_sha256", "work_sha256")):
            raise SystemExit(f"{name}: the two tokenizers did not use the same fixed fields and work")
        for model, row in paired.items():
            pair = row["work_pair"]
            joint = row["joint_witness"]
            if not all(type(n) is int and n > 0 for n in (
                row["F"], row["F0"], row["B"], row["B0_frame"], row["WB0_frame"],
                *(pair[k] for k in ("W", "W0", "WB", "prepared_reply_cap_tokens", "allocated_reply_cap_tokens")),
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
                    or not reply <= pair["allocated_reply_cap_tokens"] <= min(pair["prepared_reply_cap_tokens"], remaining)
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


def node_tap_proof(module, required):
    """Keep only expected public test names and TAP summary, not raw output."""
    try:
        run = subprocess.run([
            "node", "--test", "--test-reporter=tap", f"web/tests/{module}.test.js",
        ], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"accepted": False, "state": type(error).__name__}
    counts = {}
    for key in ("tests", "pass", "fail", "cancelled", "skipped", "todo"):
        found = re.findall(rf"^# {key} ([0-9]+)$", run.stdout, re.MULTILINE)
        counts[key] = int(found[0]) if len(found) == 1 else None
    results = {name: re.findall(rf"^ok [0-9]+ - {re.escape(name)}$", run.stdout, re.MULTILINE)
               for name in required}
    passed = [name for name, lines in results.items() if len(lines) == 1]
    accepted = (run.returncode == 0 and len(passed) == len(required)
                and counts["tests"] == counts["pass"] and counts["pass"] not in (None, 0)
                and all(counts[key] == 0 for key in ("fail", "cancelled", "skipped", "todo")))
    return {"accepted": accepted, "node_exit": run.returncode, "module_counts": counts,
            "passed_required_cases": passed}


def main(argv):
    lane, runner_dir, base, base_tree, product, product_tree, overlay, run_id, attempt = argv
    runner = Path(runner_dir).resolve(strict=True)
    owned = Path(os.environ["OUROBOROS_TEST_TEMP_ROOT"]).resolve(strict=True)
    if any(not Path(os.environ[key]).resolve().is_relative_to(owned) for key in ("HOME", "OUROBOROS_DATA_DIR", "TMPDIR")):
        raise SystemExit("No isolated test environment")
    if (any(not re.fullmatch(r"[0-9a-f]{40}", x) for x in (base, base_tree, product, product_tree, overlay))
            or not re.fullmatch(r"[0-9]+", run_id) or attempt != "1" or not sys.platform.startswith("linux")):
        raise SystemExit("Unbound Ubuntu subject")
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
    if lane == "focus":
        phases = (("targeted-contracts", FINAL_FOCUS, None), ("size", SIZE, None))
    elif lane == "browser":
        composer = nodes("test_cold_return_window_park_browser",
                         "test_web_main_composer_keeps_one_saved_origin_and_first_prepared_request")
        phases = (("web-composer", composer, None), ("settings", SETTINGS, SETTINGS_FRAMES),
                  ("resume-card", RESUME, RESUME_FRAMES))
    else:
        raise SystemExit("Unrecognized qualification lane")
    scratch, upload = owned / "nano-r5", runner / "nano-proof"
    scratch.mkdir()
    upload.mkdir(exist_ok=True)
    identity = {"sha": overlay, "run_id": run_id, "run_attempt": attempt}
    report = {"schema": 3, "kind": "synthetic_only", "lane": lane, "public_base": base, "base_tree": base_tree,
              "product": product, "product_tree": product_tree, "workflow_source": identity, "phases": {},
              "full_battery": "not_run", "native_formatter_and_generation": "not_run", "physical_qwen_0_8b": "not_run",
              "pinned_hf_reference": "see separate reference lane (tokenizer/template only)",
              "context_gate": "not part of this overlay; see standard Context capacity gate workflow",
              "phase_success_requires": "exact selected_nodeids and one setup/call/teardown passed per ordinary node; no xfails, skips or missing reports",
              "optional_native_tests": "not_selected; no skip counts as pass"}
    for name, wanted, frames in phases:
        report["phases"][name] = test_phase(name, wanted, scratch, upload, identity, base, runner, frames)
        report["accepted"] = False # No positive receipt until every lane has settled and checkout still matches.
        (upload / "proof.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if lane == "browser":
        report["phases"]["task-control-node"] = node_tap_proof("task_control_menu", TASK_CONTROL_CASES)
        report["phases"]["model-roles-node"] = node_tap_proof("model_roles", MODEL_ROLES_CASES)
    report["checkout_unchanged"] = product_is_clean()
    report["accepted"] = report["checkout_unchanged"] and all(row["accepted"] for row in report["phases"].values())
    (upload / "proof.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "stage-reference":
        stage_reference(Path(sys.argv[2]))
    elif len(sys.argv) == 3 and sys.argv[1] == "check-reference":
        check_reference(sys.argv[2])
    else:
        sys.exit(main(sys.argv[1:]))
