#!/usr/bin/env python3
"""One-shot Ubuntu reference and composition evidence on an exact detached Nano product."""
from __future__ import annotations

import ast
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from urllib.request import urlopen


def nodes(file, *names):
    return tuple(f"tests/{file}.py::{name}" for name in names)


HOMES = nodes(
    "test_context_gate", "test_workflow_filters_cover_the_lane_and_request_builders", "test_context_gate_has_recovery_homes",
) + nodes(
    "test_context_start_bound", "test_a_frame_just_under_the_window_reaches_the_ordinary_fit_unchanged",
    "test_the_connections_term_is_the_form_with_fewer_escaped_characters",
    "test_the_bound_never_exceeds_the_fitted_minimum[fresh-reversed-connections-nano]",
) + nodes(
    "test_persistence_inventory", "test_scan_is_populated_and_pinned", "test_every_scanned_path_has_an_inventory_row",
    "test_every_inventory_row_is_still_real",
)
WEB_MAIN = nodes(
    "test_nano_web_main_entry", "test_routed_web_send_starts_exact_with_its_whole_input_and_its_row_said_once",
    "test_only_the_attested_row_yields_an_identical_earlier_message_stays_whole",
    "test_a_web_reply_keeps_its_row_whole_with_what_it_replies_to",
    *(f"test_a_control_found_during_prestart_contention_still_parks_the_routed_send[{case}]"
      for case in ("last_probe", "owner_pause_recheck")),
    *(f"test_a_contended_start_count_slot_keeps_the_routed_send_unstarted[{case}]"
      for case in ("pause", "release")),
)
STARTUP = nodes(
    "test_nano_startup_exact", "test_web_main_builder_to_prepared_payload_when_estimator_overcounts",
    "test_small_ceiling_input_cap_and_counter_unavailability_window_separation",
    *(f"test_start_count_waits_on_a_busy_slot_and_hands_its_control_to_the_loop[{case}]"
      for case in ("release", "pause", "stop", "deadline", "wrap_up")),
)
SAVED_WEB = nodes(
    "test_conversation_lane_saved_turn_restore",
    *(f"test_the_first_main_count_park_returns_unsent_after_boot[{case}]"
      for case in ("still_paused", "quit_before_launch")),
    *(f"test_a_pre_body_return_proves_nothing_once_a_launch_or_body_may_have_followed[{evidence}-{timing}]"
      for timing in ("before_body", "during_count") for evidence in ("launch_mark", "body_start")),
)
COLD_RETURN = nodes(
    "test_cold_return", "test_the_owners_own_pause_ends_a_chosen_end_and_its_resume_is_a_plain_return",
    "test_the_choice_is_in_the_handover_checkpoint_before_the_grant_is_spent",
    "test_the_chosen_end_of_an_attempt_is_the_size_terminal_with_nothing_sent",
)
COLD_SCENARIOS = dict(zip(("s47", "s48", "s49", "s50"), (
    "test_s47_a_returned_work_that_no_longer_fits_parks_then_resumes_once_it_fits",
    "test_s48_the_owner_ends_a_window_parks_attempt_and_the_ordinary_continue_starts_one_successor",
    "test_s49_a_window_park_offers_the_current_overview_and_only_the_owners_choice_takes_it",
    "test_s50_the_chosen_overview_outlives_an_ordinary_stop_after_its_grant_was_spent",
)))


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

SENDER = tuple(
    f"tests/test_nano_local_exact.py::test_exact_local_send_reprices_from_the_callers_ceiling[{case}]"
    for case in ("restore-room", "shrink-to-fit")
)
PHASES = {"setup", "call", "teardown"}
QA_REF = "refs/heads/qa/nano-composed-80k-check"


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


def check_guarded_publication(row, digest, limits):
    """Require independent writer and fresh-start evidence; a count of seeded files is insufficient."""
    memory_limit, work_floor, reply = limits
    proof = row["guarded_publication"]
    sources = {"identity", "scratchpad", "overview", "knowledge_index", "mark"}
    active_sources = {"identity", "scratchpad", "scratchpad_blocks", "overview", "knowledge_index", "marks", "rules", "publication"}
    active, seeds, last, refused = (proof[key] for key in ("active_sha256", "seeds", "last_accept", "refusal"))
    fixed_rules = proof["rules_fixture"]
    if (set(fixed_rules) != {"source", "owner_save", "rules_sha256", "settings_sha256", "boundary_request_sha256"}
            or fixed_rules["source"] != "initial_synthetic_settings" or fixed_rules["owner_save"] != "not_run"
            or not digest(fixed_rules["rules_sha256"]) or fixed_rules["settings_sha256"] != active["rules"]
            or fixed_rules["boundary_request_sha256"] != proof["boundary"]["request_sha256"]):
        raise SystemExit("Fixed rules input is unbound or incorrectly claimed as an owner save")
    if (set(seeds) != sources or set(active) != active_sources or not all(digest(value) for value in active.values())
            or set(proof["fresh_start_marker_sha256"]) != sources | {"rules"}
            or not all(digest(value) for value in proof["fresh_start_marker_sha256"].values())
            or last["revision_sha256"] != active["publication"] or refused["active_sha256"] != active
            or refused["outcome"] != "over_capacity" or type(refused["candidate_memory"]) is not int
            or refused["candidate_memory"] <= memory_limit or type(refused["work_shortfall"]) is not int
            or refused["work_shortfall"] < 0 or type(proof["accepted_rows"]) is not int
            or type(proof["refused_rows"]) is not int or not 32 < proof["accepted_rows"] < proof["refused_rows"] <= 640
            or proof["accepted_rows"] + 1 != proof["refused_rows"]):
        raise SystemExit("Guarded mature publication does not prove five edits plus a fixed input and the unchanged refusal")
    revisions = [entry["revision_sha256"] for entry in (*seeds.values(), last)]
    if len(set(revisions)) != 6 or not all(digest(value) for value in revisions):
        raise SystemExit("The five seed writes and the final growth did not each advance the active revision")
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
    if (report.get("instruction_admission") != "not_tested"
            or not isinstance(report.get("instruction_fixture"), str) or not report["instruction_fixture"]):
        raise SystemExit("Public instruction fixture must not be claimed as skill admission")
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
                    "anti_slop_fixture_catalogue_row_sha256", "extension_reader_sha256")):
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
                    or row["anti_slop_fixture_catalogue_row_observed"] is not (name != "fresh_bare")
                    or row["anti_slop_body_sha256"] != (None if name == "fresh_bare" else pins["ANTI_SLOP_BODY_SHA256"])
                    or row["extension_catalogue_observed"] is not True
                    or not all(digest(row[k]) for k in ("capture_sha256", "full_index_sha256", "extension_reader_sha256",
                                                        "anti_slop_fixture_catalogue_row_sha256", "tool_schema_sha256"))
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


def run_command(argv, log, *, env, seconds):
    with log.open("wb") as output:
        try:
            return subprocess.run(argv, env=env, stdout=output, stderr=subprocess.STDOUT,
                                  timeout=seconds, check=False).returncode, "exited"
        except subprocess.TimeoutExpired:
            return None, "timeout"
        except OSError:
            return None, "command_unavailable"


def read_object(path):
    try:
        value = json.loads(path.read_bytes())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def pytest_diagnostics(record, junit, wanted):
    cases = set(wanted)
    paths = {node.split("::", 1)[0] for node in wanted}
    locations = {}
    try:
        if junit.stat().st_size <= 4_000_000:
            for row in ET.parse(junit).iter("testcase"):
                name = row.get("name", "")
                for failure in (x for x in row if x.tag in {"failure", "error"}):
                    # Read only a repository-relative source locator from the short traceback.
                    matches = re.findall(r"(?m)^(?P<path>(?:tests|ouroboros|supervisor|scripts)/[a-zA-Z0-9_./-]+\.py):(?P<line>[1-9][0-9]{0,5}):", failure.text or "")
                    if matches:
                        locations[name] = ":".join(matches[-1])
    except (OSError, ET.ParseError):
        pass
    errors = {}
    for row in record.get("reports", []):
        node = row.get("nodeid")
        if node not in cases or row.get("outcome") not in ("failed", "skipped"):
            continue
        kind = row.get("error_type", "")
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", kind):
            kind = "XFail" if row.get("wasxfail") else "Skipped" if row["outcome"] == "skipped" else "UnclassifiedFailure"
        phase = row.get("phase")
        phase = phase if phase in PHASES | {"crash"} else "unknown"
        finding = {"phase": phase, "cause": kind}
        name = node.split("::")[-1]
        if name in locations:
            finding["location"] = locations[name]
        errors.setdefault(node, finding)
    collection = [x["nodeid"] for x in record.get("collection_failures", [])
                  if isinstance(x, dict) and x.get("nodeid") in paths]
    return errors, collection


def browser_pytest(binary, argv):
    """The fixture still runs unchanged; supply only the ordinary launch option and binary."""
    import pytest
    from playwright.sync_api import BrowserType

    launch = BrowserType.launch

    def sandboxed_launch(self, *args, **kwargs):
        # A fixture changing this contract needs a fresh review, never a weaker launch.
        if (self.name != "chromium" or args or kwargs.get("chromium_sandbox") is False
                or kwargs.get("executable_path") not in (None, binary)
                or kwargs.get("ignore_default_args") or kwargs.get("args")):
            raise ValueError("Unsupported launch options for the selected synthetic browser lane")
        return launch(self, **(kwargs | {"chromium_sandbox": True, "executable_path": binary}))

    BrowserType.launch = sandboxed_launch
    try:
        return pytest.main(argv)
    finally:
        BrowserType.launch = launch


def browser_probe(binary, scratch):
    """One ordinary launch, with no secondary binary or policy changes on failure."""
    code, state = run_command([sys.executable, "-c", """
import sys
import json
from playwright.sync_api import Error, sync_playwright
try:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, executable_path=sys.argv[1],
                                             chromium_sandbox=True, timeout=30_000)
        version = browser.version
        browser.close()
    print(json.dumps({"version": version}))
except Error as error:
    # Project a known category, never the message, environment or browser stderr.
    text = str(error).lower()
    reason = "sandbox_unavailable" if any(key in text for key in (
        'no usable sandbox', 'failed to move to new namespace', 'sandbox is not supported',
        'operation not permitted', 'permission denied')) else "playwright_launch_failed"
    print(json.dumps({"reason": reason}))
    sys.exit(1)
""", binary], scratch / "browser-probe.private.log", env=os.environ, seconds=50)
    row = read_object(scratch / "browser-probe.private.log")
    version = row.get("version")
    accepted = code == 0 and isinstance(version, str) and re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,4}", version)
    reason = row.get("reason")
    if reason not in {"sandbox_unavailable", "playwright_launch_failed"}:
        reason = state if state != "exited" else "no_browser_version_observed"
    return {"accepted": bool(accepted), "status": "ready" if accepted else "not_run",
            "binary": Path(binary).name, "sandbox_requested": True,
            "version": version if accepted else None, "reason": None if accepted else reason}


def test_phase(name, wanted, scratch, identity, base, *, binary=None, upload=None, frames=None):
    evidence = scratch / name
    evidence.mkdir()
    env, args, shots = dict(os.environ), [], evidence / "frames"
    if name == "size":
        # This is a local parent/child comparison on Ubuntu, not a main/PR event.
        env.pop("GITHUB_EVENT_NAME", None)
        env.update(OURO_SIZE_RATCHET_BASE_REF=base, OURO_BOOK_GROWTH_MODE="block")
        args = ["-m", "size_ratchet"]
    elif name in COLD_SCENARIOS:
        env["OUROBOROS_E2E_DEEP"] = "mock"
        args = ["-m", "integration"]
    elif binary:
        env["OUROBOROS_UI_BROWSER_EXECUTABLE"] = binary
        if name == "settings":
            shots.mkdir()
            env.update(OUROBOROS_RUN_UI_SMOKE="1", OUROBOROS_UI_BROWSER_ENGINE="chromium",
                       OUROBOROS_UI_EVIDENCE_DIR=str(shots))
            args = ["-m", "ui_browser"]
        else:
            env["OUROBOROS_E2E_DEEP"] = "mock"
            args = ["-m", "browser"]
            if name == "resume-card":
                shots.mkdir()
                env["OUROBOROS_BROWSER_EVIDENCE_OUT"] = str(shots)
                shots = shots / RESUME_SCENE
    junit = evidence / "private-junit.xml"
    command = ([sys.executable, os.path.abspath(__file__), "browser-pytest", binary]
               if binary else [sys.executable, "-m", "pytest"])
    code, state = run_command([*command, "-o", "addopts=", "-p", "no:cacheprovider", "-q",
                               "--tb=short", "--junitxml", str(junit), "--ci-evidence-dir", str(evidence), *args, *wanted],
                              evidence / "private.log", env=env,
                              seconds=1850 if name in COLD_SCENARIOS else 480 if binary else 210)
    record = read_object(evidence / "results.json")
    proof = node_proof(record, wanted, code, identity)
    errors, collection = pytest_diagnostics(record, junit, wanted)
    proof.update(process_state=state, report_available=bool(record), first_failure_by_node=errors,
                 collection_failed=collection)
    if frames:
        checked, complete = frame_proof(shots, frames)
        proof["accepted"] &= complete
        proof["missing_or_invalid_frames"] = sorted(set(frames) - checked.keys())
        proof["frames"] = {file: meta for file, (_, meta) in checked.items()}
        if checked:
            target = upload / name
            target.mkdir()
            for file, (blob, _) in checked.items():
                (target / file).write_bytes(blob)
    return proof


def static_phase(script, args, scratch):
    code, state = run_command([sys.executable, script, *args], scratch / (Path(script).stem + ".private.log"),
                              env=os.environ, seconds=120)
    # Generator output may contain paths or values: retain the check name and typed stage only.
    return {"accepted": code == 0, "process_state": state, "exit": code, "check": script,
            "cause": None if code == 0 else "source_or_checked_in_inventory_differs" if code == 1 else "generator_check_did_not_complete"}


def summarize(directory, lane):
    path = Path(directory)
    print(f"## Nano 80k · {lane} · synthetic Ubuntu only")
    if stages := (path / "stages.txt").read_text(encoding="utf-8"):
        print("```text\n" + stages.rstrip() + "\n```")
    proof = read_object(path / "proof.json")
    if browser := proof.get("browser"):
        detail = browser.get("reason") or f"{browser['binary']} {browser['version']}; sandbox requested"
        print(f"- Chromium Settings / Resume / Send: **{browser['status']}** ({detail})")
    for name, phase in proof.get("phases", {}).items():
        count = phase.get("selected_count")
        suffix = "" if count is None else f" ({len(phase['passed_nodes'])}/{count} passed)"
        state = phase.get("process_state")
        if state and state != "exited":
            suffix += f" ({state})"
        print(f"- **{name}**: {'pass' if phase.get('accepted') else 'not verified'}{suffix}")
        if "pytest_exit" in phase and not phase.get("accepted"):
            print(f"  - pytest exit: {phase['pytest_exit']}; report available: {phase['report_available']}; "
                  f"exact selection: {phase['selection_matches']}")
        for node, reason in phase.get("first_failure_by_node", {}).items():
            location = f" · {reason['location']}" if "location" in reason else ""
            print(f"  - `{node}`: {reason['phase']} · {reason['cause']}{location}")
        if phase.get("collection_failed"):
            print("  - Collection: " + ", ".join(phase["collection_failed"]))
        for node in phase.get("incomplete_nodes", []):
            if node not in phase["first_failure_by_node"]:
                print(f"  - `{node}`: incomplete ({phase['process_state']})")
        if phase.get("missing_or_invalid_frames"):
            print("  - Missing/invalid frames: " + ", ".join(phase["missing_or_invalid_frames"]))
        if phase.get("cause"):
            print(f"  - {phase['cause']} (exit {phase.get('exit')})")
    for variant in ("normal", "growth", "missing_instruction", "missing_reference_pins"):
        diag = read_object(path / f"{variant}.diag.json")
        if diag:
            print(f"- `{variant}`: {diag['state']} at `{diag['stage']}`; "
                  f"profile={diag.get('profile')}, model={diag.get('model')}, "
                  f"exception={diag.get('exception')}, reason={diag.get('reason')}")
    if lane == "reference" and (path / "reference-exits.txt").is_file():
        print("```text\n" + (path / "reference-exits.txt").read_text(encoding="utf-8").rstrip() + "\n```")


def main(argv):
    lane, runner_dir, base, base_tree, product, product_tree, overlay, run_id, attempt = argv
    runner = Path(runner_dir).resolve(strict=True)
    owned = Path(os.environ["OUROBOROS_TEST_TEMP_ROOT"]).resolve(strict=True)
    if any(not Path(os.environ[key]).resolve().is_relative_to(owned) for key in ("HOME", "OUROBOROS_DATA_DIR", "TMPDIR")):
        raise SystemExit("No isolated test environment")
    identity = {"sha": overlay, "run_id": run_id, "run_attempt": attempt}
    if (lane not in {"focus", "browser", *COLD_SCENARIOS} or os.environ.get("GITHUB_ACTIONS") != "true"
            or os.environ.get("GITHUB_EVENT_NAME") != "push" or os.environ.get("GITHUB_REF") != QA_REF
            or any(os.environ.get("GITHUB_" + key.upper()) != value for key, value in identity.items())
            or any(not re.fullmatch(r"[0-9a-f]{40}", x) for x in (base, base_tree, product, product_tree, overlay))
            or not re.fullmatch(r"[0-9]+", run_id) or attempt != "1" or not sys.platform.startswith("linux")):
        raise SystemExit("Unbound Ubuntu composition subject")
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
    scratch, upload = owned / "nano-composed", runner / "nano-proof"
    scratch.mkdir()
    upload.mkdir(exist_ok=True)
    if lane == "focus":
        phases = {"conflict-homes": HOMES, "size": SIZE, "web-main": WEB_MAIN,
                  "nano-startup": STARTUP, "web-return-before-first-send": SAVED_WEB,
                  "cold-return-contract": COLD_RETURN, "sender-reply-cap": SENDER}
    elif lane == "browser":
        phases = {
            "settings": SETTINGS, "resume-card": RESUME,
            "web-composer": nodes("test_cold_return_window_park_browser",
                                  "test_web_main_composer_keeps_one_saved_origin_and_first_prepared_request"),
        }
    else:
        phases = {lane: (f"tests/system_e2e/test_cold_return_window_park.py::{COLD_SCENARIOS[lane]}",)}
    report = {"schema": 3, "kind": "synthetic_only", "lane": lane, "public_base": base, "base_tree": base_tree,
              "product": product, "product_tree": product_tree, "workflow_source": identity, "phases": {},
              "full_battery": "not_run", "native_formatter_and_generation": "not_run", "physical_qwen_0_8b": "not_run",
              "phase_success_requires": "exact nodeids with setup/call/teardown passed, no skips, xfails or missing reports",
              "accepted": False}

    def save():
        path = upload / "proof.json"
        pending = upload / ".proof.pending.json"
        pending.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        pending.replace(path)

    def observe(name, check):
        report["phases"][name] = {"accepted": False, "process_state": "started_without_terminal_observation"}
        save()
        report["phases"][name] = check()
        save()

    if lane == "focus":
        observe("check-domains", lambda: static_phase("scripts/check_domains.py", (), scratch))
        observe("check-inventories", lambda: static_phase("scripts/regenerate_inventories.py", ("--check",), scratch))
    binary = None
    visual = {"settings": SETTINGS_FRAMES, "resume-card": RESUME_FRAMES}
    if lane == "browser":
        binary = next((path for path in ("/usr/bin/google-chrome", "/usr/bin/chromium")
                       if Path(path).is_file() and os.access(path, os.X_OK)), None)
        report["browser"] = (browser_probe(binary, scratch) if binary else
                             {"accepted": False, "status": "not_run", "reason": "no preinstalled browser"})
        if not report["browser"]["accepted"]:
            for name, wanted in phases.items():
                report["phases"][name] = {"accepted": False, "process_state": "sandboxed_browser_unavailable",
                                          "incomplete_nodes": wanted, "first_failure_by_node": {}}
            save()
            return 1
        save()
    for name, wanted in phases.items():
        observe(name, lambda: test_phase(name, wanted, scratch, identity, base, binary=binary,
                                         upload=upload, frames=visual.get(name)))
    if binary:
        report["browser"]["status"] = "passed" if all(v["accepted"] for v in report["phases"].values()) else "not_verified"
    report["checkout_unchanged"] = product_is_clean()
    report["accepted"] = report["checkout_unchanged"] and all(row["accepted"] for row in report["phases"].values())
    save()
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "browser-pytest":
        # Match `python -m pytest` when the QA-only script lives outside the checkout.
        sys.path.insert(0, str(Path.cwd()))
        sys.exit(browser_pytest(sys.argv[2], sys.argv[3:]))
    elif len(sys.argv) == 4 and sys.argv[1] == "summarize":
        summarize(*sys.argv[2:])
    elif len(sys.argv) == 3 and sys.argv[1] == "stage-reference":
        stage_reference(Path(sys.argv[2]))
    elif len(sys.argv) == 3 and sys.argv[1] == "check-reference":
        check_reference(sys.argv[2])
    elif len(sys.argv) == 4 and sys.argv[1] == "check-reference-controls":
        check_reference_controls(sys.argv[2], sys.argv[3])
    else:
        sys.exit(main(sys.argv[1:]))
