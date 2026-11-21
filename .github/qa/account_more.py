"""One-off hosted Mac proof. An absent, skipped, or unrelated red case is never evidence."""
import json
import math
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
MODULE = "tests/test_account_resources_browser.py"
POINTER = "test_more_stays_under_pointer_through_poll_and_close"
KEYBOARD = "test_poll_keeps_reading_row_and_offscreen_keyboard_focus"
# Refresh this manifest from pytest --collect-only on the reviewed product if its journeys change.
CASES = {
    f"fixed-{engine}": [f"{MODULE}::{name}" for name in (
        f"{POINTER}[{engine}-grows-on-poll]", f"{POINTER}[{engine}-tall-at-open]", f"{KEYBOARD}[{engine}]",
        *(f"test_poll_grows_upper_panel_without_moving_lower_more[{engine}-{gesture}]"
          for gesture in ("click", "tab-space", "held-space")),
        f"test_poll_wraps_metadata_above_pressed_more_on_phone[{engine}]",
        *(f"test_last_more_closes_at_wheeled_position_until_reader_moves[{engine}-{kind}]"
          for kind in ("long-list", "fits-when-closed")),
        *(f"test_more_press_survives_quota_answer_between_down_and_up[{engine}-{kind}]"
          for kind in ("quota", "reorder", "remove")),
        f"test_quota_repaint_keeps_rotation_switch_and_next_tab[{engine}]",
        *(f"test_more_press_keeps_point_when_previous_focus_yields[{engine}-{control}]"
          for control in ("switch", "refresh", "more")),
        f"test_more_press_survives_sign_in_reorder_to_first[{engine}]",
        f"test_passive_shrink_keeps_last_reading_row_past_natural_end[{engine}]",
        f"test_passive_growth_keeps_visible_budget_inside_resources[{engine}]",
        f"test_pointed_balance_survives_another_balance_arriving_above[{engine}]",
        f"test_quota_glance_change_does_not_move_open_more[{engine}]",
        *(f"test_resource_before_next_more[{engine}-{choice}]"
          for choice in ("reading-edge", "pointed-more", "wheel-during-space", "restored-focus")),
        *(f"test_pointed_resource_wins_over_natural_end[{engine}-{part}]" for part in ("label", "renewal")),
        f"test_offscreen_more_activation_keeps_visible_reader_and_focus[{engine}]",
        *(f"test_sign_in_again_shows_its_card_unless_the_reader_wheeled_away[{engine}-{reader}]"
          for reader in ("waits", "wheels-away", "tabs-back")),
        f"test_refused_cancel_shows_the_open_sign_in_again[{engine}]",
        *(f"test_repeated_sign_in_again_shows_the_one_sign_in[{engine}-{held}]"
          for held in ("cancel", "create")),
        *(f"test_pointing_at_another_account_outranks_a_pending_sign_in[{engine}-{part}]"
          for part in ("more", "rotation-switch")),
        f"test_sign_in_card_keeps_focus_and_press_across_job_polls[{engine}]",
        f"test_first_unchanged_poll_after_typing_cannot_land_inside_a_press_on_submit[{engine}]",
        f"test_moving_to_another_more_after_the_card_is_shown_ends_the_reveal[{engine}]",
        *(f"test_shown_sign_in_card_outlasts_a_pointer_that_only_drifts[{engine}-{drift}]"
          for drift in ("tremor", "row-gap")),
        f"test_pending_refusal_and_status_promotion_keep_the_pointed_rotation_switch[{engine}]",
        *(f"test_rotation_switch_press_survives_an_answer_between_down_and_up[{engine}-{kind}]"
          for kind in ("wrap", "promotion")),
        *(f"test_changed_poll_during_a_press_on_submit_still_releases_one_real_click[{engine}-{where}]"
          for where in ("form", "advanced-just-opened")),
        f"test_job_settling_during_a_press_on_submit_never_sends_the_obsolete_code[{engine}]",
        f"test_sign_in_card_keeps_open_focused_advanced_across_changed_answer[{engine}]",
        f"test_pointing_at_the_more_a_repaint_kept_outranks_a_pending_sign_in[{engine}]",
        f"test_double_click_on_sign_in_again_starts_one_sign_in[{engine}]",
        f"test_add_account_form_grows_into_view_below_the_fold[{engine}]",
        *(f"test_shorter_state_line_keeps_the_held_submit_under_the_pointer_on_a_phone[{engine}-{phase}]"
          for phase in ("checking", "settled")),
        f"test_rebuilt_sign_in_card_gives_focus_back_inside_the_advanced_the_owner_opened[{engine}]",
    )] + [f"tests/test_harness_maintenance_browser.py::{name}" for name in (
        "test_login_job_poll_preserves_offscreen_code_selection",
        *(f"test_missing_executable_is_one_explained_section_fact_until_a_real_fresh_read[{layout}]"
          for layout in ("1440-light", "1440-dark", "390-light")),
    )] + [f"tests/test_account_row_gestures_browser.py::{name}" for name in (
        *(f"test_{control}_gesture_keeps_its_command[{meaning}-{modality}]"
          for control in ("status_button", "rotation_switch") for meaning in ("same-command", "new-command")
          for modality in ("pointer", "space")),
        *(f"test_family_add_does_not_retarget_on_release[{transition}]"
          for transition in ("named-to-default", "default-to-named", "named-to-named")),
        "test_multi_family_login_mount_follows_the_active_job",
    )] for engine in ("webkit", "chromium")
}
CASES.update({
    "control": [f"{MODULE}::{POINTER}[webkit-grows-on-poll]"],
    "native": ["tests/qa_account_more_wk.py::test_system_wk_more"],
})


def verify(mode, xml, returncode, report):
    cases = ET.parse(xml).findall(".//testcase")
    expected = sorted(node.rsplit("::", 1)[1] for node in CASES[mode])
    report["cases"] = {name: [next((tag for tag in ("failure", "error", "skipped") if c.find(tag) is not None), "passed")
                              for c in cases if c.get("name") == name] for name in expected}
    assert sorted(case.attrib["name"] for case in cases) == expected, "missing, unexpected or duplicate cases"
    if mode == "native":
        prop = cases[0].find("./properties/property[@name='system_wk_dom_click_samples']")
        samples = json.loads(prop.attrib["value"]) if prop is not None else {}
        measurements = report["measurements"] = {}
        stages = ("before", "opened", "one_second", "after_poll", "closed")
        for stage in stages:
            if stage not in samples:
                continue  # A failed case reports only what it actually observed.
            row = samples[stage]
            values = {k: row[k] for k in ("x", "y", "rowY", "top", "height", "viewport", "width", "windowHeight", "dpr")}
            assert all(type(v) in (int, float) and math.isfinite(v) for v in values.values()), "invalid native geometry"
            assert type(row["hit"]) is bool and row["expanded"] in ("true", "false"), "invalid native control state"
            measurements[stage] = dict(values, hit=row["hit"], expanded=row["expanded"] == "true")
        if returncode == 0:
            assert all(s in measurements and measurements[s]["hit"] and
                       measurements[s]["expanded"] == (s not in ("before", "closed")) for s in stages), "native proof incomplete"
    assert all(c.find("skipped") is None and c.find("error") is None for c in cases), "skip or infrastructure error"
    if mode != "control":
        assert returncode == 0 and all(c.find("failure") is None for c in cases), "candidate failed"
        return
    failure = cases[0].find("failure")
    assert returncode == 1 and failure is not None, "old product must fail the place assertion"
    trace = failure.text or ""
    # This same test file ran on the base. Reject a hit-test failure while navigating or positioning for the click.
    frame = re.search(rf"test_account_resources_browser\.py:(\d+): in {POINTER}\n[ \t]+assert_at\(more, point\)[ \t]*\n", trace)
    assert frame, "control failed outside the required stationary hit-test call"
    lines = (HERE / MODULE).read_text(encoding="utf-8").splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith(f"def {POINTER}(")]
    assert len(starts) == 1, "test source not found"
    end = next((i for i in range(starts[0] + 1, len(lines)) if lines[i] and not lines[i][0].isspace()), len(lines))
    body = range(starts[0] + 1, end)
    opened = [i for i in body if lines[i].strip() == 'expect(more).to_have_attribute("aria-expanded", "true")']
    wheeled = [i for i in body if lines[i].strip() == "moved = wheel_to(page, selector)"]
    assert len(opened) == 1 and len(wheeled) == 1, "open or wheel marker not unique"
    at = int(frame[1]) - 1
    # The reported line must be that very call, after More opened and strictly before the owner's deliberate wheel.
    assert at in body and lines[at].strip() == "assert_at(more, point)", "reported line is not the stationary call"
    assert opened[0] < at < wheeled[0], "not in the open phase"
    assert re.search(r"test_account_resources_browser\.py:\d+: in assert_at\n\s+assert (?:button\.evaluate\([^\n]+"
                     r"elementFromPoint|center\(button\) == pytest\.approx)", trace), "not the place assertion"
    assert re.search(r"^E\s+(AssertionError:|assert )", trace, re.MULTILINE), "not an assertion failure"
    report["matched_control"] = "post-open stationary pointer or button position"


def main(mode):
    assert os.environ.get("RUNNER_ENVIRONMENT") == "github-hosted" and sys.platform == "darwin"
    temp = Path(os.environ["RUNNER_TEMP"])
    out = temp / "account-more"
    out.mkdir(exist_ok=True)
    raw = temp / "account-more-raw"
    raw.mkdir(exist_ok=True)
    xml = raw / f"{mode}.xml"
    xml.unlink(missing_ok=True)
    env = dict(os.environ, OUROBOROS_RUN_UI_SMOKE="1", PLAYWRIGHT_BROWSERS_PATH=str(temp / "browsers"),
               PLAYWRIGHT_NODEJS_PATH=str(temp / "node/bin/node"))
    env["OUROBOROS_UI_BROWSER_ENGINE"] = mode.removeprefix("fixed-") if mode.startswith("fixed-") else "webkit"
    root = HERE / "control" if mode == "control" else HERE
    selected = CASES[mode]
    if mode.startswith("fixed-"):  # Execute all parameters and require an exact set of results.
        selected = list(dict.fromkeys(node.split("[", 1)[0] for node in selected))
    cmd = [sys.executable, "-I", "-S", "scripts/safe_test.py", "--temp-parent", "/tmp",
           "--evidence-dir", str(raw), "--timeout", "600", "--", sys.executable,
           "-m", "pytest", "-o", "addopts=", "-o", "junit_family=xunit1", "-q", "--tb=short",
           f"--junitxml={xml}", *selected]
    with (raw / f"{mode}.log").open("wb") as log:
        rc = subprocess.run(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    report = {"mode": mode, "exit_code": rc, "verified": False}
    try:
        verify(mode, xml, rc, report)
        report["verified"] = True
    finally:
        # Never publish JUnit (including the expected failure), stdout, or arbitrary test properties.
        (out / f"{mode}.json").write_text(json.dumps(report, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
