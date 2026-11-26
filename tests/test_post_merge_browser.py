"""The post-merge browser check (pilot, owner choice 1 of 2026-11-25): what it selects, how it judges, whom it blames.

The selection (scripts/post_merge_browser_select.py) is deterministic and browserless; the outcome is green only when
every selected node ran and passed, "checked partially" when one never ran, never green when nothing applies; a red
is attributed only by comparison with the window's base; the session ends once the Linux and Windows legs have
ended; and the check stays its own system: its job can change neither the run's conclusion nor a leg, the window's
base or another system's issues.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import yaml

from ouroboros import update_qualification
from scripts import post_merge_browser as pmb
from scripts import post_merge_browser_select as sel
from scripts import post_merge_routing as routing
from scripts import post_merge_window
from tests._post_merge_github import FakeGitHub, tracking_issue

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = yaml.safe_load((ROOT / ".github/workflows/post-merge.yml").read_text(encoding="utf-8"))
BROWSER = WORKFLOW["jobs"]["browser"]
UI = "import pytest\npytestmark = [pytest.mark.ui_browser]\n"
LIVING, ROOM = (node for node, _ in sel.SENTINELS)
LIVING_TEST, ROOM_TEST = (node.split("[", 1)[0] for node in (LIVING, ROOM))  # what runs: both declared engines


def defining(test: str) -> str:
    return f"\n\ndef {test.split('::')[-1]}(engine):\n    assert engine\n"


MODULES = {
    "tests/test_room_work_browser.py": UI + "# drives room_work.js\n" + defining(ROOM_TEST),
    "tests/test_living_cards_browser.py": UI + "# reads web/modules/chat_cards.js\n" + defining(LIVING_TEST),
    "tests/test_composer_capsules_browser.py": UI + "from tests import separate_ui_helper\n",
    **{f"tests/test_hub_{n}_browser.py": UI + "# style.css\n" for n in range(11)},
}
TABLE = {"ui_nodes": {
    "tests/test_room_work_browser.py::test_a[chromium-282]": 60_000,
    ROOM: 2_000,
    LIVING: 31_000,
    "tests/test_living_cards_browser.py::test_b[chromium]": 400_000,
    "tests/test_composer_capsules_browser.py::test_c": 30_000,
}}


def commits(*files_per_commit):
    return [{"sha": f"{index:040x}", "files": list(files)} for index, files in enumerate(files_per_commit, 1)]


# --- selection --------------------------------------------------------------------------------------------------

def test_only_web_or_browser_test_changes_apply_and_nothing_else_is_green():
    modules = sel.ui_modules({**MODULES, "tests/test_plain.py": "def test_x(): pass\n",
                              "tests/test_docker_browser.py": "pytestmark = [pytest.mark.ui_browser_docker]\n"})
    assert set(modules) == set(MODULES)  # neither a plain module nor the docker lane's
    assert sel.select(commits(["docs/DEVELOPMENT.md", "ouroboros/loop.py"]), modules)["applicable"] is False
    plan = sel.select(commits(["docs/x.md"]), modules)
    assert plan["args"] == [] and plan["selected"] == [] and "no commit of the window" in plan["reason"]
    assert sel.select(commits(["web/tests/chat.test.js"]), modules)["applicable"] is True
    assert sel.select(commits(["tests/test_room_work_browser.py"]), modules)["applicable"] is True
    assert sel.select(commits(["tests/separate_ui_helper.py"]), modules)["applicable"] is True  # a helper they name
    assert sel.select(commits(["tests/test_plain.py"]), modules)["applicable"] is False  # an ordinary module
    gone = sel.select(commits(["tests/test_gone_browser.py"]), modules, base_modules={"tests/test_gone_browser.py"})
    assert gone["applicable"] is True


def test_tiers_order_changed_tests_then_name_stems_then_narrow_mentions_and_hubs_select_nothing():
    plan = sel.select(commits(["web/style.css", "web/modules/chat_cards.js"], ["web/room_work.css"],
                              ["tests/test_composer_capsules_browser.py"]), MODULES, table=TABLE)
    rows = {row["module"]: row for row in plan["selected"]}
    assert [row["module"] for row in plan["selected"]] == [
        "tests/test_composer_capsules_browser.py", "tests/test_room_work_browser.py", "tests/test_living_cards_browser.py"]
    assert rows["tests/test_composer_capsules_browser.py"]["tier"] == 1
    assert rows["tests/test_room_work_browser.py"]["tier"] == 2  # web/room_work.css by the module's name
    assert rows["tests/test_living_cards_browser.py"]["tier"] == 3  # it names chat_cards.js
    assert plan["wide"] == [{"file": "web/style.css", "modules": 11, "by": "mention"}]  # a hub: the release lane's
    assert not any("hub" in module for module in rows)


def test_a_hub_stem_selects_only_the_module_named_exactly_after_it():
    modules = {**{f"tests/test_chat_{n}_browser.py": UI for n in range(11)}, "tests/test_chat_browser.py": UI}
    plan = sel.select(commits(["web/modules/chat.js"]), modules)
    assert [row["module"] for row in plan["selected"]] == ["tests/test_chat_browser.py"]
    assert plan["wide"] == [{"file": "web/modules/chat.js", "modules": 12, "by": "name"}]


def test_the_commits_of_a_burst_take_turns_within_a_tier():
    modules = {f"tests/test_{name}_browser.py": UI for name in ("a1", "a2", "a3", "b")}
    table = {"ui_nodes": {f"tests/test_{name}_browser.py::test_x": seconds * 1000
                          for name, seconds in (("a1", 10), ("a2", 20), ("a3", 30), ("b", 50))}}
    plan = sel.select(commits(["tests/test_a1_browser.py", "tests/test_a2_browser.py", "tests/test_a3_browser.py"],
                              ["tests/test_b_browser.py"]), modules, table=table, budget_seconds=70)
    # Cheapest-first alone would run a1, a2 and a3 and drop the other merge's only test.
    assert [row["module"] for row in plan["selected"]] == ["tests/test_a1_browser.py", "tests/test_b_browser.py"]
    assert [row["module"] for row in plan["dropped"]] == ["tests/test_a2_browser.py", "tests/test_a3_browser.py"]


def test_a_changed_module_runs_only_the_tests_that_hold_its_changes():
    text = ("import pytest\npytestmark = [pytest.mark.ui_browser]\n\n\n@pytest.fixture\ndef page():\n    return 1\n\n\n"
            "@pytest.mark.parametrize('engine', ['chromium', 'webkit'])\ndef test_x(page, engine):\n    assert page\n\n\n"
            "def test_y(page):\n    assert page\n\n\nclass TestK:\n    def test_m(self):\n        assert True\n")
    module = "tests/test_k_browser.py"
    assert sel.changed_tests(module, text, {12}) == [f"{module}::test_x"]
    assert sel.changed_tests(module, text, {10}) == [f"{module}::test_x"]  # its decorator
    assert sel.changed_tests(module, text, {12, 21}) == [f"{module}::TestK::test_m", f"{module}::test_x"]
    assert sel.changed_tests(module, text, {7}) is None  # the fixture: every test may read it
    helper = text + "\n\ndef testing_helper():\n    return 2\n\n\nclass Helper:\n    def test_n(self):\n        pass\n"
    assert sel.changed_tests(module, helper, {25}) is None and sel.changed_tests(module, helper, {30}) is None
    assert [name for _, _, name in sel.collectable_spans(module, helper)] == [
        f"{module}::test_x", f"{module}::test_y", f"{module}::TestK::test_m"]  # what pytest collects, nothing else
    table = {"ui_nodes": {f"{module}::test_x[chromium]": 5000, f"{module}::test_x[webkit]": 7000,
                          f"{module}::test_y": 90_000}}
    plan = sel.select(commits([module]), {module: text}, table=table, lines={module: {12}})
    [row] = plan["selected"]
    assert row["tests"] == [f"{module}::test_x"] and row["estimate_ms"] == 12_000
    assert row["nodes"] == {"chromium": 1, "webkit": 1, "other": 0}
    assert plan["args"] == [f"{module}::test_x"]


def test_changed_lines_reads_the_head_side_of_the_window(tmp_path):
    def git(*args):
        return subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True, text=True).stdout.strip()

    git("init", "-q")
    git("config", "user.email", "t@example.invalid")
    git("config", "user.name", "t")
    (tmp_path / "tests").mkdir()
    module = tmp_path / "tests/test_k_browser.py"
    module.write_text("a\nb\nc\nd\n")
    git("add", ".")
    git("commit", "-qm", "base")
    base = git("rev-parse", "HEAD")
    module.write_text("a\nB\nc\nd\ne\n")
    git("commit", "-qam", "head")
    assert sel.changed_lines(base, "HEAD", {"tests/test_k_browser.py"}, cwd=tmp_path) == {
        "tests/test_k_browser.py": {2, 5}}
    assert sel.changed_lines(base, "HEAD", set(), cwd=tmp_path) == {}


def test_the_two_webkit_sentinels_always_run_first_and_a_selection_that_covers_one_replaces_it():
    plan = sel.select(commits(["web/tests/x.test.js"]), MODULES, table=TABLE)
    assert plan["selected"] == [] and plan["args"] == [LIVING_TEST, ROOM_TEST]  # as tests: never a stale param id
    assert [row["guards"] for row in plan["sentinels"]] == [["web/modules/chat_cards.js"], ["web/room_work.css"]]
    assert plan["estimate_ms"] == 33_000
    whole = sel.select(commits(["web/room_work.css"]), MODULES, table=TABLE)
    assert whole["args"] == [LIVING_TEST, "tests/test_room_work_browser.py"]  # the module covers its sentinel
    living = "tests/test_living_cards_browser.py"
    unread = sel.select(commits([living]), MODULES, table=TABLE, lines={living: set()})
    assert unread["args"] == [ROOM_TEST, living]  # no changed line read: the module runs whole and covers it
    narrowed = sel.select(commits([living]), MODULES, table=TABLE, lines={living: {7}})  # the test's body
    assert narrowed["selected"][0]["tests"] == [LIVING_TEST] and narrowed["args"] == [ROOM_TEST, LIVING_TEST]


def test_a_sentinel_the_head_no_longer_defines_is_named_and_never_passed():
    renamed = {**MODULES, "tests/test_room_work_browser.py": UI + defining("tests/x.py::test_renamed")}
    plan = sel.select(commits(["web/tests/x.test.js"]), renamed, table=TABLE)
    assert plan["args"] == [LIVING_TEST]  # pytest would refuse the whole session over a test it cannot find
    assert [(row["test"], row["present"]) for row in plan["sentinels"]] == [(LIVING_TEST, True), (ROOM_TEST, False)]
    text = "\n".join(pmb.summary_lines(plan, {**pmb.judge(Path("/nonexistent"), {}), "outcome": "green",
                                               "planned": 1, "passed": 1, "not_run": []}, None))
    assert f"`{ROOM_TEST}`" in text and "no longer defines" in text


def test_the_budget_and_the_module_cap_drop_the_lowest_priority_and_name_it():
    plan = sel.select(commits(["web/modules/chat_cards.js", "web/room_work.css"]), MODULES, table=TABLE,
                      budget_seconds=120)
    assert [row["module"] for row in plan["selected"]] == ["tests/test_room_work_browser.py"]
    assert [(row["module"], row["why"]) for row in plan["dropped"]] == [
        ("tests/test_living_cards_browser.py", "over the time budget")]
    capped = sel.select(commits(["web/modules/chat_cards.js", "web/room_work.css"]), MODULES, table=TABLE,
                        max_modules=1)
    assert capped["dropped"][0]["why"] == "over the cap of 1 modules"
    unknown = sel.select(commits(["tests/test_composer_capsules_browser.py"]), MODULES, table={"ui_nodes": {}})
    assert unknown["unknown_modules"] == 1 and unknown["selected"][0]["timed"] is False
    assert sel.session_seconds(sel.BUDGET_SECONDS) == round(sel.BUDGET_SECONDS * sel.SESSION_FACTOR)


def test_a_plan_with_nothing_to_run_is_not_applicable():
    modules = {"tests/test_other_browser.py": UI}  # neither sentinel module exists any more
    plan = sel.select(commits(["web/tests/x.test.js"]), modules)
    assert plan["args"] == [] and plan["applicable"] is False
    assert plan["reason"].endswith("no browser test matches it and no sentinel is present")


def test_the_module_that_needs_caddy_is_excluded_and_named():
    modules = {**MODULES, "tests/test_separate_ui_browser.py": UI}
    plan = sel.select(commits(["tests/test_separate_ui_browser.py"]), modules)
    assert plan["selected"] == [] and plan["excluded"][0]["module"] == "tests/test_separate_ui_browser.py"


def test_the_scope_command_runs_on_a_bare_interpreter():
    done = subprocess.run([sys.executable, "-I", "-S", str(ROOT / "scripts/post_merge_browser_select.py"), "scope",
                           "--base", "HEAD", "--head", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0 and done.stdout.startswith("applicable=false\nwhy="), done.stdout + done.stderr


# --- outcome ----------------------------------------------------------------------------------------------------

PLAN = {"applicable": True, "base": "b" * 40, "head": "h" * 40, "args": ["tests/test_a_browser.py"],
        "selected": [{"module": "tests/test_a_browser.py", "files": ["web/a.js"], "reasons": ["x"]}],
        "sentinels": [], "commits": [{"sha": "1" * 40, "files": ["web/a.js"]}, {"sha": "2" * 40, "files": ["x.py"]}]}


def judged(tmp_path, monkeypatch, *, selected, outcomes, verdict, retried=None):
    head = tmp_path / "head"
    head.mkdir()
    (head / "verdict.json").write_text(json.dumps(verdict))
    if retried is not None:
        (head / "retry-1").mkdir()
    sessions = {head: {"exported": selected is not None, "selected": selected, "outcomes": outcomes},
                head / "retry-1": {"exported": True, "selected": None, "outcomes": retried or {}}}
    monkeypatch.setattr(pmb, "session", lambda directory: sessions[Path(directory)])
    return pmb.judge(tmp_path, PLAN)


def test_green_only_when_every_selected_node_ran_and_passed(tmp_path, monkeypatch):
    passed = {"outcome": "passed"}
    result = judged(tmp_path, monkeypatch, selected=["a", "b"], outcomes={"a": passed, "b": passed},
                    verdict={"verdict": "green", "failed": [], "flaky": []})
    assert result["outcome"] == "green" and result["passed"] == 2 and result["not_run"] == []
    assert result["legs_ended"] is False


def test_a_retry_pass_is_flaky_and_a_second_failure_is_red(tmp_path, monkeypatch):
    failed, passed = {"outcome": "failed"}, {"outcome": "passed"}
    result = judged(tmp_path, monkeypatch, selected=["a", "b", "c"], outcomes={"a": failed, "b": failed, "c": failed},
                    retried={"a": passed, "b": failed, "c": failed},
                    verdict={"verdict": "red", "flaky": [{"nodeid": "a"}], "accepted": [{"nodeid": "c"}],
                             "failed": [{"nodeid": "b", "attempts": [{}, {}]}, {"nodeid": "c", "attempts": [{}, {}]}]})
    assert result["outcome"] == "red" and result["flaky"] == ["a"] and result["failed"] == ["b"]
    assert result["accepted"] == ["c"] and result["partial"] is False  # an owner-accepted red is listed, not red


def test_a_stopped_session_is_checked_partially_and_names_what_never_ran(tmp_path, monkeypatch):
    (tmp_path / pmb.STOP).touch()  # the legs ended first
    result = judged(tmp_path, monkeypatch, selected=["a", "b", "c"],
                    outcomes={"a": {"outcome": "passed"}, "b": {"outcome": "failed"}},
                    verdict={"verdict": "red", "flaky": [], "reason": "1 collected test(s) never ran",
                             "failed": [{"nodeid": "b", "attempts": [{}]}]})
    assert result["outcome"] == "partial" and result["not_run"] == ["c"] and result["legs_ended"] is True
    assert result["unretried"] == ["b"] and result["failed"] == []  # no retry: reported, never handed over
    text = "\n".join(pmb.summary_lines(PLAN, result, None, {"run": "success"}))
    assert "CHECKED PARTIALLY" in text and "legs ended first" in text and "- `c`" in text


def test_an_unexpected_skip_or_a_lost_session_is_never_green(tmp_path, monkeypatch):
    skip = judged(tmp_path, monkeypatch, selected=["tests/test_x.py::t"], outcomes={"tests/test_x.py::t": {
        "outcome": "skipped"}}, verdict={"verdict": "green", "failed": [], "flaky": []})
    assert skip["outcome"] == "partial" and skip["skipped"] == ["tests/test_x.py::t"]
    monkeypatch.undo()
    lost = pmb.judge(tmp_path / "nothing", PLAN)
    assert lost["outcome"] == "partial" and "unknown" in lost["not_run"][0]
    assert pmb.judge(tmp_path, {})["outcome"] == "partial"  # the job stopped before its plan
    assert pmb.judge(tmp_path, {"applicable": False, "reason": "r"})["outcome"] == "not-applicable"


def test_the_summary_fails_the_job_unless_green_or_flaky_and_names_unfinished_steps(tmp_path, monkeypatch, capsys):
    (tmp_path / "plan.json").write_text(json.dumps(PLAN))
    for outcome, code in (("green", 0), ("flaky", 0), ("red", 1), ("partial", 1), ("not-applicable", 0)):
        monkeypatch.setattr(pmb, "judge", lambda evidence, plan, o=outcome: {
            "outcome": o, "partial": o == "partial", "failed": [], "unretried": [], "flaky": [], "accepted": [],
            "not_run": ["n"] if o == "partial" else [], "skipped": [], "passed": 1, "planned": 1})
        args = pmb.argparse.Namespace(evidence_dir=str(tmp_path), plan=str(tmp_path / "plan.json"))
        output = tmp_path / f"output-{outcome}"
        assert pmb.command_summary(args, environ={"GITHUB_OUTPUT": str(output),
                                                  "BROWSER_INSTALL_OUTCOME": "failure"}) == code
        assert output.read_text() == f"outcome={outcome}\n"
    assert "Steps that did not finish: install (failure)." in capsys.readouterr().out


# --- the legs bound the session ----------------------------------------------------------------------------------

def test_the_legs_end_the_session_only_when_both_have_ended():
    def job(leg, status, conclusion=None):
        return {"name": f"post-merge ({leg}) from {'b' * 40}", "status": status, "conclusion": conclusion}

    browser = {"name": "post-merge browser check (pilot)", "status": "in_progress", "conclusion": None}
    window = {"name": "window", "status": "completed", "conclusion": "success"}
    assert pmb.legs_ended([window, browser, job("linux", "completed", "failure"), job("windows", "completed", "success")])
    assert not pmb.legs_ended([window, browser, job("linux", "completed", "success"), job("windows", "in_progress")])
    # A dispatch off `ouroboros` runs no leg: nothing bounds the trial.
    assert not pmb.legs_ended([window, browser, job("linux", "completed", "skipped"),
                               job("windows", "completed", "skipped")])
    assert not pmb.legs_ended([browser])


def test_the_plugin_stops_between_tests_and_a_later_session_runs_nothing(tmp_path):
    stop, deselected = tmp_path / "stop", []
    config = SimpleNamespace(getoption=lambda name, default=None: str(stop),
                             hook=SimpleNamespace(pytest_deselected=lambda items: deselected.extend(items)))
    items = ["a", "b"]
    pmb.pytest_collection_modifyitems(None, config, items)
    item = SimpleNamespace(config=config, session=SimpleNamespace(shouldstop=False))
    pmb.pytest_runtest_teardown(item, "b")
    assert items == ["a", "b"] and item.session.shouldstop is False and deselected == []
    stop.touch()
    pmb.pytest_runtest_teardown(item, None)  # the last test: nothing left to stop
    assert item.session.shouldstop is False
    pmb.pytest_runtest_teardown(item, "b")
    assert "legs have ended" in item.session.shouldstop
    pmb.pytest_collection_modifyitems(None, config, items)
    assert items == [] and deselected == ["a", "b"]


def test_a_real_session_stops_after_the_test_that_saw_the_legs_end(tmp_path):
    stop = tmp_path / "stop"
    (tmp_path / "test_three.py").write_text(
        f"from pathlib import Path\n\ndef test_a():\n    Path({str(stop)!r}).touch()\n\n"
        "def test_b():\n    pass\n\ndef test_c():\n    pass\n")
    done = subprocess.run([sys.executable, "-m", "pytest", str(tmp_path / "test_three.py"), "-p",
                           "scripts.post_merge_browser", f"--browser-stop-file={stop}", "-o", "addopts=",
                           "-p", "no:cacheprovider", "-q", f"--rootdir={tmp_path}"],
                          cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert done.returncode == 2, done.stdout + done.stderr  # interrupted, as pytest-timeout's session timeout ends
    assert "1 passed" in done.stdout and "legs have ended" in done.stdout


# --- attribution ------------------------------------------------------------------------------------------------

def test_attribution_compares_with_the_base_and_names_a_merge_only_when_one_touched_the_files(monkeypatch):
    monkeypatch.setattr(pmb, "git", lambda *args, cwd=None: "0" * 40)
    node = "tests/test_a_browser.py::test_x"
    [pre] = pmb.attribute(PLAN, [node], {node: "failed"})
    assert pre == {"id": node, "verdict": "pre-existing"}
    [one] = pmb.attribute(PLAN, [node], {node: "passed"})
    assert one["verdict"] == "culprit" and one["culprit"] == "1" * 40 and one["last_good"] == "0" * 40
    [absent] = pmb.attribute(PLAN, [node], {node: "absent"})
    assert absent["verdict"] == "culprit"
    two = {**PLAN, "commits": [{"sha": "1" * 40, "files": ["web/a.js"]}, {"sha": "2" * 40, "files": ["web/a.js"]}]}
    [many] = pmb.attribute(two, [node], {node: "passed"})
    assert many["verdict"] == "unattributed" and many["candidates"] == ["1" * 40, "2" * 40]
    none = {**PLAN, "commits": [{"sha": "1" * 40, "files": ["x.py"]}, {"sha": "2" * 40, "files": ["y.py"]}]}
    [anyone] = pmb.attribute(none, [node], {node: "passed"})
    assert anyone["verdict"] == "unattributed" and anyone["candidates"] == ["1" * 40, "2" * 40]
    for at_base in ("flaky", "unknown"):
        [unsure] = pmb.attribute(PLAN, [node], {node: at_base})
        assert unsure["verdict"] == "unattributed" and "candidates" not in unsure


def test_a_sentinel_is_blamed_through_the_files_it_guards(monkeypatch):
    monkeypatch.setattr(pmb, "git", lambda *args, cwd=None: "0" * 40)
    plan = {**PLAN, "selected": [], "sentinels": [{"node": ROOM, "guards": ["web/room_work.css"]}],
            "commits": [{"sha": "1" * 40, "files": ["web/a.js"]}, {"sha": "2" * 40, "files": ["web/room_work.css"]}]}
    [verdict] = pmb.attribute(plan, [ROOM], {ROOM: "passed"})
    assert verdict["verdict"] == "culprit" and verdict["culprit"] == "2" * 40


def test_browser_reds_are_their_own_system_and_never_touch_a_legs_issues():
    linux = tracking_issue(7, {"os": "linux", "kind": "unattributed", "ids": ["tests/test_a_browser.py::test_x"]})
    github = FakeGitHub(issues=[linux], label=True)
    record = routing.route(github, classification={"failed": [{"id": "tests/test_a_browser.py::test_x"}]},
                           attribution={"verdicts": [{"id": "tests/test_a_browser.py::test_x", "verdict": "unattributed",
                                                      "candidates": ["1" * 40, "2" * 40]}]},
                           head="f" * 40, run_url="run", recheck=lambda ids: {}, system="browser", owed=())
    [created] = record["created"]
    issue = github.issues[created]
    assert issue["title"] == "Post-merge red on the browser check (Ubuntu): failing tests without a found culprit"
    assert "candidate merges, none confirmed: `" + "1" * 40 in issue["body"]
    assert routing.parse_machine_block(issue["body"])["os"] == "browser"
    assert github.issues[7]["state"] == "open" and 7 not in record["updated"]
    # A Linux leg that sees the same test pass leaves the browser issue alone.
    later = routing.route(github, classification={"failed": []}, attribution={"verdicts": []}, head="f" * 40,
                          run_url="run", recheck=lambda ids: dict.fromkeys(ids, "passed"), system="linux")
    assert created not in later["closed"] and github.issues[created]["state"] == "open"


def test_the_browser_system_is_no_leg():
    """scripts/post_merge_window.py waits for a receipt of each leg in SYSTEMS before the base advances: the browser
    check owns issues under its own name and is never one of them."""
    assert routing.SYSTEMS == {"linux": "Linux", "windows": "Windows"}
    assert routing.NAMES == {**routing.SYSTEMS, "browser": "the browser check (Ubuntu)"}
    assert post_merge_window.SYSTEMS is routing.SYSTEMS


def test_a_legs_issue_never_gains_candidates():
    github = FakeGitHub(label=True)
    routing.route(github, classification={"failed": [{"id": "tests/t.py::test_a"}]},
                  attribution={"verdicts": [{"id": "tests/t.py::test_a", "verdict": "unattributed"}]},
                  head="f" * 40, run_url="run", recheck=lambda ids: {}, system="linux")
    [issue] = github.issues.values()
    assert "candidates" not in routing.parse_machine_block(issue["body"])
    assert issue["title"] == "Post-merge red on Linux: failing tests without a found culprit"


def test_raw_launcher_evidence_stays_out_of_the_uploaded_directory(tmp_path):
    evidence = tmp_path / "post-merge-browser"
    command = pmb.session_command(tmp_path, ["tests/test_a_browser.py::t"], evidence / "base", seconds=60, retry=True)
    safe = next(index for index, arg in enumerate(command) if arg.endswith("scripts/safe_test.py"))
    assert command[safe + 1] == "--evidence-dir" and command[4:6] == ["--evidence-dir", str(evidence / "base")]
    assert Path(command[safe + 2]) == tmp_path / "post-merge-browser-raw" / "base"  # as the run step's DIR-raw/head
    assert command[:3] == ["python", "-I", "-S"] and command[3].endswith("scripts/ci_retry.py")
    assert command[command.index("-m", command.index("pytest")) + 1] == "ui_browser"
    assert "--session-timeout=60" in command and "-p" not in command  # no stop: the base comparison is bounded


def test_dry_routing_reads_github_and_writes_nothing(monkeypatch):
    calls = []
    monkeypatch.setattr(routing.GitHub, "_call", lambda self, method, url, body=None: calls.append(method) or ([], ""))
    dry = pmb.DryGitHub("razzant/ouroboros", "")
    dry.pages("issues?state=open")
    dry.post("issues", {"title": "t", "body": "b", "labels": []})
    dry.patch("issues/1", {"state": "closed"})
    assert calls == ["GET"] and [w["method"] for w in dry.writes] == ["POST", "PATCH"]


# --- the job and the readers of the run --------------------------------------------------------------------------

def test_the_browser_job_is_one_non_blocking_ubuntu_runner_beside_the_legs():
    assert BROWSER["needs"] == "window" and BROWSER["runs-on"] == "ubuntu-latest" and "strategy" not in BROWSER
    assert BROWSER["continue-on-error"] is True  # its red is never the run's conclusion the update letter reads
    # A job ceiling cancels the steps after it and could end the run cancelled: it stays above the step ceilings (and
    # ten minutes for the unbounded checkout and setup), so only a step ever times out. The legs end it first.
    ceilings = sum(step.get("timeout-minutes", 0) for step in BROWSER["steps"])
    assert BROWSER["timeout-minutes"] >= ceilings + 10, (BROWSER["timeout-minutes"], ceilings)
    assert all("timeout-minutes" in step for step in BROWSER["steps"]
               if step.get("id") in ("weights", "plan", "install_browsers", "run", "compare", "route", "summary"))
    assert "needs.window.outputs.browser != 'false'" in BROWSER["if"]
    assert "github.ref != 'refs/heads/ouroboros' && github.event_name == 'workflow_dispatch'" in BROWSER["if"]
    assert update_qualification._LEG.fullmatch(BROWSER["name"]) is None
    assert not BROWSER["name"].startswith("post-merge (")  # `legs_ended` reads the legs by that prefix
    window = {step.get("id"): step for step in WORKFLOW["jobs"]["window"]["steps"]}
    assert window["browser_scope"]["continue-on-error"] is True
    assert window["browser_scope"]["run"].startswith("python3 -I -S scripts/post_merge_browser_select.py scope")
    for step in WORKFLOW["jobs"]["window"]["steps"]:  # what the window readers take from its step names
        name = step.get("name") or ""
        if "browser" in name:
            assert not name.startswith(post_merge_window.REPLAY_STEP)
            assert post_merge_window._BASE_STEP.fullmatch(name) is None
    steps = {step.get("id") or step.get("name"): step for step in BROWSER["steps"]}
    triggers = WORKFLOW.get("on", WORKFLOW.get(True))
    assert triggers["workflow_dispatch"]["inputs"]["browser_budget"]["required"] is False
    # Only a trial off `ouroboros` may shrink the budget (the partial case of the test-branch plan).
    assert steps["plan"]["env"]["TRIAL_BUDGET"] == (
        "${{ github.ref != 'refs/heads/ouroboros' && inputs.browser_budget || '' }}")
    assert '${TRIAL_BUDGET:+--budget-seconds "$TRIAL_BUDGET"}' in steps["plan"]["run"]
    run = steps["run"]["run"]
    assert "--session-timeout=\"$SESSION\"" in run and "scripts/ci_retry.py" in run
    assert "-p scripts.post_merge_browser" in run and '--browser-stop-file="$evidence/stop"' in run
    assert "post_merge_browser.py legs" in run and 'touch "$evidence/stop"' in run
    assert "--require-ui-browser" not in run  # a subset proves no lane
    assert "chromium webkit" in steps["install_browsers"]["run"]
    assert set(steps["summary"]["env"]) == {f"BROWSER_{step.upper()}_OUTCOME" for step in pmb.STEPS}
    legs = str(WORKFLOW["jobs"]["post-merge"])
    assert "post_merge_browser" not in legs and "ci_browser_install" not in legs


def test_a_failed_browser_job_changes_no_measured_window():
    run = {"id": 1, "run_attempt": 1, "head_sha": "a" * 40, "display_title": update_qualification.RANGE_TITLE,
           "head_branch": "ouroboros", "event": "push", "status": "completed", "conclusion": "success"}

    def job(name, conclusion, steps=()):
        return {"name": name, "run_id": 1, "run_attempt": 1, "head_sha": "a" * 40, "status": "completed",
                "conclusion": conclusion, "id": hash(name) & 0xFFFF,
                "steps": [{"name": s, "status": "completed", "conclusion": "success"} for s in steps]}

    legs = [job(f"post-merge ({leg}) from {'b' * 40}", "success", [update_qualification.MEASURED_STEP])
            for leg in ("linux", "windows")]

    def reader(jobs):
        return lambda path: {"jobs": jobs, "total_count": len(jobs)}

    browser = job("post-merge browser check (pilot)", "failure", ["Post-merge browser outcome red"])
    assert (update_qualification.measured_window(run, reader(legs))
            == update_qualification.measured_window(run, reader([browser, *legs])) == ("b" * 40, False))
