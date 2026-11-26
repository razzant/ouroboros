"""The post-merge browser check (pilot, owner choice 1 of 2026-11-25): what it selects, how it judges, whom it names.

The selection (scripts/post_merge_browser_select.py) is deterministic and browserless; the outcome is green only when
every selected node ran and passed, "checked partially" when anything did not run or is not explained, never green
when nothing applies; a merge is named only when the window holds one first-parent commit and the base confirms it;
the check runs as its own workflow (.github/workflows/post-merge-browser.yml) that no post-merge run, leg, base or
update reader depends on, and its issues are its own system's.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

import yaml

from scripts import post_merge_browser as pmb
from scripts import post_merge_browser_select as sel
from scripts import post_merge_routing as routing
from scripts import post_merge_window
from tests._post_merge_github import FakeGitHub, tracking_issue

ROOT = Path(__file__).resolve().parents[1]
FLOW = yaml.safe_load((ROOT / ".github/workflows/post-merge-browser.yml").read_text(encoding="utf-8"))
POST_MERGE = yaml.safe_load((ROOT / ".github/workflows/post-merge.yml").read_text(encoding="utf-8"))
UI = "import pytest\npytestmark = [pytest.mark.ui_browser]\n"
LIVING, ROOM = (scenario for scenario, _ in sel.SENTINELS)
LIVING_TEST, ROOM_TEST = (scenario.split("[", 1)[0] for scenario in (LIVING, ROOM))  # what runs: every engine


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


def git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def repository(tmp_path):
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.email", "t@example.invalid")
    git(tmp_path, "config", "user.name", "t")
    (tmp_path / "tests").mkdir()
    return tmp_path


# --- selection --------------------------------------------------------------------------------------------------

def test_a_browser_module_is_one_whose_code_declares_the_marker():
    texts = {
        "tests/test_listed.py": UI,
        "tests/test_single.py": "import pytest\npytestmark = pytest.mark.ui_browser\n",
        "tests/test_function.py": "import pytest\n\n@pytest.mark.ui_browser\ndef test_x():\n    pass\n",
        "tests/test_class.py": "import pytest\n\n@pytest.mark.ui_browser\nclass TestK:\n    def test_x(self):\n        pass\n",
        "tests/test_method.py": "from pytest import mark\n\nclass TestK:\n    @mark.ui_browser\n    def test_x(self):\n        pass\n",
        "tests/test_param.py": ("import pytest\n\n@pytest.mark.parametrize('e', [pytest.param(1, marks=pytest.mark.ui_browser)])\n"
                                "def test_x(e):\n    pass\n"),
        "tests/test_broken.py": "import pytest\npytestmark = [pytest.mark.ui_browser\n",  # does not parse: its text
        # A mention declares nothing: this file's own fixtures, a comment, a non-test helper, the Docker lane's marker.
        "tests/test_string.py": 'UI = "import pytest\\npytestmark = [pytest.mark.ui_browser]\\n"\n',
        "tests/test_comment.py": "# pytest.mark.ui_browser runs in the UI lane\ndef test_x():\n    pass\n",
        "tests/test_helper.py": "import pytest\n\n@pytest.mark.ui_browser\ndef helper():\n    pass\n",
        "tests/test_docker.py": "import pytest\npytestmark = [pytest.mark.ui_browser_docker]\n",
        "tests/helpers.py": UI,  # not a test module
    }
    assert set(sel.ui_modules(texts)) == {"tests/test_listed.py", "tests/test_single.py", "tests/test_function.py",
                                          "tests/test_class.py", "tests/test_method.py", "tests/test_param.py",
                                          "tests/test_broken.py"}
    # This very module names the marker in its strings only: it is no browser module, and touching it applies nothing.
    own = (ROOT / "tests/test_post_merge_browser.py").read_text(encoding="utf-8")
    assert "pytest.mark.ui_browser" in own and not sel.declares_ui_browser(own)


def test_only_interface_changes_apply_and_nothing_else_is_green():
    modules = sel.ui_modules({**MODULES, "tests/test_plain.py": "def test_x(): pass\n"})
    assert set(modules) == set(MODULES)
    plan = sel.select(commits(["docs/x.md", "ouroboros/loop.py"]), modules)
    assert plan["applicable"] is False and plan["args"] == [] and "no commit of the window" in plan["reason"]
    assert sel.select(commits(["web/tests/chat.test.js"]), modules)["applicable"] is True
    assert sel.select(commits(["tests/test_room_work_browser.py"]), modules)["applicable"] is True
    assert sel.select(commits(["tests/separate_ui_helper.py"]), modules)["applicable"] is True  # a helper they name
    assert sel.select(commits(["tests/test_plain.py"]), modules)["applicable"] is False  # an ordinary module
    gone = sel.select(commits(["tests/test_gone_browser.py"]), modules, base_modules={"tests/test_gone_browser.py"})
    assert gone["applicable"] is True


def test_a_change_to_the_shared_browser_infrastructure_runs_the_sentinels_and_says_why():
    for path in ("tests/conftest.py", "tests/browser_lane.py", "tests/ci_evidence.py", "scripts/ci_retry.py", "uv.lock"):
        plan = sel.select(commits([path]), MODULES, table=TABLE)
        assert plan["applicable"] is True and plan["shared"] == [path], path
        assert plan["selected"] == [] and plan["args"] == [LIVING_TEST, ROOM_TEST], path
        assert "shared browser infrastructure" in plan["reason"]
    judged = {**pmb.judge(Path("/nonexistent"), {}), "outcome": "green", "partial": [], "planned": 2, "passed": 2}
    text = "\n".join(pmb.summary_lines(sel.select(commits(["uv.lock"]), MODULES), judged, [], {}, {}))
    assert "Shared browser infrastructure changed (`uv.lock`)" in text


def test_tiers_order_changed_tests_then_name_stems_then_narrow_mentions_and_hubs_select_nothing():
    plan = sel.select(commits(["web/style.css", "web/modules/chat_cards.js"], ["web/room_work.css"],
                              ["tests/test_composer_capsules_browser.py"]), MODULES, table=TABLE)
    rows = {row["module"]: row for row in plan["selected"]}
    assert [row["module"] for row in plan["selected"]] == [
        "tests/test_composer_capsules_browser.py", "tests/test_room_work_browser.py", "tests/test_living_cards_browser.py"]
    assert [rows[module]["tier"] for module in rows] == [1, 2, 3]
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


K_TEXT = ("import pytest\npytestmark = [pytest.mark.ui_browser]\n\n\n@pytest.fixture\ndef page():\n    return 1\n\n\n"
          "@pytest.mark.parametrize('engine', ['chromium', 'webkit'])\ndef test_x(page, engine):\n    assert page\n\n\n"
          "def test_y(page):\n    assert page\n\n\nclass TestK:\n    def test_m(self):\n        assert True\n")
K = "tests/test_k_browser.py"


def test_a_module_chosen_only_for_its_own_change_runs_the_tests_that_hold_it():
    assert sel.changed_tests(K, K_TEXT, {12}) == [f"{K}::test_x"]
    assert sel.changed_tests(K, K_TEXT, {10}) == [f"{K}::test_x"]  # its decorator
    assert sel.changed_tests(K, K_TEXT, {12, 21}) == [f"{K}::TestK::test_m", f"{K}::test_x"]
    assert sel.changed_tests(K, K_TEXT, {7}) is None  # the fixture: every test may read it
    assert sel.changed_tests(K, K_TEXT, set()) is None and sel.changed_tests(K, K_TEXT, None) is None  # unknown lines
    helper = K_TEXT + "\n\ndef testing_helper():\n    return 2\n\n\nclass Helper:\n    def test_n(self):\n        pass\n"
    assert sel.changed_tests(K, helper, {25}) is None and sel.changed_tests(K, helper, {30}) is None
    assert [name for _, _, name in sel.collectable_spans(K, helper)] == [
        f"{K}::test_x", f"{K}::test_y", f"{K}::TestK::test_m"]  # what pytest collects, nothing else
    table = {"ui_nodes": {f"{K}::test_x[chromium]": 5000, f"{K}::test_x[webkit]": 7000, f"{K}::test_y": 90_000}}
    plan = sel.select(commits([K]), {K: K_TEXT}, table=table, lines={K: {12}})
    [row] = plan["selected"]
    assert row["tests"] == [f"{K}::test_x"] and row["estimate_ms"] == 12_000
    assert row["nodes"] == {"chromium": 1, "webkit": 1, "other": 0} and plan["args"] == [f"{K}::test_x"]


def test_a_test_edit_never_narrows_what_an_independent_product_change_selects():
    """The scopes of every reason are joined before the budget: a product file that selects the whole module keeps it
    whole, however narrow the edit of one of its tests was."""
    module = "tests/test_room_work_browser.py"
    alone = sel.select(commits([module]), MODULES, lines={module: {7}})  # a line of the sentinel test's body
    assert alone["selected"][0]["tests"] == [ROOM_TEST]
    both = sel.select(commits(["web/room_work.css"], [module]), MODULES, lines={module: {7}})
    [row] = both["selected"]
    assert row["tests"] is None and row["whole"] is True and row["tier"] == 1
    assert both["args"] == [LIVING_TEST, module]


def test_a_helper_change_in_a_browser_module_reaches_the_modules_that_import_it():
    defining_module = "tests/test_storage_manual_browser.py"
    modules = {defining_module: UI + "\n\ndef _no_overflow(page):\n    return page\n\n\ndef test_manual():\n    pass\n",
               "tests/test_storage_payloads_browser.py": UI + "from tests.test_storage_manual_browser import _no_overflow\n",
               "tests/test_other_browser.py": UI}
    helper = sel.select(commits([defining_module]), modules, lines={defining_module: {5}})
    assert [row["module"] for row in helper["selected"]] == [defining_module, "tests/test_storage_payloads_browser.py"]
    assert helper["selected"][1]["reasons"] == ["imports from test_storage_manual_browser.py"]
    # A change confined to one of its tests reaches no importer.
    body = sel.select(commits([defining_module]), modules, lines={defining_module: {10}})
    assert [row["module"] for row in body["selected"]] == [defining_module]


def test_changed_lines_and_the_base_modules_read_both_revisions(tmp_path):
    repo = repository(tmp_path)
    (repo / "tests/test_k_browser.py").write_text("a\nb\nc\nd\n")
    (repo / "tests/test_gone_browser.py").write_text(UI)
    (repo / "tests/test_unmarked_browser.py").write_text(UI)
    (repo / "tests/test_mention.py").write_text('TEXT = "pytestmark = [pytest.mark.ui_browser]"\n')
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "base")
    base = git(repo, "rev-parse", "HEAD")
    (repo / "tests/test_k_browser.py").write_text("a\nB\nc\nd\ne\n")
    (repo / "tests/test_unmarked_browser.py").write_text("def test_x():\n    pass\n")
    git(repo, "rm", "-q", "tests/test_gone_browser.py", "tests/test_mention.py")
    git(repo, "commit", "-qam", "head")
    assert sel.changed_lines(base, "HEAD", {"tests/test_k_browser.py"}, cwd=repo) == {"tests/test_k_browser.py": {2, 5}}
    assert sel.changed_lines(base, "HEAD", set(), cwd=repo) == {}
    changed = ["tests/test_gone_browser.py", "tests/test_k_browser.py", "tests/test_mention.py",
               "tests/test_unmarked_browser.py"]
    # Deleted or unmarked by the window, a browser module of the base applies; a mention in a string never did.
    assert sel.base_modules(base, changed, {}, cwd=repo) == {"tests/test_gone_browser.py",
                                                             "tests/test_unmarked_browser.py"}


def test_the_sentinels_run_first_and_every_node_counts_once():
    plan = sel.select(commits(["web/tests/x.test.js"]), MODULES, table=TABLE)
    assert plan["selected"] == [] and plan["args"] == [LIVING_TEST, ROOM_TEST]  # as tests: never a stale param id
    assert [row["scenario"] for row in plan["sentinels"]] == [LIVING, ROOM]
    assert [row["guards"] for row in plan["sentinels"]] == [["web/modules/chat_cards.js"], ["web/room_work.css"]]
    assert plan["estimate_ms"] == 33_000
    # The room module covers its sentinel: it moves to the front, and its sentinel's two seconds count once.
    whole = sel.select(commits(["web/room_work.css"]), MODULES, table=TABLE)
    [room] = whole["selected"]
    assert room["estimate_ms"] == 60_000 and whole["estimate_ms"] == 33_000 + 60_000
    assert whole["args"] == [LIVING_TEST, "tests/test_room_work_browser.py"]
    # A budget that holds both sentinels and the module's other nodes keeps the module (it was dropped when the
    # sentinel's time was charged twice).
    fits = sel.select(commits(["web/room_work.css"]), MODULES, table=TABLE, budget_seconds=93)
    assert [row["module"] for row in fits["selected"]] == ["tests/test_room_work_browser.py"]
    # A selected module that holds a sentinel comes before every other candidate.
    living = sel.select(commits(["tests/test_composer_capsules_browser.py"], ["web/modules/chat_cards.js"]),
                        MODULES, table=TABLE)
    assert living["args"] == ["tests/test_living_cards_browser.py", ROOM_TEST,
                              "tests/test_composer_capsules_browser.py"]


def test_a_sentinel_the_head_no_longer_defines_is_named_and_never_passed():
    renamed = {**MODULES, "tests/test_room_work_browser.py": UI + defining("tests/x.py::test_renamed")}
    plan = sel.select(commits(["web/tests/x.test.js"]), renamed, table=TABLE)
    assert plan["args"] == [LIVING_TEST]  # pytest would refuse the whole session over a test it cannot find
    assert [(row["test"], row["present"]) for row in plan["sentinels"]] == [(LIVING_TEST, True), (ROOM_TEST, False)]


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
    assert (sel.BUDGET_SECONDS, sel.MAX_MODULES) == (2100, 24)


def test_a_plan_with_nothing_to_run_is_not_applicable_and_caddy_stays_with_the_full_lane():
    plan = sel.select(commits(["web/tests/x.test.js"]), {"tests/test_other_browser.py": UI})
    assert plan["args"] == [] and plan["applicable"] is False
    assert plan["reason"].endswith("no browser test matches it and no sentinel is present")
    modules = {**MODULES, "tests/test_separate_ui_browser.py": UI}
    caddy = sel.select(commits(["tests/test_separate_ui_browser.py"]), modules)
    assert caddy["selected"] == [] and caddy["excluded"][0]["module"] == "tests/test_separate_ui_browser.py"


def test_the_scope_command_runs_on_a_bare_interpreter():
    done = subprocess.run([sys.executable, "-I", "-S", str(ROOT / "scripts/post_merge_browser_select.py"), "scope",
                           "--base", "HEAD", "--head", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0 and done.stdout.startswith("applicable=false\nwhy="), done.stdout + done.stderr


# --- the outcome ------------------------------------------------------------------------------------------------

PLAN = {"applicable": True, "base": "b" * 40, "head": "h" * 40, "args": ["tests/test_a_browser.py"],
        "selected": [{"module": "tests/test_a_browser.py", "files": ["web/a.js"], "reasons": ["x"]}],
        "sentinels": [], "commits": [{"sha": "1" * 40, "files": ["web/a.js"]}, {"sha": "2" * 40, "files": ["x.py"]}]}


def row(node, outcome="passed", phase="call"):
    return {"nodeid": node, "phase": phase, "outcome": outcome}


def whole(node):
    return [row(node, phase="setup"), row(node), row(node, phase="teardown")]


def evidence(tmp_path, monkeypatch, *, selected, rows, verdict, rounds=(), skips=None):
    """DIR/head as the run step leaves it: the first session, ci_retry's rounds and verdict.json."""
    head = tmp_path / "head"
    head.mkdir(parents=True)
    if verdict is not None:
        (head / "verdict.json").write_text(json.dumps(verdict))
    sessions = {head: {"rows": rows, "selected": selected}}
    for number, round_rows in enumerate(rounds, 1):
        (head / f"retry-{number}").mkdir()
        sessions[head / f"retry-{number}"] = {"rows": round_rows, "selected": None}
    if skips is not None:
        (head / pmb.SKIPS).write_text(json.dumps({"format": 1, "permitted": skips}))
    monkeypatch.setattr(pmb.pmc, "read_session", lambda directory: sessions.get(
        Path(directory), {"rows": [], "selected": None}))
    return tmp_path


def test_green_only_when_every_selected_node_ran_and_passed(tmp_path, monkeypatch):
    found = pmb.judge(evidence(tmp_path, monkeypatch, selected=["a", "b"], rows=whole("a") + whole("b"),
                               verdict={"verdict": "green", "failed": [], "flaky": []}), PLAN)
    assert found["outcome"] == "green" and found["passed"] == 2 and found["partial"] == [] and found["not_run"] == []


def test_red_needs_two_observed_failures(tmp_path, monkeypatch):
    first = [row("red", "failed"), row("flaky", "failed"), row("skipped-retry", "failed")]
    retry = [row("red", "failed"), row("flaky"), row("skipped-retry", "skipped", "setup")]
    found = pmb.judge(evidence(tmp_path, monkeypatch, selected=["red", "flaky", "skipped-retry"], rows=first,
                               rounds=[retry], verdict={
                                   "verdict": "red", "flaky": [{"nodeid": "flaky"}],
                                   "failed": [{"nodeid": "red", "attempts": [{"phase": "call", "error_type": "E"},
                                                                             {"phase": "call", "error_type": "E"}]},
                                              {"nodeid": "skipped-retry", "attempts": [
                                                  {"phase": "call", "error_type": "E"},
                                                  {"phase": "setup", "error_type": "Skipped"}]}]}), PLAN)
    assert found["red"] == ["red"] and found["flaky"] == ["flaky"]
    assert found["failed_then_skipped"] == ["skipped-retry"]  # a failure and a skip are no two failures: not red
    assert found["outcome"] == "red" and "a failure skipped on its retry" in found["partial"]
    assert found["skipped"] == []  # its skip is named with its failure, not twice


def test_a_failure_without_a_second_observation_is_partial_and_names_its_cause(tmp_path, monkeypatch):
    verdict = {"verdict": "red", "flaky": [], "reason": "3 collected test(s) never ran, and no lane block names them",
               "failed": [{"nodeid": "tests/test_broken.py", "attempts": [{"phase": "collect",
                                                                          "error_type": "CollectionError"}]},
                          {"nodeid": "killed", "attempts": [{"phase": "crash", "error_type": "InFlight"}]},
                          {"nodeid": "stopped", "attempts": [{"phase": "call", "error_type": "AssertionError"}]}]}
    found = pmb.judge(evidence(tmp_path, monkeypatch, selected=["killed", "stopped", "never"],
                               rows=[row("stopped", "failed")], verdict=verdict), PLAN)
    assert found["outcome"] == "partial" and found["red"] == []
    assert found["once"] == {"tests/test_broken.py": "did not collect",
                             "killed": "was running when its session was killed, so it got no retry",
                             "stopped": f"failed once and got no retry ({verdict['reason']})"}
    assert found["not_run"] == ["killed", "never"]


def test_an_empty_selection_a_lost_verdict_or_an_unexplained_one_is_never_green(tmp_path, monkeypatch):
    empty = pmb.judge(evidence(tmp_path / "e", monkeypatch, selected=[], rows=[],
                               verdict={"verdict": "green", "failed": [], "flaky": []}), PLAN)
    assert empty["outcome"] == "partial" and "the session selected no test of the plan" in empty["partial"]
    lost = pmb.judge(evidence(tmp_path / "l", monkeypatch, selected=["a"], rows=whole("a"), verdict=None), PLAN)
    assert lost["outcome"] == "partial" and lost["partial"][0].startswith("scripts/ci_retry.py left no verdict")
    unexplained = pmb.judge(evidence(tmp_path / "u", monkeypatch, selected=["a"], rows=whole("a"), verdict={
        "verdict": "red", "failed": [], "flaky": [],
        "reason": "the session ended with exit status 3, but no test failed or went unexecuted"}), PLAN)
    assert unexplained["outcome"] == "partial" and unexplained["partial"] == [
        "scripts/ci_retry.py says red and no test explains it: the session ended with exit status 3, but no test "
        "failed or went unexecuted"]
    no_export = pmb.judge(evidence(tmp_path / "x", monkeypatch, selected=None, rows=whole("a"),
                                   verdict={"verdict": "green", "failed": [], "flaky": []}), PLAN)
    assert no_export["outcome"] == "partial" and no_export["planned"] is None
    assert pmb.judge(tmp_path, {})["outcome"] == "partial"  # the job stopped before its plan
    assert pmb.judge(tmp_path, {"applicable": False, "reason": "r"})["outcome"] == "partial"  # never green


def test_a_sentinel_case_the_session_did_not_collect_or_complete_is_named(tmp_path, monkeypatch):
    plan = {**PLAN, "sentinels": [{"scenario": ROOM, "test": ROOM_TEST, "present": True, "guards": []},
                                  {"scenario": LIVING, "test": LIVING_TEST, "present": True, "guards": []}]}
    other = ROOM.replace("webkit-282", "webkit-320")  # the parameter id left; its siblings still run green
    found = pmb.judge(evidence(tmp_path, monkeypatch, selected=[other, LIVING], rows=whole(other) + whole(LIVING),
                               verdict={"verdict": "green", "failed": [], "flaky": []}), plan)
    assert found["outcome"] == "partial" and found["missing_sentinels"] == [
        f"{ROOM}: the session did not collect this case"]
    gone = pmb.judge(evidence(tmp_path / "g", monkeypatch, selected=[LIVING], rows=whole(LIVING),
                              verdict={"verdict": "green", "failed": [], "flaky": []}),
                     {**plan, "sentinels": [{**plan["sentinels"][0], "present": False}, plan["sentinels"][1]]})
    assert gone["missing_sentinels"] == [f"{ROOM}: the head no longer defines {ROOM_TEST}"]


def test_only_the_lanes_permitted_skips_pass_and_their_reason_decides(tmp_path, monkeypatch):
    pinch = "tests/test_shell_drawer_browser.py::test_a_pinch_magnifies_the_room[chromium]"
    pmb._PERMITTED.clear()

    def report(nodeid, reason):
        return SimpleNamespace(nodeid=nodeid, skipped=True, longrepr=("file.py", 1, f"Skipped: {reason}"))

    pmb.pytest_runtest_logreport(report(pinch, "Headless Chromium on Linux applies no synthetic touch pinch"))
    pmb.pytest_runtest_logreport(report(pinch.replace("chromium", "webkit"), "the fixture could not start"))
    pmb.pytest_runtest_logreport(report("tests/test_x_browser.py::test_y", "applies no synthetic touch pinch"))
    session = SimpleNamespace(config=SimpleNamespace(option=SimpleNamespace(ci_evidence_dir=str(tmp_path / "skips"))))
    pmb.pytest_sessionfinish(session)
    recorded = json.loads((tmp_path / "skips" / pmb.SKIPS).read_text())["permitted"]
    assert recorded == {pinch: "applies no synthetic touch pinch"}  # the reviewed entry, never the free text
    webkit = pinch.replace("chromium", "webkit")
    found = pmb.judge(evidence(tmp_path / "j", monkeypatch, selected=[pinch, webkit],
                               rows=[row(pinch, "skipped", "setup"), row(webkit, "skipped", "setup")],
                               verdict={"verdict": "green", "failed": [], "flaky": []}, skips=recorded), PLAN)
    assert found["skipped"] == [webkit] and found["outcome"] == "partial"  # same prefix, another reason
    pmb._PERMITTED.clear()


# --- the base comparison and the recheck in real sessions ---------------------------------------------------------

BROWSER_MODULE = ("import pytest\npytestmark = [pytest.mark.ui_browser]\n\n\ndef test_red():\n    assert False\n\n\n"
                  "def test_green():\n    pass\n\n\n@pytest.mark.parametrize('case', ['kept', 'other'])\n"
                  "def test_param(case):\n    pass\n")


def suite(tmp_path):
    """A checkout with one browser module, and the launcher a test can give `run_session`: pytest with the
    repository's own conftest (tests/ci_evidence.py among its plugins) over that checkout."""
    tree = tmp_path / "tree"
    (tree / "tests").mkdir(parents=True)
    (tree / "tests/test_x_browser.py").write_text(BROWSER_MODULE, encoding="utf-8")
    entry = tmp_path / "run_pytest.py"
    entry.write_text("import sys\n"
                     f"sys.path.insert(0, {str(ROOT)!r})\n"
                     "import pytest\n"
                     f"raise SystemExit(pytest.main(['-p', 'tests.conftest', '-o', 'addopts=', '--rootdir', {str(tree)!r},"
                     f" '--confcutdir', {str(tree)!r}, '-p', 'no:cacheprovider', *sys.argv[1:]]))\n", encoding="utf-8")
    return tree, [sys.executable, str(entry)]


def test_an_id_a_tree_does_not_hold_is_absent_and_stops_no_sibling_in_a_real_session(tmp_path):
    tree, launcher = suite(tmp_path)
    module = "tests/test_x_browser.py"
    ids = [f"{module}::test_red", f"{module}::test_missing", f"{module}::test_param[renamed]",
           f"{module}::test_param[kept]", f"{module}::test_green", "tests/test_gone_browser.py::test_y"]
    codes = []

    def run(args, directory):
        codes.append(pmb.run_session(tree, args, directory, pytest=launcher))
        return codes[-1]

    found = pmb.pmc.recheck_ids(ids, tmp_path / "evidence" / "base", head="b" * 40, budget=120, run=run,
                                read_source=lambda path: (tree / path).read_text() if (tree / path).is_file() else None)
    assert codes == [1], codes  # one session ran, red, never refused ("not found" is exit 4)
    assert found == {f"{module}::test_red": "failed", f"{module}::test_missing": "absent",
                     f"{module}::test_param[renamed]": "absent", f"{module}::test_param[kept]": "passed",
                     f"{module}::test_green": "passed", "tests/test_gone_browser.py::test_y": "absent"}
    exported = json.loads((tmp_path / "evidence" / "base" / "results.json").read_text(encoding="utf-8"))
    assert sorted(exported["selected_nodeids"]) == sorted(node for node, answer in found.items() if answer != "absent")


def test_compare_runs_nothing_where_the_lock_differs_and_never_trusts_a_worktree_it_did_not_verify(
        tmp_path, monkeypatch):
    red = ["tests/test_x_browser.py::test_red"]
    blobs = {"b" * 40: "1", "h" * 40: "2"}
    monkeypatch.setattr(pmb.pmc, "_blob", lambda sha, path: blobs[sha])
    differ = pmb.compare(tmp_path, PLAN, red, run=lambda args, directory: 0 / 0)
    assert differ["dependencies_differ"] is True and differ["results"] == {}
    blobs["h" * 40] = "1"
    added, removed = [], []
    monkeypatch.setattr(pmb, "protected_projection", lambda head, base, git: ("same", ""))
    monkeypatch.setattr(pmb.pmc, "add_worktree", lambda tree, sha, omit_protected: (tree.mkdir(), added.append(sha)))
    monkeypatch.setattr(pmb.pmc, "remove_worktree", lambda tree: removed.append(tree))
    monkeypatch.setattr(pmb.pmc, "_git", lambda *args, cwd=None: "0" * 40)  # some other commit
    stale = pmb.compare(tmp_path, PLAN, red, run=lambda args, directory: 0 / 0)
    assert stale["results"] == {red[0]: "unknown"} and "not at the window's base" in stale["error"]
    assert added == ["b" * 40] and len(removed) == 1 and not removed[0].parent.exists()  # removed in finally


def test_the_recheck_settles_the_open_issues_by_whole_passes_first_and_skips_what_this_run_saw_fail(
        tmp_path, monkeypatch):
    issue = tracking_issue(5, {"os": "browser", "kind": "unattributed", "ids": ["a", "b", "c", "d"]})
    legs = tracking_issue(6, {"os": "linux", "kind": "unattributed", "ids": ["e"]})
    monkeypatch.setattr(pmb, "same_run_passes", lambda directory, ids: {"a": "passed"} if "a" in ids else {})
    asked = []
    monkeypatch.setattr(pmb.pmc, "recheck_ids", lambda ids, directory, **options: asked.append(ids) or dict.fromkeys(
        ids, "absent"))
    judged = {"red": ["b"], "once": {"c": "did not collect"}, "failed_then_skipped": []}
    found = pmb.recheck(tmp_path, PLAN, judged, FakeGitHub(issues=[issue, legs]))
    assert found["listed"] is True and found["issues"] == [5]
    assert found["same_run"] == {"a": "passed"} and found["rechecked"] == {"d": "absent"} and asked == [["d"]]
    refused = pmb.recheck(tmp_path, PLAN, judged, FakeGitHub(fail=[("GET", None)]))
    assert refused["listed"] is False and refused["rechecked"] == {} and "did not read" in refused["warning"]


# --- attribution ------------------------------------------------------------------------------------------------

def test_a_merge_is_named_only_for_a_window_of_one_merge_that_the_base_confirms():
    node = "tests/test_a_browser.py::test_x"
    one = {**PLAN, "commits": PLAN["commits"][:1]}
    for at_base in ("passed", "absent"):  # absent: introduced with that merge
        [named] = pmb.attribute(one, [node], {"results": {node: at_base}})
        assert named == {"id": node, "verdict": "culprit", "culprit": "1" * 40, "last_good": "b" * 40,
                         "at_base": at_base}
    [pre] = pmb.attribute(PLAN, [node], {"results": {node: "failed"}})
    assert pre == {"id": node, "verdict": "pre-existing"}
    # Two merges: the base proves only that the window introduced it. The one that touched its files comes first.
    [many] = pmb.attribute(PLAN, [node], {"results": {node: "passed"}})
    assert many["verdict"] == "unattributed" and many["candidates"] == ["1" * 40] and many["at_base"] == "passed"
    untouched = {**PLAN, "commits": [{"sha": "1" * 40, "files": ["x.py"]}, {"sha": "2" * 40, "files": ["y.py"]}]}
    [anyone] = pmb.attribute(untouched, [node], {"results": {node: "absent"}})
    assert anyone["candidates"] == ["1" * 40, "2" * 40]
    for answer in ("flaky", "unknown", "skipped", None):
        [unsure] = pmb.attribute(one, [node], {"results": {node: answer} if answer else {}})
        assert unsure["verdict"] == "unattributed" and unsure["at_base"] == "unknown", answer
    [locked] = pmb.attribute(one, [node], {"results": {node: "passed"}, "dependencies_differ": True})
    assert locked["verdict"] == "unattributed" and locked["at_base"] == "dependencies" and "uv.lock" in locked["why"]


def test_a_sentinel_is_compared_through_the_files_it_guards():
    plan = {**PLAN, "selected": [], "sentinels": [{"scenario": ROOM, "test": ROOM_TEST, "guards": ["web/room_work.css"],
                                                   "present": True}],
            "commits": [{"sha": "1" * 40, "files": ["web/a.js"]}, {"sha": "2" * 40, "files": ["web/room_work.css"]}]}
    [verdict] = pmb.attribute(plan, [ROOM], {"results": {ROOM: "passed"}})
    assert verdict["verdict"] == "unattributed" and verdict["candidates"] == ["2" * 40]


# --- the hand-over ----------------------------------------------------------------------------------------------

DATA = {"base": "b" * 40, "head": "f" * 40, "red": ["tests/test_a_browser.py::test_x", "tests/test_b_browser.py::t"],
        "flaky": [], "same_run": {}, "rechecked": {}, "recheck_listed": True,
        "verdicts": [{"id": "tests/test_a_browser.py::test_x", "verdict": "unattributed", "at_base": "passed",
                      "candidates": ["1" * 40]},
                     {"id": "tests/test_b_browser.py::t", "verdict": "unattributed", "at_base": "unknown",
                      "candidates": ["1" * 40, "2" * 40]}]}


def test_browser_reds_are_their_own_system_with_each_tests_base_answer_and_candidates():
    linux = tracking_issue(7, {"os": "linux", "kind": "unattributed", "ids": ["tests/test_a_browser.py::test_x"]})
    github = FakeGitHub(issues=[linux], label=True)
    record = pmb.hand_over(DATA, github, run_url="run", retire=True)
    [created] = record["created"]
    issue = github.issues[created]
    assert issue["title"] == "Post-merge red on the browser check (Ubuntu): failing tests without a found culprit"
    block = routing.parse_machine_block(issue["body"])
    assert block["os"] == "browser" and block["candidates"] == {"tests/test_a_browser.py::test_x": ["1" * 40],
                                                                 "tests/test_b_browser.py::t": ["1" * 40, "2" * 40]}
    assert block["at_base"] == {"tests/test_a_browser.py::test_x": "passed", "tests/test_b_browser.py::t": "unknown"}
    assert ("- `tests/test_a_browser.py::test_x`: passed at the window's base; candidate merges `" + "1" * 40 + "`"
            in issue["body"])
    assert "- `tests/test_b_browser.py::t`: not settled at the window's base; candidate merges" in issue["body"]
    assert "passed at, or were absent from," not in issue["body"]  # one of them did not
    assert github.issues[7]["state"] == "open" and 7 not in record["updated"] and record["untracked"] == []
    one = {**DATA, "red": DATA["red"][:1], "verdicts": DATA["verdicts"][:1]}
    alone = FakeGitHub(label=True)
    pmb.hand_over(one, alone, run_url="run", retire=True)
    [body] = [issue["body"] for issue in alone.issues.values()]
    assert "They passed at, or were absent from, the window's base" in body
    # A Linux leg that sees the same test pass leaves the browser issue alone.
    later = routing.route(github, classification={"failed": []}, attribution={"verdicts": []}, head="f" * 40,
                          run_url="run", recheck=lambda ids: dict.fromkeys(ids, "passed"), system="linux")
    assert created not in later["closed"] and github.issues[created]["state"] == "open"


def test_an_issue_keeps_each_tests_own_notes_as_tests_join_and_leave_it():
    first = {"format": 1, "os": "browser", "kind": "unattributed", "ids": ["a", "b"], "at_base": {"a": "passed", "b": "x"},
             "candidates": {"a": ["1" * 40], "b": ["2" * 40]}}
    github = FakeGitHub(issues=[tracking_issue(9, first)], label=True)
    joined = {**DATA, "red": ["c"], "verdicts": [{"id": "c", "verdict": "unattributed", "at_base": "absent",
                                                  "candidates": ["3" * 40]}]}
    record = pmb.hand_over(joined, github, run_url="run", retire=True, ancestor=lambda older, newer: True)
    assert record["updated"] == [9]
    block = routing.parse_machine_block(github.issues[9]["body"])
    assert block["ids"] == ["a", "b", "c"] and block["candidates"] == {"a": ["1" * 40], "b": ["2" * 40],
                                                                       "c": ["3" * 40]}
    narrowed = routing.route(github, classification={"failed": []}, attribution={"verdicts": []}, head="f" * 40,
                             run_url="run", recheck=lambda ids: {"a": "passed"}, system="browser",
                             ancestor=lambda older, newer: True)
    assert narrowed["narrowed"] == [{"issue": 9, "ids": ["a"]}]
    rest = routing.parse_machine_block(github.issues[9]["body"])
    assert rest["ids"] == ["b", "c"] and set(rest["candidates"]) == set(rest["at_base"]) == {"b", "c"}


def test_a_legs_issue_never_gains_notes():
    github = FakeGitHub(label=True)
    routing.route(github, classification={"failed": [{"id": "tests/t.py::test_a"}]},
                  attribution={"verdicts": [{"id": "tests/t.py::test_a", "verdict": "unattributed"}]},
                  head="f" * 40, run_url="run", recheck=lambda ids: {}, system="linux")
    [issue] = github.issues.values()
    assert not set(routing.PER_ID) & set(routing.parse_machine_block(issue["body"]))
    assert issue["title"] == "Post-merge red on Linux: failing tests without a found culprit"


def test_the_browser_system_is_no_leg():
    """scripts/post_merge_window.py waits for a receipt of each leg in SYSTEMS before the base advances: the browser
    check owns issues under its own name and is never one of them."""
    assert routing.SYSTEMS == {"linux": "Linux", "windows": "Windows"}
    assert routing.NAMES == {**routing.SYSTEMS, "browser": "the browser check (Ubuntu)"}
    assert post_merge_window.SYSTEMS is routing.SYSTEMS


class Tip(FakeGitHub):
    def __init__(self, tip, **options):
        super().__init__(**options)
        self.tip = tip

    def get(self, path):
        if path == f"git/ref/heads/{routing.BRANCH}":
            return {"object": {"sha": self.tip}}
        return super().get(path)


def route(tmp_path, monkeypatch, github, data, *, ref="refs/heads/ouroboros"):
    monkeypatch.setattr(pmb.routing, "GitHub", lambda repo, token: github)
    summary = tmp_path / "summary.md"
    summary.unlink(missing_ok=True)  # the step summary is appended to: each call reads its own
    environ = {"ROUTING": pmb.encode(data), "GITHUB_REF": ref, "GITHUB_REPOSITORY": "razzant/ouroboros",
               "GITHUB_STEP_SUMMARY": str(summary), "GITHUB_RUN_ID": "77", "GITHUB_RUN_ATTEMPT": "1"}
    args = pmb.argparse.Namespace(run_url="run/77", out=str(tmp_path / "routing.json"), dry_run=False)
    code = pmb.command_route(args, environ=environ)
    return code, summary.read_text(encoding="utf-8"), json.loads((tmp_path / "routing.json").read_text())


def test_a_refused_hand_over_fails_the_route_job_and_names_what_waits(tmp_path, monkeypatch):
    refused = Tip("f" * 40, label=True, fail=[("POST", "issues")])
    code, text, record = route(tmp_path, monkeypatch, refused, DATA)
    assert code == 1 and record["record"]["untracked"] == sorted(DATA["red"])
    assert "Not handed over (2)" in text and "create an issue" in text and "gh run rerun" in text
    code, text, _ = route(tmp_path, monkeypatch, Tip("f" * 40, label=True), DATA)
    assert code == 0 and "Not handed over" not in text


def test_only_a_run_at_the_branch_tip_retires_an_issue(tmp_path, monkeypatch):
    issue = tracking_issue(5, {"os": "browser", "kind": "unattributed", "ids": ["tests/test_old_browser.py::t"]})
    data = {**DATA, "red": [], "verdicts": [], "rechecked": {"tests/test_old_browser.py::t": "absent"}}
    moved = Tip("e" * 40, issues=[issue], label=True)
    code, text, record = route(tmp_path, monkeypatch, moved, data)
    assert code == 0 and record["retire"] is False and moved.issues[5]["state"] == "open" and "moved past" in text
    monkeypatch.setattr(pmb, "_ancestor", lambda older, newer: True)
    current = Tip("f" * 40, issues=[issue], label=True)
    code, _, record = route(tmp_path, monkeypatch, current, data)
    assert code == 0 and record["record"]["closed"] == [5] and current.issues[5]["state"] == "closed"


def test_route_without_its_input_hands_nothing_over_and_fails(tmp_path, monkeypatch):
    summary = tmp_path / "summary.md"
    args = pmb.argparse.Namespace(run_url="run", out=str(tmp_path / "routing.json"), dry_run=False)
    assert pmb.command_route(args, environ={"ROUTING": "", "GITHUB_STEP_SUMMARY": str(summary)}) == 1
    assert "Nothing was handed over" in summary.read_text()
    assert pmb.decode(pmb.encode(DATA)) == DATA and pmb.decode("not base64!") is None


def test_dry_routing_reads_github_and_writes_nothing(monkeypatch):
    calls = []
    monkeypatch.setattr(routing.GitHub, "_call", lambda self, method, url, body=None: calls.append(method) or ([], ""))
    dry = pmb.DryGitHub("razzant/ouroboros", "")
    dry.pages("issues?state=open")
    dry.post("issues", {"title": "t", "body": "b", "labels": []})
    dry.patch("issues/1", {"state": "closed"})
    assert calls == ["GET"] and [w["method"] for w in dry.writes] == ["POST", "PATCH"]


def test_the_outcome_step_hands_its_input_to_route_and_the_verdict_fails_unless_green(tmp_path, monkeypatch):
    (tmp_path / "plan.json").write_text(json.dumps(PLAN))
    empty = pmb.judge(Path("/nonexistent"), {})
    for outcome, code in (("green", 0), ("flaky", 0), ("red", 1), ("partial", 1)):
        monkeypatch.setattr(pmb, "judge", lambda evidence, plan, o=outcome: {
            **empty, "outcome": o, "partial": [], "planned": 1, "passed": 1})
        output = tmp_path / f"output-{outcome}"
        args = pmb.argparse.Namespace(evidence_dir=str(tmp_path), plan=str(tmp_path / "plan.json"))
        assert pmb.command_outcome(args, environ={"GITHUB_OUTPUT": str(output),
                                                  "BROWSER_INSTALL_OUTCOME": "failure"}) == 0
        values = dict(line.split("=", 1) for line in output.read_text().splitlines())
        assert values["outcome"] == outcome and pmb.decode(values["routing"])["outcome"] == outcome
        assert pmb.command_verdict(pmb.argparse.Namespace(evidence_dir=str(tmp_path))) == code
    assert pmb.command_verdict(pmb.argparse.Namespace(evidence_dir=str(tmp_path / "none"))) == 1


# --- the workflow and the readers of the post-merge runs ---------------------------------------------------------

JOBS = FLOW["jobs"]
STEPS = {name: {step.get("id") or step.get("uses") or step["name"]: step for step in job["steps"]}
         for name, job in JOBS.items()}


def test_the_window_job_starts_the_check_only_on_ouroboros_and_only_when_it_applies():
    window = POST_MERGE["jobs"]["window"]
    steps = {step.get("id"): step for step in window["steps"]}
    assert set(POST_MERGE["jobs"]) == {"window", "post-merge"}  # no browser job holds the post-merge group
    assert window["permissions"] == {"contents": "read", "actions": "write"}  # write: the dispatch alone
    scope, start = steps["browser_scope"], steps["browser_dispatch"]
    assert scope["run"].startswith("python3 -I -S scripts/post_merge_browser_select.py scope")
    assert scope["continue-on-error"] is True and start["continue-on-error"] is True  # neither can stop a leg
    assert start["if"] == "github.ref == 'refs/heads/ouroboros' && steps.browser_scope.outputs.applicable == 'true'"
    assert re.fullmatch(r'gh workflow run post-merge-browser\.yml --repo "\$GITHUB_REPOSITORY" --ref ouroboros '
                        r'-f base="\$BASE" -f head="\$GITHUB_SHA"', start["run"]), start["run"]
    assert start["env"] == {"GH_TOKEN": "${{ github.token }}", "BASE": "${{ steps.base.outputs.base }}"}
    order = [step.get("id") for step in window["steps"]]
    assert order.index("base") < order.index("browser_scope") < order.index("browser_dispatch")
    for step in window["steps"]:  # what the window readers take from its step names
        name = step.get("name") or ""
        if "browser" in name:
            assert not name.startswith(post_merge_window.REPLAY_STEP)
            assert post_merge_window._BASE_STEP.fullmatch(name) is None
    text = (ROOT / ".github/workflows/post-merge.yml").read_text(encoding="utf-8")
    assert "browser_budget" not in text and "--browser-stop-file" not in text and "legs --repo" not in text


def test_the_route_jobs_run_no_test_and_only_the_trial_routes_dry():
    """Permissions, tokens and pinning: tests/test_post_merge_workflow.py; ceilings and the install's bound:
    tests/test_ci_diagnostic_workflows.py."""
    for name in ("route", "route-trial"):
        assert JOBS[name]["needs"] == ["scope", "check"] and "needs.check.result != 'skipped'" in JOBS[name]["if"]
        run = STEPS[name]["route"]["run"]
        assert run.startswith("python3 -I -S scripts/post_merge_browser.py route") and "pytest" not in run
        assert ("--dry-run" in run) == (name == "route-trial")
        assert STEPS[name]["route"]["env"]["ROUTING"] == "${{ needs.check.outputs.routing }}"
    assert JOBS["check"]["if"] == "needs.scope.outputs.applicable == 'true'"  # not applicable: skipped, never green
    run = STEPS["check"]["run"]["run"]
    assert "-p scripts.post_merge_browser" in run and "--session-timeout=\"$SESSION\"" in run
    assert "scripts/ci_retry.py" in run and "--require-ui-browser" not in run  # a subset proves no lane


def test_the_check_writes_its_outcome_before_the_upload_and_fails_only_after_it():
    order = [step.get("id") or step.get("name") for step in JOBS["check"]["steps"]]
    upload = next(step for step in JOBS["check"]["steps"] if "upload-artifact" in step.get("uses", ""))
    assert order.index("outcome") < order.index(upload["name"]) < order.index("verdict") == len(order) - 1
    assert upload["if"] == "${{ always() }}" and STEPS["check"]["outcome"]["if"] == "${{ !cancelled() }}"
    assert JOBS["check"]["outputs"] == {"outcome": "${{ steps.outcome.outputs.outcome }}",
                                        "routing": "${{ steps.outcome.outputs.routing }}"}
    assert set(STEPS["check"]["outcome"]["env"]) == {f"BROWSER_{step.upper()}_OUTCOME" for step in pmb.STEPS}
