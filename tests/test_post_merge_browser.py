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
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
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


def test_a_marker_through_any_name_of_pytest_mark_declares_a_browser_module():
    """Round 6 (b1): `from pytest import mark as m` with `@m.ui_browser` or `pytestmark = m.ui_browser` (listed or
    called) marks the module whatever the base name is; the lane keeps those tests and the window applies."""
    alias = "from pytest import mark as m\n"
    texts = {"tests/test_alias_decorator.py": alias + "\n@m.ui_browser\ndef test_x():\n    pass\n",
             "tests/test_alias_single.py": alias + "pytestmark = m.ui_browser\n\n\ndef test_x():\n    pass\n",
             "tests/test_alias_listed.py": alias + "pytestmark = [m.ui_browser]\n\n\ndef test_x():\n    pass\n",
             "tests/test_alias_called.py": ("import pytest\nmarks = pytest.mark\n\n\nclass TestK:\n    @marks.ui_browser()\n"
                                            "    def test_x(self):\n        pass\n"),
             "tests/test_alias_docker.py": alias + "pytestmark = [m.ui_browser_docker]\n"}
    found = sel.ui_modules(texts)
    assert set(found) == set(texts) - {"tests/test_alias_docker.py"}
    for path, text in found.items():
        assert all(lane for _, _, _, lane in sel.lane_tests(path, text)), path
    assert sel.applicability(["tests/test_alias_decorator.py"], found)[0]


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
    for path in ("tests/conftest.py", "tests/browser_lane.py", "conftest.py"):
        plan = sel.select(commits([path]), MODULES, table=TABLE)
        assert plan["applicable"] is True and plan["shared"] == [path], path
        assert plan["selected"] == [] and plan["args"] == [LIVING_TEST, ROOM_TEST], path
        assert "shared browser infrastructure" in plan["reason"]
    judged = {**pmb.judge(Path("/nonexistent"), {}), "outcome": "green", "partial": [], "planned": 2, "passed": 2}
    text = "\n".join(pmb.summary_lines(sel.select(commits(["tests/browser_lane.py"]), MODULES), judged, [], {}, {}))
    assert "Shared browser infrastructure changed (`tests/browser_lane.py`)" in text


def test_only_the_owners_scope_starts_the_check_automatically():
    """Owner choice 1: web/, browser tests, their helpers and the browser lane. Dependencies, CI runners and the setup
    action stay with the release and manual lanes (findings 4 and 14): a window of them alone is not applicable."""
    for path in ("uv.lock", "pyproject.toml", "scripts/ci_retry.py", "scripts/run_tests.py", "tests/ci_evidence.py",
                 "tests/ci_verdicts.py", "tests/ci_accepted.py", ".github/actions/setup-python-env/action.yml",
                 "tests/system_e2e/conftest.py", "scripts/post_merge_browser.py"):
        assert sel.select(commits([path]), MODULES)["applicable"] is False, path


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
    # Line 2 replaces a base line that no base test holds: line 0, the module runs whole (Carbon, round 4).
    assert sel.changed_lines(base, "HEAD", {"tests/test_k_browser.py"}, cwd=repo) == {"tests/test_k_browser.py": {0, 2, 5}}
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
    assert whole["args"] == [LIVING_TEST, "tests/test_room_work_browser.py"]  # the room sentinel: inside its module
    # A budget that holds both sentinels and the module's other nodes keeps the module (it was dropped when the
    # sentinel's time was charged twice).
    fits = sel.select(commits(["web/room_work.css"]), MODULES, table=TABLE, budget_seconds=93)
    assert [row["module"] for row in fits["selected"]] == ["tests/test_room_work_browser.py"]
    # The arguments keep the plan's priority; the head session's plugin puts the two exact cases first (below).
    living = sel.select(commits(["tests/test_composer_capsules_browser.py"], ["web/modules/chat_cards.js"]),
                        MODULES, table=TABLE)
    assert living["args"] == [ROOM_TEST, "tests/test_composer_capsules_browser.py", "tests/test_living_cards_browser.py"]


def test_the_head_session_runs_the_two_sentinel_cases_first_whatever_argument_collected_them():
    """pytest keeps a module's own order where arguments overlap (a test and its whole module), so the plugin, not the
    argument order, puts the exact WebKit cases first (advisory 7), each once."""
    items = [SimpleNamespace(nodeid=node) for node in (
        "tests/test_living_cards_browser.py::test_a[chromium]", ROOM.replace("webkit", "chromium"),
        "tests/test_composer_capsules_browser.py::test_c", ROOM, "tests/test_living_cards_browser.py::test_z", LIVING)]
    pmb.pytest_collection_modifyitems(None, items)
    assert [item.nodeid for item in items] == [
        LIVING, ROOM, "tests/test_living_cards_browser.py::test_a[chromium]", ROOM.replace("webkit", "chromium"),
        "tests/test_composer_capsules_browser.py::test_c", "tests/test_living_cards_browser.py::test_z"]


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


def test_an_eligible_window_with_nothing_to_run_is_checked_partially_naming_the_missing_sentinels(
        tmp_path, monkeypatch, capsys):
    """Decision 6 (A5): scope reads applicability itself, so a web/ window whose plan is empty (both sentinel modules
    gone, a hub file only) still starts the check, which ends CHECKED PARTIALLY before installing a browser."""
    modules = {f"tests/test_hub_{n}_browser.py": UI + "# style.css\n" for n in range(11)}
    plan = sel.select(commits(["web/style.css"]), modules)
    assert plan["applicable"] is True and plan["runnable"] is False and plan["args"] == []
    assert plan["reason"].endswith(f"no browser test matches it and the head defines no sentinel (missing: {LIVING}, "
                                   f"{ROOM})")
    monkeypatch.setattr(sel, "window_commits", lambda base, head: commits(["web/style.css"]))
    monkeypatch.setattr(sel, "tree_modules", lambda root: modules)
    monkeypatch.setattr(sel, "base_modules", lambda base, changed, head_modules: set())
    assert sel.main(["scope", "--base", "b" * 40, "--head", "h" * 40]) == 0
    assert capsys.readouterr().out.startswith("applicable=true\nwhy=1 changed file(s) under web/")
    found = pmb.judge(tmp_path, {**plan, "base": "b" * 40, "head": "h" * 40})
    assert found["outcome"] == "partial" and found["missing_sentinels"] == [
        f"{LIVING}: the head no longer defines {LIVING_TEST}", f"{ROOM}: the head no longer defines {ROOM_TEST}"]
    assert found["partial"][0].startswith("the plan has nothing to run: 1 changed file(s) under web/")
    text = "\n".join(pmb.summary_lines(plan, found, [], {}, {}))
    assert "CHECKED PARTIALLY" in text and f"- {ROOM}: the head no longer defines {ROOM_TEST}" in text
    install = STEPS["check"]["install_browsers"]["if"]
    assert "steps.plan.outputs.runnable == 'true'" in install  # nothing to run: no browser is installed


CLIENT = "tests/test_client_views_browser.py"
CLIENT_TEXT = UI + "".join(f"\n\ndef {name}():\n    pass\n" for name in (
    "test_example_message_ids_work_on_insecure_and_secure_origins",
    "test_independent_pages_sign_in_configure_and_use_the_real_core",
    "test_copied_builtin_parent_recovers_a_broken_selected_setup",
    "test_builtin_recovery_is_a_whole_working_page_before_and_after_auth"))


def test_only_the_tests_that_start_a_caddy_stay_with_the_full_lane():
    """Decision 7 (A6): the separate-interface module needs Caddy whole; in tests/test_client_views_browser.py only the
    three tests that start a Caddy proxy do, and the rest runs by its ids on its declared engines."""
    caddy = sel.select(commits(["tests/test_separate_ui_browser.py"]), {**MODULES, "tests/test_separate_ui_browser.py": UI})
    assert caddy["selected"] == [] and caddy["excluded"] == [{"module": "tests/test_separate_ui_browser.py",
                                                              "why": sel.CADDY}]
    proxied = [f"{CLIENT}::{name}" for name in ("test_independent_pages_sign_in_configure_and_use_the_real_core",
                                                "test_copied_builtin_parent_recovers_a_broken_selected_setup",
                                                "test_builtin_recovery_is_a_whole_working_page_before_and_after_auth")]
    for lines in (None, {CLIENT: {1}}):  # the module whole (a change outside its tests), and that same reach by line
        plan = sel.select(commits([CLIENT]), {**MODULES, CLIENT: CLIENT_TEXT}, lines=lines)
        [row] = [row for row in plan["selected"] if row["module"] == CLIENT]
        assert row["tests"] == [f"{CLIENT}::test_example_message_ids_work_on_insecure_and_secure_origins"]
        assert row["tests"][0] in plan["args"] and CLIENT not in plan["args"]
        assert sorted(item["module"] for item in plan["excluded"]) == sorted(proxied)
    narrowed = sel.select(commits([CLIENT]), {**MODULES, CLIENT: CLIENT_TEXT}, lines={CLIENT: {9}})  # in a Caddy test
    assert all(row["module"] != CLIENT for row in narrowed["selected"]) and narrowed["excluded"] == [
        {"module": proxied[0], "why": sel.CADDY}]


DOCKER = ("import pytest\n\n\n@pytest.mark.ui_browser\ndef test_page():\n    pass\n\n\n"
          "@pytest.mark.ui_browser_docker\ndef test_docker():\n    pass\n")
D = "tests/test_smoke_browser.py"


def test_a_changed_test_the_lane_deselects_is_excluded_by_the_lane_and_never_passed():
    """Finding 9: a Docker-only test of a browser module (tests/test_ui_smoke_playwright.py) is named as excluded by
    the lane; it is neither run (`-m ui_browser` deselects it) nor counted green."""
    assert [(name, marked) for _, _, name, marked in sel.lane_tests(D, DOCKER)] == [
        (f"{D}::test_page", True), (f"{D}::test_docker", False)]
    only = sel.select(commits([D]), {**MODULES, D: DOCKER}, lines={D: {10}})
    assert only["selected"] == [] and only["excluded"] == [{"module": f"{D}::test_docker", "why": sel.DESELECTED}]
    assert only["args"] == [LIVING_TEST, ROOM_TEST]
    both = sel.select(commits([D]), {**MODULES, D: DOCKER}, lines={D: {6, 10}})
    assert both["selected"][0]["tests"] == [f"{D}::test_page"] and f"{D}::test_page" in both["args"]


def test_a_hunk_that_only_deletes_lines_runs_its_module_whole(tmp_path):
    """Findings 5 and 19: a deleted helper or fixture after a test leaves `@@ -a,n +b,0 @@`, whose line b is the last
    line of the test before it; that test alone would leave the deletion's other readers unrun."""
    repo = repository(tmp_path)
    text = UI + "\n\ndef test_a():\n    pass\n\n\ndef test_b():\n    assert helper()\n"
    (repo / K).write_text(text + "\n\ndef helper():\n    return True\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "base")
    base = git(repo, "rev-parse", "HEAD")
    (repo / K).write_text(text)
    git(repo, "commit", "-qam", "head")
    lines = sel.changed_lines(base, "HEAD", {K}, cwd=repo)
    assert lines == {K: {0}} and sel.changed_tests(K, text, lines[K]) is None
    plan = sel.select(commits([K]), {**MODULES, K: text}, lines=lines)
    [row] = plan["selected"]
    assert row["module"] == K and row["tests"] is None and K in plan["args"]
    # An ordinary edit inside one test keeps the narrow path.
    (repo / K).write_text(text.replace("assert helper()", "assert helper() is True"))
    git(repo, "commit", "-qam", "edit")
    edited = sel.changed_lines("HEAD~1", "HEAD", {K}, cwd=repo)
    assert sel.changed_tests(K, text, edited[K]) == [f"{K}::test_b"]


def test_a_hunk_that_removes_a_base_helper_beside_a_test_runs_its_module_whole(tmp_path):
    """Round 4 advisory (Carbon 5): `-4,2 +4` deletes a module-level line and rewrites a test's first line; the head
    side alone lies in that test, but the removed base line lies in no base test: the module runs whole."""
    repo = repository(tmp_path)
    (repo / K).write_text(UI + "\nY = 2\ndef test_a():\n    assert True\n\n\ndef test_b():\n    assert Y\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "base")
    base = git(repo, "rev-parse", "HEAD")
    text = UI + "\ndef test_a(page=None):\n    assert True\n\n\ndef test_b():\n    assert Y\n"
    (repo / K).write_text(text)
    git(repo, "commit", "-qam", "head")
    lines = sel.changed_lines(base, "HEAD", {K}, cwd=repo)
    assert 0 in lines[K] and sel.changed_tests(K, text, lines[K]) is None


def test_an_unparseable_mixed_module_runs_whole_and_its_collection_error_is_never_green(tmp_path, monkeypatch):
    """Decisions 3 and 10 (round 4, Astra 3, Neon 10): tests/test_client_views_browser.py, whose three Caddy tests
    the lane excludes, does not parse: it stays a candidate, passed whole (no per-test exclusion it could not
    compute), so its collection error makes the check partial, never GREEN on the sentinels alone."""
    mixed = "tests/test_client_views_browser.py"
    broken = UI + "\ndef test_independent_pages_sign_in_configure_and_use_the_real_core(:\n    pass\n"
    plan = sel.select(commits([mixed]), {**MODULES, mixed: broken}, lines={mixed: {4}})
    [row_] = [row_ for row_ in plan["selected"] if row_["module"] == mixed]
    assert row_["tests"] is None and mixed in plan["args"]
    assert not any(excluded["module"].startswith(mixed) for excluded in plan["excluded"])
    plan = {**plan, "base": "b" * 40, "head": "h" * 40}
    nodes = [LIVING, ROOM]
    path = evidence(tmp_path, monkeypatch, selected=nodes, rows=[r for node in nodes for r in whole(node)],
                    verdict={"verdict": "red", "flaky": [], "failed": [
                        {"nodeid": mixed, "attempts": [{"phase": "collect", "error_type": "CollectionError"}]}]})
    sessions = {path / "head": {"rows": [r for node in nodes for r in whole(node)], "selected": nodes,
                                "collection": [mixed], "collection_skips": []}}
    monkeypatch.setattr(pmb.pmc, "read_session", lambda d: {"running": [], **sessions.get(
        Path(d), {"rows": [], "selected": None})})
    found = pmb.judge(path, plan)
    assert found["outcome"] == "partial" and found["once"] == {mixed: "did not collect"}


def test_a_base_browser_module_the_head_cannot_parse_runs_whole_and_is_never_green(tmp_path, monkeypatch):
    """Round 6 (b2): a browser module at the base that the head breaks, with a marker its text test misses
    (`pytest.mark .ui_browser`), is planned whole, so its collection error makes the check partial or red, never
    GREEN on the sentinels alone."""
    (tmp_path / "repo").mkdir()
    repo = repository(tmp_path / "repo")
    (repo / "scripts").mkdir()
    shutil.copy(ROOT / "scripts/post_merge_browser_select.py", repo / "scripts")
    for module in ("tests/test_room_work_browser.py", "tests/test_living_cards_browser.py"):
        (repo / module).write_text(MODULES[module])
    broken = "tests/test_broken_browser.py"
    (repo / broken).write_text(UI + "\n\ndef test_a():\n    pass\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "base")
    base = git(repo, "rev-parse", "HEAD")
    (repo / broken).write_text("import pytest\npytestmark = [pytest.mark .ui_browser]\n\n\ndef test_a(:\n    pass\n")
    git(repo, "commit", "-qam", "head")
    out = tmp_path / "plan.json"
    done = subprocess.run([sys.executable, "-I", "-S", str(repo / "scripts/post_merge_browser_select.py"), "plan",
                           "--base", base, "--head", "HEAD", "--out", str(out)], cwd=repo, capture_output=True,
                          text=True, timeout=120)
    assert done.returncode == 0, done.stdout + done.stderr
    plan = {**json.loads(out.read_text()), "base": "b" * 40, "head": "h" * 40}
    assert plan["applicable"] and broken in plan["args"], plan["args"]
    assert [row_["tests"] for row_ in plan["selected"] if row_["module"] == broken] == [None]
    nodes = [LIVING, ROOM]
    path = evidence(tmp_path / "run", monkeypatch, selected=nodes, rows=[r for node in nodes for r in whole(node)],
                    verdict={"verdict": "red", "flaky": [], "failed": [
                        {"nodeid": broken, "attempts": [{"phase": "collect", "error_type": "CollectionError"}]}]})
    sessions = {path / "head": {"rows": [r for node in nodes for r in whole(node)], "selected": nodes,
                                "collection": [broken], "collection_skips": []}}
    monkeypatch.setattr(pmb.pmc, "read_session", lambda d: {"running": [], **sessions.get(
        Path(d), {"rows": [], "selected": None})})
    found = pmb.judge(path, plan)
    assert found["outcome"] == "partial" and found["once"] == {broken: "did not collect"}


def test_non_ascii_and_control_character_paths_are_read_whole_and_outputs_stay_one_line(tmp_path):
    """Decision 5 (round 4, Astra 5): git quotes such paths in its display format; `-z` reads them as they are, and
    a path written into a single-line Actions output is escaped."""
    repo = repository(tmp_path)
    (repo / "web").mkdir()
    (repo / "web/a.css").write_text("0")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "base")
    base = git(repo, "rev-parse", "HEAD")
    (repo / "web/настройки.css").write_text("1")
    # A TAB is not a legal Windows file name character: record it in the index only (plumbing), so git holds the exact
    # path bytes the parser must read whole on every platform without the file ever existing on disk.
    blob = subprocess.run(["git", "hash-object", "-w", "--stdin"], cwd=repo, input="1", check=True,
                          capture_output=True, text=True).stdout.strip()
    git(repo, "add", ".")
    git(repo, "update-index", "--add", "--cacheinfo", f"100644,{blob},web/tab\tname.css")
    git(repo, "commit", "-qm", "head")
    [commit] = sel.window_commits(base, "HEAD", cwd=repo)
    assert commit["files"] == ["web/tab\tname.css", "web/настройки.css"]
    applies, why = sel.applicability(["web/a\nb=1.css"], {})
    assert applies and "\n" not in sel.one_line(why) and sel.one_line(why).endswith("web/a\\nb=1.css")
    assert sel.one_line("a\\b\x01") == "a\\\\b\\x01"


def test_the_scope_command_runs_on_a_bare_interpreter():
    done = subprocess.run([sys.executable, "-I", "-S", str(ROOT / "scripts/post_merge_browser_select.py"), "scope",
                           "--base", "HEAD", "--head", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0 and done.stdout.startswith("applicable=false\nwhy="), done.stdout + done.stderr


# --- the outcome ------------------------------------------------------------------------------------------------

PLAN = {"applicable": True, "base": "b" * 40, "head": "h" * 40, "args": [],
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
    monkeypatch.setattr(pmb.pmc, "read_session", lambda directory: {"running": [], **sessions.get(
        Path(directory), {"rows": [], "selected": None})})
    return tmp_path


def test_green_only_when_every_selected_node_ran_and_passed(tmp_path, monkeypatch):
    found = pmb.judge(evidence(tmp_path, monkeypatch, selected=["a", "b"], rows=whole("a") + whole("b"),
                               verdict={"verdict": "green", "failed": [], "flaky": []}), PLAN)
    assert found["outcome"] == "green" and found["passed"] == 2 and found["partial"] == [] and found["not_run"] == []


def test_a_dropped_changed_or_same_named_module_is_checked_partially(tmp_path, monkeypatch):
    """Findings 8 and 20: a tier-1 or tier-2 module the budget or the cap dropped is not checked: CHECKED PARTIALLY,
    named. A dropped tier-3 module or a hub is only named, and a plan within its budget stays green."""
    green = {"verdict": "green", "failed": [], "flaky": []}
    plan = sel.select(commits(["web/room_work.css"], ["tests/test_composer_capsules_browser.py"]), MODULES,
                      table=TABLE, budget_seconds=70)
    assert {(row["module"], row["tier"]) for row in plan["dropped"]} == {
        ("tests/test_room_work_browser.py", 2)} and plan["args"] == [LIVING_TEST, ROOM_TEST,
                                                                     "tests/test_composer_capsules_browser.py"]
    plan = {**plan, "base": "b" * 40, "head": "h" * 40}
    nodes = [LIVING, ROOM, "tests/test_composer_capsules_browser.py::test_c"]
    found = pmb.judge(evidence(tmp_path / "d", monkeypatch, selected=nodes,
                               rows=[r for node in nodes for r in whole(node)], verdict=green), plan)
    assert found["outcome"] == "partial" and found["dropped"] == [
        "tests/test_room_work_browser.py: over the time budget"]
    assert "a changed or same-named browser module was not checked (budget or cap)" in found["partial"]
    text = "\n".join(pmb.summary_lines(plan, found, [], {}, {}))
    assert "CHECKED PARTIALLY" in text and "- tests/test_room_work_browser.py: over the time budget" in text
    third = {**plan, "dropped": [{"module": "tests/test_x_browser.py", "tier": 3, "why": "over the time budget"}]}
    named = pmb.judge(evidence(tmp_path / "t", monkeypatch, selected=nodes,
                               rows=[r for node in nodes for r in whole(node)], verdict=green), third)
    assert named["outcome"] == "green" and named["dropped"] == []


def test_a_requested_collector_skipped_or_never_collected_is_partial_beside_a_passing_sibling(tmp_path, monkeypatch):
    """Findings 1 and 9: ci_retry can say green when one selected module is skipped at collection and its sibling
    passes; the session's selection then omits it. What was requested and never collected is CHECKED PARTIALLY;
    what the plan excludes by the lane is not."""
    green = {"verdict": "green", "failed": [], "flaky": []}
    plan = {**PLAN, "args": ["tests/test_a_browser.py", "tests/test_b_browser.py", "tests/test_c_browser.py::test_y"],
            "excluded": [{"module": "tests/test_c_browser.py::test_docker", "why": sel.DESELECTED}]}
    node = "tests/test_a_browser.py::test_x"
    path = evidence(tmp_path / "s", monkeypatch, selected=[node], rows=whole(node), verdict=green)
    sessions = {path / "head": {"rows": whole(node), "selected": [node], "collection_skips": ["tests/test_b_browser.py"],
                                "collection": []}}
    monkeypatch.setattr(pmb.pmc, "read_session", lambda d: {"running": [], **sessions.get(
        Path(d), {"rows": [], "selected": None})})
    found = pmb.judge(path, plan)
    assert found["outcome"] == "partial" and found["collection_skipped"] == ["tests/test_b_browser.py"]
    assert found["not_collected"] == {"tests/test_c_browser.py::test_y": "the session collected no test it names"}
    assert {"a requested collector was skipped at collection", "a requested test was never collected"} <= set(
        found["partial"])
    text = "\n".join(pmb.summary_lines(plan, found, [], {}, {}))
    assert "Not checked: skipped at collection:\n- `tests/test_b_browser.py`" in text
    assert "Excluded by the lane (neither checked nor partial):\n- `tests/test_c_browser.py::test_docker`" in text
    # Every requested argument collected and passed: green, the lane's exclusion named only.
    sessions[path / "head"] = {"rows": whole(node), "selected": [node], "collection_skips": [], "collection": []}
    assert pmb.judge(path, {**plan, "args": ["tests/test_a_browser.py"]})["outcome"] == "green"
    # A class skipped at collection inside a module whose other tests ran is named too.
    sessions[path / "head"] = {"rows": whole(node), "selected": [node], "collection": [],
                               "collection_skips": ["tests/test_a_browser.py::TestSkipped"]}
    assert pmb.judge(path, {**plan, "args": ["tests/test_a_browser.py"]})["collection_skipped"] == [
        "tests/test_a_browser.py::TestSkipped"]


def test_an_unfinished_retry_is_partial_with_its_first_failure_and_never_routed(tmp_path, monkeypatch):
    """Findings 2 and 10: a real failure whose retry was killed (InFlight, a start-only journal) has one completed
    observation: CHECKED PARTIALLY, the first failure shown, nothing handed over. Two completed failures stay RED."""
    verdict = {"verdict": "red", "flaky": [], "failed": [
        {"nodeid": "cut", "attempts": [{"phase": "call", "error_type": "AssertionError"},
                                       {"phase": "crash", "error_type": "InFlight"}]},
        {"nodeid": "red", "attempts": [{"phase": "call", "error_type": "AssertionError"},
                                       {"phase": "setup", "error_type": "TimeoutError"}]}]}
    found = pmb.judge(evidence(tmp_path, monkeypatch, selected=["cut", "red"],
                               rows=[row("cut", "failed"), row("red", "failed")], rounds=[[row("red", "failed")]],
                               verdict=verdict), PLAN)
    assert found["red"] == ["red"] and found["outcome"] == "red"
    assert found["once"] == {"cut": "failed (call: AssertionError); its retry did not finish (crash: InFlight)"}
    assert "a failure got no second completed observation" in found["partial"]
    alone = {**verdict, "failed": verdict["failed"][:1]}
    cut = pmb.judge(evidence(tmp_path / "c", monkeypatch, selected=["cut"], rows=[row("cut", "failed")],
                             verdict=alone), PLAN)
    assert cut["outcome"] == "partial" and cut["red"] == []
    text = "\n".join(pmb.summary_lines(PLAN, cut, [], {}, {}))
    assert "- `cut`: failed (call: AssertionError); its retry did not finish" in text


def test_a_pass_counts_only_whole_in_its_attempt_and_an_interrupted_test_is_named(tmp_path, monkeypatch):
    """Decision 3 (A2): one failure passed its retry while another test was interrupted in teardown and stayed in the
    session's running list. Its passing call is no completed test: CHECKED PARTIALLY, named, never counted passed."""
    first = [row("flaky", "failed"), row("cut", phase="setup"), row("cut")]  # its teardown never reported
    found = pmb.judge(evidence(tmp_path, monkeypatch, selected=["flaky", "cut"], rows=first, rounds=[whole("flaky")],
                               verdict={"verdict": "flaky", "failed": [], "flaky": [{"nodeid": "flaky"}]}), PLAN)
    assert found["outcome"] == "partial" and found["unfinished"] == ["cut"] and found["passed"] == 0
    assert found["flaky"] == ["flaky"] and "a selected test did not run whole (setup, call and teardown)" in found[
        "partial"]
    text = "\n".join(pmb.summary_lines(PLAN, found, [], {}, {}))
    assert "Not checked: did not run whole (setup, call and teardown):\n- `cut`" in text
    # The same test whole: GREEN with FLAKY tests.
    done = pmb.judge(evidence(tmp_path / "w", monkeypatch, selected=["flaky", "cut"],
                              rows=[row("flaky", "failed"), *whole("cut")], rounds=[whole("flaky")],
                              verdict={"verdict": "flaky", "failed": [], "flaky": [{"nodeid": "flaky"}]}), PLAN)
    assert done["outcome"] == "flaky" and done["unfinished"] == [] and done["passed"] == 1


def test_a_failure_the_time_limit_left_unretried_is_run_again_and_a_second_failure_is_handed_over(
        tmp_path, monkeypatch):
    """Decision 8 (Opus): a session that reaches its time limit leaves ci_retry no lane block, so it retries nothing.
    The confirm step runs each completed single failure once more: a second failure is RED and handed over beside
    what stays partial, a whole pass is FLAKY, an unclear answer keeps it reported only."""
    reason = "3 collected test(s) never ran, and no lane block or crash record names them"
    verdict = {"verdict": "red", "flaky": [], "reason": reason, "failed": [
        {"nodeid": node, "attempts": [{"phase": "call", "error_type": "AssertionError"}]} for node in ("a", "b", "c")]
        + [{"nodeid": "k", "attempts": [{"phase": "crash", "error_type": "InFlight"}]}]}
    path = evidence(tmp_path, monkeypatch, selected=["a", "b", "c", "k", "never"],
                    rows=[row(node, "failed") for node in "abc"], verdict=verdict)
    judged = pmb.judge(path, PLAN)
    assert judged["unconfirmed"] == ["a", "b", "c"] and judged["red"] == []  # a killed test has no completed failure
    asked = []
    monkeypatch.setattr(pmb.pmc, "recheck_ids", lambda ids, directory, **options: asked.append(ids) or {
        "a": "failed", "b": "passed", "c": "unknown"})
    monkeypatch.setattr(pmb, "exact_answers", lambda directory, answers, at: ({}, {}))
    (path / pmb.CONFIRM).write_text(json.dumps(pmb.confirm(path, PLAN, judged)))
    assert asked == [["a", "b", "c"]]
    found = pmb.judge(path, PLAN)
    assert found["red"] == ["a"] and found["confirmed"] == ["a"] and found["flaky"] == ["b"]
    assert found["outcome"] == "red" and found["partial"] and set(found["once"]) == {"c", "k"}
    assert found["not_run"] == ["k", "never"]
    text = "\n".join(pmb.summary_lines(PLAN, found, [], {}, {}))
    assert "RED, AND CHECKED PARTIALLY" in text and "- `a` (its second failure came from the confirm step" in text
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "output"))
    (path / "plan.json").write_text(json.dumps(PLAN))
    pmb.command_outcome(SimpleNamespace(evidence_dir=str(path), plan=str(path / "plan.json")),
                        environ={"GITHUB_OUTPUT": str(tmp_path / "output")})
    routed = pmb.decode((tmp_path / "output").read_text().split("routing=", 1)[1].strip())
    assert routed["red"] == ["a"] and routed["outcome"] == "red"  # handed over although the run was partial
    assert pmb.confirm(path, PLAN, {"unconfirmed": []}) == {"format": 1, "head": "h" * 40, "ids": [], "results": {},
                                                             "inconclusive": {}}


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
    outside = pmb.judge(tmp_path, {"applicable": False, "reason": "r"})  # a manual dispatch outside the scope
    assert outside["outcome"] == "not-applicable" and outside["outcome"] not in pmb.PASSING
    assert "\n".join(pmb.summary_lines({"reason": "r"}, outside, [], {}, {})).endswith(
        "No browser test ran, so nothing here is green.")


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


def test_without_a_verdict_each_failure_counts_its_observations_per_attempt(tmp_path, monkeypatch):
    """Decision 4 (round 4, Astra 4): with the aggregate verdict lost, two completed failures in two attempts stay
    RED and routed, a failure its retry passed whole is FLAKY, and only a failure observed once goes to the confirm
    step; its answer never touches the confirmed red (no third run)."""
    first = [row("red", "failed"), row("flaky", "failed"), row("once", "failed")]
    retry = [row("red", "failed"), *whole("flaky")]
    path = evidence(tmp_path, monkeypatch, selected=["red", "flaky", "once"], rows=first, rounds=[retry], verdict=None)
    found = pmb.judge(path, PLAN)
    assert found["red"] == ["red"] and found["flaky"] == ["flaky"] and found["unconfirmed"] == ["once"]
    assert found["once"] == {"once": "failed; its retry never ran"} and found["outcome"] == "red"
    asked = []
    monkeypatch.setattr(pmb.pmc, "recheck_ids", lambda ids, directory, **options: asked.append(ids) or dict.fromkeys(
        ids, "passed"))
    monkeypatch.setattr(pmb, "exact_answers", lambda directory, answers, at: ({}, {}))
    (path / pmb.CONFIRM).write_text(json.dumps(pmb.confirm(path, PLAN, found)))
    assert asked == [["once"]]
    again = pmb.judge(path, PLAN)
    assert again["red"] == ["red"] and again["flaky"] == ["flaky", "once"] and again["outcome"] == "red"


def test_a_failure_the_confirm_step_resolved_is_neither_a_missing_sentinel_nor_an_unexplained_red(
        tmp_path, monkeypatch):
    """Decision 11 (round 4, Neon 11, Carbon 6): a sentinel that failed once at the time limit and passed whole in
    the confirm step is FLAKY, not "did not complete"; ci_retry's red verdict is explained by it. An accepted
    sentinel completed too."""
    plan = {**PLAN, "sentinels": [{"scenario": ROOM, "test": ROOM_TEST, "present": True, "guards": []}]}
    verdict = {"verdict": "red", "flaky": [], "reason": "the session reached its time limit",
               "failed": [{"nodeid": ROOM, "attempts": [{"phase": "call", "error_type": "AssertionError"}]}]}
    path = evidence(tmp_path, monkeypatch, selected=[ROOM], rows=[row(ROOM, "failed")], verdict=verdict)
    (path / pmb.CONFIRM).write_text(json.dumps({"results": {ROOM: "passed"}}))
    found = pmb.judge(path, plan)
    assert found["flaky"] == [ROOM] and found["missing_sentinels"] == [] and found["partial"] == []
    assert found["outcome"] == "flaky"
    accepted = {"verdict": "accepted", "flaky": [], "failed": [], "accepted": [{"nodeid": ROOM}]}
    path = evidence(tmp_path / "a", monkeypatch, selected=[ROOM], rows=[row(ROOM, "failed")],
                    rounds=[[row(ROOM, "failed")]], verdict=accepted)
    assert pmb.judge(path, plan)["missing_sentinels"] == []


def test_a_public_id_shared_by_two_tests_is_no_pass_of_one(tmp_path, monkeypatch):
    """Round 4 advisory (Neon 14): the judge applies the legs' `_shared` rule: two tests behind one public id pass
    as neither."""
    found = pmb.judge(evidence(tmp_path, monkeypatch, selected=["a", "a"], rows=whole("a") + whole("a"),
                               verdict={"verdict": "green", "failed": [], "flaky": []}), PLAN)
    assert found["outcome"] == "partial" and found["unfinished"] == ["a"] and found["passed"] == 0


def test_an_expected_failure_is_neither_a_skip_nor_a_pass(tmp_path, monkeypatch):
    """Round 4 advisory (Carbon 6): an XFAIL row is reported skipped with `wasxfail`; it ran as its marker expects,
    so it is listed, never an unpermitted skip."""
    rows = [row("x", phase="setup"), {**row("x", "skipped"), "wasxfail": "known"}, row("x", phase="teardown")]
    found = pmb.judge(evidence(tmp_path, monkeypatch, selected=["x"], rows=rows,
                               verdict={"verdict": "green", "failed": [], "flaky": []}), PLAN)
    assert found["skipped"] == [] and found["xfailed"] == ["x"] and found["outcome"] == "green"
    assert "XFAIL (failed as their marker expects):\n- `x`" in "\n".join(pmb.summary_lines(PLAN, found, [], {}, {}))


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
    assert found["permitted"] == {pinch: "applies no synthetic touch pinch"}
    text = "\n".join(pmb.summary_lines(PLAN, found, [], {}, {}))  # decision 10: the permitted skips are listed
    assert ("Skipped as the lane permits (tests/browser_lane.py PLATFORM_SKIPS):\n"
            f"- `{pinch}`: applies no synthetic touch pinch") in text
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


def test_a_real_session_collects_the_sentinel_cases_first_and_each_case_once(tmp_path):
    """pytest 9 keeps a module's order where a test and its module are both arguments; the plugin moves the exact
    WebKit case to the front of the collected items, and no case is collected twice."""
    tree, launcher = suite(tmp_path)
    module = LIVING.split("::", 1)[0]
    name = LIVING_TEST.split("::")[-1]
    (tree / module).write_text(UI + "\n\ndef test_first():\n    pass\n\n\n"
                               f"@pytest.mark.parametrize('engine', ['chromium', 'webkit'])\ndef {name}(engine):\n"
                               "    pass\n", encoding="utf-8")
    done = subprocess.run([*launcher, "-p", "scripts.post_merge_browser", "--co", "-q", "-m", "ui_browser",
                           LIVING_TEST, module], cwd=tree, capture_output=True, text=True, timeout=120)
    collected = [line for line in done.stdout.splitlines() if line.startswith("tests/") and "::" in line]
    assert collected == [LIVING, f"{module}::test_first", LIVING.replace("webkit", "chromium")], done.stdout + done.stderr


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
    monkeypatch.setattr(pmb.pmc, "add_worktree", lambda tree, sha: (tree.mkdir(), added.append(sha)))
    monkeypatch.setattr(pmb.pmc, "remove_worktree", lambda tree: removed.append(tree))
    monkeypatch.setattr(pmb.pmc, "_git", lambda *args, cwd=None: "0" * 40)  # some other commit
    stale = pmb.compare(tmp_path, PLAN, red, run=lambda args, directory: 0 / 0)
    assert stale["results"] == {red[0]: "unknown"} and "not at the window's base" in stale["error"]
    assert added == ["b" * 40] and len(removed) == 1 and not removed[0].parent.exists()  # removed in finally


def test_a_base_that_needs_the_protected_file_held_off_disk_is_never_run_and_blames_nobody(tmp_path, monkeypatch):
    """Decision 5 (A9, Carbon): the base would need a skip-worktree index, which the browser fixtures refuse in setup
    (CandidateError): every red is unknown, so not attributed, never pre-existing. A base without the file at all is
    an ordinary checkout and runs."""
    red = ["tests/test_living_cards_browser.py::test_new[webkit]"]
    monkeypatch.setattr(pmb.pmc, "_blob", lambda sha, path: "1")
    monkeypatch.setattr(pmb, "protected_projection", lambda head, base, git: (
        "different", "`tests/test_repo_health_smoke.py` differs from the head"))
    monkeypatch.setattr(pmb.pmc, "add_worktree", lambda *args, **options: 0 / 0)
    found = pmb.compare(tmp_path, PLAN, red, run=lambda args, directory: 0 / 0)
    assert found["results"] == {red[0]: "unknown"} and "the browser fixtures refuse" in found["inconclusive"][red[0]]
    one = {**PLAN, "commits": PLAN["commits"][:1]}
    [verdict] = pmb.attribute(one, red, found)
    assert verdict["verdict"] == "unattributed" and verdict["at_base"] == "unknown"
    started = []

    def checkout(tree, sha):
        started.append(sha)
        raise OSError("stop after the checkout began")

    monkeypatch.setattr(pmb, "protected_projection", lambda head, base, git: ("absent", "gone at this revision"))
    monkeypatch.setattr(pmb, "comparable", lambda files, **options: (red, {}, None))
    monkeypatch.setattr(pmb.pmc, "add_worktree", checkout)
    monkeypatch.setattr(pmb.pmc, "remove_worktree", lambda tree: None)
    assert pmb.compare(tmp_path, PLAN, red)["results"] == {red[0]: "unknown"} and started == ["b" * 40]


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


def test_a_test_that_failed_here_is_never_rechecked_nor_retired_by_this_run(tmp_path, monkeypatch):
    """Decision 12 (round 4, Neon 12): a test that failed in this run's head session (FLAKY, ACCEPTED) is no
    recheck's to settle, and the hand-over never closes or narrows an issue for it, whatever answer it carries."""
    issue = tracking_issue(5, {"os": "browser", "kind": "unattributed", "ids": ["accepted", "d", "flaky"]})
    asked = []
    monkeypatch.setattr(pmb, "same_run_passes", lambda directory, ids: {})
    monkeypatch.setattr(pmb.pmc, "recheck_ids", lambda ids, directory, **options: asked.append(ids) or dict.fromkeys(
        ids, "passed"))
    monkeypatch.setattr(pmb, "exact_answers", lambda directory, answers, at: ({}, {}))
    judged = {"red": [], "once": {}, "failed_then_skipped": [], "flaky": ["flaky"], "accepted": ["accepted"]}
    assert pmb.recheck(tmp_path, PLAN, judged, FakeGitHub(issues=[issue]))["rechecked"] == {"d": "passed"}
    assert asked == [["d"]] and pmb.failed_here(judged) == ["accepted", "flaky"]
    github = FakeGitHub(issues=[issue], label=True)
    data = {**DATA, "red": [], "verdicts": [], "flaky": ["flaky"], "failed_here": ["accepted", "flaky"],
            "rechecked": {"accepted": "passed", "d": "passed"}, "same_run": {"flaky": "passed"}}
    record = pmb.hand_over(data, github, run_url="run", retire=True, ancestor=lambda older, newer: True)
    assert record["closed"] == [] and record["narrowed"] == [{"issue": 5, "ids": ["d"]}]
    assert routing.parse_machine_block(github.issues[5]["body"])["ids"] == ["accepted", "flaky"]


# --- attribution ------------------------------------------------------------------------------------------------

def test_a_merge_is_named_only_for_a_window_of_one_merge_whose_base_ran_the_red_green():
    node = "tests/test_a_browser.py::test_x"
    one = {**PLAN, "commits": PLAN["commits"][:1], "head_parent": "b" * 40}
    [named] = pmb.attribute(one, [node], {"results": {node: "passed"}})
    assert named == {"id": node, "verdict": "culprit", "culprit": "1" * 40, "last_good": "b" * 40,
                     "at_base": "passed", "window": "b" * 40}
    # Finding 6: absent at the base proves no merge (a renamed test that already failed): not attributed.
    [absent] = pmb.attribute(one, [node], {"results": {node: "absent"}})
    assert absent["verdict"] == "unattributed" and absent["why"] == "not present at the base"
    assert absent["candidates"] == ["1" * 40] and absent["at_base"] == "absent"
    [pre] = pmb.attribute(PLAN, [node], {"results": {node: "failed"}})
    assert pre == {"id": node, "verdict": "pre-existing", "at_base": "failed", "window": "b" * 40}
    # Finding 11: a failure the base did not observe for this test is inconclusive, never pre-existing.
    [unsure] = pmb.attribute(one, [node], {"results": {node: "unknown"},
                                           "inconclusive": {node: "did not collect at the base"}})
    assert unsure["verdict"] == "unattributed" and unsure["at_base"] == "unknown"
    assert unsure["why"] == "its answer at the base was inconclusive: did not collect at the base"
    # Two merges: the base proves only that the window introduced it. The one that touched its files comes first.
    [many] = pmb.attribute(PLAN, [node], {"results": {node: "passed"}})
    assert many["verdict"] == "unattributed" and many["candidates"] == ["1" * 40] and many["at_base"] == "passed"
    untouched = {**PLAN, "commits": [{"sha": "1" * 40, "files": ["x.py"]}, {"sha": "2" * 40, "files": ["y.py"]}]}
    [anyone] = pmb.attribute(untouched, [node], {"results": {node: "absent"}})
    assert anyone["candidates"] == ["1" * 40, "2" * 40] and anyone["why"] == "not present at the base"
    for answer in ("flaky", "unknown", "skipped", None):
        [unsure] = pmb.attribute(one, [node], {"results": {node: answer} if answer else {}})
        assert unsure["verdict"] == "unattributed" and unsure["at_base"] == "unknown", answer
    [locked] = pmb.attribute(one, [node], {"results": {node: "passed"}, "dependencies_differ": True})
    assert locked["verdict"] == "unattributed" and locked["at_base"] == "dependencies" and "uv.lock" in locked["why"]


def test_a_merge_is_named_only_when_its_first_parent_is_the_windows_base():
    """Round 4, Astra 2: a manual dispatch whose base is a sibling of the head's parent lists one commit in
    `base..head`, yet that commit's first parent is not the base: the base's pass proves nothing about the merge.
    Only `head^1 == base` names it; the summary says what the writer records."""
    node = "tests/test_a_browser.py::test_x"
    sibling = {**PLAN, "commits": PLAN["commits"][:1], "head_parent": "c" * 40}
    [verdict] = pmb.attribute(sibling, [node], {"results": {node: "passed"}})
    assert verdict["verdict"] == "unattributed" and verdict["why"] == (
        "its one commit's first parent is not the window's base") and verdict["candidates"] == ["1" * 40]
    unknown = {**PLAN, "commits": PLAN["commits"][:1]}  # a plan without the parent never names a merge
    assert pmb.attribute(unknown, [node], {"results": {node: "passed"}})[0]["verdict"] == "unattributed"
    judged = {**pmb.judge(Path("/nonexistent"), {}), "outcome": "red", "red": [node]}
    text = "\n".join(pmb.summary_lines(sibling, judged, [verdict], {}, {}))
    assert "introduced by" not in text and "not attributed (its one commit's first parent is not" in text


def test_a_masked_id_never_names_a_merge():
    """Round 6 (c1): the evidence keeps redacted node ids (`***`), so two different raw parameters, the base's and the
    head's, can share one public id: the base's answer for it may be another case's. Such a red is not attributed and
    lists its candidates, whatever the base answered; an unmasked id still names the one merge."""
    masked, plain = "tests/test_a_browser.py::test_x[***]", "tests/test_a_browser.py::test_x[case]"
    one = {**PLAN, "commits": PLAN["commits"][:1], "head_parent": "b" * 40}
    for answer in ("passed", "failed", "absent"):
        [verdict] = pmb.attribute(one, [masked], {"results": {masked: answer}})
        assert verdict["verdict"] == "unattributed" and verdict["candidates"] == ["1" * 40], answer
        assert verdict["at_base"] == "unknown" and "masked" in verdict["why"], answer
    assert pmb.attribute(one, [plain], {"results": {plain: "passed"}})[0]["verdict"] == "culprit"


def test_the_selector_reads_the_heads_first_parent(tmp_path):
    repo = repository(tmp_path)
    (repo / "f").write_text("0")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "root")
    root = git(repo, "rev-parse", "HEAD")
    (repo / "f").write_text("1")
    git(repo, "commit", "-qam", "next")
    assert sel.first_parent("HEAD", cwd=repo) == root and sel.first_parent(root, cwd=repo) == ""


def test_candidates_say_when_no_merge_touched_the_files_that_selected_the_test():
    """Round 4, Astra advisory 1: with no touched file matching, every commit is listed, and the summary and the
    issue say so instead of claiming those commits touched the selecting files."""
    node = "tests/test_a_browser.py::test_x"
    untouched = {**PLAN, "commits": [{"sha": "1" * 40, "files": ["x.py"]}, {"sha": "2" * 40, "files": ["y.py"]}]}
    [verdict] = pmb.attribute(untouched, [node], {"results": {node: "absent"}})
    assert verdict["fallback"] is True and verdict["candidates"] == ["1" * 40, "2" * 40]
    [matched] = pmb.attribute(PLAN, [node], {"results": {node: "absent"}})
    assert "fallback" not in matched
    judged = {**pmb.judge(Path("/nonexistent"), {}), "outcome": "red", "red": [node]}
    text = "\n".join(pmb.summary_lines(untouched, judged, [verdict], {}, {}))
    assert "none touched the files that selected it, so all of the window's" in text
    github = FakeGitHub(label=True)
    pmb.hand_over({**DATA, "red": [node], "verdicts": [verdict]}, github, run_url="run", retire=True)
    [issue] = github.issues.values()
    assert ("candidate merges (none touched the files that selected it, so all of its window's) `" + "1" * 40
            in issue["body"])


def test_a_base_failure_needs_a_failed_report_of_that_test(tmp_path, monkeypatch):
    """Finding 11: `recheck_ids` answers `failed` for every id of a module that did not collect, or of a test still
    running; the browser check keeps those inconclusive (unknown) and a real failure pre-existing. The shared
    recheck semantics the legs use stay as they are."""
    reds = ["tests/test_x_browser.py::test_red", "tests/test_x_browser.py::test_new", "tests/test_y_browser.py::t",
            "tests/test_z_browser.py::test_hang"]
    session = {"rows": [row(reds[0], "failed"), row(reds[3], "failed", "crash")], "running": [], "final": False,
               "collection": ["tests/test_y_browser.py"], "selected": None}
    monkeypatch.setattr(pmb.pmc, "read_session", lambda directory: session)
    found, why = pmb.exact_answers(tmp_path, dict.fromkeys(reds, "failed") | {reds[1]: "absent"}, at="at the base")
    assert found == {reds[2]: "unknown", reds[3]: "unknown"}
    assert why == {reds[2]: "did not collect at the base", reds[3]: "did not finish at the base"}
    monkeypatch.setattr(pmb.pmc, "_blob", lambda sha, path: "1")
    monkeypatch.setattr(pmb, "protected_projection", lambda head, base, git: ("same", ""))
    monkeypatch.setattr(pmb, "comparable", lambda files, **options: ([reds[0], reds[2], reds[3]], {}, None))
    monkeypatch.setattr(pmb.pmc, "add_worktree", lambda tree, sha: tree.mkdir())
    monkeypatch.setattr(pmb.pmc, "remove_worktree", lambda tree: None)
    monkeypatch.setattr(pmb.pmc, "_git", lambda *args, cwd=None: "b" * 40)
    monkeypatch.setattr(pmb.pmc, "recheck_ids", lambda ids, directory, **options: dict.fromkeys(ids, "failed"))
    compared = pmb.compare(tmp_path, PLAN, [reds[0], reds[2], reds[3]])
    assert compared["results"] == {reds[0]: "failed", reds[2]: "unknown", reds[3]: "unknown"}
    verdicts = {row["id"]: row for row in pmb.attribute(PLAN, [reds[0], reds[2]], compared)}
    assert verdicts[reds[0]]["verdict"] == "pre-existing" and verdicts[reds[2]]["verdict"] == "unattributed"


def test_only_the_exact_node_answers_at_the_base_and_in_a_recheck(tmp_path, monkeypatch):
    """Decision 4 (A3): `--ci-listed` keeps a test's parametrizations under its plain id and `recheck_ids` aggregates
    them; a parametrized relative never counts. A pass needs the exact node, once and whole; a failure its own report;
    a fixture's CandidateError in setup is a refusal of the checkout, not the test's failure."""
    ok, bad, refused, twice = ("tests/test_x_browser.py::test_ok", "tests/test_x_browser.py::test_bad",
                               "tests/test_x_browser.py::test_refused", "tests/test_x_browser.py::test_twice")
    rows = (whole(f"{ok}[base-only]") + [row(f"{bad}[base-only]", "failed")]
            + [{**row(refused, "failed", "setup"), "error_type": "CandidateError"}] + whole(twice) + whole(twice))
    session = {"rows": rows, "running": [], "final": True, "collection": [],
               "selected": [f"{ok}[base-only]", f"{bad}[base-only]", refused, twice, twice]}
    monkeypatch.setattr(pmb.pmc, "read_session", lambda directory: session)
    answers = {ok: "passed", bad: "failed", refused: "failed", twice: "passed"}
    found, why = pmb.exact_answers(tmp_path, answers, at="at the base")
    assert found == {ok: "absent", bad: "absent", refused: "unknown", twice: "unknown"}
    assert why == {refused: "its fixture refused the checkout at the base (CandidateError)",
                   twice: "that exact test did not run whole, once, at the base"}
    one = {**PLAN, "commits": PLAN["commits"][:1]}
    verdicts = {row["id"]: row for row in pmb.attribute(one, list(answers), {"results": {**answers, **found},
                                                                             "inconclusive": why})}
    assert {node: row["verdict"] for node, row in verdicts.items()} == dict.fromkeys(answers, "unattributed")
    # A module that did not collect at the base selects nothing: its `failed` is that, never absent.
    session.update(rows=[], selected=[], collection=["tests/test_x_browser.py"])
    assert pmb.exact_answers(tmp_path, {bad: "failed"}, at="at the base") == (
        {bad: "unknown"}, {bad: "did not collect at the base"})
    session["collection"] = []
    # The exact node itself: a whole pass names the one merge, its own failure is pre-existing.
    session.update(rows=whole(ok) + [row(bad, "failed")], selected=[ok, bad])
    assert pmb.exact_answers(tmp_path, {ok: "passed", bad: "failed"}, at="at the base") == ({}, {})
    # The open issues' recheck takes the same exact answers.
    issue = tracking_issue(5, {"os": "browser", "kind": "unattributed", "ids": [ok]})
    session.update(rows=whole(f"{ok}[other]"), selected=[f"{ok}[other]"])
    monkeypatch.setattr(pmb.pmc, "recheck_ids", lambda ids, directory, **options: dict.fromkeys(ids, "passed"))
    rechecked = pmb.recheck(tmp_path, PLAN, {"red": [], "once": {}, "failed_then_skipped": []},
                            FakeGitHub(issues=[issue]))
    assert rechecked["rechecked"] == {ok: "absent"}


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
                      "window": "b" * 40, "candidates": ["1" * 40]},
                     {"id": "tests/test_b_browser.py::t", "verdict": "unattributed", "at_base": "unknown",
                      "window": "b" * 40, "candidates": ["1" * 40, "2" * 40]}]}


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
    assert block["window"] == dict.fromkeys(block["ids"], "b" * 40)
    assert ("- `tests/test_a_browser.py::test_x`: passed at its window's base `bbbbbbbbbbbb`; candidate merges `"
            + "1" * 40 + "`" in issue["body"])
    assert "- `tests/test_b_browser.py::t`: not settled at its window's base `bbbbbbbbbbbb`; candidate" in issue["body"]
    # Advisories 5 and 16: the candidates are the merges that touched the files that selected the test, no more.
    assert ("the merges that touched the files that selected the test, or all of them where none did. The cause may "
            "lie elsewhere." in issue["body"])
    assert github.issues[7]["state"] == "open" and 7 not in record["updated"] and record["untracked"] == []
    # A Linux leg that sees the same test pass leaves the browser issue alone.
    later = routing.route(github, classification={"failed": []}, attribution={"verdicts": []}, head="f" * 40,
                          run_url="run", recheck=lambda ids: dict.fromkeys(ids, "passed"), system="linux")
    assert created not in later["closed"] and github.issues[created]["state"] == "open"


def test_an_issue_keeps_each_tests_own_notes_as_tests_join_and_leave_it():
    """Finding 18 and advisory 11: a test that joins an issue from a later window keeps its own base and comparison;
    the rows render each against its own commit."""
    first = {"format": 1, "os": "browser", "kind": "unattributed", "ids": ["a", "b"], "at_base": {"a": "passed", "b": "x"},
             "window": {"a": "1" * 40, "b": "1" * 40}, "candidates": {"a": ["1" * 40], "b": ["2" * 40]},
             "base": "1" * 40}
    github = FakeGitHub(issues=[tracking_issue(9, first)], label=True)
    joined = {**DATA, "base": "9" * 40, "red": ["c"], "verdicts": [{
        "id": "c", "verdict": "unattributed", "at_base": "absent", "window": "9" * 40, "candidates": ["3" * 40]}]}
    record = pmb.hand_over(joined, github, run_url="run", retire=True, ancestor=lambda older, newer: True)
    assert record["updated"] == [9]
    block = routing.parse_machine_block(github.issues[9]["body"])
    assert block["ids"] == ["a", "b", "c"] and block["candidates"] == {"a": ["1" * 40], "b": ["2" * 40],
                                                                       "c": ["3" * 40]}
    assert block["window"] == {"a": "1" * 40, "b": "1" * 40, "c": "9" * 40}
    body = github.issues[9]["body"]
    assert "- `a`: passed at its window's base `111111111111`" in body
    assert "- `c`: not present at its window's base `999999999999`" in body
    narrowed = routing.route(github, classification={"failed": []}, attribution={"verdicts": []}, head="f" * 40,
                             run_url="run", recheck=lambda ids: {"a": "passed"}, system="browser",
                             ancestor=lambda older, newer: True)
    assert narrowed["narrowed"] == [{"issue": 9, "ids": ["a"]}]
    rest = routing.parse_machine_block(github.issues[9]["body"])
    assert rest["ids"] == ["b", "c"] and set(rest["candidates"]) == set(rest["at_base"]) == {"b", "c"}


def test_a_browser_issue_is_read_again_right_before_each_write():
    """Finding 16: route writers take turns (one concurrency group), and each reads the issue again right before it
    writes: an update merges against the issue as it is now, a narrowing of an issue changed since its read waits."""
    first = {"format": 1, "os": "browser", "kind": "unattributed", "ids": ["a"]}
    github = FakeGitHub(issues=[tracking_issue(9, first)], label=True)
    listed = routing.open_issues(github)
    github.issues[9]["body"] = tracking_issue(9, {**first, "ids": ["a", "z"]})["body"]  # another window wrote
    github.pages = lambda path, listed=listed: [dict(github.issues[9], body=tracking_issue(9, first)["body"])]
    joined = {**DATA, "red": ["c"], "verdicts": [{"id": "c", "verdict": "unattributed", "at_base": "absent",
                                                  "window": "b" * 40, "candidates": ["3" * 40]}]}
    pmb.hand_over(joined, github, run_url="run", retire=False, ancestor=lambda older, newer: True)
    assert routing.parse_machine_block(github.issues[9]["body"])["ids"] == ["a", "c", "z"]  # z kept
    stale = FakeGitHub(issues=[tracking_issue(5, {**first, "ids": ["a", "b"]})], label=True)
    stale.pages = lambda path: [dict(stale.issues[5], body=tracking_issue(5, {**first, "ids": ["a"]})["body"])]
    record = routing.route(stale, classification={"failed": []}, attribution={"verdicts": []}, head="f" * 40,
                           run_url="run", recheck=lambda ids: dict.fromkeys(ids, "passed"), system="browser",
                           ancestor=lambda older, newer: True)
    assert record["closed"] == [] and stale.issues[5]["state"] == "open"
    assert any("changed since it was read" in warning for warning in record["warnings"])


def test_browser_reds_reach_ouroboros_through_the_post_merge_red_issues_it_reads():
    """Decision 7 (finding 15 declined by the owner's choice): the consumer of a red whose author's task ended is
    Ouroboros, which reads the open post-merge-red issues from the update facts; a browser issue is one of them."""
    from ouroboros import update_qualification

    github = FakeGitHub(label=True)
    pmb.hand_over(DATA, github, run_url="run", retire=True)
    [issue] = github.issues.values()
    assert {"name": routing.LABEL} in issue["labels"] and routing.LABEL == update_qualification.RED_LABEL
    asked = []
    read = update_qualification.post_merge_reds("https://github.com/razzant/ouroboros", get_json=lambda url: asked.append(
        url) or [dict(issue, created_at="2026-11-26T00:00:00Z")])
    assert f"labels={routing.LABEL}" in asked[0] and read["issues"][0]["number"] == issue["number"]
    assert read["issues"][0]["title"].startswith("Post-merge red on the browser check (Ubuntu)")


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
               "BASE": data["base"], "HEAD_SHA": data["head"],
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
    monkeypatch.setattr(pmb, "_chain", lambda ref: ["e" * 40, "f" * 40, "b" * 40])
    code, text, record = route(tmp_path, monkeypatch, moved, data)
    assert code == 0 and record["retire"] is False and moved.issues[5]["state"] == "open" and "moved past" in text
    monkeypatch.setattr(pmb, "_ancestor", lambda older, newer: True)
    current = Tip("f" * 40, issues=[issue], label=True)
    code, _, record = route(tmp_path, monkeypatch, current, data)
    assert code == 0 and record["record"]["closed"] == [5] and current.issues[5]["state"] == "closed"


class History(Tip):
    """Tip with closed issues and every issue's comments readable, as GitHub keeps them."""

    def pages(self, path):
        if path.startswith("issues?") and "state=all" in path:
            self._check("GET", path)
            return [issue for issue in self.issues.values() if {"name": routing.LABEL} in issue["labels"]]
        if path.startswith("issues/") and path.endswith("/comments"):
            self._check("GET", path)
            return [{"body": body} for number, body in self.comments if number == int(path.split("/")[1])]
        return super().pages(path)


def test_an_older_red_routed_after_a_newer_green_files_and_comments_nothing(tmp_path, monkeypatch):
    """Decision 1 (round 4, Astra 1, Carbon, Neon): a run of an older window that reaches the writer after a newer
    head closed the issue as green neither files, reopens nor comments for that test. Any other stale red is filed as
    usual, so the current head's recheck settles it; one the open issue of its kind cannot take (that issue records a
    newer observation) gets its own issue instead of failing every rerun (round 6, a1)."""
    older, newer, settled, own, filed = "f" * 40, "e" * 40, "tests/test_x_browser.py::t", "tests/test_z_browser.py::t", \
        "tests/test_w_browser.py::t"
    chain = [newer, older, "b" * 40]
    github = History(newer, issues=[tracking_issue(5, {"os": "browser", "kind": "unattributed", "ids": [settled]},
                                                   head=older)], label=True)
    monkeypatch.setattr(pmb, "_ancestor", lambda first, second: chain.index(first) >= chain.index(second))
    monkeypatch.setattr(pmb, "_chain", lambda ref: chain, raising=False)
    current = {**DATA, "head": newer, "red": [], "verdicts": [], "rechecked": {settled: "passed"}}
    code, _, record = route(tmp_path, monkeypatch, github, current)
    assert code == 0 and record["record"]["closed"] == [5]
    github.issues[7] = dict(tracking_issue(7, {"os": "browser", "kind": "unattributed", "ids": ["tests/test_y.py::t"]},
                                           head=newer), state="open", labels=[{"name": routing.LABEL}])
    comments, issues = len(github.comments), set(github.issues)
    late = {**DATA, "head": older, "red": [settled, own, filed], "verdicts": [
        {"id": settled, "verdict": "unattributed", "at_base": "passed", "window": "b" * 40, "candidates": ["1" * 40]},
        {"id": own, "verdict": "unattributed", "at_base": "absent", "window": "b" * 40, "candidates": ["1" * 40]},
        {"id": filed, "verdict": "pre-existing", "at_base": "failed", "window": "b" * 40}]}
    code, text, record = route(tmp_path, monkeypatch, github, late)
    assert code == 0 and record["retire"] is False
    created = record["record"]["created"]
    assert set(github.issues) - issues == set(created) and sorted(routing.parse_machine_block(
        github.issues[number]["body"])["ids"] for number in created) == [[filed], [own]]  # never the settled test's
    assert github.issues[5]["state"] == "closed" and github.issues[7]["body"].count(own) == 0
    assert not any(settled in body or own in body for _, body in github.comments[comments:])
    assert f"`{settled}` passed at the newer `eeeeeeeeeeee`: not handed over." in text and "is held" not in text
    # What newer heads recorded did not read: nothing is handed over, and a rerun offers it again.
    blind = History(newer, issues=[], label=True, fail=[("GET", "issues")])
    code, text, _ = route(tmp_path, monkeypatch, blind, late)
    assert code == 1 and "Nothing handed over: what newer heads recorded did not read" in text and not blind.posts


def test_a_late_red_no_newer_head_recorded_is_handed_over_beside_a_newer_issue_of_its_kind(tmp_path, monkeypatch):
    """Round 6 (a1): a stale run's red that no newer head recorded is handed over, never held: the open issue of its
    kind records a newer observation of another test and cannot take an older one, so the red gets its own issue
    under its own head, which the current head's recheck settles. Only a test a newer head recorded failing is held.
    Nothing is closed or reopened."""
    older, newer = "f" * 40, "e" * 40
    chain = [newer, older, "b" * 40]
    lost, other, seen = ("tests/test_a_browser.py::test_lost", "tests/test_b_browser.py::test_other",
                         "tests/test_c_browser.py::test_seen")
    github = History(newer, issues=[tracking_issue(7, {"os": "browser", "kind": "unattributed", "ids": [other]},
                                                   head=newer)], label=True)
    github.issues[9] = {**tracking_issue(9, {"os": "browser", "kind": "unattributed", "ids": [seen]}, head=newer),
                        "state": "closed", "labels": [{"name": routing.LABEL}]}
    monkeypatch.setattr(pmb, "_ancestor", lambda first, second: chain.index(first) >= chain.index(second))
    monkeypatch.setattr(pmb, "_chain", lambda ref: chain)
    late = {**DATA, "head": older, "red": [lost, seen], "verdicts": [
        {"id": node, "verdict": "unattributed", "at_base": "passed", "window": "b" * 40, "candidates": ["1" * 40]}
        for node in (lost, seen)]}
    code, text, record = route(tmp_path, monkeypatch, github, late)
    assert code == 0 and record["retire"] is False and not record["record"]["untracked"], text
    [created] = record["record"]["created"]
    assert routing.parse_machine_block(github.issues[created]["body"])["ids"] == [lost]
    assert github.issues[created]["state"] == "open" and f"Found failing at `{older}`" in github.issues[created]["body"]
    assert routing.parse_machine_block(github.issues[7]["body"])["ids"] == [other]
    assert github.issues[7]["state"] == "open" and github.issues[9]["state"] == "closed"
    assert record["record"]["closed"] == [] and record["record"]["narrowed"] == []
    assert f"`{lost}` is held" not in text
    assert f"`{seen}` is held for the check at the current head: the newer `eeeeeeeeeeee` recorded it failing." in text


def test_the_writer_hands_over_only_the_window_its_run_was_given(tmp_path, monkeypatch):
    """Round 4 advisory (Astra 4, Neon 14): the route job compares the hand-over's base and head with the trusted
    pair the workflow passes (BASE, HEAD_SHA) before any write."""
    github = Tip("f" * 40, label=True)
    monkeypatch.setattr(pmb.routing, "GitHub", lambda repo, token: github)
    summary = tmp_path / "summary.md"
    environ = {"ROUTING": pmb.encode(DATA), "GITHUB_REF": "refs/heads/ouroboros", "GITHUB_STEP_SUMMARY": str(summary),
               "BASE": DATA["base"], "HEAD_SHA": "e" * 40}
    args = pmb.argparse.Namespace(run_url="run", out=str(tmp_path / "routing.json"), dry_run=False)
    assert pmb.command_route(args, environ=environ) == 1 and not github.posts
    assert "not the window of this run" in summary.read_text()
    for name in ("route", "route-trial"):
        env = STEPS[name]["route"]["env"]
        assert env["BASE"] == "${{ needs.check.outputs.base }}" and env["HEAD_SHA"] == "${{ needs.check.outputs.head }}"


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
    window, dispatch = POST_MERGE["jobs"]["window"], POST_MERGE["jobs"]["browser-dispatch"]
    steps = {step.get("id"): step for step in window["steps"]}
    assert set(POST_MERGE["jobs"]) == {"window", "post-merge", "browser-dispatch"}  # none holds the post-merge group
    assert window["permissions"] == {"contents": "read", "actions": "read"}  # as before the pilot (advisories 6, 14)
    scope = steps["browser_scope"]
    assert scope["run"].startswith("set -o pipefail\npython3 -I -S scripts/post_merge_browser_select.py scope")
    assert scope["continue-on-error"] is True and dispatch["continue-on-error"] is True  # neither can stop a leg
    assert dispatch["needs"] == "window" and dispatch["permissions"] == {"actions": "write"}
    assert dispatch["if"] == ("github.repository == 'razzant/ouroboros' && github.ref == 'refs/heads/ouroboros' "
                              "&& needs.window.outputs.browser == 'true'")
    [start] = dispatch["steps"]
    assert re.search(r'\ngh workflow run post-merge-browser\.yml --repo "\$GITHUB_REPOSITORY" --ref ouroboros \\\n'
                     r' *-f base="\$BASE" -f head="\$GITHUB_SHA"\n$', start["run"]), start["run"]
    assert start["env"] == {"GH_TOKEN": "${{ github.token }}", "BASE": "${{ needs.window.outputs.browser_base }}",
                            "PILOT_UNTIL": "2026-12-10", "PILOT_RELEASE": "v7.12.1"}
    order = [step.get("id") for step in window["steps"]]
    assert order.index("base") < order.index("browser_scope")
    for step in window["steps"]:  # what the window readers take from its step names
        name = step.get("name") or ""
        if "browser" in name:
            assert not name.startswith(post_merge_window.REPLAY_STEP)
            assert post_merge_window._BASE_STEP.fullmatch(name) is None
    text = (ROOT / ".github/workflows/post-merge.yml").read_text(encoding="utf-8")
    assert "browser_budget" not in text and "--browser-stop-file" not in text and "legs --repo" not in text


@pytest.mark.skipif(sys.platform == "win32", reason="the workflow steps run under bash on Ubuntu")
def test_the_pilot_ends_on_its_date_or_at_the_first_release_after_v7_12_1(tmp_path):
    """Decision 7 (round 4, Astra 7): the dispatch pins PILOT_UNTIL=2026-12-10 and PILOT_RELEASE=v7.12.1; after that
    date, or once a release tag without a pre-release suffix newer than v7.12.1 exists, it starts nothing and says so
    in one summary line. Both workflow headers name the same end."""
    [step] = POST_MERGE["jobs"]["browser-dispatch"]["steps"]
    assert step["env"]["PILOT_UNTIL"] == "2026-12-10" and step["env"]["PILOT_RELEASE"] == "v7.12.1"
    assert "until 2026-12-10 or the first release after v7.12.1" in (
        ROOT / ".github/workflows/post-merge-browser.yml").read_text(encoding="utf-8")
    assert "a pilot until PILOT_UNTIL or the first release after PILOT_RELEASE" in (
        ROOT / ".github/workflows/post-merge.yml").read_text(encoding="utf-8")

    def dispatch(today, tags, *, unread=None):
        tools = tmp_path / today / ("-".join(tags) or "unread")
        tools.mkdir(parents=True)
        listed = "".join(f"{index:040x}\trefs/tags/{tag}\n" for index, tag in enumerate(tags))
        tags_read = f"printf '{listed}'" if unread is None else f"echo 'fatal: unable to access' >&2; exit {unread}"
        for name, body in (("date", f"echo {today}"), ("git", tags_read),
                           ("gh", f'echo "$@" >> "{tools}/gh.log"')):
            (tools / name).write_text(f"#!/bin/sh\n{body}\n")
            (tools / name).chmod(0o755)
        summary = tools / "summary"
        subprocess.run(["bash", "-e", "-c", step["run"]], check=True, env={
            "PATH": f"{tools}:/usr/bin:/bin", "GITHUB_STEP_SUMMARY": str(summary), "GITHUB_REPOSITORY": "o/r",
            "GITHUB_SHA": "h" * 40, "BASE": "b" * 40, **step["env"]}, capture_output=True)
        log = tools / "gh.log"
        return (log.read_text() if log.exists() else ""), (summary.read_text() if summary.exists() else "")

    started, said = dispatch("2026-12-10", ["v7.12.0", "v7.12.1", "v7.12.2-alpha.0"])
    assert started.startswith("workflow run post-merge-browser.yml") and said == ""
    assert dispatch("2026-12-11", ["v7.12.1"]) == ("", "browser pilot ended: date (2026-12-10 passed); re-enable by PR.\n")
    assert dispatch("2026-12-01", ["v7.12.1", "v7.12.2"])[1] == "browser pilot ended: release (v7.12.2); re-enable by PR.\n"
    assert dispatch("2026-12-02", ["v7.12.1", "v7.12.10", "v7.13.0-alpha.1"])[1].startswith(
        "browser pilot ended: release (v7.12.10)")
    # Round 6 (d1): release tags that cannot be read end nothing by themselves and start nothing either.
    assert dispatch("2026-12-03", [], unread=128) == (
        "", "browser pilot skipped: release tags could not be read (exit 128)\n")
    assert dispatch("2026-12-11", [], unread=128)[1].startswith("browser pilot ended: date")


def test_the_route_jobs_run_no_test_and_only_the_trial_routes_dry():
    """Permissions, tokens and pinning: tests/test_post_merge_workflow.py; ceilings and the install's bound:
    tests/test_ci_diagnostic_workflows.py."""
    for name in ("route", "route-trial"):
        assert JOBS[name]["needs"] == "check" and "needs.check.outputs.applicable == 'true'" in JOBS[name]["if"]
        run = STEPS[name]["route"]["run"]
        assert run.startswith("python3 -I -S scripts/post_merge_browser.py route") and "pytest" not in run
        assert ("--dry-run" in run) == (name == "route-trial")
        assert STEPS[name]["route"]["env"]["ROUTING"] == "${{ needs.check.outputs.routing }}"
        # Decision 2: the dispatching workflow's own revision, never the window's head, and no checkout of it later.
        checkout = next(step for step in JOBS[name]["steps"] if "actions/checkout" in step.get("uses", ""))
        assert "ref" not in checkout["with"] and checkout["with"]["persist-credentials"] is False
        assert not any("git checkout" in step.get("run", "") for step in JOBS[name]["steps"])
    assert set(JOBS) == {"register", "check", "route", "route-trial"}  # decision 12: no separate scope job
    run = STEPS["check"]["run"]["run"]
    assert "-p scripts.post_merge_browser" in run and "--session-timeout=\"$SESSION\"" in run
    assert "scripts/ci_retry.py" in run and "--require-ui-browser" not in run  # a subset proves no lane


def _scope_step(steps):
    return next(step for step in steps if step.get("id") in ("scope", "browser_scope") and "run" in step)


@pytest.mark.skipif(sys.platform == "win32", reason="the workflow steps run under bash on Ubuntu")
def test_a_selector_that_crashes_fails_its_scope_step_in_both_workflows(tmp_path):
    """Finding 12: under the default `bash -e`, `selector | tee` exits with tee's status; pipefail makes a crash fail
    the step instead of an empty "not applicable"."""
    for steps in (POST_MERGE["jobs"]["window"]["steps"], JOBS["check"]["steps"]):
        script = _scope_step(steps)["run"].replace("python3 -I -S scripts/post_merge_browser_select.py scope",
                                                    "python3 -c 'raise SystemExit(1)'")
        done = subprocess.run(["bash", "-e", "-c", script], env={"GITHUB_OUTPUT": str(tmp_path / "out"),
                                                                "PATH": "/usr/bin:/bin"}, capture_output=True)
        assert done.returncode != 0, script


@pytest.mark.skipif(sys.platform == "win32", reason="the workflow steps run under bash on Ubuntu")
def test_the_writer_hands_over_only_a_window_on_the_first_parent_chain_of_ouroboros(tmp_path):
    """Decision 2 (A1, Carbon, Opus): base and head both on origin/ouroboros's first-parent chain, base first, or
    nothing is handed over and the summary says why. A head merged through a second parent never qualifies."""
    (tmp_path / "r").mkdir()
    repo = repository(tmp_path / "r")
    (repo / "f").write_text("0")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "base")
    base = git(repo, "rev-parse", "HEAD")
    git(repo, "checkout", "-qb", "topic")
    (repo / "g").write_text("1")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "topic")
    topic = git(repo, "rev-parse", "HEAD")
    git(repo, "checkout", "-q", "-")
    git(repo, "merge", "-q", "--no-ff", "-m", "merge", "topic")
    head = git(repo, "rev-parse", "HEAD")
    git(repo, "update-ref", "refs/remotes/origin/ouroboros", head)
    script = STEPS["route"]["trusted"]["run"]

    def trusted(base_sha, head_sha):
        out, summary = tmp_path / "out", tmp_path / "summary"
        out.write_text("")
        summary.write_text("")
        subprocess.run(["bash", "-e", "-c", script], cwd=repo, check=True, env={
            "BASE": base_sha, "HEAD_SHA": head_sha, "GITHUB_OUTPUT": str(out), "GITHUB_STEP_SUMMARY": str(summary),
            "RUNNER_TEMP": str(tmp_path), "PATH": "/usr/bin:/bin:/opt/homebrew/bin"})
        return out.read_text().strip(), summary.read_text()

    assert trusted(base, head) == ("trusted=true", "")
    assert git(repo, "rev-parse", "HEAD") == head  # nothing of the window is checked out
    for window in ((base, topic), (topic, head), (head, base), (head, head)):  # second parent, reversed, empty
        off, text = trusted(*window)
        assert off == "trusted=false" and "not a window on the first-parent chain of ouroboros" in text, window
    git(repo, "checkout", "-qb", "elsewhere", base)
    (repo / "h").write_text("2")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "unmerged")
    assert trusted(base, git(repo, "rev-parse", "HEAD"))[0] == "trusted=false"  # not in ouroboros


def test_the_run_step_ends_just_above_the_session_so_the_outcome_is_still_judged():
    """Advisory 1: ci_retry's retry runs without `--session-timeout`; the run step's ceiling cuts it a few minutes after
    the session's own limit, and the judge reads that retry as unfinished (CHECKED PARTIALLY)."""
    session = sel.session_seconds(sel.BUDGET_SECONDS)
    ceiling = STEPS["check"]["run"]["timeout-minutes"] * 60
    assert session < ceiling <= session + 300


def test_the_check_writes_its_outcome_before_the_upload_and_fails_only_after_it():
    order = [step.get("id") or step.get("name") for step in JOBS["check"]["steps"]]
    upload = next(step for step in JOBS["check"]["steps"] if "upload-artifact" in step.get("uses", ""))
    assert order.index("outcome") < order.index(upload["name"]) < order.index("verdict") == len(order) - 1
    applicable = "steps.scope.outputs.applicable == 'true'"
    assert upload["if"] == "${{ always() }}" and STEPS["check"]["outcome"]["if"] == f"${{{{ !cancelled() && {applicable} }}}}"
    assert order.index("window") < order.index("scope") < order.index("setup_python") < order.index("install_browsers")
    assert order.index("run") < order.index("confirm") < order.index("compare")
    for step in ("setup_python", "weights", "plan"):  # a manual dispatch outside the scope installs nothing
        assert STEPS["check"][step]["if"] == applicable, step
    assert JOBS["check"]["outputs"] == {"applicable": "${{ steps.scope.outputs.applicable }}",
                                        "base": "${{ steps.window.outputs.base }}",
                                        "head": "${{ steps.window.outputs.head }}",
                                        "outcome": "${{ steps.outcome.outputs.outcome }}",
                                        "routing": "${{ steps.outcome.outputs.routing }}"}
    assert set(STEPS["check"]["outcome"]["env"]) == {f"BROWSER_{step.upper()}_OUTCOME" for step in pmb.STEPS}
