"""The test-retirement instruments: census rules, protected set, coverage digest."""
import json
import sqlite3
from contextlib import closing

import pytest

from devtools import coverage_census as cov
from devtools import protected_tests as protect
from devtools import test_census as census

_TREE = {
    "tests/test_prose_pins.py": '''
import subprocess
from pathlib import Path


def test_wording():
    assert "word" in Path("docs/a.md").read_text()


def test_wording_of_a_defect():
    """Guards #2784."""
    assert "word" in Path("docs/a.md").read_text()


def test_wording_beside_a_process():
    subprocess.run(["true"])
    assert "word" in Path("docs/a.md").read_text()
''',
    "tests/test_matrix.py": '''
import pytest
from ouroboros.alpha import run

CASES = list(range(8))


@pytest.mark.parametrize("a", [1, 2, 3])
@pytest.mark.parametrize("b", [1, 2])
def test_cartesian(a, b):
    assert run(a, b)


@pytest.mark.parametrize("a", [1, 2, 3, 4, 5])
def test_case_list(a):
    assert run(a)


@pytest.mark.parametrize(["a"], [(1,), (2,), (3,)])
def test_short_list(a):
    assert run(a)


@pytest.mark.parametrize("a", CASES)
def test_cheap_enumeration(a):
    assert run(a)


@pytest.mark.parametrize("a", CASES)
def test_heavy_enumeration(a):
    assert run(a)


@pytest.mark.parametrize("a", [1, 2, 3, 4, 5])
def test_named_in_a_red_record(a):
    assert run(a)


@pytest.mark.integration
def test_outside_the_default_lane():
    assert run()
''',
    "tests/test_facets.py": '''
import pytest
from ouroboros.beta import build


@pytest.fixture
def world():
    return build()


def test_cheap_world(world):
    assert world.a


@pytest.mark.usefixtures("world")
def test_costly_world(world):
    assert world.b
''',
    "tests/test_preflight_gate.py": '''
import subprocess
from ouroboros.gate import kept_step, rewritten_step


def test_cheapest_spawn():
    subprocess.run(["true"])
    rewritten_step()


def test_dearer_spawn():
    subprocess.run(["true"])
    rewritten_step()


def test_dearer_spawn_of_kept_machinery():
    subprocess.run(["true"])
    kept_step()


def test_dearer_spawn_scrubs_the_token():
    subprocess.run(["true"])
    rewritten_step()
''',
    "tests/test_alpha_batch4.py": '''
from ouroboros.alpha import run


def test_event_pin():
    assert run()


def test_event_pin_keeps_the_token_out():
    assert run()
''',
    "tests/test_beta_batch4.py": '''
from ouroboros.beta import build


def test_event_pin_of_a_barely_reached_module():
    assert build()
''',
    "tests/test_floor_witness.py": '''
import pytest
from ouroboros.alpha import run


@pytest.mark.parametrize("a", [1, 2, 3, 4, 5])
def test_protected_by_its_file(a):
    assert run(a)
''',
}
_JOURNAL = {
    ("tests/test_matrix.py", "test_cheap_enumeration"): {"items": 8, "seconds": 0.8, "setup_s": 0.1},
    ("tests/test_matrix.py", "test_heavy_enumeration"): {"items": 8, "seconds": 80.0, "setup_s": 1.0},
    ("tests/test_preflight_gate.py", "test_cheapest_spawn"): {"items": 1, "seconds": 1.0, "setup_s": 0.1},
    ("tests/test_preflight_gate.py", "test_dearer_spawn"): {"items": 1, "seconds": 2.0, "setup_s": 0.1},
    ("tests/test_preflight_gate.py", "test_dearer_spawn_of_kept_machinery"): {"items": 1, "seconds": 3.0, "setup_s": 0.1},
    ("tests/test_preflight_gate.py", "test_dearer_spawn_scrubs_the_token"): {"items": 1, "seconds": 4.0, "setup_s": 0.1},
    ("tests/test_facets.py", "test_cheap_world"): {"items": 1, "seconds": 1.0, "setup_s": 0.2},
    ("tests/test_facets.py", "test_costly_world"): {"items": 1, "seconds": 3.0, "setup_s": 2.0},
}


def _write_tree(root, files):
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text, encoding="utf-8")


def test_each_function_goes_to_its_first_rule_and_the_protected_set_wins(tmp_path):
    _write_tree(tmp_path, _TREE)
    rows = census.join_lane(census.scan_tests(tmp_path), _JOURNAL, None)
    unproven = census.decide([dict(row) for row in rows], set(), set())  # no --base: nothing is proven rewritten
    assert next(row for row in unproven if row["qual"] == "test_dearer_spawn")["cluster"] != "D6_machinery"
    census.decide(rows, {"tests/test_floor_witness.py"}, {("tests/test_matrix.py", "test_named_in_a_red_record")},
                  {"ouroboros.gate.rewritten_step"})
    claimed = {row["qual"]: (row["cluster"], row["items_removed"]) for row in rows}
    assert "test_outside_the_default_lane" not in claimed
    assert {name: claimed[name] for name in (
        "test_wording", "test_cartesian", "test_case_list", "test_short_list", "test_cheap_enumeration",
        "test_heavy_enumeration", "test_named_in_a_red_record", "test_protected_by_its_file",
        "test_cheapest_spawn", "test_dearer_spawn", "test_event_pin", "test_cheap_world", "test_costly_world",
    )} == {
        "test_wording": ("D4_wording", 1), "test_cartesian": ("D1_cartesian", 3), "test_case_list": ("D2_cases", 2),
        "test_short_list": ("", 0), "test_cheap_enumeration": ("D3_enumeration", 7),
        "test_heavy_enumeration": ("D2_cases", 5), "test_named_in_a_red_record": ("P_postmerge_protected", 0),
        "test_protected_by_its_file": ("K_protected", 0), "test_cheapest_spawn": ("D7_facets", 1),
        "test_dearer_spawn": ("D6_machinery", 1), "test_event_pin": ("D5_event_file", 1),
        "test_cheap_world": ("D7_facets", 1), "test_costly_world": ("D7_survivor", 0),
    }
    assert claimed["test_wording_of_a_defect"][0] != "D4_wording" != claimed["test_wording_beside_a_process"][0]
    assert claimed["test_event_pin_of_a_barely_reached_module"][0] != "D5_event_file"
    # Machinery that was not rewritten may be what a spawning test witnesses; a secret test is never retired by name.
    assert claimed["test_dearer_spawn_of_kept_machinery"][0] != "D6_machinery"
    assert claimed["test_dearer_spawn_scrubs_the_token"][0] != "D6_machinery"
    assert claimed["test_event_pin_keeps_the_token_out"][0] != "D5_event_file"
    assert next(row for row in rows if row["qual"] == "test_cheap_world")["survivor"] == "test_costly_world"
    report = census.totals(rows)
    assert report["deleted"]["items_removed"] == 1 + 3 + 2 + 5 + 1 + 1
    # the two refused wording pins share a world, and so do the three gate tests no rule retired
    assert report["folded"]["items_removed"] == 7 + 1 + 1 + 2
    assert report["composite"]["items_removed"] == report["deleted"]["items_removed"] + 11


def test_a_rerun_gives_the_same_bytes_from_journal_and_collection(tmp_path):
    _write_tree(tmp_path, _TREE)
    journal = tmp_path / "events.jsonl"
    reports = [("tests/test_matrix.py::test_case_list[1]@serial3", "setup", 2.0),
               ("tests/test_matrix.py::test_case_list[1]@serial3", "call", 1.0),
               ("tests/test_matrix.py::test_case_list[2]@tests/test_matrix.py", "call", 1.0)]
    journal.write_text("".join(json.dumps({"event": "report", "nodeid": node, "phase": phase,
                                           "duration_seconds": seconds}) + "\n"
                               for node, phase, seconds in reports), encoding="utf-8")
    older = tmp_path / "older.jsonl"
    older.write_text(json.dumps({"event": "report", "nodeid": "tests/test_matrix.py::test_case_list[1]",
                                 "phase": "call", "duration_seconds": 50.0}) + "\n", encoding="utf-8")
    assert census.load_journals([journal, older]) == {
        ("tests/test_matrix.py", "test_case_list"): {"items": 2, "seconds": 4.0, "setup_s": 2.0}}
    collected = tmp_path / "ids.txt"
    collected.write_text("".join(f"tests/test_matrix.py::test_cartesian[{case}]\n" for case in range(6))
                         + "6 tests collected\n", encoding="utf-8")
    protected = tmp_path / "protected.json"
    protected.write_text(json.dumps({"files": {}, "functions": {}}), encoding="utf-8")
    argv = ["--root", str(tmp_path), "--protected", str(protected), "--journal", str(journal),
            "--collected", str(collected)]
    written = []
    for name in ("first", "second"):
        assert census.main([*argv, "--out", str(tmp_path / name)]) == 0
        written.append((tmp_path / name / "decisions.csv").read_bytes())
    assert written[0] == written[1]
    assert written[0].splitlines()[1:] and b"test_cartesian" in written[0] and b"test_case_list" not in written[0]


def test_the_protected_set_takes_red_records_at_function_level(tmp_path):
    records = [{"number": 7, "body": "- `tests/test_a.py::test_x[case-1]`\n"
                                     "- `tests/system_e2e/test_b.py::TestC::test_y`\n"},
               {"number": 9, "body": "size: `size:tests/test_a.py::test_x` and test_d.py::test_z, not tests/test_e.py"}]
    assert protect.named_functions(records, [{"nodeid": "tests/test_f.py::test_w[1]", "issue": 11}]) == {
        "tests/test_a.py::test_x": [7, 9], "tests/system_e2e/test_b.py::TestC::test_y": [7],
        "tests/test_d.py::test_z": [9], "tests/test_f.py::test_w": [11]}
    keep = tmp_path / "keep.csv"
    keep.write_text("file,reason\ntests/test_kept.py,real-process system scenario (one witness)\n"
                    'tests/test_cited.py,"cites issue(s) #1 in a test docstring: keep the closest witness"\n',
                    encoding="utf-8")
    assert protect.keep_list_files(keep) == {"tests/test_kept.py": "real-process system scenario"}
    floor = sorted(protect.FLOOR_TESTS)[0]
    _write_tree(tmp_path, {floor: "", "tests/test_kept.py": "", "tests/test_carried.py": ""})
    files = protect.protected_files(tmp_path, {"tests/test_carried.py": "a reason", "tests/test_gone.py": "a reason",
                                               "tests/test_left_the_floor.py": "gate floor: old"},
                                    protect.keep_list_files(keep))
    assert set(files) == {floor, "tests/test_kept.py", "tests/test_carried.py"}
    assert files[floor].startswith("gate floor: ")


def _numbits(lines):
    return bytes(sum(1 << (line & 7) for line in lines if line >> 3 == byte) for byte in range(max(lines) // 8 + 1))


def _coverage_file(path, rows):
    with closing(sqlite3.connect(path)) as database:
        database.executescript("CREATE TABLE file (id INTEGER PRIMARY KEY, path TEXT);"
                               "CREATE TABLE context (id INTEGER PRIMARY KEY, context TEXT);"
                               "CREATE TABLE line_bits (file_id, context_id, numbits);")
        for index, (measured, context, lines) in enumerate(rows):
            database.execute("INSERT INTO file VALUES (?, ?)", (index, measured))
            database.execute("INSERT INTO context VALUES (?, ?)", (index, context))
            database.execute("INSERT INTO line_bits VALUES (?, ?, ?)", (index, index, _numbits(lines)))
        database.commit()


def test_the_digest_answers_what_a_deletion_loses(tmp_path):
    source = "import os\n\n\ndef alpha():\n    one = 1\n    return one\n\n\ndef beta():\n    return 2\n"
    _write_tree(tmp_path, {"ouroboros/x.py": source})
    # A hermetic copy measured on Windows records its own separators.
    checkout, copy = str(tmp_path / "ouroboros/x.py"), "C:\\hermetic\\repo\\ouroboros\\x.py"
    one, two, three = "tests/test_a.py::test_one", "tests/test_a.py::test_two[1]", "tests/test_b.py::test_three"
    (tmp_path / "shard1").mkdir()
    (tmp_path / "shard2").mkdir()
    _coverage_file(tmp_path / "shard1" / ".coverage", [
        (checkout, f"{one}|run", [1, 4, 5, 6]), (copy, f"{two} (call)", [5, 6]), (checkout, "", [1, 4, 9, 10])])
    (tmp_path / "shard1" / ".coverage.killed").write_bytes(b"not a database")
    _coverage_file(tmp_path / "shard2" / ".coverage", [(checkout, f"{three}|teardown", [6, 10])])
    folds = [cov.fold(tmp_path / "shard1", tmp_path), cov.fold(tmp_path / "shard2", tmp_path)]
    # Every product root is measured, the launcher and the bundled skills too; a test file never is.
    _write_tree(tmp_path, {"launcher.py": "", "skills/telegram/plugin.py": "", "tests/test_a.py": ""})
    assert cov._product_file(tmp_path, "/runner/copy/launcher.py") == "launcher.py"
    assert cov._product_file(tmp_path, "/runner/copy/skills/telegram/plugin.py") == "skills/telegram/plugin.py"
    assert cov._product_file(tmp_path, "/runner/copy/tests/test_a.py") == ""
    assert [fold["unreadable"] for fold in folds] == [1, 0]
    report = cov.digest(folds)
    assert report["totals"] == {"tests_measured": 3, "lines_covered": 3, "covering_sets": 3, "kept_sets": 2,
                                "single_witness_lines": 1, "unreadable_files": 1}
    assert report["nodes"] == sorted([one, two, three])
    assert cov.query(report, {one}) == []  # test_two still covers everything it did
    assert cov.query(report, {three}) == [{"node": three, "lines": "ouroboros/x.py:10"}]
    assert [row["node"] for row in cov.query(report, {one, two})] == [one]
    # Line 6 is in no kept set: returning the witnesses of the kept sets covers it too.
    assert sorted(row["node"] for row in cov.query(report, {one, two, three})) == [one, three]
    decisions = tmp_path / "decisions.csv"
    decisions.write_text("file,qual,cluster\ntests/test_a.py,test_two,D2_cases\ntests/test_a.py,test_one,D4_wording\n"
                         "tests/test_b.py,test_three,D7_facets\n", encoding="utf-8")
    (tmp_path / "digest.json").write_text(json.dumps(report), encoding="utf-8")
    assert cov.main(["query", str(tmp_path / "digest.json"), "--decisions", str(decisions),
                     "--out", str(tmp_path / "returned.csv")]) == 0
    assert (tmp_path / "returned.csv").read_text(encoding="utf-8").splitlines() == [
        "node,cluster,lines", f"{one},D4_wording,ouroboros/x.py:5"]
    with pytest.raises(SystemExit):  # no candidates is no question, never a clean answer
        cov.main(["query", str(tmp_path / "digest.json"), "--out", str(tmp_path / "returned.csv")])


def test_a_custom_child_environment_still_names_the_test(monkeypatch):
    calls = []
    monkeypatch.setattr(cov, "_popen_init", lambda self, *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setenv("COVERAGE_PROCESS_START", "/rc")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/test_a.py::test_b (call)")
    carried = {"A": "1", "COVERAGE_PROCESS_START": "/rc", "PYTEST_CURRENT_TEST": "tests/test_a.py::test_b (call)"}
    cov._measured_init(object(), ["x"], env={"A": "1"})
    assert calls.pop()[1]["env"] == carried
    cov._measured_init(object(), ["x"], *[None] * 9, {"A": "1"})
    assert calls.pop()[0][10] == carried
    cov._measured_init(object(), ["x"])  # no env: the child inherits both
    assert calls.pop() == ((["x"],), {})
