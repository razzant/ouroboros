"""The warm bytecode prefix of a run (ouroboros/test_bytecode.py): what the owner compiles once, who reads it."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from ouroboros import test_bytecode


def test_shards_interleave_the_files_and_never_hand_out_an_empty_one():
    files = [f"tests/test_{index}.py" for index in range(5)]
    assert test_bytecode.shards(files, 2) == [files[0::2], files[1::2]]
    assert test_bytecode.shards(files, 9) == [[name] for name in files]
    assert test_bytecode.shards(files, 0) == [files]
    assert test_bytecode.shards([], 3) == []


def test_the_warm_lifts_the_write_ban_for_itself_alone_and_the_owner_names_its_shards():
    env = {"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPYCACHEPREFIX": "/p", "OUROBOROS_TEST_EVIDENCE_ROOT": "/e",
           "OUROBOROS_TEST_DURATIONS": "/d", "OURO_TEST_WORKER_BUDGET": "2", "PYTEST_XDIST_AUTO_NUM_WORKERS": "2",
           "PYTEST_XDIST_WORKER": "gw3", "PYTEST_XDIST_TESTRUNUID": "u", "OUROBOROS_TEST_TEMP_ROOT": "/t", "HOME": "/h"}
    assert test_bytecode.writing_env(env) == {"PYTHONPYCACHEPREFIX": "/p", "OUROBOROS_TEST_TEMP_ROOT": "/t", "HOME": "/h"}
    argv = test_bytecode.command("py", "/repo", workers=0, files_from="/sel.txt")
    assert argv[:2] == ["py", "-c"] and "runpy.run_path" in argv[2]
    assert argv[3:] == [str(test_bytecode.MODULE_PATH), "--repo", "/repo", "--workers", "1", "--files-from", "/sel.txt"]
    assert test_bytecode.command("py", "/repo", workers=3)[-2:] == ["--workers", "3"]


def test_a_walker_hands_pytest_one_directory_and_keeps_to_its_shard(tmp_path, monkeypatch):
    """Handed its modules one by one, pytest re-collects ``tests`` once per module and a walker with the collector
    paused held every discarded node (4.3 GB for a quarter of the tree, run 42126810926): the walker names the
    directory once, a module outside it by name, and its plugin passes over every test module not in its shard."""
    argv = test_bytecode.walker_command("py", ["tests/test_a.py", "tests/sub/test_b.py", "skills/x/test_c.py"], "plug")
    assert argv[argv.index("--no-header") + 1:] == ["-p", "plug", "tests", "skills/x/test_c.py"]
    tests = tmp_path / "tests"
    for name in ("test_a.py", "sub/test_b.py", "test_other.py", "_helper.py", "test_pkg.py/test_inside.py"):
        (tests / name).parent.mkdir(parents=True, exist_ok=True)
        (tests / name).write_text("", encoding="utf-8")
    listed = tmp_path / "shard.txt"
    listed.write_text("tests/test_a.py\ntests/sub/test_b.py\ntests/test_pkg.py/test_inside.py\n", encoding="utf-8")
    config = SimpleNamespace(invocation_params=SimpleNamespace(dir=tmp_path))
    ignore = test_bytecode.pytest_ignore_collect
    assert ignore(tests / "test_other.py", config) is None  # outside a walker: decides nothing
    monkeypatch.setenv(test_bytecode.SHARD_ENV, str(listed))
    assert ignore(tests / "test_a.py", config) is None and ignore(tests / "sub" / "test_b.py", config) is None
    assert ignore(tests / "test_other.py", config) is True
    assert ignore(tests / "_helper.py", config) is None and ignore(tests / "sub", config) is None
    assert ignore(tests / "test_pkg.py", config) is None  # a package named like a module stays walkable


def test_the_warm_root_compiles_the_checkout_and_collects_the_walked_modules_into_the_runs_prefix(tmp_path):
    """A real root over a tiny checkout: compileall's checked-hash bytecode of the checkout's modules and pytest's
    rewritten test modules land under the run's prefix, a selection narrows the walk, the facts line says what ran,
    and a later process of the run, its write ban kept, reads the bytecode instead of compiling."""
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "ouroboros").mkdir()
    (repo / "ouroboros" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "ouroboros" / "thing.py").write_text("VALUE = 1\n", encoding="utf-8")
    for name in ("test_a.py", "test_b.py", "test_c.py"):
        (repo / "tests" / name).write_text("def test_one():\n    assert True\n", encoding="utf-8")
    # A candidate module under a fixed plugin name would shadow the walkers' plugin from their cwd and pass over all.
    (repo / "ouroboros_warm_walk.py").write_text("def pytest_ignore_collect(collection_path, config):\n"
                                                 "    return True\n", encoding="utf-8")
    prefix = tmp_path / "prefix"
    env = {key: value for key, value in os.environ.items()
           if key not in ("PYTEST_ADDOPTS", "OUROBOROS_TEST_EVIDENCE_ROOT", "OUROBOROS_TEST_DURATIONS")}
    env.update(PYTHONPYCACHEPREFIX=str(prefix), PYTHONDONTWRITEBYTECODE="1")
    selection = tmp_path / "files.txt"
    selection.write_text("tests/test_a.py\ntests/test_b.py\n", encoding="utf-8")
    result = subprocess.run(test_bytecode.command(sys.executable, repo, workers=2, files_from=selection), cwd=repo,
                            env=env, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    facts = json.loads(result.stdout.strip().splitlines()[-1].removeprefix("WARM_BYTECODE "))
    assert (facts["files"], facts["shards"], facts["compile_exit"], facts["collect_exits"]) == (2, 2, 0, [0])
    assert (prefix / "OUROBOROS_TEST_SCRATCH.txt").read_text(encoding="utf-8").startswith("bytecode-prefix\n")
    tag, below = sys.implementation.cache_tag, Path(*repo.resolve().parts[1:]).as_posix()
    cached = sorted(path.relative_to(prefix).as_posix() for path in prefix.rglob("*.pyc"))
    compiled = f"{below}/ouroboros/thing.{tag}.pyc"
    assert compiled in cached, cached  # compileall: the whole checkout
    rewritten = [name for name in cached if name.startswith(f"{below}/tests/") and "-pytest-" in name]
    assert [name.rsplit("/", 1)[1].split(".")[0] for name in rewritten] == ["test_a", "test_b"], rewritten  # the walk
    before = len(cached)
    probe = subprocess.run([sys.executable, "-v", "-c", "import ouroboros.thing"], cwd=repo, env=env,
                           capture_output=True, text=True, timeout=120)
    # `python -v` names the pyc it read with repr(): the exact file, as that form spells it on every OS.
    assert probe.returncode == 0 and f"code object from {str(prefix / compiled)!r}" in probe.stderr, probe.stderr[-800:]
    assert len(list(prefix.rglob("*.pyc"))) == before, "the run's processes keep the write ban"
