"""Capacity is information at the real health/readiness consumers, not a gate."""

import subprocess
from types import SimpleNamespace

from ouroboros import review
from ouroboros.tools import health, review_helpers


def test_near_limit_module_is_not_hidden_by_registered_giants(monkeypatch):
    giants = tuple(review.GatedModule(f"old{i}.py", 2000 + i, 5000) for i in range(12))
    monkeypatch.setattr(review, "GIANT_PATHS", frozenset(m.path for m in giants))
    inventory = review.SizeRatchetInventory(
        modules=(*giants, review.GatedModule("near.py", review.MAX_MODULE_LINES - 1, 199999)),
        functions=(), giant_paths=frozenset(), function_debt=frozenset(),
        band_paths=frozenset(), byte_debt={},
    )

    lines = review.size_headroom_lines(inventory)

    assert lines[1].startswith("near.py:")
    assert "1 remaining" in lines[1]
    assert any("registered line debt" in line for line in lines)
    assert any("8 more modules omitted" in line for line in lines)
    assert review.size_headroom_lines(inventory, paths=["near.py"])[1:] == [lines[1]]


def test_headroom_uses_live_limits_and_retains_zero_and_negative(monkeypatch):
    monkeypatch.setattr(review, "MAX_MODULE_LINES", 500)
    monkeypatch.setattr(review, "MAX_MODULE_BYTES", 100_000)
    monkeypatch.setattr(review, "MAX_FUNCTION_LINES", 200)
    module = review.GatedModule("sample.py", review.MAX_MODULE_LINES, review.MAX_MODULE_BYTES + 1)
    functions = (review.GatedFunction("sample.py", "a", 1, review.MAX_FUNCTION_LINES),
                 review.GatedFunction("sample.py", "b", 2, review.MAX_FUNCTION_LINES + 1))
    inventory = review.SizeRatchetInventory((module,), functions, frozenset(), frozenset(), frozenset(), {})

    lines = review.size_headroom_lines(inventory)

    assert lines[0] == "Runtime functions: 2 (descriptive)."
    assert "lines (0 remaining)" in lines[1]
    assert "UTF-8 bytes (-1 remaining)" in lines[1]
    assert "b:" in lines[2] and "-1 remaining" in lines[2]
    assert "a:" in lines[3] and "0 remaining" in lines[3]


def test_headroom_counts_normalized_utf8_source(tmp_path):
    source = "# Snowman: ☃\r\ndef sample():\r\n    return 1\r\n"
    (tmp_path / "sample.py").write_bytes(source.encode("utf-8"))
    inventory = review.collect_size_ratchet_inventory(tmp_path)

    line = review.size_headroom_lines(inventory)[1]

    assert "sample.py: 3/" in line
    assert f"{len(source.replace(chr(13), '').encode('utf-8'))}/" in line


def _repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "ouroboros").mkdir()
    # A module at the cap carries its band reason, which a change that touches it owes.
    (root / "ouroboros/size_ratchet_manifest.py").write_text(
        'BASELINE_SOURCE_SHA = "' + "0" * 40 + '"\n'
        'GIANT_PATHS = ()\nFUNCTION_DEBT = ()\nBAND_BASELINE_PATHS = ()\n'
        'BAND_PATHS = {"sample.py": "one module at the cap"}\nBYTE_BASELINE_DEBT = {}\nBYTE_DEBT = {}\n',
        encoding="utf-8")
    (root / "sample.py").write_text("def sample():\n    return 1\n" + "#\n" * (review.MAX_MODULE_LINES - 3),
                                    encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "-c", "commit.gpgsign=false", "commit", "-qm", "fixture"], cwd=root, check=True)
    return root


def test_real_readiness_keeps_information_out_of_warning_and_reuses_inventory(tmp_path, monkeypatch):
    root = _repo(tmp_path)
    information = []
    assert review_helpers.check_worktree_readiness(root, information=information) == [
        "No uncommitted changes detected — nothing to review."]
    assert information == []
    (root / "sample.py").write_text("# useful capacity fact\n" * (review.MAX_MODULE_LINES - 1), encoding="utf-8")
    real_collect = review.collect_size_ratchet_inventory
    calls = []

    def collect(*args, **kwargs):
        calls.append(args[0])
        return real_collect(*args, **kwargs)

    monkeypatch.setattr(review, "collect_size_ratchet_inventory", collect)
    assert review_helpers.check_worktree_readiness(root, information=information) == []
    assert len(calls) == 1
    assert any("sample.py:" in line for line in information)
    assert any("lines (1 remaining)" in line for line in information)
    assert not any("size_ratchet_manifest.py:" in line for line in information)
    calls.clear()

    result = health._codebase_health(SimpleNamespace(repo_dir=root))

    assert len(calls) == 1
    assert "Size Headroom (information; official CI enforces the limits)" in result
    assert "lines (1 remaining)" in result
    assert "manifest matches the live tree" in result
