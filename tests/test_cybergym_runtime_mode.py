"""Explicit runtime access survives the benchmark's settings projection."""
import json

import pytest

from devtools.benchmarks.cybergym.run_cybergym import (
    _prepare_applied_settings,
    parse_args,
)


@pytest.mark.parametrize(("timeout", "mode"), [(10800, "pro"), (21600, "cyber_pro")])
def test_runtime_mode_reaches_applied_settings(tmp_path, mode, timeout):
    args = parse_args(["--runtime-mode", mode, "--budget-usd", "200",
                       "--per-task-cost-usd", "10", "--workers", "32",
                       "--timeout-sec", str(timeout)])
    template = tmp_path / "template.json"
    template.write_text('{"OUROBOROS_RUNTIME_MODE": "advanced"}', encoding="utf-8")
    output = tmp_path / "run"
    output.mkdir()
    path, metadata = _prepare_applied_settings(template, output, args)
    applied = json.loads(path.read_text(encoding="utf-8"))
    assert applied["OUROBOROS_RUNTIME_MODE"] == mode
    assert metadata["runtime_mode"] == mode
    assert metadata["effective_overrides"]["OUROBOROS_RUNTIME_MODE"] == mode
    assert applied["OUROBOROS_MAX_WORKERS"] == 32
    assert applied["TOTAL_BUDGET"] == 200
    assert applied["OUROBOROS_TASK_ABS_CEILING_SEC"] == timeout
    assert metadata["task_abs_ceiling_sec"] == timeout
    assert applied["OUROBOROS_PER_TASK_COST_USD"] == 10
    assert applied["OUROBOROS_MAX_ROUNDS"] == 600


def test_runtime_mode_default_remains_pro():
    assert parse_args([]).runtime_mode == "pro"


@pytest.mark.parametrize("mode", ["advanced", "light", "typo"])
def test_runtime_mode_refuses_unsupported_benchmark_modes(mode):
    with pytest.raises(SystemExit):
        parse_args(["--runtime-mode", mode])


def test_default_six_hour_benchmark_settings_do_not_mutate_user_defaults(tmp_path, monkeypatch):
    import copy
    import os

    from ouroboros.settings_defaults import SETTINGS_DEFAULTS

    before = copy.deepcopy(SETTINGS_DEFAULTS)
    monkeypatch.setenv("OUROBOROS_TASK_ABS_CEILING_SEC", "1800")
    template = tmp_path / "template.json"
    template.write_text('{"OUROBOROS_TASK_ABS_CEILING_SEC": "unlimited"}', encoding="utf-8")
    original = template.read_bytes()
    output = tmp_path / "run"
    output.mkdir()
    path, metadata = _prepare_applied_settings(template, output, parse_args(["--per-task-cost-usd", "10"]))
    applied = json.loads(path.read_text(encoding="utf-8"))
    assert applied["OUROBOROS_TASK_ABS_CEILING_SEC"] == 21600
    assert metadata["task_abs_ceiling_sec"] == 21600
    assert SETTINGS_DEFAULTS == before
    assert os.environ["OUROBOROS_TASK_ABS_CEILING_SEC"] == "1800"
    assert template.read_bytes() == original
