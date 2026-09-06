"""Pins of the live E2E stand runner (``devtools/e2e_live``). Default lane: no server, no sockets, no network (the
provider probes are monkeypatched and the lane pool is replaced by a fake). The one real-server test at the end is
the keyless ``--stub`` rehearsal of SM1 and carries the same three gates as the system_e2e lane."""
from __future__ import annotations

import collections
import dataclasses
import json
import os
import pathlib
import subprocess
import sys
import threading
import time
import types
import urllib.error
import urllib.request

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from devtools.benchmarks.common import launcher_audit  # noqa: E402
from devtools.benchmarks.common.manifests import repo_provenance  # noqa: E402
from devtools.e2e_live import run_live_lanes, scenarios, stub_lane, ui_probe  # noqa: E402

FAKE_KEY = "sk-or-v1-e2e-live-test-key-value-never-printed-0123456789"


def _commit(repo: pathlib.Path, message: str) -> str:
    subprocess.run(["git", "add", "-A"], cwd=str(repo), check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@e.invalid", "commit", "-q", "-m", message],
                   cwd=str(repo), check=True)
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo), check=True,
                          capture_output=True, text=True).stdout.strip()


def _git_seed(root: pathlib.Path, *, dirty: bool = False) -> pathlib.Path:
    """A tiny SOURCE checkout: one committed VERSION, optionally a TRACKED uncommitted edit."""
    seed = root / "source"
    seed.mkdir()
    (seed / "VERSION").write_text("7.0.0-test\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=str(seed), check=True)
    _commit(seed, "seed")
    if dirty:
        (seed / "VERSION").write_text("7.0.0-dirty\n", encoding="utf-8")  # describe says -dirty
    return seed


def _fake_lane(job, args, out, template, stagger, states, seed, budget=None, *, key="", seed_sha=""):
    sid, attempt = job
    lane = out / "lanes" / f"{sid}_a{attempt}"
    ceiling = budget.ceiling(job) if budget is not None else None   # the real lane reads it before spending
    lane.mkdir(parents=True)
    row = {"scenario": sid, "attempt": attempt, "status": "pass", "checks": {"fake": True}, "error": "",
           "duration_sec": 0.1, "model_slots": {"OUROBOROS_MODEL": template.get("OUROBOROS_MODEL")},
           "lane_total_budget_usd": ceiling,
           "template_has_key": "OPENROUTER_API_KEY" in template, "key_handed": bool(key), "seed_sha": seed_sha}
    (lane / "result.json").write_text(json.dumps(row), encoding="utf-8")
    return row


class _Response:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return self._body


def _fake_urlopen(key_body: bytes, credits_body: bytes, calls: list):
    def fake(req, timeout=0):
        assert req.headers["Authorization"] == f"Bearer {FAKE_KEY}"
        calls.append(req.full_url)
        return _Response(key_body if req.full_url.endswith("/key") else credits_body)
    return fake


def _short_tmp(monkeypatch) -> None:
    monkeypatch.setattr(run_live_lanes.tempfile, "gettempdir", lambda: "/tmp/short")


# --------------------------------------------------------------------------- #
# Structure: the shared launcher gate, the table, the argv bounds
# --------------------------------------------------------------------------- #

def test_launcher_passes_the_shared_structural_gate():
    """Admission is the outer boundary, confinement follows the handed source, only the seam
    publishes the manifest — the SAME gate the benchmark family is held to, by source."""
    source = (REPO_ROOT / "devtools" / "e2e_live" / "run_live_lanes.py").read_text(encoding="utf-8")
    assert launcher_audit.audit_source(source, name="run_live_lanes.py") == []


def test_scenario_table_shape():
    assert set(scenarios.SCENARIOS) == {"SM1", "SW1", "SK1"}
    for sid, row in scenarios.SCENARIOS.items():
        assert row.id == sid and row.prompt.strip() and row.title.strip()
        assert isinstance(row.settings_overrides, dict)
        assert callable(row.acceptance) and callable(row.stub_script)
    sm1 = scenarios.SCENARIOS["SM1"].settings_overrides
    assert sm1 == {"OUROBOROS_RUNTIME_MODE": "advanced", "OUROBOROS_REVIEW_ENFORCEMENT": "blocking"}
    roster = json.loads(scenarios.SCENARIOS["SW1"].overrides("openai-compatible::mock-child")["OUROBOROS_SUBAGENTS"])
    assert roster["items"][0]["route"]["target_id"] == "openai-compatible::mock-child"
    assert scenarios.SCENARIOS["SW1"].overrides("x")["OUROBOROS_MAX_SUBAGENT_DEPTH"] == 1
    # The budget reservation unit: SK1 mints two root tasks (author + dispatch), the others one
    # (SW1's scouts spend under their single root's ceiling).
    assert {sid: row.root_tasks for sid, row in scenarios.SCENARIOS.items()} == {"SM1": 1, "SW1": 1, "SK1": 2}
    # Only SM1 lands a commit the post-task evolution absorbs: the --self-mod wait and check follow this flag.
    assert {sid: row.expects_absorb for sid, row in scenarios.SCENARIOS.items()} == {"SM1": True, "SW1": False, "SK1": False}
    # Stub scripts are role-keyed queues; SW1 needs every role the swarm wire interleaves.
    sw1 = scenarios.SCENARIOS["SW1"].stub_script(REPO_ROOT)
    assert set(sw1) == {"router", "agent", "child", "probe"}
    assert [s["tool"] for s in sw1["agent"] if isinstance(s, dict) and "tool" in s] == [
        "plan_task", "schedule_subagent", "schedule_subagent"]
    assert sum(1 for s in sw1["agent"] if callable(s)) == 3  # wait_tasks + two child dispositions


def test_css_accent_helpers_only_touch_the_root_token():
    css = ":root {\n    --accent: #c93545;\n    --accent-light: #f07a86;\n}\n.x { --accent: red; }\n"
    assert scenarios.accent_value(css) == "#c93545"
    changed = scenarios.css_with_accent(css, "#2f7de1")
    assert scenarios.accent_value(changed) == "#2f7de1"
    assert changed.count("#2f7de1") == 1 and "--accent-light: #f07a86" in changed and "--accent: red" in changed


def test_lane_count_and_stagger_bounds(monkeypatch):
    _short_tmp(monkeypatch)
    assert run_live_lanes.parse_args(["--stub"]).lanes == 4
    assert run_live_lanes.parse_args(["--stub", "--lanes", "6"]).lanes == 6
    assert run_live_lanes.parse_args(["--stub", "--stagger", "10"]).stagger == 3.0
    assert run_live_lanes.parse_args(["--stub", "--stagger", "0.1"]).stagger == 2.0
    assert run_live_lanes.parse_args(["--stub", "--stagger", "2.4"]).stagger == 2.4
    for argv in (["--lanes", "7"], ["--lanes", "0"], ["--model", "x/y"],   # --model: the stub IS the model
                 ["--scenarios", "SM1,NOPE"], ["--attempts", "2", "--pass-of", "3"]):
        with pytest.raises(SystemExit):
            run_live_lanes.parse_args(["--stub", *argv])
    args = run_live_lanes.parse_args(["--total-budget", "30"])
    assert args.min_credit_usd == 30.0 and args.key_env == run_live_lanes.DEFAULT_KEY_ENV
    assert args.seed == "HEAD" and args.source_repo == ""


def test_money_and_interval_arguments_must_be_finite_and_positive(monkeypatch):
    """A non-positive TOTAL_BUDGET means NO cap to the runtime and a non-positive tick is a hot
    loop: both are argument-shaped refusals, before anything touches the world."""
    _short_tmp(monkeypatch)
    for argv in (["--total-budget", "0"], ["--total-budget", "-5"], ["--total-budget", "inf"],
                 ["--total-budget", "nan"], ["--per-task-usd", "0"], ["--min-credit-usd", "0"],
                 ["--min-credit-usd", "inf"], ["--watch-interval", "0"], ["--watch-interval", "-1"],
                 ["--watch-interval", "nan"], ["--watch-interval", "1"], ["--seed", " "]):
        with pytest.raises(SystemExit):
            run_live_lanes.parse_args(["--stub", *argv])
    ok = run_live_lanes.parse_args(["--stub", "--watch-interval", str(run_live_lanes.WATCH_INTERVAL_MIN_SEC)])
    assert ok.watch_interval == run_live_lanes.WATCH_INTERVAL_MIN_SEC


def test_tmpdir_length_guard_refuses_fail_closed(monkeypatch):
    monkeypatch.setattr(run_live_lanes.tempfile, "gettempdir", lambda: "/tmp/" + "x" * 80)
    with pytest.raises(SystemExit):
        run_live_lanes.parse_args(["--stub"])


# --------------------------------------------------------------------------- #
# Effective settings: the tree's defaults, the budget knobs as settings keys, no env guesses
# --------------------------------------------------------------------------- #

def test_budget_and_per_task_caps_are_written_into_the_applied_settings(monkeypatch):
    _short_tmp(monkeypatch)
    args = run_live_lanes.parse_args(["--total-budget", "30", "--per-task-usd", "8"])
    cfg = run_live_lanes.effective_settings(args, FAKE_KEY)
    assert cfg["TOTAL_BUDGET"] == 30.0 and cfg["OUROBOROS_PER_TASK_COST_USD"] == 8.0
    assert cfg["OPENROUTER_API_KEY"] == FAKE_KEY and cfg["OUROBOROS_RUNTIME_MODE"] == "advanced"
    from ouroboros.provider_models import declared_model_settings
    for key, value in declared_model_settings({}).items():
        assert cfg[key] == value  # the defaults of the tree under test, written explicitly
    pinned = run_live_lanes.effective_settings(run_live_lanes.parse_args(["--model", "argv/model-y"]), FAKE_KEY)
    assert pinned["OUROBOROS_MODEL"] == "argv/model-y"
    # The run-root copy is redacted: the key survives only in memory and in the lane files.
    redacted = run_live_lanes.redacted_template(cfg)
    assert "OPENROUTER_API_KEY" not in redacted and redacted["OUROBOROS_MODEL"] == cfg["OUROBOROS_MODEL"]
    assert run_live_lanes.template_credentials(cfg) == {"OPENROUTER_API_KEY": FAKE_KEY}


def test_self_mod_is_off_by_default(monkeypatch):
    _short_tmp(monkeypatch)
    off = run_live_lanes.effective_settings(run_live_lanes.parse_args(["--stub"]), "")
    assert off["OUROBOROS_POST_TASK_EVOLUTION"] == "false" and "OUROBOROS_POST_TASK_EVOLUTION_CADENCE" not in off
    on = run_live_lanes.effective_settings(run_live_lanes.parse_args(["--stub", "--self-mod"]), "")
    assert on["OUROBOROS_POST_TASK_EVOLUTION"] == "true" and on["OUROBOROS_POST_TASK_EVOLUTION_CADENCE"] == "every_n:1"


def test_preflight_worker_cap_reaches_every_lane_and_is_recorded(tmp_path, monkeypatch):
    """The commit gate's hermetic pytest pass runs INSIDE the lane server and resolves ``-n auto``
    to the host CPU count (the 2026-09-04 paid run fanned out to >= 104 xdist workers per lane): the
    stand must set the runtime's own lever to ``max(2, 16 // lanes)`` in the process every lane
    server inherits, override an ambient value, and record the applied number in the manifest
    and in each lane row. The runtime reads exactly that key (pinned here, not modified)."""
    from ouroboros import preflight_runner
    _short_tmp(monkeypatch)
    assert run_live_lanes.PREFLIGHT_WORKERS_ENV == preflight_runner._PREFLIGHT_WORKERS_ENV
    assert run_live_lanes.PREFLIGHT_WORKERS_FLOOR == preflight_runner._MIN_PREFLIGHT_WORKERS
    assert run_live_lanes.parse_args(["--stub", "--lanes", "1"]).preflight_test_workers == 16
    assert run_live_lanes.parse_args(["--stub", "--lanes", "4"]).preflight_test_workers == 4
    assert run_live_lanes.parse_args(["--stub", "--lanes", "6"]).preflight_test_workers == 2   # floor at MAX_LANES
    monkeypatch.setenv(run_live_lanes.PREFLIGHT_WORKERS_ENV, "128")   # the operator shell must lose
    seen: dict = {}

    def lane(job, args, out, template, stagger, states, seed, budget=None, *, key="", seed_sha=""):
        seen[job] = (os.environ.get(run_live_lanes.PREFLIGHT_WORKERS_ENV),
                     run_live_lanes._lane_row(job, args)["preflight_test_workers"],
                     preflight_runner._preflight_worker_count())
        return _fake_lane(job, args, out, template, stagger, states, seed, budget, key=key, seed_sha=seed_sha)

    monkeypatch.setattr(run_live_lanes, "run_lane", lane)
    out = tmp_path / "out"
    rc = run_live_lanes.main(["--stub", "--source-repo", str(_git_seed(tmp_path)), "--out", str(out),
                              "--scenarios", "SM1,SW1", "--lanes", "3", "--watch-interval", "600"])
    assert rc == 0
    assert seen == {("SM1", 1): ("5", 5, 5), ("SW1", 1): ("5", 5, 5)}
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["lanes"] == 3 and manifest["extra"]["preflight_test_workers"] == 5


def test_isolated_server_forwards_the_preflight_worker_cap_through_the_authoritative_sweep(tmp_path, monkeypatch):
    """The lane servers start in settings-authoritative mode, which strips the whole OUROBOROS_
    namespace; the worker cap is the one operational lever that must survive, while an ambient
    model slot still does not."""
    from devtools.benchmarks.common.server_runner import _AUTHORITATIVE_ENV_KEEP, IsolatedServer
    assert run_live_lanes.PREFLIGHT_WORKERS_ENV in _AUTHORITATIVE_ENV_KEEP
    settings = tmp_path / "settings.json"
    settings.write_text("{}", encoding="utf-8")
    monkeypatch.setenv(run_live_lanes.PREFLIGHT_WORKERS_ENV, "4")
    monkeypatch.setenv("OUROBOROS_MODEL", "ambient/model")
    env = IsolatedServer(tmp_path / "clone", tmp_path / "data", settings, settings_authoritative_env=True)._env()
    assert env[run_live_lanes.PREFLIGHT_WORKERS_ENV] == "4" and "OUROBOROS_MODEL" not in env


def test_stub_template_carries_only_the_loopback_slots(monkeypatch):
    _short_tmp(monkeypatch)
    cfg = run_live_lanes.effective_settings(run_live_lanes.parse_args(["--stub"]), "")
    assert cfg["OUROBOROS_MODEL"] == stub_lane.STUB_MODEL_SLUG == cfg["OUROBOROS_MODEL_LIGHT"]
    assert not any(k.startswith("OUROBOROS_MODEL") and v and v != stub_lane.STUB_MODEL_SLUG for k, v in cfg.items())
    assert "OPENROUTER_API_KEY" not in cfg


def test_config_sha256_is_secret_free_and_key_independent():
    base = {"OUROBOROS_MODEL": "m", "TOTAL_BUDGET": 1.0}
    a = run_live_lanes.config_sha256({**base, "OPENROUTER_API_KEY": "key-one"})
    b = run_live_lanes.config_sha256({**base, "OPENROUTER_API_KEY": "key-two"})
    assert a != b  # a different key is a different (fingerprinted) config...
    assert a == run_live_lanes.config_sha256({**base, "OPENROUTER_API_KEY": "key-one"})
    assert a != run_live_lanes.config_sha256({**base, "OPENROUTER_API_KEY": "key-one", "TOTAL_BUDGET": 2.0})


# --------------------------------------------------------------------------- #
# The run-wide budget ledger
# --------------------------------------------------------------------------- #

def test_lane_spend_sums_durable_llm_usage_rows_and_counts_unknown_costs(tmp_path):
    logs = tmp_path / "data" / "logs"
    logs.mkdir(parents=True)
    rows = [{"type": "llm_usage", "cost": 1.5}, {"type": "llm_usage", "cost": 0.25},
            {"type": "llm_usage", "cost": None, "cost_known": False}, {"type": "task_done", "cost": 99.0},
            {"type": "llm_usage", "cost": True}]
    (logs / "events.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    assert run_live_lanes.lane_spend(tmp_path / "data") == (1.75, 2)
    assert run_live_lanes.lane_spend(tmp_path / "absent") == (0.0, 0)


def _ask(budget, job, root_tasks, root, waits: list | None = None, *, index: int = 0):
    """``admit`` on its own thread (it may block): ``(thread, box)``; ``box["r"]`` is the answer."""
    box: dict = {}
    thread = threading.Thread(target=lambda: box.__setitem__("r", budget.admit(
        job, root_tasks, root, dispatch_index=index, on_wait=waits.append if waits is not None else None)), daemon=True)
    thread.start()
    thread.join(0.3)
    return thread, box


def test_run_budget_waits_on_in_flight_reservations_and_refuses_only_what_can_never_fit(tmp_path):
    """Per attempt: spent (durable, re-read) + reservation > cap -> refused, no run-wide halt; fits the cap but not
    the reservations in flight -> waits and re-asks after EVERY settle (the first paid run wrote SW1/SK1 off at t=+21
    min behind two SM1 reservations still in flight); spent only grows, so a waiter can end refused with its wait
    recorded. A lane's TOTAL_BUDGET is its OWN reservation: the ceilings in flight are disjoint and settled spend +
    in-flight ceilings never exceeds the cap (the first draft handed each lane cap - others' reservations). The
    reservation unit is per-task x root tasks: $8 per task reserves $8 per root, $16 for SK1's two."""
    spend = {}
    budget = run_live_lanes.RunBudget(20.0, 8.0, reader=lambda root: (spend.get(root.name, 0.0), 0))
    assert budget.reservation(1) == 8.0 and budget.reservation(2) == 16.0 and budget.reservation(0) == 8.0
    ok, facts = budget.admit(("SM1", 1), 1, tmp_path / "a", dispatch_index=0)
    assert ok and facts == {"cap_usd": 20.0, "spent_usd": 0.0, "reserved_usd": 0.0, "reservation_usd": 8.0,
                            "unknown_cost_rows": 0, "waited_sec": 0.0}
    assert budget.ceiling(("SM1", 1)) == 8.0            # its own reservation, never the whole cap
    ok, facts = budget.admit(("SW1", 1), 1, tmp_path / "b", dispatch_index=1)
    assert ok and facts["reserved_usd"] == 8.0
    assert budget.ceiling(("SW1", 1)) == 8.0            # disjoint from lane a: 8 + 8 + spent 0 <= cap 20
    assert budget.ceiling(("SM1", 1)) + budget.ceiling(("SW1", 1)) <= 20.0
    spend["a"] = 5.0                                    # lane a spends while in flight: visible now
    ok, facts = budget.admit(("SK1", 1), 2, tmp_path / "c", dispatch_index=2)   # 5 + 16 > 20: can NEVER fit -> refused at once
    assert not ok and facts["spent_usd"] == 5.0 and facts["waited_sec"] == 0.0 and budget.not_run == ["SK1_a1"]
    waits: list = []
    thread, box = _ask(budget, ("SM1", 2), 1, tmp_path / "d", waits, index=3)   # 5 + 8 <= 20 but 5 + 16 + 8 > 20: waits
    assert thread.is_alive() and budget.not_run == ["SK1_a1"]   # not refused: the blocker is in flight
    assert waits == ["waiting — in flight reserved $16.00, needs $8.00, spent $5.00, cap $20.00"]
    budget.settle(("SM1", 1))                           # 5 + 8 + 8 > 20: re-asked, still waiting
    thread.join(0.3)
    assert thread.is_alive() and waits[1:] == []        # told once per wait, not per wake-up
    budget.settle(("SW1", 1))                           # 5 + 0 + 8: admitted after the wait
    thread.join(5.0)
    assert not thread.is_alive() and box["r"][0] and box["r"][1]["reserved_usd"] == 0.0 and box["r"][1]["waited_sec"] > 0
    thread, box = _ask(budget, ("SM1", 3), 1, tmp_path / "e", index=4)   # 5 + 8 <= 20 but 5 + 8 + 8 > 20: waits
    assert thread.is_alive()
    spend["d"] = 10.0                                   # the lane in flight overruns: spent 15 on the next question
    budget.settle(("SM1", 2))                           # 15 + 8 > 20: refused AFTER the wait
    thread.join(5.0)
    assert not thread.is_alive() and not box["r"][0] and box["r"][1]["spent_usd"] == 15.0 and box["r"][1]["waited_sec"] > 0
    snap = budget.snapshot()
    assert snap["spent_usd"] == 15.0 and snap["reserved_usd"] == 0.0 and snap["lanes_settled"] == 3
    assert snap["attempts_not_run"] == ["SK1_a1", "SM1_a3"] and snap["first_refused"] == "SK1_a1" and "halted" not in snap
    assert [(r["attempt"], r["reason"], r["reservation_usd"]) for r in snap["refusals"]] == [
        ("SK1_a1", "budget_cap", 16.0), ("SM1_a3", "budget_cap", 8.0)]
    assert snap["refusals"][0]["waited_sec"] == 0.0 and snap["refusals"][1]["waited_sec"] > 0
    assert snap["reservation_rule"] == run_live_lanes.RESERVATION_RULE
    # The ceiling ignores what OTHER lanes spend (it is this lane's reservation), and the floor
    # keeps it positive (the runtime reads a non-positive TOTAL_BUDGET as NO cap).
    tiny = run_live_lanes.RunBudget(10.0, 8.0, reader=lambda root: (20.0, 0))
    assert tiny.admit(("SM1", 1), 1, tmp_path / "x", dispatch_index=0)[0]
    assert tiny.ceiling(("SM1", 1)) == 8.0
    assert tiny.ceiling(("never", 9)) == run_live_lanes.LANE_BUDGET_FLOOR_USD   # not admitted: the floor, not the cap
    # The floor is part of the ONE effective ceiling: admission reserves it, the lane receives it,
    # so micro reservations cannot sum past the cap (5 x 0.01 fit a 0.05 cap, the 6th waits on them).
    micro = run_live_lanes.RunBudget(0.05, 0.001, reader=lambda root: (0.0, 0))
    assert micro.reservation(1) == run_live_lanes.LANE_BUDGET_FLOOR_USD
    for n in range(5):
        ok, facts = micro.admit(("SM1", n), 1, tmp_path / f"m{n}", dispatch_index=n)
        assert ok and facts["reservation_usd"] == 0.01 and micro.ceiling(("SM1", n)) == 0.01
    sixth, box = _ask(micro, ("SM1", 5), 1, tmp_path / "m5", index=5)
    assert sixth.is_alive() and micro.not_run == []
    assert sum(micro.ceiling(("SM1", n)) for n in range(5)) <= 0.05
    for n in range(5):
        micro.settle(("SM1", n))
    sixth.join(5.0)
    assert not sixth.is_alive() and box["r"][0]       # admitted once the five settled at $0
    below = run_live_lanes.RunBudget(0.005, 0.001, reader=lambda root: (0.0, 0))
    assert not below.admit(("SM1", 1), 1, tmp_path / "z", dispatch_index=0)[0]   # the floored reservation exceeds the cap
    # Fractional reservations are never rounded upward (round(0.01006, 4) would hand out 0.0101):
    # two exact 0.01006 reservations fill a 0.02012 cap and each lane receives exactly 0.01006.
    frac = run_live_lanes.RunBudget(0.02012, 0.01006, reader=lambda root: (0.0, 0))
    assert frac.admit(("SM1", 1), 1, tmp_path / "f1", dispatch_index=0)[0] and frac.admit(("SM1", 2), 1, tmp_path / "f2", dispatch_index=1)[0]
    assert frac.ceiling(("SM1", 1)) == 0.01006 and frac.ceiling(("SM1", 2)) == 0.01006
    assert frac.ceiling(("SM1", 1)) + frac.ceiling(("SM1", 2)) <= 0.02012
    third, box = _ask(frac, ("SM1", 3), 1, tmp_path / "f3", index=2)
    assert third.is_alive()                             # full: waits, not refused
    frac.settle(("SM1", 1))
    frac.settle(("SM1", 2))
    third.join(5.0)
    assert not third.is_alive() and box["r"][0]


def test_admission_is_fifo_by_dispatch_index_and_a_refused_head_frees_the_line(tmp_path):
    """A later-dispatched attempt that WOULD fit waits while an earlier one is still asking (the freed lane's next
    job can no longer leapfrog the woken waiter); a head that can never fit is refused and leaves the line. cap 30 /
    per-task 8: SK1 #0 (16) in flight; SK1 #1 (16) waits on it (32 > 30); SM1 #2 (8) would fit (24 <= 30) but waits
    behind #1, its wait naming that; lane #0 spends 15 and settles: #1 refused (31 > 30), #2 admitted at reserved $0."""
    spend: dict = {}
    budget = run_live_lanes.RunBudget(30.0, 8.0, reader=lambda root: (spend.get(root.name, 0.0), 0))
    assert budget.admit(("SK1", 1), 2, tmp_path / "a", dispatch_index=0)[0]
    head_waits, later_waits = [], []
    head, head_box = _ask(budget, ("SK1", 2), 2, tmp_path / "b", head_waits, index=1)
    later, later_box = _ask(budget, ("SM1", 1), 1, tmp_path / "c", later_waits, index=2)
    assert head.is_alive() and later.is_alive() and budget.not_run == []
    assert head_waits == ["waiting — in flight reserved $16.00, needs $16.00, spent $0.00, cap $30.00"]
    assert later_waits == ["waiting — behind SK1_a2 in dispatch order, needs $8.00, spent $0.00, cap $30.00"]
    spend["a"] = 15.0
    budget.settle(("SK1", 1))
    head.join(5.0)
    later.join(5.0)
    assert not head.is_alive() and not head_box["r"][0] and head_box["r"][1]["spent_usd"] == 15.0
    assert not later.is_alive() and later_box["r"][0] and later_box["r"][1]["reserved_usd"] == 0.0
    assert budget.not_run == ["SK1_a2"] and budget.ceiling(("SM1", 1)) == 8.0 and later_box["r"][1]["waited_sec"] > 0


def test_reservation_counts_roots_plus_the_evolution_root_and_is_the_lane_total_budget(tmp_path, monkeypatch):
    """EQUALITY pins of the rc.14/rc.15 finding: the reservation is per-task x root tasks, +1 with --self-mod (the one
    post-task cycle; rc.14: SM1_a1 task $3.84 + cycles $12.40 + $2.84 of $20 — the lane's TOTAL_BUDGET is the fence); the 2x
    factor and its product import are gone and no bench budget profile is projected. Per-task $20 and one root reserve
    $20 ($40 for the absorbing SM1 root with --self-mod; SK1's two roots stay $40, it does not promote) and that exact number reaches the lane's settings file as
    TOTAL_BUDGET through ``run_lane`` (never the run-wide cap)."""
    _short_tmp(monkeypatch)
    rule = run_live_lanes.RESERVATION_RULE
    assert not hasattr(run_live_lanes, "HARD_STOP_INVERSE") and rule == run_live_lanes.RunBudget(1, 1).snapshot()["reservation_rule"]
    assert rule.startswith("max(0.01, per_task_usd x (root_tasks + 1 if --self-mod and the scenario absorbs else root_tasks))")
    assert "post-task cycle of a lane that promotes" in rule and "the true fence" in rule and "cost_hard_stop" not in rule
    budget = run_live_lanes.RunBudget(100.0, 20.0, reader=lambda root: (0.0, 0))
    assert budget.reservation(1) == 20.0 and budget.reservation(2) == 40.0 and not budget.self_mod
    evolving = run_live_lanes.RunBudget(100.0, 20.0, reader=lambda root: (0.0, 0), self_mod=True)
    assert evolving.reservation(1, absorbs=True) == 40.0 and evolving.reservation(2) == 40.0 and evolving.reservation(1) == 20.0
    seed = _git_seed(tmp_path)
    out, job = tmp_path / "out", ("SM1", 1)
    ok, facts = budget.admit(job, 1, out / "lanes" / "SM1_a1" / "data", dispatch_index=0)
    assert ok and facts["reservation_usd"] == 20.0 and budget.ceiling(job) == 20.0

    class _NoServer(_NoopServer):                       # the real path up to the written settings, then stop
        def start(self, **_k) -> None:
            raise RuntimeError("no server in this pin: the settings file on disk is the evidence")

    monkeypatch.setattr(run_live_lanes, "IsolatedServer", _NoServer)
    args = run_live_lanes.parse_args(["--per-task-usd", "20", "--total-budget", "100", "--scenarios", "SM1",
                                      "--out", str(out), "--watch-interval", "600"])
    template = run_live_lanes.effective_settings(args, FAKE_KEY)
    assert template["TOTAL_BUDGET"] == 100.0             # the run cap; every lane rewrites it with its ceiling
    row = run_live_lanes.run_lane(job, args, out, template, run_live_lanes.Stagger(2.0), {}, seed, budget,
                                  key=FAKE_KEY, seed_sha=run_live_lanes.head_sha(seed))
    applied = json.loads((out / "lanes" / "SM1_a1" / "data" / "settings.json").read_text(encoding="utf-8"))
    assert applied["TOTAL_BUDGET"] == 20.0 == budget.ceiling(job) == budget.reservation(1)
    assert applied["OUROBOROS_PER_TASK_COST_USD"] == 20.0 and applied["OPENROUTER_API_KEY"] == FAKE_KEY
    assert row["budget"] == {"reservation_usd": 20.0, "lane_total_budget_usd": 20.0, "per_task_usd": 20.0,
                             "spent_usd": 0.0, "unknown_cost_rows": 0}
    assert row["status"] == "infra_error" and row["refusal"]["type"] == "RuntimeError"


def test_submit_injects_no_budget_profile_into_the_stand_roots(monkeypatch):
    """Every stand root runs under the PRODUCT'S default in-task ceiling: no ``metadata.budget_profile``
    (a bench profile would change the pacing path under test; with a reservation >= 2 x per-task the
    per-task axis binds first). Only the stand's identity and the scenario's own metadata are sent."""
    bodies: list = []
    monkeypatch.setattr(scenarios, "_api", lambda base, method, path, payload=None, timeout=0:
                        bodies.append((method, path, payload)) or {"task_id": "t-1"})
    assert _ctx().submit("do it", metadata={"force_plan": True, "force_plan_source": "swarm"}) == "t-1"
    (method, path, body), = bodies
    assert (method, path) == ("POST", "/api/tasks") and body["timeout_sec"] == 1
    assert body["metadata"] == {"source": "e2e_live", "delegation_role": "root", "force_plan": True, "force_plan_source": "swarm"}
    assert not hasattr(scenarios, "STAND_BUDGET_PROFILE") and "budget_profile" not in json.dumps(body)


def test_budget_preflight_refuses_reservations_that_can_never_all_be_admitted(tmp_path, monkeypatch):
    """The rc.15 plan under the 2x rule (cap 200, SK1 reserving the whole cap, attempts 3) would have burned SM1/SW1 and
    refused every SK1 attempt by construction. The preflight refuses BEFORE any spend a reservation above the cap, or
    equal to it with attempts >= 2 (the second can never be admitted after any spend), in the credit preflight's typed
    shape, before the key, the seed or a lane; no override. The per-ROUND worst case is the --lanes largest reservations,
    ONE attempt per scenario: the owner's cap 300 / per-task 50 / --self-mod / 3 lanes = $250 (SM1 100, SK1 100, SW1 50)."""
    def rows(budget, attempts, ids=("SM1", "SW1", "SK1"), lanes=3):
        pre = run_live_lanes.budget_preflight(budget, list(ids), attempts, lanes)
        return ({r["scenario"]: r["reservation_usd"] for r in pre["scenarios"]}, pre["worst_case_usd"], pre["unreachable"],
                pre["round_worst_case_usd"])

    reader = lambda root: (0.0, 0)   # noqa: E731 - a stub reader
    assert rows(run_live_lanes.RunBudget(300.0, 50.0, reader, self_mod=True), 3) == ({"SM1": 100.0, "SW1": 50.0, "SK1": 100.0}, 750.0, [], 250.0)
    assert rows(run_live_lanes.RunBudget(300.0, 50.0, reader, self_mod=True), 3, lanes=2)[3] == 200.0
    assert rows(run_live_lanes.RunBudget(90.0, 50.0, reader, self_mod=True), 1) == ({"SM1": 100.0, "SW1": 50.0, "SK1": 100.0}, 250.0, ["SM1", "SK1"], 250.0)
    at_cap = run_live_lanes.RunBudget(100.0, 50.0, reader, self_mod=True)   # SM1/SK1 == cap: one attempt fits at $0, never a second
    assert rows(at_cap, 1)[2] == [] and rows(at_cap, 2)[2] == ["SM1", "SK1"]
    assert rows(run_live_lanes.RunBudget(200.0, 100.0, reader), 3)[2] == ["SK1"]                 # the shipped 2x rule's SK1 = 200 of 200
    pre = run_live_lanes.budget_preflight(run_live_lanes.RunBudget(200.0, 50.0, reader), ["SK1"], 2, 4)
    assert pre == {"cap_usd": 200.0, "per_task_usd": 50.0, "self_mod": False, "reservation_rule": run_live_lanes.RESERVATION_RULE,
                   "scenarios": [{"scenario": "SK1", "root_tasks": 2, "reservation_usd": 100.0, "attempts": 2,
                                  "worst_case_usd": 200.0, "unreachable": False}], "worst_case_usd": 200.0, "lanes": 4,
                   "round_worst_case_usd": 100.0, "unreachable": []}
    out, manifest = _fake_run(tmp_path, monkeypatch, ["--total-budget", "90", "--per-task-usd", "50", "--self-mod",
                                                      "--scenarios", "SM1,SW1,SK1"],
                              lane=lambda *a, **k: pytest.fail("a lane started after a budget refusal"), expect_rc=3)
    refusal = manifest["extra"]["refusal"]
    assert refusal["stage"] == "budget_preflight" and refusal["reason"] == "reservation_unreachable"
    assert refusal["unreachable"] == ["SM1", "SK1"] and refusal["cap_usd"] == 90.0 and refusal["self_mod"] is True
    assert manifest["extra"]["budget_preflight"]["unreachable"] == ["SM1", "SK1"] and manifest["extra"]["exit_code"] == 3
    assert "credential_fingerprint" not in manifest["extra"] and not (out / "seed").exists() and not (out / "lanes").exists()
    assert manifest["requested_task_ids"] == ["SM1_a1", "SW1_a1", "SK1_a1"]   # SM1/SW1 = $100 = cap: one attempt fits


def test_jobs_are_dispatched_round_robin_by_attempt_largest_reservation_first_within_a_round(tmp_path, monkeypatch):
    """``dispatch_order``: a1 of every scenario, then a2 (the verdict is pass-of PER scenario: the order protects the
    MINIMUM admitted per scenario); within a round SK1 (two roots) asks before SM1 and SW1 (stable among equals), and
    admission keeps that order (FIFO by index). Requested ids keep the argument order; the per-round worst case is recorded."""
    order: list = []

    def lane(job, *a, **k):
        order.append(f"{job[0]}_a{job[1]}")
        return _fake_lane(job, *a, **k)

    _out, manifest = _fake_run(tmp_path, monkeypatch, ["--total-budget", "200", "--per-task-usd", "50", "--lanes", "1",
                                                       "--scenarios", "SM1,SW1,SK1", "--attempts", "2"], lane=lane)
    assert order == ["SK1_a1", "SM1_a1", "SW1_a1", "SK1_a2", "SM1_a2", "SW1_a2"]
    assert manifest["requested_task_ids"] == ["SM1_a1", "SM1_a2", "SW1_a1", "SW1_a2", "SK1_a1", "SK1_a2"]
    assert manifest["extra"]["budget_preflight"]["unreachable"] == [] and manifest["extra"]["outcome"] == "completed"
    assert manifest["extra"]["budget_preflight"]["worst_case_usd"] == 400.0
    assert manifest["extra"]["budget_preflight"]["lanes"] == 1 and manifest["extra"]["budget_preflight"]["round_worst_case_usd"] == 100.0


# --------------------------------------------------------------------------- #
# Feasibility pins with POSITIVE spends: the audit's driver over the REAL ledger
# --------------------------------------------------------------------------- #

class _Driver:
    """``main()``'s pool replaced by a virtual clock over the REAL ``RunBudget``: ``admit``/``settle`` as ``run_attempt``
    makes them (indices from ``dispatch_order``), spend visible at settle, lane durations in virtual minutes (rc.14 SM1
    22-54, rc.11 SW1 ~10, SK1 ~7). The ledger's ``wait`` is a park the driver releases one thread at a time, so the
    schedule is the runner's OBSERVED one, never the OS's: after a settle the freed lane's next job asks FIRST (it wins
    the lock on CPython, 300/300), then the parked attempts re-ask in dispatch order; a refusal frees its lane at once."""
    DURATION = {"SM1": 50, "SW1": 10, "SK1": 7}

    def __init__(self, cap, per_task, scenario_ids, attempts, spends, *, lanes=3, self_mod=False) -> None:
        self.spend: dict = {}
        self.budget = run_live_lanes.RunBudget(cap, per_task, reader=lambda root: (self.spend.get(root.name, 0.0), 0),
                                               self_mod=self_mod)
        requested = [(sid, n) for sid in scenario_ids for n in range(1, attempts + 1)]
        self.pending = collections.deque(enumerate(run_live_lanes.dispatch_order(self.budget, requested)))
        self.lanes, self.spends, self.now = lanes, spends, 0.0
        self.in_flight, self.parked, self.threads, self.admitted, self.refused = {}, {}, {}, [], []
        lock = self.budget._lock

        def park(*_a, **_k) -> None:   # the ledger's wait: release the lock, hold until the driver wakes this thread
            gate = self.parked[threading.current_thread().name] = threading.Event()
            lock.release()
            gate.wait()
            lock.acquire()

        lock.wait = park

    def _ask(self, index: int, job) -> None:
        name, box, row = f"{job[0]}_a{job[1]}", {}, scenarios.SCENARIOS[job[0]]
        thread = threading.Thread(name=name, daemon=True, target=lambda: box.__setitem__("r", self.budget.admit(
            job, row.root_tasks, pathlib.Path("/x") / name, dispatch_index=index, absorbs=row.expects_absorb)))
        self.threads[name] = (index, job, thread, box)
        thread.start()
        self._settle_thread(name)

    def _settle_thread(self, name: str) -> None:
        """Spin until the thread has answered or parked; the deadline is a hang guard, never a timing assumption."""
        _index, job, thread, box = self.threads[name]
        deadline = time.monotonic() + 10.0
        while thread.is_alive() and name not in self.parked:
            assert time.monotonic() < deadline, f"{name} neither answered nor parked"
            time.sleep(0.0005)
        if not thread.is_alive():                                  # answered: admitted (in flight now) or refused
            del self.threads[name]
            (self.admitted if box["r"][0] else self.refused).append(name)
            if box["r"][0]:
                self.in_flight[job] = self.now + self.DURATION[job[0]]

    def run(self) -> tuple[list, list, float]:   # (admitted in admission order, refused in refusal order, spend)
        while self.pending or self.in_flight:
            while self.pending and len(self.in_flight) + len(self.parked) < self.lanes:
                self._ask(*self.pending.popleft())
            assert self.in_flight, "parked attempts with nothing in flight (the ledger contract forbids it)"
            job = min(self.in_flight, key=lambda j: (self.in_flight[j], j))
            self.now = self.in_flight.pop(job)
            self.spend[f"{job[0]}_a{job[1]}"] = self.spends[job[0]]
            self.budget.settle(job)
            if self.pending:                                       # the freed lane's next job asks before the line re-asks
                self._ask(*self.pending.popleft())
            for name in sorted(self.parked, key=lambda n: self.threads[n][0]):   # then the line, earliest first
                self.parked.pop(name).set()
                self._settle_thread(name)
        return self.admitted, self.refused, round(self.budget.snapshot()["spent_usd"], 2)


REALISTIC_SPEND = {"SM1": 30.0, "SW1": 8.0, "SK1": 15.0}    # assumed per-attempt spends: rc.14 SM1 lanes, rc.11 SW1/SK1
PESSIMISTIC_SPEND = {"SM1": 45.0, "SW1": 8.0, "SK1": 30.0}
OWNER_CONFIGURATION = dict(cap=300.0, per_task=50.0, scenario_ids=["SM1", "SW1", "SK1"], attempts=3, lanes=3, self_mod=True)
DISPATCH_ORDER = ["SM1_a1", "SK1_a1", "SW1_a1", "SM1_a2", "SK1_a2", "SW1_a2", "SM1_a3", "SK1_a3", "SW1_a3"]


def test_owner_configuration_cap_300_per_task_50_three_attempts_self_mod_is_exact_under_fifo_admission():
    """The live configuration (cap 300, per-task 50, attempts 3, pass-of 2, 3 lanes, --self-mod: SM1 reserves 100 — its
    root plus the post-task cycle only it promotes — SK1 100 for two roots, SW1 50; round 1 = 250 fits) — EXACT
    sequences, no wake-order range. Realistic spends: all nine admitted in dispatch order, $159. Pessimistic: SK1_a3
    refused ($219 + 100 > 300): 8/9 at $219, every scenario keeping two = pass-of. Largest-first dispatch under the
    earlier +1-for-every-lane rule refused all of SW1 ($225, 0/3): the handbook's traced reason, prose, not a pin."""
    assert _Driver(spends=REALISTIC_SPEND, **OWNER_CONFIGURATION).run() == (DISPATCH_ORDER, [], 159.0)
    admitted, refused, spent = _Driver(spends=PESSIMISTIC_SPEND, **OWNER_CONFIGURATION).run()
    assert (admitted, refused, spent) == (DISPATCH_ORDER[:7] + ["SW1_a3"], ["SK1_a3"], 219.0)
    assert {s: sum(n.startswith(s) for n in admitted) for s in ("SM1", "SW1", "SK1")} == {"SM1": 3, "SW1": 3, "SK1": 2}


# --------------------------------------------------------------------------- #
# The watcher's key probe: informational, bounded, backing off, never on the tick's path
# --------------------------------------------------------------------------- #

def test_key_probe_failures_are_informational_and_back_off():
    stop = threading.Event()
    calls = {"n": 0}

    def flaky() -> float | None:
        calls["n"] += 1
        if calls["n"] <= 2:
            raise TimeoutError("timed out")
        return 3.0

    probe = run_live_lanes.KeyProbe(flaky, floor=5.0, interval=30.0, stop=stop)
    assert probe.interval == run_live_lanes.PROBE_MIN_INTERVAL_SEC   # never more often than the floor
    assert probe.fragment() == "key probe pending"
    probe.poll_once()
    assert probe.failures == 1 and "ALERT" not in probe.fragment()
    assert probe.fragment().startswith("key probe failed: TimeoutError") and "informational" in probe.fragment()
    assert probe.next_wait() == 2 * run_live_lanes.PROBE_MIN_INTERVAL_SEC
    probe.poll_once()
    assert probe.failures == 2 and probe.next_wait() == 4 * run_live_lanes.PROBE_MIN_INTERVAL_SEC
    probe.failures = 10
    assert probe.next_wait() == run_live_lanes.PROBE_BACKOFF_MAX_SEC
    probe.poll_once()                                    # a good reading resets the back-off
    assert probe.failures == 0 and probe.next_wait() == run_live_lanes.PROBE_MIN_INTERVAL_SEC
    assert probe.fragment() == "key remaining $3.00 ALERT"   # ALERT only on a GOOD reading under the floor
    probe.seed(None)
    assert probe.fragment() == "key uncapped"


def test_watcher_tick_never_waits_on_the_key_probe(capsys):
    """A probe stuck in a provider call must not delay the tick: the watcher reads the probe's
    last fragment and prints the ledger's spend regardless."""
    stop, release = threading.Event(), threading.Event()

    def stuck() -> float | None:
        release.wait(10)
        return None

    probe = run_live_lanes.KeyProbe(stuck, floor=1.0, interval=30.0, stop=stop)
    probe.interval = 0.01
    probe.start()
    budget = run_live_lanes.RunBudget(50.0, 16.0, reader=lambda root: (2.5, 0))
    budget.admit(("SM1", 1), 1, pathlib.Path("/nonexistent/lane/data"), dispatch_index=0)
    states = {("SM1", 1): ("running scenario", time.time())}
    thread = threading.Thread(target=run_live_lanes.watcher, args=(stop, states, 0.05, budget, probe), daemon=True)
    thread.start()
    seen = ""
    deadline = time.time() + 5
    while "[watch]" not in seen and time.time() < deadline:
        time.sleep(0.05)
        seen += capsys.readouterr().out
    stop.set()
    release.set()
    thread.join(timeout=5)
    line = next(ln for ln in seen.splitlines() if "[watch]" in ln)
    assert "spent $2.50/$50.00 reserved $16.00" in line and "SM1_a1=running scenario" in line   # $16 per task, one root
    assert "key probe pending" in line and "ALERT" not in line


# --------------------------------------------------------------------------- #
# Self-modification: a confirmed absorb, never an assumed one
# --------------------------------------------------------------------------- #

def _campaign(data_root: pathlib.Path, cycles: int, tx: dict | None = None) -> None:
    (data_root / "state").mkdir(parents=True, exist_ok=True)
    (data_root / "state" / "evolution_campaign.json").write_text(json.dumps(
        {"absorbed_cycles_done": cycles, "transaction_history": [tx] if tx else []}), encoding="utf-8")


def test_confirm_absorb_requires_positive_evidence(tmp_path, monkeypatch):
    clone = _git_seed(tmp_path)
    (clone / "f").write_text("1\n", encoding="utf-8")
    first = _commit(clone, "one")
    data_root = tmp_path / "data"
    _campaign(data_root, 0)
    state = {"sha": first[:8], "uptime": 100}
    monkeypatch.setattr(run_live_lanes, "_api", lambda base, method, path, payload=None, timeout=0: dict(state))
    pre = run_live_lanes.self_mod_snapshot(_FakeServer(), clone, data_root)
    assert pre["head"] == first and pre["sha"] == first[:8] and pre["cycles"] == 0 and pre["state_read"]
    # No promotion: the runtime declined, and a liveness check alone would have said PASS.
    out = run_live_lanes.confirm_absorb(_FakeServer(absorb={"absorbed": False, "reason": "no_promotion"}), clone,
                                        data_root, pre, timeout=1, ready_timeout=1)
    assert out["confirmed"] is False and out["reason"] == "no_promotion" and out["head_moved"] is False
    # The wait said absorbed and the counter advanced, but the served uptime never reset: not restarted.
    (clone / "f").write_text("2\n", encoding="utf-8")
    second = _commit(clone, "two")
    _campaign(data_root, 1, {"commit_sha": second, "cycle_outcome": "absorbed", "restart_verified": True,
                             "verified_by": "boot_reconciliation"})
    state.update({"sha": second[:8], "uptime": 100})
    out = run_live_lanes.confirm_absorb(_FakeServer(absorb={"absorbed": True, "reason": "absorbed"}), clone,
                                        data_root, pre, timeout=1, ready_timeout=1)
    assert out["confirmed"] is False and out["reason"] == "not_restarted" and out["head_moved"] is True
    assert out["transaction"] == {"commit_sha": second, "cycle_outcome": "absorbed", "restart_verified": True,
                                  "verified_by": "boot_reconciliation"}
    # Every fact present: counter advanced, sha moved, uptime reset, healthy, serving the clone HEAD.
    state["uptime"] = 0
    out = run_live_lanes.confirm_absorb(_FakeServer(absorb={"absorbed": True, "reason": "absorbed"}), clone,
                                        data_root, pre, timeout=1, ready_timeout=1)
    assert out["confirmed"] is True and out["reason"] == "absorbed" and out["serving_head"] is True
    assert out["post"]["cycles"] == 1 and out["post"]["head"] == second
    unhealthy = run_live_lanes.confirm_absorb(_FakeServer(absorb={"absorbed": True, "reason": "absorbed"}, healthy=False),
                                              clone, data_root, pre, timeout=1, ready_timeout=1)
    assert unhealthy["confirmed"] is False and unhealthy["reason"] == "unhealthy"


# --------------------------------------------------------------------------- #
# Scenario contracts: per-task check keys, the dispatch verdict, typed refusal facts, SM1 parity
# --------------------------------------------------------------------------- #

class _FakeServer:   # the scenario-facing surface (wait_task/cancel_task) and the one confirm_absorb reads
    base_url = "http://127.0.0.1:1"

    def __init__(self, status: str = "completed", *, absorb: dict | None = None, healthy: bool = True) -> None:
        self.status, self.absorb, self.healthy = status, dict(absorb or {}), healthy

    def wait_task(self, task_id, timeout=0):
        return {"status": self.status, "reason_code": "final_message" if self.status == "completed" else "deadline_local"}

    def cancel_task(self, task_id):
        return {}

    def wait_for_absorb(self, prev_sha, prev_absorbed, timeout=0):
        return dict(self.absorb)

    def wait_for_health(self, timeout=0):
        return self.healthy


class _FakeHarness:
    @staticmethod
    def wait_durable_result(oracle, task_id, timeout=0):
        return {"status": "completed", "reason_code": "final_message", "task_id": task_id}


def _ctx(server=None, *, ui_resolver=None, restart=lambda: None, shots=pathlib.Path("/s")) -> scenarios.LaneContext:
    return scenarios.LaneContext(server=server or _FakeServer(), clone=pathlib.Path("/x"), data_root=pathlib.Path("/y"),
                                 oracle=None, harness=_FakeHarness(), ui_resolver=ui_resolver, ui_reason="", shots=shots,
                                 log=lambda m: None, task_timeout=1, restart=restart)


try:
    from playwright.sync_api import TargetClosedError  # newer Playwright re-exports it
except ImportError:  # pragma: no cover - depends on the installed Playwright
    try:
        from playwright._impl._errors import TargetClosedError
    except ImportError:
        class TargetClosedError(Exception):  # type: ignore[no-redef]
            """Stand-in with Playwright's class name when Playwright is not installed."""


class _FakeUI:
    """A UI client recording its lifecycle; ``goto`` raises ``fail_goto`` when given (the dead
    target of the rc.14 incident: the chrome died during the absorb wait, the driver lived)."""

    def __init__(self, base_url: str, calls: list, fail_goto: Exception | None = None) -> None:
        self.base_url, self.calls, self.fail_goto = base_url, calls, fail_goto

    def open(self):
        self.calls.append(("open", self.base_url))
        return self

    def goto(self, path="/"):
        self.calls.append(("goto", path))
        if self.fail_goto is not None:
            raise self.fail_goto

    def computed_property(self, selector, prop):
        self.calls.append(("computed_property", selector, prop))
        return "#123456"

    send_chat = rebind = lambda self, *a, **k: None

    def screenshot(self, path):
        self.calls.append(("screenshot", str(path)))

    def close(self):
        self.calls.append(("close", self.base_url))


def _ui_resolver(calls: list, fail_goto: Exception | None = None):
    def resolve(base_url: str):
        return _FakeUI(base_url, calls, fail_goto).open(), ""
    return resolve


def _sm1_ui_tail(ctx: scenarios.LaneContext) -> None:
    """The UI tail of ``run_sm1`` verbatim (restart, the ``ctx.ui`` truthiness gate, goto, the
    computed property, the check, the screenshot): what the lazy guarded client must carry."""
    ctx.restart()
    if ctx.ui is None:
        ctx.check("ui_computed_style", False, ui_reason=ctx.ui_reason)
        return
    ctx.ui.goto("/")
    observed = str(ctx.ui.computed_property(":root", "--accent") or "").strip()
    ctx.check("ui_computed_style", observed == "#123456", accent_computed=observed)
    ctx.screenshot("sm1_after_restart")


def test_ui_client_opens_on_first_use_and_restart_reopens_against_the_new_server(tmp_path):
    calls: list = []
    servers = [_FakeServer(), _FakeServer()]
    servers[1].base_url = "http://127.0.0.1:2"
    ctx = _ctx(servers[0], ui_resolver=_ui_resolver(calls), restart=lambda: servers[1], shots=tmp_path)
    assert calls == []                              # nothing opened at construction
    assert ctx.ui is not None and calls == [("open", "http://127.0.0.1:1")]
    assert ctx.ui is not None and len(calls) == 1   # one client per open, not one per access
    _sm1_ui_tail(ctx)
    assert calls[1] == ("close", "http://127.0.0.1:1") and calls[2] == ("open", "http://127.0.0.1:2")
    assert ctx.checks == {"ui_computed_style": True} and ctx.ui_reason == "" and "ui_reason" not in ctx.facts
    assert ctx.screenshots == [str(tmp_path / "sm1_after_restart.png")]
    ctx.close_ui()
    ctx.close_ui()                                   # idempotent
    assert [c for c in calls if c[0] == "close"] == [("close", "http://127.0.0.1:1"), ("close", "http://127.0.0.1:2")]


def test_ui_open_failure_is_a_typed_reason_and_never_retried_before_restart():
    attempts: list = []

    def refuse(base_url):
        attempts.append(base_url)
        return None, "ui_unavailable:browser_missing"

    ctx = _ctx(ui_resolver=refuse, restart=_FakeServer)
    assert ctx.ui is None and ctx.ui is None and attempts == ["http://127.0.0.1:1"]
    assert ctx.ui_reason == "ui_unavailable:browser_missing"
    ctx.restart()                                    # a restart is the one re-resolve point
    assert ctx.ui is None and len(attempts) == 2


def test_closed_target_degrades_the_ui_checks_typed_and_keeps_every_other_check(tmp_path):
    calls: list = []
    ctx = _ctx(ui_resolver=_ui_resolver(calls, TargetClosedError("Target page, context or browser has been closed")),
               restart=_FakeServer, shots=tmp_path)
    ctx.check("commit_landed", True)
    _sm1_ui_tail(ctx)                                # no exception escapes
    assert ctx.checks == {"commit_landed": True, "ui_computed_style": False}
    assert ctx.ui_reason == ctx.facts["ui_reason"] == "ui_unavailable:TargetClosedError"
    assert ctx.facts["ui_errors"] == ["TargetClosedError: Target page, context or browser has been closed"]
    assert ctx.facts["accent_computed"] == "" and ctx.screenshots == []
    assert [c[0] for c in calls] == ["open", "goto", "close"]   # closed on the failure, later calls no-ops


class _NoopServer:
    base_url, attestation = "http://127.0.0.1:1", {}
    __init__ = start = stop = lambda self, *a, **k: None


def _attempt_row(tmp_path, monkeypatch, sid: str, acceptance=lambda ctx: ctx.check("scenario_ok", True), *,
                 flags: str = "") -> dict:
    """One ``run_attempt`` of ``sid`` under ``<tmp_path>/<sid>``: a fresh git seed, the real template, the no-op
    server, ``acceptance`` in the scenario's place (default: one passing check); ``<sid>/out`` keeps the artifacts."""
    _short_tmp(monkeypatch)
    monkeypatch.setattr(run_live_lanes, "IsolatedServer", _NoopServer)
    monkeypatch.setitem(run_live_lanes.SCENARIOS, sid, dataclasses.replace(scenarios.SCENARIOS[sid], acceptance=acceptance))
    root = tmp_path / sid
    root.mkdir()
    seed = _git_seed(root)
    args = run_live_lanes.parse_args(["--out", str(root / "out"), "--watch-interval", "600", *flags.split()])
    return run_live_lanes.run_attempt((sid, 1), args, root / "out", run_live_lanes.effective_settings(args, ""),
                                      run_live_lanes.Stagger(0.0), {}, seed,
                                      run_live_lanes.RunBudget(100.0, 8.0, reader=lambda root: (0.0, 0)),
                                      dispatch_index=0, key="", seed_sha=repo_provenance(seed)["head"])


def test_lane_with_a_dead_browser_target_is_checks_failed_not_infra_error(tmp_path, monkeypatch):
    """The rc.14 incident at lane level: the probe at lane start opens and closes, the client the
    scenario uses opens after the restart, its ``goto`` meets a closed target — the lane row is
    ``fail/checks_failed`` with the UI check typed, the task-side checks kept, no ``refusal``."""
    calls: list = []

    def acceptance(ctx):
        ctx.check("commit_landed", True)
        _sm1_ui_tail(ctx)

    monkeypatch.setattr(run_live_lanes, "resolve_ui_client",
                        _ui_resolver(calls, TargetClosedError("Target page, context or browser has been closed")))
    row = _attempt_row(tmp_path, monkeypatch, "SM1", acceptance)
    assert row["status"] == "fail" and row["reason_code"] == "checks_failed" and row["error"] == ""
    assert "refusal" not in row
    assert row["checks"]["commit_landed"] is True and row["checks"]["ui_computed_style"] is False
    assert row["facts"]["ui_reason"] == "ui_unavailable:TargetClosedError"
    assert row["ui"] == {"available": False, "reason": "ui_unavailable:TargetClosedError"}
    # lane start: availability probe opened and closed; use: opened after the restart, dead, closed
    assert [c[0] for c in calls] == ["open", "close", "open", "goto", "close"]
    stored = json.loads((tmp_path / "SM1" / "out" / "lanes" / "SM1_a1" / "result.json").read_text(encoding="utf-8"))
    assert stored["status"] == "fail" and stored["facts"]["ui_reason"] == "ui_unavailable:TargetClosedError"


def test_absorb_wait_and_check_follow_the_scenarios_expects_absorb(tmp_path, monkeypatch):
    """The rc.15 paid stand (2026-09-05, SK1_a1): every ``--self-mod`` lane waited ``--task-timeout`` for an absorb
    only SM1's commit could trigger, then failed ``self_mod_absorb_confirmed`` by construction. Now SM1 waits and
    carries the check; SW1/SK1 stop right after the scenario with ``{"expected": False}``, no check, post-task
    evolution OFF in their settings; every lane seeds ``owner_chat_id`` ONLY, never a campaign (run2's t=0 cycles)."""
    waits: list = []
    monkeypatch.setattr(run_live_lanes, "resolve_ui_client", lambda base_url: (None, "ui_unavailable:test"))
    monkeypatch.setattr(run_live_lanes, "self_mod_snapshot", lambda server, clone, data_root: {"pre": True})
    monkeypatch.setattr(run_live_lanes, "confirm_absorb", lambda server, clone, data_root, pre, **kw: (
        waits.append(pre) or {"confirmed": False, "reason": "no_promotion", "healthy": True}))
    sm1 = _attempt_row(tmp_path, monkeypatch, "SM1", flags="--self-mod")
    assert waits == [{"pre": True}] and sm1["status"] == "fail" and sm1["checks"]["self_mod_absorb_confirmed"] is False
    assert sm1["self_mod_absorb"] == {"expected": True, "confirmed": False, "reason": "no_promotion", "healthy": True}
    for sid in ("SM1", "SW1", "SK1"):
        if sid != "SM1":
            row = _attempt_row(tmp_path, monkeypatch, sid, flags="--self-mod")
            assert row["status"] == "pass" and "self_mod_absorb_confirmed" not in row["checks"], row["checks"]
            assert row["self_mod_absorb"] == {"expected": False} and row["self_mod"] is True and waits == [{"pre": True}]
        lane = tmp_path / sid / "out" / "lanes" / f"{sid}_a1" / "data"
        state = json.loads((lane / "state" / "state.json").read_text(encoding="utf-8"))
        assert json.loads((lane / "settings.json").read_text())["OUROBOROS_POST_TASK_EVOLUTION"] == ("true" if sid == "SM1" else "false")
        assert state["owner_chat_id"] == 1 and "evolution_mode_enabled" not in state, state
        assert not (lane / "state" / "evolution_campaign.json").exists(), sid


def test_wait_task_namespaces_checks_per_task_and_check_refuses_overwrites():
    ctx = _ctx()
    ctx.wait_task("t1", label="author")
    ctx.wait_task("t2", label="dispatch")
    assert set(ctx.checks) == {"author_http_terminal_completed", "author_durable_terminal_completed",
                               "dispatch_http_terminal_completed", "dispatch_durable_terminal_completed"}
    assert all(ctx.checks.values())
    assert ctx.facts["author_terminal"]["task_id"] == "t1" and ctx.facts["dispatch_terminal"]["task_id"] == "t2"
    assert ctx.facts["author_http_status"] == "completed" and ctx.facts["runtime_result"]["task_id"] == "t2"
    with pytest.raises(scenarios.DuplicateCheckKey):
        ctx.wait_task("t3", label="author")
    with pytest.raises(scenarios.DuplicateCheckKey):
        ctx.check("author_http_terminal_completed", True)
    # An unlabeled await keeps the plain keys for single-task scenarios.
    plain = _ctx()
    plain.wait_task("t9")
    assert set(plain.checks) == {"http_terminal_completed", "durable_terminal_completed"}


def test_dispatch_verdict_requires_ok_status_and_the_exact_echo():
    gen = "f773dad013e846c793dccd7938188b46"
    failed = [{"tool": "ext_x", "status": "error", "result_preview": "boom",
               "tool_result_meta": {"extension_generation": gen, "physical_dispatch": True}}]
    verdict = scenarios.dispatch_verdict(failed, scenarios.SK1_ECHO_EXPECTED)
    assert verdict["generation_ok"] and verdict["status"] == "error" and not verdict["echo_ok"]
    good = [{"tool": "ext_x", "status": "ok", "result_preview": "echo: ping-e2e-live\n",
             "tool_result_meta": {"extension_generation": gen, "physical_dispatch": True}}]
    verdict = scenarios.dispatch_verdict(good, scenarios.SK1_ECHO_EXPECTED)
    assert verdict == {"row_present": True, "status": "ok", "generation": gen, "generation_ok": True,
                       "physical_dispatch": True, "echo_ok": True}
    assert scenarios.dispatch_verdict([], scenarios.SK1_ECHO_EXPECTED)["row_present"] is False
    assert scenarios.SK1_ECHO_EXPECTED == f"echo: {scenarios.SK1_ECHO_MESSAGE}"
    assert scenarios.SCENARIOS["SK1"].stub_script(REPO_ROOT)["agent"][4]["arguments"]["message"] == scenarios.SK1_ECHO_MESSAGE
    # The relayed line opens one owner-chat turn on the stub wire: a second closing final absorbs it.
    assert [list(s)[0] for s in scenarios.SCENARIOS["SK1"].stub_script(REPO_ROOT)["agent"]] == [
        "tool", "tool", "tool", "final", "tool", "final", "final"]


def test_sk1_fixture_declares_exactly_the_permissions_its_plugin_exercises():
    """The SK1 manifest is honest by construction: every declared permission maps to source the
    plugin actually runs, the ONLY owner-granted one (``inject_chat``) is what the stand grants,
    and the prose states that narrow purpose. The first paid run declared ``inject_chat`` over an
    echo-only plugin and the skill review refused it 3/3 on ``permissions_honesty`` +
    ``inject_chat_minimization`` — a fixture defect, so this pins the fixture, not the reviewer."""
    import ast

    from ouroboros.contracts.skill_manifest import parse_skill_manifest_text
    from ouroboros.skill_loader import requested_skill_permissions

    manifest = parse_skill_manifest_text(scenarios.SK1_SKILL_MD)
    assert manifest.name == scenarios.SK1_SKILL and manifest.type == "extension" and manifest.entry == "plugin.py"
    exercised_by = {   # permission -> the source that performs it
        "tool": "api.register_tool(",
        "inject_chat": "/chat/inject",
        "net": "urllib.request",
    }
    assert set(manifest.permissions) == set(exercised_by)
    for permission, marker in exercised_by.items():
        assert marker in scenarios.SK1_PLUGIN, (permission, marker)
    assert requested_skill_permissions(list(manifest.permissions)) == scenarios.SK1_GRANTS == ["inject_chat"]
    # Host-token discipline (checklist item 12): the token is revealed at the request site only.
    assert scenarios.SK1_PLUGIN.count("get_skill_token().use_in_request()") == 1
    assert "print(" not in scenarios.SK1_PLUGIN and "log(" not in scenarios.SK1_PLUGIN
    # Owner binding: the destination is a module constant, never a tool argument.
    assert f"OWNER_CHAT_ID = {scenarios.SK1_OWNER_CHAT_ID}" in scenarios.SK1_PLUGIN and scenarios.SK1_OWNER_CHAT_ID == 1
    assert "'chat_id': OWNER_CHAT_ID" in scenarios.SK1_PLUGIN
    tree = ast.parse(scenarios.SK1_PLUGIN)
    register_call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                         and isinstance(n.func, ast.Attribute) and n.func.attr == "register_tool")
    schema = ast.literal_eval(next(k.value for k in register_call.keywords if k.arg == "schema"))
    assert set(schema["properties"]) == {"message"}
    # The prose names the purpose and no longer denies what the code does.
    body = scenarios.SK1_SKILL_MD.split("---", 2)[2]
    assert "/chat/inject" in body and f"chat_id {scenarios.SK1_OWNER_CHAT_ID}" in body and "127.0.0.1" in body
    assert "no host or network access" not in body


class _FakeExtensionApi:
    """Only the two PluginAPI members the probe plugin touches (the token object: ``use_in_request`` alone)."""

    def __init__(self, token: str) -> None:
        self.token, self.tools = token, {}

    def register_tool(self, name, handler, *, description, schema, timeout_sec=60):
        self.tools[name] = handler

    def get_skill_token(self):
        return types.SimpleNamespace(use_in_request=lambda: self.token)


def _inject_sink(status: int, hits: list):
    """A loopback HTTP server standing in for the Host Service ``/chat/inject`` route."""
    import http.server

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            hits.append({"path": self.path, "token": self.headers.get("X-Skill-Token"),
                         "body": json.loads(body.decode("utf-8"))})
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok": true}')

        def log_message(self, *_args):  # keep pytest output clean
            return

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_sk1_plugin_relays_one_bounded_line_into_the_owner_chat(monkeypatch):
    """The plugin text the model is told to write, executed: one POST to the loopback
    ``/chat/inject`` per call, owner chat pinned, the skill token only in the header, the text
    bounded, the same text returned — and a Host Service refusal surfaces as a tool error."""
    hits: list = []
    sink = _inject_sink(202, hits)
    try:
        monkeypatch.setenv("HOST_SERVICE_URL", f"http://127.0.0.1:{sink.server_port}")
        namespace: dict = {}
        exec(compile(scenarios.SK1_PLUGIN, "plugin.py", "exec"), namespace)  # noqa: S102 - the fixture under test
        api = _FakeExtensionApi("tok-e2e")
        namespace["register"](api)
        echo = api.tools["echo"]
        assert echo(None, message=scenarios.SK1_ECHO_MESSAGE) == scenarios.SK1_ECHO_EXPECTED
        assert hits == [{"path": "/chat/inject", "token": "tok-e2e", "body": {
            "text": scenarios.SK1_ECHO_EXPECTED, "chat_id": scenarios.SK1_OWNER_CHAT_ID,
            "sender_label": scenarios.SK1_SKILL}}]
        long = echo(None, message="x" * (scenarios.SK1_ECHO_MAX_CHARS + 50))
        assert long == hits[-1]["body"]["text"] == ("echo: " + "x" * scenarios.SK1_ECHO_MAX_CHARS)[:scenarios.SK1_ECHO_MAX_CHARS]
        assert len(long) == scenarios.SK1_ECHO_MAX_CHARS   # the cap bounds the FINAL text, prefix included
        multi = echo(None, message="first\r\nsecond\nthird")
        assert multi == hits[-1]["body"]["text"] == "echo: first second third"   # ONE line: breaks collapse
        assert len(hits) == 3   # exactly one line per call, no retry
    finally:
        sink.shutdown()
    refusing = _inject_sink(403, [])
    try:
        monkeypatch.setenv("HOST_SERVICE_URL", f"http://127.0.0.1:{refusing.server_port}")
        with pytest.raises(urllib.error.HTTPError):
            echo(None, message="denied")
    finally:
        refusing.shutdown()
    # The acceptance reads the HOST's attribution of that line, never the plugin's claim.
    rows = [{"direction": "in", "chat_id": 1, "source": f"skill:{scenarios.SK1_SKILL}", "text": scenarios.SK1_ECHO_EXPECTED},
            {"direction": "out", "chat_id": 1, "source": f"skill:{scenarios.SK1_SKILL}", "text": scenarios.SK1_ECHO_EXPECTED},
            {"direction": "in", "chat_id": 1, "source": "web", "text": scenarios.SK1_ECHO_EXPECTED},
            {"direction": "in", "chat_id": 2, "source": f"skill:{scenarios.SK1_SKILL}", "text": scenarios.SK1_ECHO_EXPECTED},
            {"direction": "in", "chat_id": 1, "source": f"skill:{scenarios.SK1_SKILL}", "text": "echo: other"}]
    assert scenarios.owner_chat_relay_rows(rows, scenarios.SK1_SKILL, scenarios.SK1_ECHO_EXPECTED) == rows[:1]


def test_commit_refusal_facts_name_every_typed_refusal():
    ledger = {"attempts": [
        {"attempt": 1, "phase": "preflight", "status": "blocked", "block_reason": "tests_preflight_blocked"},
        {"attempt": 2, "phase": "blocking_review", "status": "blocked", "block_reason": "scope_blocked"},
        {"attempt": 3, "phase": "late_wait", "status": "reviewing", "block_reason": "review_late_result_pending"}],
        "advisory_runs": [{"status": "stale"}, {"status": "bypassed"}]}
    tools = [
        {"tool": "preflight_review", "status": "ok",
         "result_preview": '{\n  "status": "preflight_blocked",\n  "error": "⚠️ PREFLIGHT_BLOCKED: VERSION is not in scope'},
        {"tool": "commit_reviewed", "status": "blocked", "result_preview": "⚠️ TESTS_PREFLIGHT_BLOCKED: Tests must pass"},
        {"tool": "commit_reviewed", "status": "blocked", "result_preview": "⚠️ SCOPE_REVIEW_BLOCKED: the review pack"},
        {"tool": "write_file", "status": "ok", "result_preview": "⚠️ NOT_A_REVIEW_TOOL: ignored"},
        {"tool": "commit_reviewed", "status": "ok", "result_preview": "⚠️ REVIEW_PENDING: physical reviewer work"}]
    facts = scenarios.commit_refusal_facts(ledger, tools, {"status": "failed", "reason_code": "budget_exhausted"})
    assert facts["refusal_codes"] == ["PREFLIGHT_BLOCKED", "REVIEW_PENDING", "SCOPE_REVIEW_BLOCKED", "TESTS_PREFLIGHT_BLOCKED"]
    assert [a["block_reason"] for a in facts["commit_attempts"]] == [
        "tests_preflight_blocked", "scope_blocked", "review_late_result_pending"]
    assert facts["advisory_run_statuses"] == ["stale", "bypassed"]
    assert facts["review_tool_calls"][1] == {"tool": "commit_reviewed", "status": "blocked", "code": "TESTS_PREFLIGHT_BLOCKED"}
    assert facts["terminal_status"] == "failed" and facts["terminal_reason_code"] == "budget_exhausted"


def test_sm1_changes_both_stylesheets_and_keeps_the_mirror_parity():
    """web/onboarding.css mirrors web/style.css BY VALUE (tests/test_web_typography_static.py):
    the scenario edits, commits and validates both files, in the prompt, the stub and the
    acceptance. Both take the FULL user path: no ``skip_tests``, no ``skip_advisory_review``,
    no "do not bump" (the first paid run's narrower prompt was refused by the commit gate on
    exactly the version bump, the design system and the missing UI evidence)."""
    assert scenarios.SM1_CSS_PATHS == ("web/style.css", "web/onboarding.css")
    prompt = scenarios.sm1_prompt()
    assert "web/style.css" in prompt and "web/onboarding.css" in prompt and "docs/DESIGN.md" in prompt
    assert "skip_" not in prompt.lower() and "do not bump" not in prompt.lower() and "bumped in the same diff" in prompt
    script = scenarios.sm1_stub_script(REPO_ROOT)["agent"]
    writes = [s for s in script if s.get("tool") == "write_file"]
    written = [w["arguments"]["path"] for w in writes]
    assert written[:2] == list(scenarios.SM1_CSS_PATHS)
    commit = next(s for s in script if s.get("tool") == "commit_reviewed")["arguments"]
    assert commit["paths"] == written and "commit_message" in commit
    assert not any(key.startswith("skip_") for key in commit), "the stub rehearsal takes the full user path like the paid prompt"
    edited = {w["arguments"]["path"]: w["arguments"]["content"] for w in writes if w["arguments"]["path"] in scenarios.SM1_CSS_PATHS}
    for path, text in edited.items():
        original = (REPO_ROOT / path).read_text(encoding="utf-8")
        assert scenarios.accent_value(text) == scenarios.SM1_NEW_ACCENT
        # The tree under test may ALREADY carry the target accent (this suite runs inside the
        # SM1 candidate's hermetic preflight after the model applied the change): then the
        # stub edit is a no-op by design, not a failure.
        if scenarios.accent_value(original) != scenarios.SM1_NEW_ACCENT:
            assert text != original
        assert len(text.splitlines()) == len(original.splitlines())
    assert scenarios.css_mirror_drift(edited["web/style.css"], edited["web/onboarding.css"]) == {}
    # The parser reads the SAME :root block the invariant reads, and the invariant is real.
    style_tokens = scenarios.css_root_tokens((REPO_ROOT / "web/style.css").read_text(encoding="utf-8"))
    onboarding_tokens = scenarios.css_root_tokens((REPO_ROOT / "web/onboarding.css").read_text(encoding="utf-8"))
    assert "--accent" in style_tokens and "--accent" in onboarding_tokens and len(set(style_tokens) & set(onboarding_tokens)) > 20
    lopsided = scenarios.css_with_accent(edited["web/style.css"], "#000000")
    assert scenarios.css_mirror_drift(lopsided, edited["web/onboarding.css"]) == {"--accent": ("#000000", scenarios.SM1_NEW_ACCENT)}


def test_sm1_stub_bumps_the_release_carriers_through_the_sync_ssot(tmp_path):
    """The stub's bump is a strictly-greater release version whose carriers come from
    ``release_sync`` (no hand list) and pass the product's own release admission gate; the
    acceptance's advisory-row and vision-evidence readers tell the real rows from the audited ones."""
    from ouroboros.commit_admission import release_metadata_preflight
    from ouroboros.tools.release_sync import CARRIER_SPAN_PATHS

    writes = [s for s in scenarios.sm1_stub_script(REPO_ROOT)["agent"] if s.get("tool") == "write_file"]
    carriers = {w["arguments"]["path"]: w["arguments"]["content"] for w in writes
                if w["arguments"]["path"] not in scenarios.SM1_CSS_PATHS}
    assert {"VERSION", "README.md"} <= set(carriers) <= CARRIER_SPAN_PATHS
    seed = (REPO_ROOT / "VERSION").read_text(encoding="utf-8").strip()
    bumped = carriers["VERSION"].strip()
    assert scenarios.version_is_bumped(seed, bumped) and f"| {bumped} |" in carriers["README.md"]
    root = tmp_path / "carriers"
    for rel in sorted(CARRIER_SPAN_PATHS):
        if (REPO_ROOT / rel).is_file():
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_text(carriers.get(rel) or (REPO_ROOT / rel).read_text(encoding="utf-8"), encoding="utf-8")
    assert release_metadata_preflight(root, scenarios.SM1_COMMIT_MESSAGE, ["VERSION"]) is None
    assert scenarios.sm1_next_version("7.0.0-rc.14") == "7.0.0-rc.15" and scenarios.sm1_next_version("7.0.0") == "7.0.1"
    # A seed cloned from an older ref carries the newer tags: the stub skips taken versions.
    assert scenarios.sm1_next_version("7.0.0-rc.14", {"v7.0.0-rc.15", "v7.0.0-rc.16"}) == "7.0.0-rc.17"
    assert scenarios.sm1_next_version("7.0.0", {"v7.0.1"}) == "7.0.2"
    assert not scenarios.version_is_bumped("7.0.0-rc.14", "7.0.0-rc.14") and not scenarios.version_is_bumped("7.0.0-rc.14", "7.0.0-rc.13")
    assert scenarios.advisory_run_is_real({"status": "fresh"}) and scenarios.advisory_run_is_real({"status": "stale", "raw_result": "[]"})
    assert not scenarios.advisory_run_is_real({"status": "bypassed", "bypass_reason": "skip_advisory_review"})
    assert not scenarios.advisory_run_is_real({"status": "stale", "raw_result": "⚠️ ADVISORY_SKIPPED: prompt too large"})
    rows = [{"tool": "vlm_query"}, {"tool": "browser_action", "args": {"action": "click"}}, {"tool": "read_file"}]
    assert [r["tool"] for r in scenarios.vision_evidence_rows(rows)] == ["vlm_query"]


# --------------------------------------------------------------------------- #
# Seed: a clean detached clone of the requested ref, never the operator's live worktree
# --------------------------------------------------------------------------- #

def test_materialize_seed_is_a_clean_detached_clone_of_the_ref(tmp_path):
    source = _git_seed(tmp_path)
    first = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(source), check=True, capture_output=True, text=True).stdout.strip()
    (source / "VERSION").write_text("7.0.1-test\n", encoding="utf-8")
    second = _commit(source, "bump")
    (source / "VERSION").write_text("7.0.2-wip\n", encoding="utf-8")   # dirty source, never under test
    seed = tmp_path / "seed"
    record = run_live_lanes.materialize_seed(source, "HEAD~1", seed)
    assert record["resolved_sha"] == first and record["policy"] == run_live_lanes.SEED_POLICY
    assert (seed / "VERSION").read_text(encoding="utf-8") == "7.0.0-test\n"
    detached = subprocess.run(["git", "symbolic-ref", "-q", "HEAD"], cwd=str(seed), check=False, capture_output=True)
    assert detached.returncode != 0   # no branch checked out
    provenance = repo_provenance(seed)
    assert run_live_lanes.seed_is_clean(provenance, first) and not run_live_lanes.seed_is_clean(provenance, second)
    assert not provenance["describe"].endswith("-dirty")
    with pytest.raises(run_live_lanes.SeedMaterializeRefused) as exists:
        run_live_lanes.materialize_seed(source, "HEAD", seed)
    assert exists.value.reason == "seed_dir_exists"
    with pytest.raises(run_live_lanes.SeedMaterializeRefused) as bogus:
        run_live_lanes.materialize_seed(source, "no-such-ref", tmp_path / "seed2")
    assert bogus.value.reason == "ref_unresolved"


def test_dirty_source_runs_the_committed_ref_from_a_clean_detached_seed(tmp_path, monkeypatch):
    _short_tmp(monkeypatch)
    monkeypatch.setattr(run_live_lanes, "run_lane", _fake_lane)
    source = _git_seed(tmp_path, dirty=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(source), check=True, capture_output=True, text=True).stdout.strip()
    out = tmp_path / "out"
    rc = run_live_lanes.main(["--stub", "--source-repo", str(source), "--out", str(out), "--scenarios", "SM1",
                              "--watch-interval", "600"])
    assert rc == 0
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["source"]["dirty"] is True and manifest["seed_gate"]["allow_dirty_seed"] is True
    assert manifest["seed"]["resolved_sha"] == head and manifest["seed"]["clean"] is True
    assert manifest["seed"]["requested_ref"] == "HEAD" and manifest["extra"]["seed_policy"] == run_live_lanes.SEED_POLICY
    assert manifest["extra"]["seed_head"] == head and not manifest["extra"]["seed_describe"].endswith("-dirty")
    assert (out / "seed" / "VERSION").read_text(encoding="utf-8") == "7.0.0-test\n"   # committed, not the edit
    row = json.loads((out / "lanes" / "SM1_a1" / "result.json").read_text(encoding="utf-8"))
    assert row["seed_sha"] == head


def test_unresolvable_seed_ref_is_a_typed_refusal_before_any_lane(tmp_path, monkeypatch):
    _short_tmp(monkeypatch)
    monkeypatch.setattr(run_live_lanes, "run_lane", lambda *a, **k: pytest.fail("a lane started without a seed"))
    source = _git_seed(tmp_path)
    out = tmp_path / "out"
    rc = run_live_lanes.main(["--stub", "--source-repo", str(source), "--seed", "no-such-ref", "--out", str(out),
                              "--watch-interval", "600"])
    assert rc == 3
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["outcome"] == "refused" and manifest["extra"]["exit_code"] == 3
    assert manifest["extra"]["refusal"]["stage"] == "seed_materialize"
    assert manifest["extra"]["refusal"]["reason"] == "ref_unresolved"
    assert not (out / "lanes").exists() and not (out / "effective_settings.json").exists()


# --------------------------------------------------------------------------- #
# Admission and the typed refusals (persisted manifest, no footprint)
# --------------------------------------------------------------------------- #

def test_run_root_confinement_refuses_before_anything_is_created(tmp_path, monkeypatch):
    _short_tmp(monkeypatch)
    source = _git_seed(tmp_path)
    with pytest.raises(ValueError, match="must not be under repo/"):
        run_live_lanes.main(["--stub", "--source-repo", str(source), "--out", str(source / "inside")])
    assert not (source / "inside").exists()


def test_missing_key_env_is_a_typed_refusal_before_any_lane_starts(tmp_path, monkeypatch):
    _short_tmp(monkeypatch)
    monkeypatch.delenv("E2E_TEST_KEY_ENV", raising=False)
    monkeypatch.setattr(run_live_lanes, "run_lane", lambda *a, **k: pytest.fail("a lane started without a key"))
    source = _git_seed(tmp_path)
    out = tmp_path / "out"
    rc = run_live_lanes.main(["--source-repo", str(source), "--out", str(out), "--key-env", "E2E_TEST_KEY_ENV"])
    assert rc == 3
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["outcome"] == "refused" and manifest["extra"]["exit_code"] == 3
    assert manifest["extra"]["refusal"] == {"stage": "credential", "reason": "key_env_absent", "env": "E2E_TEST_KEY_ENV"}
    assert not (out / "lanes").exists() and not (out / "effective_settings.json").exists() and not (out / "seed").exists()


def test_credit_preflight_takes_the_min_of_both_planes(tmp_path, monkeypatch):
    """Key limit says $50, the account behind it holds $1: the run is bounded by $1 and refused
    below a $5 floor. Both numbers are recorded; the key value never is."""
    _short_tmp(monkeypatch)
    monkeypatch.setenv("E2E_TEST_KEY_ENV", FAKE_KEY)
    monkeypatch.setattr(run_live_lanes, "run_lane", lambda *a, **k: pytest.fail("a lane started under the floor"))
    calls: list = []
    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen(
        b'{"data":{"limit_remaining":50.0}}', b'{"data":{"total_credits":10.0,"total_usage":9.0}}', calls))
    source = _git_seed(tmp_path)
    out = tmp_path / "out"
    rc = run_live_lanes.main(["--source-repo", str(source), "--out", str(out), "--key-env", "E2E_TEST_KEY_ENV",
                              "--min-credit-usd", "5"])
    assert rc == 3
    assert calls == ["https://openrouter.ai/api/v1/key", "https://openrouter.ai/api/v1/credits"]
    raw = (out / "run_manifest.json").read_bytes()
    assert FAKE_KEY.encode() not in raw
    manifest = json.loads(raw)
    refusal = manifest["extra"]["refusal"]
    assert refusal["stage"] == "credit_preflight" and refusal["reason"] == "insufficient_remaining"
    assert refusal["remaining_usd"] == 1.0 and refusal["key_limit_remaining_usd"] == 50.0
    assert refusal["account_credits_usd"] == 1.0 and refusal["floor_usd"] == 5.0
    assert manifest["extra"]["credential_fingerprint"].startswith("sha256:")


def test_openrouter_account_credits_is_the_second_bound_only(monkeypatch):
    from devtools.benchmarks.common.manifests import openrouter_account_credits, openrouter_key_remaining

    calls: list = []
    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen(
        b'{"data":{"limit":null}}', b'{"data":{"total_credits":12.5,"total_usage":2.5}}', calls))
    assert openrouter_key_remaining(FAKE_KEY) is None            # uncapped key: not "$0", not "plenty"
    assert openrouter_account_credits(FAKE_KEY) == 10.0
    assert run_live_lanes.credit_preflight(FAKE_KEY)["remaining_usd"] == 10.0
    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen(b'{"data":{"limit":null}}', b'{"data":{}}', []))
    assert run_live_lanes.credit_preflight(FAKE_KEY, timeout=3) == {
        "key_limit_remaining_usd": None, "account_credits_usd": None, "remaining_usd": None}


# --------------------------------------------------------------------------- #
# The manifest names the APPLIED model; secrets stay out of every run-level artifact
# --------------------------------------------------------------------------- #

def _fake_run(tmp_path, monkeypatch, argv: list[str], *, lane=_fake_lane, expect_rc: int = 0) -> tuple[pathlib.Path, dict]:
    _short_tmp(monkeypatch)
    monkeypatch.setenv("E2E_TEST_KEY_ENV", FAKE_KEY)
    monkeypatch.setattr(run_live_lanes, "run_lane", lane)
    monkeypatch.setattr(run_live_lanes, "credit_preflight", lambda key, **_kw: {
        "key_limit_remaining_usd": None, "account_credits_usd": None, "remaining_usd": None})
    source = _git_seed(tmp_path)
    out = tmp_path / "out"
    rc = run_live_lanes.main(["--source-repo", str(source), "--out", str(out), "--key-env", "E2E_TEST_KEY_ENV",
                              "--watch-interval", "600", *argv])
    assert rc == expect_rc
    return out, json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))


def test_manifest_names_the_effective_model_not_argv(tmp_path, monkeypatch):
    """EQUALITY pin: the manifest's model is the one in the APPLIED settings file. argv pins
    model Y, the applied file carries X -> the manifest says X."""
    real = run_live_lanes.effective_settings

    def applied_differs(args, key):
        return {**real(args, key), "OUROBOROS_MODEL": "applied/model-x"}

    monkeypatch.setattr(run_live_lanes, "effective_settings", applied_differs)
    out, manifest = _fake_run(tmp_path, monkeypatch, ["--model", "argv/model-y", "--scenarios", "SM1"])
    applied = json.loads((out / "effective_settings.json").read_text(encoding="utf-8"))
    assert manifest["model_slots"]["OUROBOROS_MODEL"] == applied["OUROBOROS_MODEL"] == "applied/model-x"
    assert manifest["extra"]["effective_model"] == "applied/model-x"
    assert "argv/model-y" not in json.dumps(manifest["model_slots"])


def test_run_root_template_is_redacted_and_the_key_reaches_only_the_lanes(tmp_path, monkeypatch):
    out, manifest = _fake_run(tmp_path, monkeypatch, ["--model", "argv/model-y", "--scenarios", "SM1,SK1",
                                                      "--attempts", "2", "--pass-of", "2", "--lanes", "2"])
    assert manifest["model_slots"]["OUROBOROS_MODEL"] == "argv/model-y"
    template_path = out / "effective_settings.json"
    if os.name == "posix":
        assert (template_path.stat().st_mode & 0o777) == 0o600
    else:   # Windows: chmod only toggles read-only; the mode reads 0o666 — the redaction is the guarantee there
        assert template_path.is_file()
    template = json.loads(template_path.read_text(encoding="utf-8"))
    assert "OPENROUTER_API_KEY" not in template and template["OUROBOROS_MODEL"] == "argv/model-y"
    for artifact in (out / "run_manifest.json", template_path, *out.glob("lanes/*/result.json")):
        assert FAKE_KEY.encode() not in artifact.read_bytes(), artifact
    creds = manifest["provider_credentials"]
    assert creds["granted"] == {}                       # the file grant: nothing
    assert creds["runtime_granted"]["OPENROUTER_API_KEY"]["present"] is True
    assert creds["runtime_granted"]["OPENROUTER_API_KEY"]["fingerprint"].startswith("sha256:")
    for row_path in out.glob("lanes/*/result.json"):
        row = json.loads(row_path.read_text(encoding="utf-8"))
        assert row["template_has_key"] is False and row["key_handed"] is True   # injected per lane, in memory
    assert manifest["requested_task_ids"] == ["SM1_a1", "SM1_a2", "SK1_a1", "SK1_a2"]
    assert manifest["extra"]["scenarios"] == {
        "SM1": {"attempts": 2, "passed": 2, "infra_errors": 0, "not_run": 0, "verdict": "pass"},
        "SK1": {"attempts": 2, "passed": 2, "infra_errors": 0, "not_run": 0, "verdict": "pass"}}
    assert manifest["extra"]["outcome"] == "completed" and manifest["extra"]["exit_code"] == 0
    assert manifest["extra"]["total_budget_usd"] == 100.0 and manifest["extra"]["per_task_usd"] == 8.0
    budget = manifest["extra"]["budget"]
    assert budget["cap_usd"] == 100.0 and budget["refusals"] == [] and budget["attempts_not_run"] == []
    assert budget["first_refused"] is None and "halted" not in budget
    assert budget["reservation_rule"] == run_live_lanes.RESERVATION_RULE and "stop_reason" not in manifest["extra"]


def test_run_wide_cap_refuses_per_attempt_and_records_not_run_rows(tmp_path, monkeypatch):
    """cap $16, per-task $4 (SK1 reserves $8, SM1/SW1 $4), one lane, round-robin dispatch; a settled SK1 lane
    reads back $7, every other $2: round 1 runs SK1_a1 (0+8), SM1_a1 (7+4), SW1_a1 (9+4); SK1_a2 (11+8 > 16) is
    refused at once; SM1_a2 (11+4) still RUNS after that refusal — a refusal is per attempt, not a halt; SW1_a2
    (13+4 > 16) is refused. Every refusal is a recorded row and the stop_reason; pass-of 2 fails SW1/SK1, not SM1."""
    monkeypatch.setattr(run_live_lanes, "lane_spend", lambda root: (
        (7.0 if pathlib.Path(root).parent.name.startswith("SK1") else 2.0, 0)
        if pathlib.Path(root).parent.exists() else (0.0, 0)))
    out, manifest = _fake_run(tmp_path, monkeypatch, ["--scenarios", "SM1,SK1,SW1", "--attempts", "2", "--pass-of", "2",
                                                      "--lanes", "1", "--total-budget", "16", "--per-task-usd", "4"], expect_rc=1)
    budget = manifest["extra"]["budget"]
    assert budget["first_refused"] == "SK1_a2" and "halted" not in budget
    assert [(r["attempt"], r["spent_usd"], r["reservation_usd"], r["waited_sec"]) for r in budget["refusals"]] == [
        ("SK1_a2", 11.0, 8.0, 0.0), ("SW1_a2", 13.0, 4.0, 0.0)]
    assert budget["attempts_not_run"] == ["SK1_a2", "SW1_a2"] and budget["spent_usd"] == 13.0
    assert manifest["extra"]["stop_reason"] == "budget_cap" and manifest["extra"]["lanes_run"] == 4
    assert manifest["extra"]["scenarios"]["SW1"] == {"attempts": 2, "passed": 1, "infra_errors": 0, "not_run": 1,
                                                     "verdict": "fail"} and manifest["extra"]["scenarios"]["SM1"]["verdict"] == "pass"
    assert manifest["extra"]["scenarios"]["SK1"]["passed"] == 1 and manifest["extra"]["scenarios"]["SK1"]["not_run"] == 1
    rows = {json.loads(p.read_text(encoding="utf-8"))["attempt"]: json.loads(p.read_text(encoding="utf-8"))
            for p in out.glob("lanes/SM1_*/result.json")}
    assert rows[1]["lane_total_budget_usd"] == 4.0 and rows[2]["lane_total_budget_usd"] == 4.0   # each: its reservation
    refused = json.loads((out / "lanes" / "SK1_a2" / "result.json").read_text(encoding="utf-8"))
    assert refused["status"] == "not_run" and refused["reason_code"] == "budget_cap"
    assert refused["refusal"]["code"] == "budget_cap" and refused["budget"]["waited_sec"] == 0.0
    assert refused["budget"]["spent_usd"] == 11.0 and refused["budget"]["reservation_usd"] == 8.0
    index = [json.loads(ln) for ln in (out / "result_index.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert [(r["instance_id"], r["status"], r["reason_code"]) for r in index if r["status"] == "not_run"] == [
        ("SK1_a2", "not_run", "budget_cap"), ("SW1_a2", "not_run", "budget_cap")]


def test_self_mod_run_level_gate_fails_every_unconfirmed_absorbing_lane(tmp_path, monkeypatch):
    """The gate follows ``expects_absorb``: an unconfirmed SM1 fails the run; SW1 (no absorb to confirm) is never listed."""
    def lane(job, *a, **k):
        row = _fake_lane(job, *a, **k)
        row["self_mod_absorb"] = {"expected": True, "confirmed": False} if job[0] == "SM1" else {"expected": False}
        return row

    _out, manifest = _fake_run(tmp_path, monkeypatch, ["--self-mod", "--scenarios", "SM1,SW1", "--lanes", "1"],
                               lane=lane, expect_rc=1)
    assert manifest["extra"]["self_mod"] == {"lanes": 2, "absorb_expected": 1, "absorb_unconfirmed": ["SM1_a1"]}
    assert manifest["extra"]["outcome"] == "failed" and manifest["extra"]["exit_code"] == 1
    assert manifest["extra"]["scenarios"]["SM1"]["verdict"] == "pass"   # the lane verdict alone would have passed


def test_lane_infra_failure_is_a_typed_refusal_in_both_artifacts(tmp_path, monkeypatch):
    _short_tmp(monkeypatch)

    def refuse(seed, clone):
        raise run_live_lanes.SeedMaterializeRefused("clone_failed", "git clone exploded")

    monkeypatch.setattr(run_live_lanes, "clone_seed", refuse)
    args = run_live_lanes.parse_args(["--stub", "--out", str(tmp_path / "out"), "--watch-interval", "600"])
    out = tmp_path / "out"
    states: dict = {}
    row = run_live_lanes.run_attempt(("SM1", 1), args, out, {}, run_live_lanes.Stagger(2.0), states, tmp_path / "seed",
                                     run_live_lanes.RunBudget(100.0, 8.0, reader=lambda root: (0.0, 0)),
                                     dispatch_index=0, key="", seed_sha="abc")
    assert row["status"] == "infra_error" and row["reason_code"] == "infra_error:clone_failed"
    assert row["refusal"] == {"type": "SeedMaterializeRefused", "code": "clone_failed", "message": "git clone exploded"}
    stored = json.loads((out / "lanes" / "SM1_a1" / "result.json").read_text(encoding="utf-8"))
    assert stored["refusal"]["code"] == "clone_failed" and stored["error"].startswith("SeedMaterializeRefused:")
    index = json.loads((out / "result_index.jsonl").read_text(encoding="utf-8").strip())
    assert index["status"] == "infra_error" and index["reason_code"] == "infra_error:clone_failed"
    assert index["details"]["refusal"]["code"] == "clone_failed"


def test_stagger_gate_spaces_lane_starts(monkeypatch):
    clock = {"t": 100.0}
    slept: list = []
    monkeypatch.setattr(run_live_lanes.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(run_live_lanes.time, "sleep", lambda s: slept.append(s))
    gate = run_live_lanes.Stagger(2.5)
    gate.wait_turn()
    gate.wait_turn()
    clock["t"] += 1.0
    gate.wait_turn()
    assert slept == [0.0, 2.5, 1.5]


def test_ui_client_degrades_typed_without_playwright(monkeypatch):
    monkeypatch.setattr(ui_probe, "_suite_client", lambda base_url: None)
    monkeypatch.setitem(sys.modules, "playwright", None)
    monkeypatch.setitem(sys.modules, "playwright.sync_api", None)
    assert ui_probe.resolve_ui_client("http://127.0.0.1:1") == (None, "ui_unavailable:playwright_not_installed")


def test_ui_client_prefers_the_suite_interface_when_it_has_this_surface(monkeypatch):
    class Landed:
        def __init__(self, base_url):
            self.base_url = base_url
            self.opened = False

        def open(self):
            self.opened = True
            return self

        goto = computed_property = send_chat = screenshot = rebind = close = lambda self, *a, **k: None

    fake = type(sys)("tests.system_e2e.interfaces")
    fake.PlaywrightUIClient = Landed
    monkeypatch.setitem(sys.modules, "tests.system_e2e.interfaces", fake)
    client, reason = ui_probe.resolve_ui_client("http://127.0.0.1:1")
    assert isinstance(client, Landed) and client.opened and reason == ""


# --------------------------------------------------------------------------- #
# The keyless rehearsal: SM1 end-to-end on a real isolated server (--stub)
# --------------------------------------------------------------------------- #

@pytest.mark.integration
@pytest.mark.serial
def test_stub_sm1_end_to_end_on_a_real_isolated_server(tmp_path):
    """Real server, loopback stub model, no key: the commit lands through the review organ
    (both stylesheets plus the release-carrier bump, no skip flags, through the same hermetic
    tests preflight as the paid prompt), the durable rows and receipts exist, the
    seed is a clean detached clone of this tree's HEAD and the manifest names the stub as
    the model."""
    if str(os.environ.get("OUROBOROS_E2E_DEEP") or "").strip().lower() != "mock":
        pytest.skip("set OUROBOROS_E2E_DEEP=mock to run the stub rehearsal (spawns a real isolated server)")
    out = tmp_path / "out"
    rc = run_live_lanes.main(["--stub", "--lanes", "1", "--scenarios", "SM1", "--source-repo", str(REPO_ROOT),
                              "--seed", "HEAD", "--out", str(out), "--watch-interval", "600"])
    row = json.loads((out / "lanes" / "SM1_a1" / "result.json").read_text(encoding="utf-8"))
    failed = sorted(k for k, v in row["checks"].items() if not v and not k.startswith("ui_"))
    assert rc == 0 and failed == [], (row["status"], failed, row["error"])
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_slots"]["OUROBOROS_MODEL"] == stub_lane.STUB_MODEL_SLUG == row["model_slots"]["OUROBOROS_MODEL"]
    assert row["digests"]["pre_head"] == manifest["seed"]["resolved_sha"] != row["digests"]["post_head"]
    assert len(row["digests"]["diff_sha256"]) == 64 and not row["digests"]["seed_describe"].endswith("-dirty")
    assert (out / "result_index.jsonl").read_text(encoding="utf-8").count("\n") == 1


def test_orphan_after_stop_fails_a_passing_lane_with_a_typed_reason(tmp_path):
    """A process still carrying the lane's data root after stop flips a passing lane to fail with
    reason_code=checks_failed, in result.json AND in result_index.jsonl — never an empty reason."""
    def row():
        return {"scenario": "SM1", "attempt": 1, "status": "pass", "reason_code": "", "checks": {"fake": True},
                "error": "", "duration_sec": 1.0, "budget": {}, "refusal": None, "runtime_outcome": "completed"}
    clean = row()
    run_live_lanes._apply_orphan_scan(clean, [])
    assert clean["status"] == "pass" and clean["reason_code"] == "" and clean["checks"]["no_orphans_after_stop"] is True
    assert "orphans" not in clean
    absent = row()
    run_live_lanes._apply_orphan_scan(absent, None)   # no procfs (macOS, Windows): a typed fact, no check
    assert absent["status"] == "pass" and absent["orphan_scan"] == "unavailable:no_procfs"
    assert absent["no_orphans_after_stop"] is None and "no_orphans_after_stop" not in absent["checks"]
    dirty = row()
    run_live_lanes._apply_orphan_scan(dirty, [os.getpid()])
    assert dirty["status"] == "fail" and dirty["reason_code"] == "checks_failed"
    assert dirty["checks"]["no_orphans_after_stop"] is False and dirty["no_orphans_after_stop"] is False
    # The survivors are NAMED: pid + the head of its cmdline (this very interpreter here).
    assert [o["pid"] for o in dirty["orphans"]] == [os.getpid()] and "orphans_omitted" not in dirty
    assert len(dirty["orphans"][0]["cmdline"]) <= 120
    if run_live_lanes.PROCFS_AVAILABLE:   # the cmdline text is read from /proc: Linux only; elsewhere it is the typed ""
        assert "python" in dirty["orphans"][0]["cmdline"]
    crowded = row()
    run_live_lanes._apply_orphan_scan(crowded, [os.getpid()] + [2 ** 22 + n for n in range(24)])
    assert len(crowded["orphans"]) == 20 and crowded["orphans_omitted"] == 5
    assert crowded["orphans"][1] == {"pid": 2 ** 22, "cmdline": ""}   # a pid gone by read time: typed empty
    out = tmp_path / "run"
    lane = out / "lanes" / "SM1_a1"
    lane.mkdir(parents=True)
    run_live_lanes._record_row(out, lane, dirty)
    recorded = json.loads((lane / "result.json").read_text(encoding="utf-8"))
    assert recorded["status"] == "fail" and recorded["reason_code"] == "checks_failed"
    index = [json.loads(ln) for ln in (out / "result_index.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert [(r["instance_id"], r["status"], r["reason_code"]) for r in index] == [("SM1_a1", "fail", "checks_failed")]
