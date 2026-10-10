"""The two monitoring examples in docs/examples follow the record passport and stay safe.

The collector reads only the passport's planes and forwards only metadata the passport lists;
the error tracker loads in process and starts its SDK with nothing that leaks secrets.
"""
from __future__ import annotations

import ast
import logging
import re
import shutil
import socket
import sys
import types
from pathlib import Path

import pytest
import yaml

from ouroboros.contracts import record_contract as rc
from tests._extension_loader_shared import _clear_loader_state as _clear_loader_state

REPO = Path(__file__).resolve().parents[1]
COLLECTOR = REPO / "docs/examples/log_collector/vector.yaml"
TRACKER = REPO / "docs/examples/error_tracker"
_SECRET = "sk-or-v1-" + "a" * 64


def _vrl_list(remap_source: str, name: str) -> list[str]:
    match = re.search(rf"{name} = (\[.*?\])", remap_source, re.S)
    assert match, f"the remap names its {name} list"
    return ast.literal_eval(match.group(1))


def test_the_collector_reads_the_passport_planes_and_survives_long_rows_rotation_and_downtime():
    config = yaml.safe_load(COLLECTOR.read_text(encoding="utf-8"))
    [source] = config["sources"].values()
    include, root = set(source["include"]), "/data"  # the Compose example's data root
    assert all(path.startswith(root + "/") for path in include)
    logs = {row.log for row in rc.ANCHOR_ROWS}
    assert {f"{root}/logs/{log}.jsonl" for log in logs} <= include
    assert {f"{root}/archive/{log}_*.jsonl" for log in logs} <= include  # generations rotated while down
    drive = {row.log for row in rc.ANCHOR_ROWS if row.plane == "drive_logs"}
    assert drive and {f"{root}/state/headless_tasks/*/data/logs/{log}.jsonl" for log in drive} <= include
    assert not any("headless_tasks" in path and path.endswith("/tools.jsonl") for path in include)  # REPLICA_RULE
    assert source["fingerprint"]["strategy"] == "device_and_inode"
    assert source["max_line_bytes"] >= 1024 * 1024 and source["ignore_older_secs"] > 0


def test_the_collector_forwards_only_anchor_rows_and_their_metadata_and_no_money():
    config = yaml.safe_load(COLLECTOR.read_text(encoding="utf-8"))
    [transform] = config["transforms"].values()
    assert set(_vrl_list(transform["source"], "anchors")) == set(rc.ANCHOR_BY_TYPE)
    allowed = set(_vrl_list(transform["source"], "allowed"))
    metadata = set().union(*(row.metadata for row in rc.ANCHOR_ROWS))
    content = set().union(*(row.content for row in rc.ANCHOR_ROWS))
    assert allowed <= metadata, sorted(allowed - metadata)
    assert not allowed & content
    assert not {name for name in allowed if "cost" in name or name.endswith("_usd")}  # ACCOUNTING_RULE
    assert set(rc.CORRELATION_IDS) <= allowed  # every join key ships
    for row in rc.ANCHOR_ROWS:
        assert set(row.natural_key) <= allowed, row.type


class _Integration:
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs


@pytest.fixture
def stub_sentry(monkeypatch):
    calls = {}

    def init(**kwargs):
        calls.setdefault("init", []).append(kwargs)

    sdk = types.ModuleType("sentry_sdk")
    sdk.init = init
    sdk.get_client = lambda: types.SimpleNamespace(close=lambda timeout=None: calls.setdefault("closed", True))
    monkeypatch.setitem(sys.modules, "sentry_sdk", sdk)
    monkeypatch.setitem(sys.modules, "sentry_sdk.integrations", types.ModuleType("sentry_sdk.integrations"))
    for name, cls in (("atexit", "AtexitIntegration"), ("dedupe", "DedupeIntegration"),
                      ("logging", "LoggingIntegration")):
        module = types.ModuleType(f"sentry_sdk.integrations.{name}")
        setattr(module, cls, type(cls, (_Integration,), {}))
        monkeypatch.setitem(sys.modules, f"sentry_sdk.integrations.{name}", module)
    return calls


@pytest.mark.serial
def test_the_error_tracker_loads_in_process_with_a_safe_sdk(tmp_path, monkeypatch, stub_sentry):
    from ouroboros import extension_loader
    from ouroboros.contracts.skill_manifest import parse_skill_manifest_text
    from ouroboros.skill_loader import SkillReviewState, find_skill, save_enabled, save_review_state

    manifest = parse_skill_manifest_text((TRACKER / "SKILL.md").read_text(encoding="utf-8"))
    declared = {"install_specs", "install", "dependencies"} & set(manifest.raw_extra)
    assert manifest.validate() == [] and not declared  # a declared dependency would force a child process
    assert not [path for path in TRACKER.iterdir() if path.suffix in {".so", ".pyd", ".dylib", ".dll"}]
    drive, skills = tmp_path / "drive", tmp_path / "skills"
    drive.mkdir()
    shutil.copytree(TRACKER, skills / "error_tracker")
    monkeypatch.setattr("ouroboros.config.get_skills_repo_path", lambda: str(skills))
    loaded = find_skill(drive, "error_tracker", repo_path=str(skills))
    save_enabled(drive, loaded.name, True)
    save_review_state(drive, loaded.name, SkillReviewState(status="pass", content_hash=loaded.content_hash))
    loaded = find_skill(drive, loaded.name, repo_path=str(skills))
    assert extension_loader.load_extension(loaded, lambda: {}, drive_root=drive, repo_path=str(skills)) is None
    [kwargs] = stub_sentry["init"]  # register() ran here, in process
    assert kwargs["include_local_variables"] is False and kwargs["include_source_context"] is False
    assert kwargs["send_default_pii"] is False and kwargs["default_integrations"] is False
    assert kwargs["max_request_body_size"] == "never" and kwargs["trace_propagation_targets"] == []
    assert kwargs["server_name"] in {"ouroboros-server", "ouroboros-worker"}
    assert kwargs["server_name"] != socket.gethostname() and kwargs["release"].startswith("ouroboros@")
    names = [type(integration).__name__ for integration in kwargs["integrations"]]
    assert names == ["LoggingIntegration", "DedupeIntegration", "AtexitIntegration"]
    assert kwargs["integrations"][0].kwargs == {"level": None, "event_level": logging.ERROR}
    before_send = kwargs["before_send"]
    leaked = {"message": f"key {_SECRET}", "exception": {"values": [{"type": "RuntimeError",
                                                                      "value": f"provider rejected {_SECRET}"}]}}
    sent = before_send(leaked, {})
    assert _SECRET not in repr(sent) and sent["exception"]["values"][0]["type"] == "RuntimeError"
    clean = {"message": "worker crashed during make_agent", "level": "error", "logger": "supervisor.worker_process"}
    assert before_send(dict(clean), {}) == clean
    carried = before_send({**clean, "extra": {"payload": "whatever a caller logged"}, "breadcrumbs": [1]}, {})
    assert carried == clean  # record extras and anything else unselected stay home
