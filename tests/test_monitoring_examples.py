"""The two monitoring examples in docs/examples follow the record passport and stay safe.

The collector reads only the passport's planes and forwards only metadata the passport lists;
the error tracker loads in process and starts its SDK with nothing that leaks secrets.
"""
from __future__ import annotations

import ast
import json
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
from tests.test_deployment_policy import install as install_fixture

install = install_fixture  # the shared isolated data root, as a fixture of this module
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


def test_a_row_written_to_both_planes_reaches_the_sink_once():
    """A replicated row the collector reads from a child drive too is deduplicated by its natural key; tools
    rows are read from the canonical file only, and every other row bypasses the dedupe."""
    config = yaml.safe_load(COLLECTOR.read_text(encoding="utf-8"))
    [source] = config["sources"].values()
    child = {path.rsplit("/", 1)[-1][:-len(".jsonl")] for path in source["include"] if "headless_tasks" in path}
    replicated = [row for row in rc.ANCHOR_ROWS if row.plane == "both" and row.log in child]
    assert [row.type for row in replicated] == ["llm_api_error"]
    route, once = config["transforms"]["ouroboros_api_errors"], config["transforms"]["ouroboros_api_error_once"]
    assert route["inputs"] == ["ouroboros_metadata"] and route["route"] == {"api_error": '.type == "llm_api_error"'}
    assert once["type"] == "dedupe" and once["inputs"] == ["ouroboros_api_errors.api_error"]
    assert set(rc.ANCHOR_BY_TYPE["llm_api_error"].natural_key) | {"type"} <= set(once["fields"]["match"])
    [sink] = config["sinks"].values()
    assert sorted(sink["inputs"]) == ["ouroboros_api_error_once", "ouroboros_api_errors._unmatched"]


def test_the_collector_forwards_only_anchor_rows_and_their_metadata_and_no_money():
    config = yaml.safe_load(COLLECTOR.read_text(encoding="utf-8"))
    transform = config["transforms"]["ouroboros_metadata"]
    assert set(_vrl_list(transform["source"], "anchors")) == set(rc.ANCHOR_BY_TYPE)
    allowed = set(_vrl_list(transform["source"], "allowed"))
    metadata = set().union(*(row.metadata for row in rc.ANCHOR_ROWS))
    content = set().union(*(row.content for row in rc.ANCHOR_ROWS))
    assert allowed <= metadata, sorted(allowed - metadata)
    assert not allowed & content
    assert not {name for name in allowed if "cost" in name or name.endswith("_usd")}  # ACCOUNTING_RULE
    assert set(rc.CORRELATION_IDS) <= allowed  # every join key ships
    assert {"record_contract_version", "prompt_tokens", "completion_tokens", "cached_tokens", "cache_write_tokens",
            "session_id", "review_wave_id", "review_slot_id", "logged_at", "provider_code", "provider_error_type"} <= allowed
    # Filtering preserves values (including null); it must never coerce null to
    # zero or stamp a historical row with the current writer's contract version.
    assert '. = filter(row)' in transform["source"]
    assert '.record_contract_version =' not in transform["source"]
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
    sdk.init, sdk.VERSION = init, "stub"
    sdk.get_client = lambda: types.SimpleNamespace(close=lambda timeout=None: calls.setdefault("closed", True))
    transport = types.ModuleType("sentry_sdk.transport")
    transport.HttpTransport = type("HttpTransport", (), {})
    monkeypatch.setitem(sys.modules, "sentry_sdk", sdk)
    monkeypatch.setitem(sys.modules, "sentry_sdk.transport", transport)
    monkeypatch.setitem(sys.modules, "sentry_sdk.integrations", types.ModuleType("sentry_sdk.integrations"))
    for name, cls in (("atexit", "AtexitIntegration"), ("dedupe", "DedupeIntegration"),
                      ("logging", "LoggingIntegration")):
        module = types.ModuleType(f"sentry_sdk.integrations.{name}")
        setattr(module, cls, type(cls, (_Integration,), {}))
        monkeypatch.setitem(sys.modules, f"sentry_sdk.integrations.{name}", module)
    return calls


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
    assert "dsn" not in kwargs, "with no setting the SDK reads SENTRY_DSN from the environment itself, as before"
    assert isinstance(kwargs["transport"], type), "a Transport class the SDK instantiates, never a factory callable"
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


def test_the_tracker_status_says_only_what_it_can_see(tmp_path, stub_sentry, monkeypatch):
    """A process whose start time cannot be read is of unknown liveness, never ended, and still counts when its last
    report failed; a provably gone process is ended and no longer a problem; records past the newest 16 are named
    and left out of every sentence."""
    import importlib.util
    import datetime
    import os
    import subprocess

    spec = importlib.util.spec_from_file_location("error_tracker_status_under_test", TRACKER / "plugin.py")
    plugin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plugin)
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    folder = tmp_path / "delivery"
    folder.mkdir()

    def write(name, pid, started, loaded_at, outcome, detail):
        (folder / f"{name}.json").write_text(json.dumps({
            "role": "worker", "pid": pid, "started": started, "loaded_at": loaded_at, "source": "settings",
            "sdk": "2.71.0", "counts": {"queued": 1, outcome: 1},
            "last": {"outcome": outcome, "detail": detail, "at": loaded_at}}), encoding="utf-8")
        stamp = datetime.datetime.fromisoformat(loaded_at.replace("Z", "+00:00")).timestamp()
        os.utime(folder / f"{name}.json", (stamp, stamp))

    me = os.getpid()
    write("running", me, plugin._birth(me), "2026-11-08T10:00:03Z", "accepted", "HTTP 200")
    write("ended", gone.pid, "an earlier start", "2026-11-08T10:00:02Z", "refused", "HTTP 500")
    write("unknown", me, "", "2026-11-08T10:00:01Z", "unreachable", "no HTTP answer")
    status = plugin._status_of(str(tmp_path))
    assert [row["process"] for row in status["processes"]] == [
        f"worker · pid {me} · running", f"worker · pid {gone.pid} · ended", f"worker · pid {me} · liveness unknown"]
    assert status["problem"] == (f"Last outcome not accepted in worker · pid {me} · liveness unknown: unreachable "
                                 "(no HTTP answer) at 2026-11-08T10:00:01Z")
    assert not status["ok"] and not status["note"] and status["processes"][0]["sdk"] == "2.71.0"

    (folder / "unknown.json").unlink()
    for index in range(17):
        write(f"older-{index}", gone.pid, "an earlier start", f"2026-11-07T00:00:{index:02d}Z", "refused", "HTTP 500")
    reads, load = [], plugin.json.load
    def counted(handle):
        reads.append(handle.name)
        return load(handle)
    monkeypatch.setattr(plugin.json, "load", counted)
    status = plugin._status_of(str(tmp_path))
    assert len(reads) == 16
    assert len(status["processes"]) == 16 and status["note"] == "3 process records are not shown and not counted above."
    assert status["ok"].startswith("The receiver answered HTTP 2xx to the last report of each process shown here")
    assert not status["problem"], "an ended process's last refusal is history, not a current problem"
    # An old record updated by a still-running process must not disappear behind newer process starts.
    write("old-server", me, plugin._birth(me), "2026-11-06T00:00:00Z", "unreachable", "no HTTP answer")
    os.utime(folder / "old-server.json", (1794220000, 1794220000))
    reads.clear()
    status = plugin._status_of(str(tmp_path))
    assert len(reads) == 16 and "unreachable" in status["problem"]
    assert status["processes"][0]["loaded"] == "2026-11-06T00:00:00Z"


@pytest.fixture
def receiver():
    """A synthetic loopback Sentry endpoint: it answers each envelope POST with the status the test sets and keeps
    the bodies. Nothing leaves this machine."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    state = types.SimpleNamespace(status=200, bodies=[])

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            state.bodies.append((self.path, self.headers.get("Content-Encoding") or "", body))
            self.send_response(state.status)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *_args):
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state.port, state.server = server.server_address[1], server
    yield state
    server.shutdown()
    server.server_close()


def _envelope_text(encoding: str, body: bytes) -> str:
    """The envelope as the receiver would decode it (the SDK compresses with Brotli when present, else gzip)."""
    import gzip

    if encoding == "br":
        import brotli

        body = brotli.decompress(body)
    elif encoding == "gzip":
        body = gzip.decompress(body)
    return body.decode("utf-8", "replace")


def test_the_real_sdk_reports_accepted_refused_and_unreachable_without_stopping_work(
        install, monkeypatch, tmp_path, receiver):
    """sentry-sdk 2.71.0 against the loopback receiver, under a policy that requires the tracker and fixes its DSN:
    the DSN reaches the SDK only through the grant; each process's status names what its transport saw, never init
    or a queue as delivery; a refusing or absent receiver changes that status, never the requirement, the work or
    the local log."""
    sentry_sdk = pytest.importorskip("sentry_sdk")
    if sentry_sdk.VERSION != "2.71.0":
        pytest.skip("the transport hooks are verified on sentry-sdk 2.71.0")
    from ouroboros import extension_loader
    from ouroboros.config import load_settings
    from ouroboros.deployment_extensions import state_rows
    from ouroboros.skill_loader import SkillReviewState, find_skill, save_review_state, save_skill_grants
    from tests.test_deployment_policy import manage

    empty = tmp_path / "no-local-skills"
    empty.mkdir()
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(empty))
    monkeypatch.delenv("SENTRY_DSN", raising=False)
    shutil.copytree(TRACKER, install.data / "skills" / "external" / "error_tracker")
    loaded = find_skill(install.data, "error_tracker")
    save_review_state(install.data, loaded.name, SkillReviewState(status="pass", content_hash=loaded.content_hash))
    dsn = f"http://fixturepublickey@127.0.0.1:{receiver.port}/1"
    manage(install, monkeypatch, {"version": 1, "extensions": {"when_unmet": "hold", "required": [{
        "name": "error_tracker", "path": str(loaded.skill_dir), "content_hash": loaded.content_hash,
        "settings": {"SENTRY_DSN": dsn}}]}})
    assert extension_loader.reconcile_extension("error_tracker", install.data, load_settings)["reason"] == "missing_grants"
    save_skill_grants(install.data, loaded.name, ["SENTRY_DSN"], content_hash=loaded.content_hash,
                      requested_keys=["SENTRY_DSN"], granted_permissions=[], requested_permissions=[])
    assert extension_loader.reconcile_extension("error_tracker", install.data, load_settings)["live_loaded"]
    status = extension_loader.list_routes()["/api/extensions/error_tracker/status"]["handler"]
    local = []
    handler = logging.Handler()
    handler.emit = local.append
    logger = logging.getLogger("tests.error_tracker")
    logger.addHandler(handler)
    try:
        first = status(None)
        assert first["idle"] == "Configured; nothing has been sent yet." and first["processes"][0]["queued"] == 0
        assert first["processes"][0]["destination"] == "Settings"
        observed = []
        for answer, outcome in ((200, "accepted"), (500, "refused")):
            receiver.status = answer
            logger.error("synthetic failure %s near %s", answer, _SECRET)  # returns: the work goes on
            sentry_sdk.flush(timeout=5)
            observed.append(status(None))
            assert observed[-1]["processes"][0]["last"].startswith(f"{outcome} (HTTP {answer})"), observed[-1]
        receiver.server.shutdown()
        receiver.server.server_close()
        logger.error("synthetic failure with the receiver gone")
        sentry_sdk.flush(timeout=5)
        gone = status(None)
    finally:
        logger.removeHandler(handler)
    [row] = gone["processes"]
    assert (row["queued"], row["accepted"], row["refused"], row["unreachable"]) == (3, 1, 1, 1), row
    assert row["last"].startswith("unreachable (no HTTP answer)") and "Last outcome not accepted" in gone["problem"]
    assert observed[0]["ok"].startswith("The receiver answered HTTP 2xx") and "not that it stored" in observed[0]["ok"]
    assert len(local) == 3, "every record still reaches the process's own log handlers"
    assert [state["state"] for state in state_rows(install.data)] == ["met"], "a sink outage is the tracker's status"
    sent = [_envelope_text(encoding, body) for path, encoding, body in receiver.bodies if path.endswith("/envelope/")]
    events = [text for text in sent if '"type":"event"' in text]
    assert len(events) == 2 and all(_SECRET not in text and '"vars"' not in text for text in events)
    assert all('"server_name":"ouroboros-' in text for text in events)
    for doc in (first, *observed, gone):
        assert dsn not in json.dumps(doc) and "fixturepublickey" not in json.dumps(doc)
