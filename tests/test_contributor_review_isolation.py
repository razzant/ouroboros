"""Contributor review data isolation (``run_external_review.py --contributor``).

The full entrypoint runs as a real subprocess against a fake LEGACY host: a
journal-era usage ledger the current store imports on first open, settings with
a lifetime ``TOTAL_BUDGET`` and agent-session reviewer rows, and an
Ouroboros-owned engine home whose loopback descriptor points at an in-process
fake engine. The wrapper runs from the installed body, a clean checkout of the
proposal's base: a copy of this runtime whose review operation is a probe. It
drives the ordinary default writers and engine funnels, then returns. Every
host byte and mtime survives; money, reviewer marker and engine traffic land
where the isolation says.

What the probe proves is the isolation of what it drives (the installed body's
review operation running in the wrapper's own process, the data-root switch,
settings pin, panel and efforts, the run cap and the attach-only engine
funnels). It dispatches no reviewer, so no outcome here is a review verdict:
every run ends INCOMPLETE (exit 3), never READY.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from ouroboros.review_run_isolation import (
    ATTACH_HOME_ENV, ISOLATION_RECORD, REVIEW_RUN_CAP_ENV, isolate_review_data,
)
from ouroboros.settings_integrity import SETTINGS_INTEGRITY_ENV

# Real loopback ports and real child processes: the serial pass.
pytestmark = pytest.mark.serial

REPO = pathlib.Path(__file__).resolve().parents[1]
_TOKEN = "fake-host-engine-token"


class _FakeEngine:
    """A loopback engine answering the authenticated handshake at the floor version."""

    def __init__(self, token: str = _TOKEN, version: str = "3.2.0") -> None:
        self.requests: list[tuple[str, str]] = []
        engine = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args) -> None:
                pass

            def _send(self, status: int, payload: dict) -> None:
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _answer(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                if length:
                    self.rfile.read(length)
                engine.requests.append((self.command, self.path))
                if self.headers.get("Authorization") != f"Bearer {token}":
                    return self._send(401, {"code": "unauthorized"})
                if self.path == "/v2/handshake":
                    return self._send(200, {"compatible": True, "protocolMajor": 3,
                                            "engine": {"version": version, "sha": "host-build"}})
                if self.command == "POST" and self.path.endswith("/control"):
                    return self._send(200, {"accepted": True})
                return self._send(404, {"code": "not_found"})

            do_GET = do_POST = do_PATCH = do_DELETE = _answer

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def engine():
    fake = _FakeEngine()
    yield fake
    fake.close()


def _engine_home(data: pathlib.Path, port: int, *, marker_data_dir: str | None = None) -> pathlib.Path:
    home = data / "claudexor"
    (home / "daemon").mkdir(parents=True, exist_ok=True)
    (home / "ouroboros-owned.json").write_text(json.dumps({
        "owner": "ouroboros", "data_dir": marker_data_dir or str(data.resolve()),
        "provisioned_at": "2026-09-07T10:17:15+00:00"}), encoding="utf-8")
    (home / "daemon" / "token").write_text(_TOKEN, encoding="utf-8")
    (home / "daemon" / "control-api.json").write_text(json.dumps({
        "host": "127.0.0.1", "port": port, "tokenPath": str(home / "daemon" / "token")}), encoding="utf-8")
    return home


def _personal_engine(home: pathlib.Path, port: int) -> None:
    """The operator's own ``~/.claudexor/v3`` daemon layout (pre-D30 discovery)."""
    daemon = home / ".claudexor" / "v3" / "daemon"
    daemon.mkdir(parents=True)
    (daemon / "token").write_text(_TOKEN, encoding="utf-8")
    (daemon / "control-api.json").write_text(json.dumps({
        "host": "127.0.0.1", "port": port, "tokenPath": str(daemon / "token")}), encoding="utf-8")


# A synthetic, provider-neutral stand-in for a credential the host keeps in its
# settings: the leak check compares exact bytes, so no provider key shape is needed.
_HOST_PROVIDER_VALUE = "isolation-fixture-host-provider-value"
_REVIEWER_SLOTS = {  # the scope row's effort is the host's surface setting
    "triad": [{"slot_id": "t1", "route": {"kind": "agent_session", "target_id": "codex=gpt-host"},
               "effort": "high"}],
    "scope": [{"slot_id": "s1", "route": {"kind": "agent_session", "target_id": "cursor=claude-host"}}],
}
# A stale projection of another panel, as a harness environment exported before
# the owner's last settings edit would carry it.
_INHERITED_PANEL = {
    "OUROBOROS_REVIEWER_SLOTS": json.dumps({
        "triad": [{"slot_id": "stale", "route": {"kind": "agent_session", "target_id": "codex=gpt-stale"},
                   "effort": "low"}],
        "scope": [{"slot_id": "stale-scope", "route": {"kind": "agent_session", "target_id": "codex=gpt-stale"}}],
    }),
    "OUROBOROS_EFFORT_SCOPE_REVIEW": "low",
}


def _legacy_host(root: pathlib.Path, port: int) -> pathlib.Path:
    """A journal-era install: the current store would import this ledger on first open."""
    host = root / "host-data"
    (host / "state").mkdir(parents=True)
    rows, base = [], {"attempt_id": "legacy-1", "kind": "attempt", "root_task_id": "legacy",
                      "provider": "local", "model": "stub", "reservation_upper_bound_usd": "0.25",
                      "pricing_known": True, "ts": "2026-01-01T00:00:00Z"}
    for state in ("reserved", "dispatched", "settled"):
        row = {**base, "state": state, "seq": len(rows) + 1}
        if state == "settled":
            row.update(cost_usd=0.25, cost_final=True)
        rows.append(row)
    (host / "state" / "usage_attempts.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    (host / "settings.json").write_text(json.dumps({
        "TOTAL_BUDGET": 500.0,
        "OPENROUTER_API_KEY": _HOST_PROVIDER_VALUE,
        "OUROBOROS_REVIEWER_SLOTS": json.dumps(_REVIEWER_SLOTS),
        "OUROBOROS_EFFORT_SCOPE_REVIEW": "xhigh",
        "OUROBOROS_REVIEW_ENFORCEMENT": "blocking",
    }, indent=2), encoding="utf-8")
    _engine_home(host, port)
    return host


def _tree_state(root: pathlib.Path) -> dict[str, tuple]:
    """Every path under ``root`` with its bytes digest and mtime (dirs: mtime only)."""
    state = {".": ("dir", root.stat().st_mtime_ns)}
    for path in sorted(root.rglob("*")):
        stat = path.lstat()
        rel = path.relative_to(root).as_posix()
        state[rel] = (("dir", stat.st_mtime_ns) if path.is_dir()
                      else (hashlib.sha256(path.read_bytes()).hexdigest(), stat.st_mtime_ns))
    return state


# The installed body's review operation in the fixture repository: the ordinary
# runtime writers and engine funnels, driven exactly as a reviewer would reach them.
_PROBE_OPERATION = '''"""Fixture review operation: exercise the default writers, report, return."""
import json, os, pathlib, subprocess
from types import SimpleNamespace

_MACHINERY = "installed"


class ReviewChangeArgumentError(ValueError):
    """The operation's typed argument refusal, part of the wrapper's contract."""


def run_review_change(ctx, **arguments):
    from ouroboros import config
    from ouroboros.claudexor_daemon import ensure_owned_gateway, read_owned_gateway
    from ouroboros.gateways.claudexor import ClaudexorGateway
    from ouroboros.reviewer_slot_config import load_reviewer_slot_config, record_reviewer_slot_executions
    from ouroboros.settings_setup_contract import resolve_total_budget_usd
    from ouroboros.usage_accounting import (
        AttemptRequest, BudgetExceeded, mark_dispatched, release_attempt, reserve_attempt,
        settle_attempt, usage_projection)

    here = pathlib.Path(__file__).resolve().parents[2]
    report = {
        "machinery": _MACHINERY, "ppid": os.getppid(),
        "arguments": {key: arguments.get(key) for key in ("root", "surface", "subject", "base", "head")},
        "machinery_sha": subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(here),
                                        capture_output=True, text=True).stdout.strip(),
        "data_dir": str(config.DATA_DIR), "settings_path": str(config.SETTINGS_PATH),
        "drive_root": str(ctx.drive_root), "global_limit_usd": resolve_total_budget_usd(),
        "saved_total_budget": config.load_settings().get("TOTAL_BUDGET"),
        "ledger_at_start": usage_projection(),
    }
    try:
        config.save_settings(config.load_settings())
        report["settings_write"] = "written"
    except config.SettingsIntegrityError as exc:
        report["settings_write"] = "refused: " + str(exc)
    with ensure_owned_gateway() as gateway:
        report["engine_version"] = gateway.engine_version
        report["own_run_cancel"] = gateway.cancel_run("probe-run", reason="probe")
    with read_owned_gateway() as gateway:
        report["read_gateway_version"] = gateway.engine_version
    with ClaudexorGateway() as gateway:
        report["bare_gateway_version"] = gateway.handshake()["engine"]["version"]
    def admit(usd):
        return reserve_attempt(AttemptRequest(model="synthetic", provider="test", task_id="probe",
                                              root_task_id="probe", reservation_usd=usd))

    def spend(usd):
        held = admit(usd)
        mark_dispatched(held)
        settle_attempt(held, cost_usd=usd, cost_final=True)

    def probe(label, step):
        try:
            step()
            report[label] = "admitted"
        except BudgetExceeded as exc:
            report[label] = "refused: " + str(exc)

    probe("spend_1", lambda: spend(1.0))
    # Known spend decides (#1487): a hold above the remainder is exposure, not spending.
    probe("hold_10", lambda: release_attempt(admit(10.0)))
    if report["spend_1"] == "admitted":
        spend(3.0)  # known spend reaches the explicit cap
    probe("at_cap", lambda: release_attempt(admit(0.01)))
    slots = {row.slot_id: row for row in load_reviewer_slot_config().triad}
    report["configured_triad"] = sorted(slots)
    record_reviewer_slot_executions(
        "review", [SimpleNamespace(slot_id="t1", status="responded", usage={})], slots)
    pathlib.Path(os.environ["ISOLATION_PROBE_OUT"]).write_text(json.dumps(report, default=str))
    return {"error": "isolation probe: no reviewer was dispatched"}
'''

_CARRIERS = ("VERSION", "pyproject.toml", "uv.lock", "web/package.json", "web/modules/api_types.js",
             "README.md", "docs/ARCHITECTURE.md", "site/install/index.html", "docs/install/index.html")


def _git(repo: pathlib.Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=str(repo), check=True,
                          capture_output=True, text=True).stdout


def _fixture_repo(root: pathlib.Path) -> pathlib.Path:
    """Base = this runtime (probe review operation), checked out as the installed body;
    branch ``proposal`` = base + one proposal file + its own review operation, which
    must never run."""
    repo = root / "repo"
    repo.mkdir()
    for args in (("init",), ("config", "user.email", "t@example.com"), ("config", "user.name", "T"),
                 ("config", "core.autocrlf", "false")):
        _git(repo, *args)
    tracked = _git(REPO, "ls-files", "-co", "--exclude-standard", "ouroboros").splitlines()
    for rel in (*tracked, *_CARRIERS, "scripts/run_external_review.py",
                "scripts/contributor_review_evidence.py"):
        target = repo / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / rel, target)
    (repo / ".gitignore").write_text("__pycache__/\n*.pyc\n", encoding="utf-8")
    operation = repo / "ouroboros" / "tools" / "review_change.py"
    operation.write_text(_PROBE_OPERATION, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    _git(repo, "branch", "base")
    operation.write_text(_PROBE_OPERATION.replace('_MACHINERY = "installed"', '_MACHINERY = "proposal"'),
                         encoding="utf-8")
    (repo / "proposal.txt").write_text("proposal\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "proposal")
    _git(repo, "branch", "proposal")
    _git(repo, "checkout", "-q", "--detach", "base")
    return repo


def _review(repo: pathlib.Path, host: pathlib.Path, out: pathlib.Path, *extra: str,
            inherited: dict | None = None):
    env = {**os.environ, "OUROBOROS_DATA_DIR": str(host),
           "OUROBOROS_SETTINGS_PATH": str(host / "settings.json"),
           "ISOLATION_PROBE_OUT": str(out / "probe.json")}
    for key in ("OUROBOROS_SETTINGS_SHA256", "OUROBOROS_REVIEW_RUN_CAP_USD",
                "OUROBOROS_CLAUDEXOR_ATTACH_HOME", "TOTAL_BUDGET", *_INHERITED_PANEL):
        env.pop(key, None)
    env.update(inherited or {})
    return subprocess.run(
        [sys.executable, str(repo / "scripts" / "run_external_review.py"), "--contributor",
         "--base-ref=base", "--head-ref=proposal", f"--output={out / 'packet'}",
         f"--drive-root={out / 'drive'}", *extra, "--", "PR title"],
        cwd=str(repo), env=env, capture_output=True, text=True, timeout=600,
    )


def test_full_entrypoint_keeps_a_legacy_host_untouched(tmp_path, engine):
    host = _legacy_host(tmp_path, engine.port)
    repo = _fixture_repo(tmp_path)
    base_sha, head_sha = (_git(repo, "rev-parse", ref).strip() for ref in ("base", "proposal"))

    # Control: the fixture IS the hazard. A default reader on (a copy of) this
    # host imports its journal into a new store and leaves lock evidence.
    control = tmp_path / "control-data"
    shutil.copytree(host, control)
    read = subprocess.run(
        [sys.executable, "-c", "import pathlib,sys; from ouroboros import usage_store; "
         "print(len(usage_store.read_usage_records(pathlib.Path(sys.argv[1]))))", str(control)],
        cwd=str(REPO), env={**os.environ, "OUROBOROS_DATA_DIR": str(control)},
        capture_output=True, text=True, timeout=120)
    assert read.returncode == 0, read.stderr[-2000:]
    assert (control / "state" / "usage.sqlite").is_file()
    assert set(_tree_state(control)) - set(_tree_state(host))  # new host-side files

    before = _tree_state(host)
    out = tmp_path / "isolated"
    # The invoking shell carries a stale projection of another panel: the panel
    # and its efforts still come from the pinned host settings.
    run = _review(repo, host, out, "--run-cap-usd=4", "--attach-host-engine", inherited=_INHERITED_PANEL)
    after = _tree_state(host)
    assert (out / "probe.json").is_file(), run.stderr[-4000:]
    report = json.loads((out / "probe.json").read_text(encoding="utf-8"))

    # Not one host byte, mtime, file, store or lock.
    assert after == before
    assert not (host / "state" / "usage.sqlite").exists()
    assert not list(host.rglob("*.lock")) and not (host / "logs").exists()
    # D31: the installed body (the base checkout) ran its own review operation over the
    # frozen base..head, in the wrapper's process; the proposal's copy never ran.
    assert (report["machinery"], report["machinery_sha"]) == ("installed", base_sha)
    assert report["ppid"] == os.getpid()  # the wrapper this test started, never a re-executed copy
    assert report["arguments"] == {"root": "system_repo", "surface": "change", "subject": "base..head",
                                   "base": base_sha, "head": head_sha}
    # The drive is the whole data root; the host settings are read pinned, never copied.
    drive = (out / "drive").resolve()
    assert pathlib.Path(report["data_dir"]).resolve() == drive
    assert pathlib.Path(report["drive_root"]).resolve() == drive
    assert pathlib.Path(report["settings_path"]).resolve() == (host / "settings.json").resolve()
    assert report["settings_write"].startswith("refused")
    assert not (drive / "settings.json").exists()
    # Money: the explicit cap is the global limit of a ledger that starts empty;
    # the saved lifetime budget and the host's journal are not.
    assert report["saved_total_budget"] == 500.0
    assert report["global_limit_usd"] == 4.0
    assert (report["ledger_at_start"]["accounted_usd"], report["ledger_at_start"]["limit_usd"]) == (0.0, 4.0)
    assert report["spend_1"] == "admitted"
    assert report["hold_10"] == "admitted"  # $1 known: a $10 hold is exposure, not spending
    assert report["at_cap"].startswith("refused")  # known $4 reached the $4 cap, not the saved $500
    assert (drive / "state" / "usage.sqlite").is_file()
    sys.path.insert(0, str(REPO))
    from ouroboros import usage_store

    def probe_rows(state: str) -> list[dict]:
        return [row for row in usage_store.read_usage_records(drive)
                if row.get("task_id") == "probe" and row.get("state") == state]

    assert [float(row["cost_usd"]) for row in probe_rows("settled")] == [1.0, 3.0]
    assert len(probe_rows("released")) == 1  # the $10 hold, admitted then let go
    # The configured rows came from the host settings; the marker landed in the drive.
    assert report["configured_triad"] == ["t1"]
    marker = json.loads((drive / "state" / "reviewer_slot_last_execution.json").read_text(encoding="utf-8"))
    assert marker["t1"]["requested"]["effort"] == "high"
    # Engine: attach at the floor version, own-run cancel allowed, nothing managed:
    # no engine home, runtime or Node prepared on the drive either.
    assert report["engine_version"] == report["read_gateway_version"] == "3.2.0"
    assert report["bare_gateway_version"] == "3.2.0"
    paths = {path for _method, path in engine.requests}
    assert paths == {"/v2/handshake", "/v2/runs/probe-run/control"}
    assert not (drive / "claudexor").exists() and not (drive / "state" / "cx").exists()
    # The packet discloses the isolation and the cap's limited authority.
    record = json.loads((drive / "contributor-review-isolation.json").read_text(encoding="utf-8"))
    assert record["run_cap_usd"] == 4.0
    assert record["settings_sha256"] == hashlib.sha256((host / "settings.json").read_bytes()).hexdigest()
    evidence = json.loads((out / "packet" / "review-evidence.json").read_text(encoding="utf-8"))
    assert evidence["trust"]["installed_body_execution"]["executing_checkout_head"] == base_sha
    # No record, no reviewer: incomplete evidence, with both facts disclosed.
    outcome = evidence["production_outcome"]
    assert (outcome["block_reason"], outcome["original_block_reason"]) == (
        "execution_receipt_mismatch", "review_record_unavailable")
    assert outcome["execution_receipt_mismatches"] == ["missing_actor:scope:s1", "missing_actor:triad:t1"]
    assert evidence["review_record"] == {"record_id": None, "available": False}
    assert evidence["budget"]["run_cap_usd"] == 4.0
    assert evidence["budget"]["authority"] == "isolated_review_ledger"
    isolation = evidence["review_config"]["data_isolation"]
    assert isolation["review_data_root"] == "$REVIEW_DRIVE"
    slots = {surface: [(row["slot_id"], row["route"]["target_id"], row["effort"])
                       for row in evidence["review_config"][f"{surface}_slots"]] for surface in ("triad", "scope")}
    assert slots == {"triad": [("t1", "codex=gpt-host", "high")], "scope": [("s1", "cursor=claude-host", "xhigh")]}
    assert run.returncode == 3  # the probe dispatched no reviewer: never READY
    # Neither the host's provider key nor its engine token reaches any output.
    outputs = [run.stdout.encode(), run.stderr.encode()]
    for path in (out / "packet").rglob("*"):
        if path.suffix == ".zip":
            with zipfile.ZipFile(path) as packet:
                outputs.extend(packet.read(name) for name in packet.namelist())
        elif path.is_file():
            outputs.append(path.read_bytes())
    host_values = (_HOST_PROVIDER_VALUE.encode(), _TOKEN.encode())
    assert len(outputs) > 3 and not [blob for blob in outputs if any(value in blob for value in host_values)]

    # Continuation with the SAME cap on the same drive: the known spend already
    # recorded counts, so the spend that fit before no longer does.
    again = _review(repo, host, out, "--run-cap-usd=4", "--attach-host-engine")
    assert again.returncode == 3, again.stderr[-4000:]
    second = json.loads((out / "probe.json").read_text(encoding="utf-8"))
    assert (second["ledger_at_start"]["settled_usd"], second["global_limit_usd"]) == (4.0, 4.0)
    assert second["spend_1"].startswith("refused") and second["hold_10"].startswith("refused")
    assert [float(row["cost_usd"]) for row in probe_rows("settled")] == [1.0, 3.0]
    # A continuation never changes the cap it was opened with.
    changed = _review(repo, host, out, "--run-cap-usd=9", "--attach-host-engine")
    assert changed.returncode == 3 and "keeps that cap" in changed.stderr
    assert _tree_state(host) == before


def test_the_proposal_checkout_is_refused_before_any_engine_or_review(tmp_path, engine):
    """D31: run from a checkout that contains the proposal, the review would run the
    proposal's own review operation; it refuses before reaching the engine or any review."""
    host = _legacy_host(tmp_path, engine.port)
    repo = _fixture_repo(tmp_path)
    _git(repo, "checkout", "-q", "proposal")
    before = _tree_state(host)
    out = tmp_path / "isolated"

    run = _review(repo, host, out, "--run-cap-usd=4", "--attach-host-engine")

    assert run.returncode == 3, run.stderr[-4000:]  # infrastructure, never "empty diff" (2)
    assert "already contains proposal" in run.stderr
    assert not (out / "probe.json").exists() and not (out / "packet").exists()
    assert engine.requests == []
    assert _tree_state(host) == before


def test_session_rows_without_attach_refuse_before_any_engine_contact(tmp_path, engine):
    host = _legacy_host(tmp_path, engine.port)
    repo = _fixture_repo(tmp_path)
    before = _tree_state(host)

    run = _review(repo, host, tmp_path / "isolated", "--run-cap-usd=4")

    assert run.returncode == 3
    assert "--attach-host-engine" in run.stderr
    assert not (tmp_path / "isolated" / "probe.json").exists()
    assert engine.requests == []
    assert _tree_state(host) == before


# ---------------------------------------------------------------------------
# In-process contracts of the runtime seams the wrapper selects.
# ---------------------------------------------------------------------------


@pytest.fixture
def owned(monkeypatch, tmp_path):
    from ouroboros import claudexor_daemon, config

    isolated = tmp_path / "review-drive"
    isolated.mkdir()
    monkeypatch.setattr(config, "DATA_DIR", isolated)
    monkeypatch.setattr(claudexor_daemon, "_MANAGER", None)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("attach-only must never manage an engine")

    for name in ("_spawn", "reconcile_rotation", "_remember_stop_targets", "stop", "stop_outcome",
                 "panic_stop", "_request_operator_stop", "_terminate_child"):
        monkeypatch.setattr(claudexor_daemon.OwnedClaudexorDaemon, name, forbidden)
    monkeypatch.setattr(claudexor_daemon, "_write_ownership_marker", forbidden)
    monkeypatch.setattr("ouroboros.claudexor_runtime.get_runtime_manager", forbidden)
    return claudexor_daemon


def _bare_handshake() -> dict:
    """Supervision/interaction consumers' bare discovery, connection closed after."""
    from ouroboros.gateways.claudexor import ClaudexorGateway

    with ClaudexorGateway() as gateway:
        return gateway.handshake()


def test_attach_only_reaches_the_running_host_engine_and_manages_nothing(owned, engine, tmp_path, monkeypatch):
    from ouroboros.gateways.claudexor import ClaudexorGateway, ClaudexorUnavailable

    host = tmp_path / "host-data"
    monkeypatch.setenv(ATTACH_HOME_ENV, str(_engine_home(host, engine.port)))

    with owned.ensure_owned_gateway() as gateway:
        assert gateway.engine_version == "3.2.0"  # the floor, whatever this checkout pins
        gateway.cancel_run("own-run")  # this process's own runs stay cancellable
    assert owned.owned_engine_version() == "3.2.0"
    with owned.read_owned_gateway() as gateway:
        assert gateway.engine_version == "3.2.0"
    with ClaudexorGateway() as gateway:  # supervision/interaction consumers' bare discovery
        gateway.handshake()
    with pytest.raises(ClaudexorUnavailable) as refused:
        owned.get_owned_daemon().ensure_running()
    assert refused.value.code == "attach_only_engine"
    assert {path for _method, path in engine.requests} == {"/v2/handshake", "/v2/runs/own-run/control"}
    assert not (tmp_path / "review-drive" / "claudexor").exists()


def test_attach_only_refuses_missing_foreign_or_dead_engines_without_fallback(owned, engine, tmp_path, monkeypatch):
    from ouroboros.gateways.claudexor import ClaudexorGateway, ClaudexorUnavailable

    # A live personal-home engine that must never be chosen in place of the host's.
    _personal_engine(tmp_path / "home", engine.port)
    host = tmp_path / "host-data"
    home = _engine_home(host, engine.port)
    monkeypatch.setenv(ATTACH_HOME_ENV, str(home))

    def refusal(call) -> str:
        with pytest.raises(ClaudexorUnavailable) as caught:
            call()
        return caught.value.code

    monkeypatch.setenv(ATTACH_HOME_ENV, str(host / "no-such-engine-home"))
    assert refusal(owned.ensure_owned_gateway) == "attach_provenance_refused"
    monkeypatch.setenv(ATTACH_HOME_ENV, str(home))
    (home / "ouroboros-owned.json").unlink()
    assert refusal(owned.ensure_owned_gateway) == "attach_provenance_refused"
    _engine_home(host, engine.port, marker_data_dir=str(tmp_path / "another-install"))
    assert refusal(owned.read_owned_gateway) == "attach_provenance_refused"
    _engine_home(host, engine.port)
    closed = _FakeEngine()
    closed.close()  # a dead engine: its port no longer answers
    _engine_home(host, closed.port)
    for call in (owned.ensure_owned_gateway, owned.read_owned_gateway,
                 lambda: ClaudexorGateway().handshake()):
        assert refusal(call) == "daemon_unreachable"
    assert engine.requests == []  # never the personal home, never a spawn


def test_attach_only_refuses_an_engine_below_the_transport_floor(owned, tmp_path, monkeypatch):
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    old = _FakeEngine(version="3.1.9")
    try:
        _personal_engine(tmp_path / "home", old.port)
        monkeypatch.setenv(ATTACH_HOME_ENV, str(_engine_home(tmp_path / "host-data", old.port)))
        for call in (owned.ensure_owned_gateway, owned.read_owned_gateway, _bare_handshake):
            with pytest.raises(ClaudexorUnavailable) as refused:
                call()
            assert refused.value.code == "engine_too_old"
        assert owned.owned_engine_version() == ""  # never proven, never rotated to
        assert {path for _method, path in old.requests} == {"/v2/handshake"}
    finally:
        old.close()


def test_default_route_without_attach_keeps_owned_management(owned, engine, tmp_path, monkeypatch):
    from ouroboros.gateways.claudexor import DaemonEndpoint, discover_daemon

    monkeypatch.delenv(ATTACH_HOME_ENV, raising=False)
    managed = []
    monkeypatch.setattr(owned.OwnedClaudexorDaemon, "ensure_running",
                        lambda self, **_kw: managed.append("ensure") or DaemonEndpoint(
                            host="127.0.0.1", port=engine.port, token=_TOKEN))
    monkeypatch.setattr(owned.OwnedClaudexorDaemon, "reconcile_rotation",
                        lambda self, gateway: managed.append("reconcile"))
    owned.ensure_owned_gateway().close()
    assert managed == ["ensure", "reconcile"]
    # An unprovisioned owned home still falls through to the operator layout.
    _personal_engine(tmp_path / "home", engine.port)
    assert discover_daemon().port == engine.port


def test_an_isolated_review_without_attach_never_starts_an_engine(owned, tmp_path, monkeypatch):
    """No attach selection: reviewer rows on Claudexor are refused up front, and any other
    Claudexor call the cycle makes (a Light model canonicalizing an unparsed verdict) gets a
    typed refusal — never a prepared runtime or a started engine under the review drive."""
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from ouroboros.review_verdict_extraction import _extract_verdict_via_light_model

    monkeypatch.setenv(REVIEW_RUN_CAP_ENV, "4")  # set only by the isolated review's launcher
    monkeypatch.delenv(ATTACH_HOME_ENV, raising=False)
    prepared, refusals = [], []

    def prepare(*_args, **_kwargs):
        prepared.append("runtime")
        raise AssertionError("an isolated review must never prepare an engine runtime")

    def gateway_refusals(real=owned.ensure_owned_gateway):
        try:
            return real()
        except ClaudexorUnavailable as exc:
            refusals.append(exc.code)
            raise

    monkeypatch.setattr("ouroboros.claudexor_runtime.get_runtime_manager", prepare)
    monkeypatch.setattr("ouroboros.llm_claudexor.ensure_owned_gateway", gateway_refusals)
    monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", "claudexor::claude=claude-light")

    assert _extract_verdict_via_light_model("The change looks fine to me.")[0] is None
    assert (prepared, refusals) == ([], ["attach_only_engine"])
    assert not (tmp_path / "review-drive" / "claudexor").exists()


def test_run_cap_is_the_global_limit_over_the_saved_budget(tmp_path, monkeypatch):
    from ouroboros import config
    from ouroboros.settings_setup_contract import resolve_total_budget_usd

    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"TOTAL_BUDGET": 500}), encoding="utf-8")
    monkeypatch.setattr(config, "SETTINGS_PATH", settings)
    monkeypatch.delenv(REVIEW_RUN_CAP_ENV, raising=False)
    assert resolve_total_budget_usd() == 500.0
    monkeypatch.setenv(REVIEW_RUN_CAP_ENV, "4")
    assert resolve_total_budget_usd() == 4.0
    for unreadable in ("0", "-3", "inf", "nan", "four"):
        monkeypatch.setenv(REVIEW_RUN_CAP_ENV, unreadable)
        assert resolve_total_budget_usd() == 0.0  # a zero allowance, never "no limit"


def test_an_unreadable_run_cap_admits_nothing(tmp_path, monkeypatch):
    """Only the launcher writes the cap (validated); anything else reads as $0, never as no limit."""
    from ouroboros import usage_accounting
    from ouroboros.usage_accounting import AttemptRequest, BudgetExceeded, reserve_attempt, usage_projection
    from ouroboros.usage_admission import review_wave_admission

    monkeypatch.setenv(REVIEW_RUN_CAP_ENV, "four")
    for request in (dict(reservation_usd=0.01), dict(force_unknown_reservation=True)):
        with pytest.raises(BudgetExceeded):
            reserve_attempt(AttemptRequest(model="synthetic", provider="test", task_id="t",
                                           root_task_id="t", drive_root=tmp_path, **request))
    assert usage_projection(tmp_path)["limit_usd"] == 0.0
    # The whole-wave pre-admission reads the same zero (a priced seat, no catalog I/O).
    monkeypatch.setattr(usage_accounting, "_reservation_cost", lambda _request: 0.5)
    wave = review_wave_admission(tmp_path, root_task_id="t", models=["synthetic"], prompt_chars=4000)
    assert (wave["fits"], wave["binding_axis"], wave["global_limit_usd"]) == (False, "global", 0.0)


def test_pinned_settings_read_takes_no_lock_and_writers_refuse(tmp_path, monkeypatch):
    from ouroboros import config

    settings = tmp_path / "live" / "settings.json"
    settings.parent.mkdir()
    settings.write_text(json.dumps({"TOTAL_BUDGET": 500}), encoding="utf-8")
    monkeypatch.setattr(config, "SETTINGS_PATH", settings)
    monkeypatch.setenv(config.SETTINGS_INTEGRITY_ENV, hashlib.sha256(settings.read_bytes()).hexdigest())
    before = _tree_state(settings.parent)

    assert config._acquire_settings_lock() is None
    assert config.load_settings()["TOTAL_BUDGET"] == 500
    with pytest.raises(config.SettingsIntegrityError):
        config.save_settings({"TOTAL_BUDGET": 1})
    assert _tree_state(settings.parent) == before  # no lock created beside the host file
    settings.write_text(json.dumps({"TOTAL_BUDGET": 900}), encoding="utf-8")  # an owner edit
    with pytest.raises(config.SettingsIntegrityError):
        config.load_settings()


def test_isolation_fails_closed_and_continuation_keeps_its_cap(tmp_path, monkeypatch):
    host = _legacy_host(tmp_path, 1).resolve()
    monkeypatch.delitem(sys.modules, "ouroboros.config")  # the check reads module presence only
    for key in ("OUROBOROS_DATA_DIR", SETTINGS_INTEGRITY_ENV, REVIEW_RUN_CAP_ENV, ATTACH_HOME_ENV):
        monkeypatch.setenv(key, "inherited")
    monkeypatch.delenv("OUROBOROS_SETTINGS_PATH", raising=False)

    def isolate(drive, cap="4", attach=False):
        return isolate_review_data(host_data=host, drive_root=str(drive), run_cap=cap,
                                   attach_host_engine=attach)

    for overlapping in (host, host / "state", tmp_path):
        with pytest.raises(RuntimeError, match="overlaps the host data root"):
            isolate(overlapping)
    foreign = tmp_path / "another-data-root"
    (foreign / "state").mkdir(parents=True)  # not a drive this lane opened: its ledger is not empty
    with pytest.raises(RuntimeError, match="not empty"):
        isolate(foreign)
    assert sorted(path.name for path in foreign.iterdir()) == ["state"]
    with pytest.raises(ValueError, match="positive finite"):
        isolate(tmp_path / "drive", cap="0")
    facts = isolate(tmp_path / "drive", attach=True)
    drive = (tmp_path / "drive").resolve()
    assert facts["review_data_root"] == os.environ["OUROBOROS_DATA_DIR"] == str(drive)
    assert os.environ["OUROBOROS_SETTINGS_PATH"] == str(host / "settings.json")
    assert os.environ[SETTINGS_INTEGRITY_ENV] == facts["settings_sha256"]
    assert os.environ[REVIEW_RUN_CAP_ENV] == "4.0"
    assert os.environ[ATTACH_HOME_ENV] == str(host / "claudexor")
    assert sorted(path.name for path in drive.iterdir()) == [ISOLATION_RECORD]  # no settings copy
    with pytest.raises(RuntimeError, match="keeps that cap"):
        isolate(drive, cap="5")
    isolate(drive)  # an inherited attach selection is not this run's choice
    assert ATTACH_HOME_ENV not in os.environ
    (host / "settings.json").unlink()
    # An engine host without settings is a mistaken host root: never the default panel.
    with pytest.raises(RuntimeError, match="no host settings"):
        isolate(drive, attach=True)
    facts = isolate(drive)
    assert facts["settings_sha256"] is None and SETTINGS_INTEGRITY_ENV not in os.environ
    assert os.environ["OUROBOROS_SETTINGS_PATH"] == str(drive / "settings.json")
    monkeypatch.setitem(sys.modules, "ouroboros.config", object())
    with pytest.raises(RuntimeError, match="imported before"):
        isolate(tmp_path / "other")


def test_another_spelling_of_the_host_root_is_refused_before_mkdir(tmp_path, monkeypatch):
    """``Path.resolve`` keeps the spelling it was given, and a spelling is not a directory.

    On a case-insensitive volume (the macOS and Windows default) another case of the host
    root IS the host root; on a case-sensitive one it is a distinct path, refused all the
    same. A macOS firmlink spelling (``/System/Volumes/Data``) is the host under any case rule.
    """
    host = _legacy_host(tmp_path, 1).resolve()
    for key in ("OUROBOROS_DATA_DIR", SETTINGS_INTEGRITY_ENV, REVIEW_RUN_CAP_ENV, ATTACH_HOME_ENV):
        monkeypatch.setenv(key, "inherited")
    variant = host.with_name(host.name.upper())
    spellings = [variant / "review", variant, host.parent.with_name(host.parent.name.upper())]
    firmlink = pathlib.Path("/System/Volumes/Data" + str(host))
    if firmlink.is_dir() and os.path.samefile(firmlink, host):  # only where that alias exists
        spellings += [firmlink / "review", firmlink]
    before = _tree_state(host)

    config = sys.modules.pop("ouroboros.config")  # the check reads module presence only
    try:  # restored at once: a teardown importing config meanwhile would load a second copy
        for spelling in spellings:
            with pytest.raises(RuntimeError, match="overlaps the host data root"):
                isolate_review_data(host_data=host, drive_root=str(spelling), run_cap="4", attach_host_engine=False)
    finally:
        sys.modules["ouroboros.config"] = config

    assert _tree_state(host) == before  # no drive, record or ledger inside the installation
    assert not variant.exists() or os.path.samefile(variant, host)  # a distinct variant was not created
    assert os.environ["OUROBOROS_DATA_DIR"] == "inherited"  # refused before selecting anything


def test_a_default_drive_is_refused_before_allocating_inside_the_host(tmp_path, monkeypatch):
    """No ``--drive-root`` and a temporary directory inside the host: nothing is left behind."""
    host = _legacy_host(tmp_path, 1).resolve()
    (host / "tmp").mkdir()
    for key in ("OUROBOROS_DATA_DIR", SETTINGS_INTEGRITY_ENV, REVIEW_RUN_CAP_ENV, ATTACH_HOME_ENV):
        monkeypatch.setenv(key, "inherited")
    monkeypatch.delenv("OUROBOROS_SETTINGS_PATH", raising=False)
    before = _tree_state(host)

    config = sys.modules.pop("ouroboros.config")  # the check reads module presence only
    try:
        monkeypatch.setattr("tempfile.tempdir", str(host / "tmp"))
        with pytest.raises(RuntimeError, match="host data root"):
            isolate_review_data(host_data=host, drive_root="", run_cap="4", attach_host_engine=False)
        assert _tree_state(host) == before and os.environ["OUROBOROS_DATA_DIR"] == "inherited"
        monkeypatch.setattr("tempfile.tempdir", str(tmp_path))  # outside the host: a fresh default drive
        drive = pathlib.Path(isolate_review_data(host_data=host, drive_root="", run_cap="4",
                                                 attach_host_engine=False)["review_data_root"])
    finally:
        sys.modules["ouroboros.config"] = config
    assert drive.parent == tmp_path.resolve() and drive.name.startswith("ouroboros-external-review-")

    # The wrapper entrypoint leaves no allocation either.
    review = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "run_external_review.py"), "--contributor", "--base-ref=HEAD",
         "--head-ref=HEAD", "--run-cap-usd=4", f"--output={tmp_path / 'packet'}", "--", "PR title"],
        cwd=str(REPO), capture_output=True, text=True, timeout=300,
        env={**{key: value for key, value in os.environ.items()
                if key not in {SETTINGS_INTEGRITY_ENV, REVIEW_RUN_CAP_ENV, ATTACH_HOME_ENV}},
             "OUROBOROS_DATA_DIR": str(host), "OUROBOROS_SETTINGS_PATH": str(host / "settings.json"),
             "TMPDIR": str(host / "tmp")})
    assert review.returncode == 3 and "host data root" in review.stderr, review.stderr[-4000:]
    # Python's writability check may touch the directory's mtime; no path or byte is added.
    assert {path: facts[0] for path, facts in _tree_state(host).items()} == {
        path: facts[0] for path, facts in before.items()}
    assert not (tmp_path / "packet").exists()


def test_wrapper_settings_load_refuses_bytes_that_changed_under_its_pin(tmp_path, monkeypatch):
    import scripts.run_external_review as module

    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"OUROBOROS_REVIEW_ENFORCEMENT": "blocking"}), encoding="utf-8")
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(settings))
    monkeypatch.setenv(SETTINGS_INTEGRITY_ENV, hashlib.sha256(b"what was pinned").hexdigest())
    with pytest.raises(RuntimeError, match="changed after this review pinned it"):
        module._load_settings_into_env()


def test_wrapper_settings_load_makes_the_pinned_document_the_whole_panel(tmp_path, monkeypatch):
    import scripts.run_external_review as module
    from ouroboros.review_run_isolation import PINNED_PANEL_KEYS
    from ouroboros.settings_defaults import RETIRED_COMMA_LIST_SETTING_KEYS

    monkeypatch.setattr(module, "_keys_file", lambda: None)
    settings = tmp_path / "settings.json"
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(settings))

    def load(document, *, pinned=True, **inherited):
        for key in (*PINNED_PANEL_KEYS, *RETIRED_COMMA_LIST_SETTING_KEYS, "TOTAL_BUDGET"):
            monkeypatch.delenv(key, raising=False)
        for key, value in inherited.items():
            monkeypatch.setenv(key, value)
        settings.write_text(json.dumps(document), encoding="utf-8")
        if pinned:
            monkeypatch.setenv(SETTINGS_INTEGRITY_ENV, hashlib.sha256(settings.read_bytes()).hexdigest())
        else:
            monkeypatch.delenv(SETTINGS_INTEGRITY_ENV, raising=False)
        module._load_settings_into_env()

    stale = {**_INHERITED_PANEL, "OUROBOROS_SUBAGENTS": "[]", "TOTAL_BUDGET": "7"}
    # The document stores the panel as an object and leaves the scope effort and
    # the subagent registry unset: the stale projection supplies none of them.
    load({"OUROBOROS_REVIEWER_SLOTS": _REVIEWER_SLOTS, "TOTAL_BUDGET": 500}, **stale)
    assert json.loads(os.environ["OUROBOROS_REVIEWER_SLOTS"]) == _REVIEWER_SLOTS
    assert not {"OUROBOROS_EFFORT_SCOPE_REVIEW", "OUROBOROS_SUBAGENTS"} & set(os.environ)
    assert os.environ["TOTAL_BUDGET"] == "7"  # outside the panel an explicit environment value still wins
    # A retired reviewer comma-list is no panel: the read seam drops the document's copy and an
    # inherited one is a stale projection, so with the panel unset the default panel stays the default.
    retired = {key: "retired/a,retired/b,retired/c" for key in RETIRED_COMMA_LIST_SETTING_KEYS}
    load(dict(retired), **retired)
    assert not set(RETIRED_COMMA_LIST_SETTING_KEYS) & set(os.environ)
    # The unpinned operator lane keeps "an explicit environment value wins" for every key.
    load({"OUROBOROS_REVIEWER_SLOTS": json.dumps(_REVIEWER_SLOTS)}, pinned=False, **stale)
    assert {key: os.environ.get(key) for key in _INHERITED_PANEL} == _INHERITED_PANEL
    # A pinned document that is not a settings object refuses instead of running the default panel.
    with pytest.raises(RuntimeError, match="not a settings object"):
        load([], **stale)


# The default panel a host task runs when its settings name none: the task-start
# view of the document (``subagent_runtime``), as a host task binds it, each row on
# the lane it dispatches on (``" (local)"`` where ``use_local``).
_HOST_TASK_PANEL = '''import json
from ouroboros.model_slots import local_lane_label
from ouroboros.review_substrate import scope_reviewer_slots
from ouroboros.reviewer_slot_config import load_reviewer_slot_config, triad_delivery_slots
from ouroboros.settings_integrity import task_settings_scope
from ouroboros.subagent_runtime import apply_task_start_settings

with task_settings_scope(apply_task_start_settings()):
    source = load_reviewer_slot_config().source
    triad, scope = ([local_lane_label(slot.model, slot.use_local) for slot in slots]
                    for slots in (triad_delivery_slots(), scope_reviewer_slots()))
print(json.dumps({"source": source, "triad": triad, "scope": scope}))
'''
# The wrapper's resolution in ``_prepare_review_configuration`` order, after its real
# isolation (no proposal read, no provider probe), frozen and then delivered as the
# review dispatches it, with the OpenRouter rows the wrapper would probe a key for.
_WRAPPER_PANEL = '''import json, sys
import scripts.run_external_review as wrapper

if sys.argv[1:]:
    wrapper.isolate_review_data(host_data=wrapper.DATA, drive_root=sys.argv[1], run_cap="4",
                                attach_host_engine=False)
wrapper._load_settings_into_env()
wrapper._apply_contributor_review_env()
frozen = wrapper._freeze_contributor_slots(wrapper._resolved_review_config(profile=wrapper._CONTRIBUTOR_PROFILE))
from ouroboros.model_slots import local_lane_label
from ouroboros.review_substrate import scope_reviewer_slots
from ouroboros.reviewer_slot_config import triad_delivery_slots

triad, scope = ([local_lane_label(slot.model, slot.use_local) for slot in slots]
                for slots in (triad_delivery_slots(), scope_reviewer_slots()))
print(json.dumps({"source": frozen["slot_config_source"], "triad": triad, "scope": scope,
                  "openrouter_probe": wrapper._configured_openrouter_models(frozen)}))
'''


def _panel_resolver(root: pathlib.Path, document: dict):
    """Run a panel script against a host whose settings are ``document``, and its pin."""
    from ouroboros.settings_defaults import RETIRED_COMMA_LIST_SETTING_KEYS, settings_env_keys

    host = root / "host-data"
    host.mkdir(parents=True)
    settings = host / "settings.json"
    settings.write_text(json.dumps(document), encoding="utf-8")
    dropped = {*settings_env_keys(), *RETIRED_COMMA_LIST_SETTING_KEYS, SETTINGS_INTEGRITY_ENV, "OUROBOROS_KEYS_FILE"}
    clean = {key: value for key, value in os.environ.items() if key not in dropped}
    clean.update(OUROBOROS_DATA_DIR=str(host), OUROBOROS_SETTINGS_PATH=str(settings))

    def resolve(code: str, *argv: str, **inherited: str) -> dict:
        run = subprocess.run([sys.executable, "-c", code, *argv], cwd=str(REPO), env={**clean, **inherited},
                             capture_output=True, text=True, timeout=300)
        assert run.returncode == 0, run.stderr[-4000:]
        result = json.loads(run.stdout.strip().splitlines()[-1])
        probe = result.pop("openrouter_probe", None)
        return result if probe is None else (result, probe)

    return resolve, {SETTINGS_INTEGRITY_ENV: hashlib.sha256(settings.read_bytes()).hexdigest()}


def test_a_pinned_document_without_a_panel_gets_its_hosts_default_panel(tmp_path):
    """The default panel's model and provider inputs come from the pinned document too."""
    resolve, pin = _panel_resolver(tmp_path, {"ANTHROPIC_API_KEY": _HOST_PROVIDER_VALUE,
                                              "OUROBOROS_MODEL": "anthropic::claude-opus-5"})

    host_panel = resolve(_HOST_TASK_PANEL, **pin)
    assert host_panel["source"] == "default"
    assert host_panel["triad"] == ["anthropic::claude-opus-5"] * 3  # the host's exclusive direct provider
    assert resolve(_WRAPPER_PANEL, str(tmp_path / "drive-clean")) == (host_panel, [])
    # A shell exported for another configuration: an older Main.
    stale = {"OUROBOROS_MODEL": "anthropic::claude-sonnet-4-5"}
    assert resolve(_WRAPPER_PANEL, **stale)[0] != host_panel  # unpinned, it selects another panel
    assert resolve(_WRAPPER_PANEL, str(tmp_path / "drive-stale"), **stale) == (host_panel, [])
    # Another provider's key is a credential the run's calls can use, so the panel is the one
    # the host derives with that credential available, whichever source supplies it.
    extra = {"OPENAI_API_KEY": "isolation-fixture-second-provider-value"}
    assert resolve(_WRAPPER_PANEL, str(tmp_path / "drive-extra"), **extra)[0] == resolve(_HOST_TASK_PANEL, **pin, **extra)


def test_a_pinned_default_panel_keeps_the_credentials_its_calls_use(tmp_path):
    """Saved models choose the panel; the credentials are the run's, from any supported source.

    A credential reaches this run from its environment or the wrapper's keys file as
    well as from the document (synthetic values; no provider is contacted).
    """
    credential = {"ANTHROPIC_API_KEY": _HOST_PROVIDER_VALUE}
    resolve, pin = _panel_resolver(tmp_path, {"OUROBOROS_MODEL": "anthropic::claude-opus-5"})
    host_panel = resolve(_HOST_TASK_PANEL, **pin, **credential)
    assert host_panel["triad"] == ["anthropic::claude-opus-5"] * 3
    keys_file = tmp_path / "keys.txt"
    keys_file.write_text(f"anthropic: {_HOST_PROVIDER_VALUE}\n", encoding="utf-8")
    for name, supplied in (("environment", credential), ("keys-file", {"OUROBOROS_KEYS_FILE": str(keys_file)}),
                           ("stale-main", {**credential, "OUROBOROS_MODEL": "anthropic::claude-sonnet-4-5"})):
        assert resolve(_WRAPPER_PANEL, str(tmp_path / f"drive-{name}"), **supplied) == (host_panel, []), name


def test_a_frozen_default_row_dispatches_on_the_lane_it_was_resolved_on(tmp_path):
    """Each frozen row dispatches, and is probed, on the lane the host would run it on."""
    # A local-only install: Main on the local lane, no remote credential saved.
    resolve, pin = _panel_resolver(tmp_path, {
        "USE_LOCAL_MAIN": True, "LOCAL_MODEL_SOURCE": "owner/local-model.gguf", "OUROBOROS_MODEL": "owner-local"})
    local_panel = resolve(_HOST_TASK_PANEL, **pin)
    assert local_panel["triad"] == ["owner-local (local)"] * 3
    assert local_panel["scope"] and set(local_panel["scope"]) == {"owner-local (local)"}
    for name, inherited in (("clean", {}), ("stale-lane-flag", {"USE_LOCAL_MAIN": "0"}),
                            ("remote-credential", {"OPENAI_API_KEY": "isolation-fixture-second-provider-value"})):
        host = resolve(_HOST_TASK_PANEL, **pin, **inherited)
        wrapper, probe = resolve(_WRAPPER_PANEL, str(tmp_path / f"drive-local-{name}"), **inherited)
        assert wrapper == host, name
        if name != "remote-credential":  # the document names the lane flag; a stale one does not move it
            assert host == local_panel and probe == [], name  # a local row needs no OpenRouter key


def test_run_identities_never_reach_tests_or_preflight():
    from ouroboros.test_environment import scrub_environment
    from tests.conftest import _NEVER_INHERITED, _isolated_child_env

    selections = {REVIEW_RUN_CAP_ENV: "4", ATTACH_HOME_ENV: "/host/claudexor"}
    assert set(selections) <= set(_NEVER_INHERITED)  # conftest's literals name the leaf's identities
    assert not set(selections) & set(scrub_environment(selections))
    assert not set(selections) & set(_isolated_child_env(selections))
    assert not set(selections) & set(os.environ)
