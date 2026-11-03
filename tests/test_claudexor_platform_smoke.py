"""The Claudexor three-platform gate's own logic, under the ordinary suite.

Only the parts that do not need a daemon: the path-shape guard, the request shape,
the on-disk mutation evidence, and the honesty block the summary carries. The
transport itself is what the CI job exercises; testing it here would mean mocking
the thing under test.

The path-shape test is the one that earns its keep. This repository has already been
bitten by a `$HOME`-MEMBERSHIP guard on Windows, where the runner's temp directory
lives under the user profile and membership therefore passes for paths a POSIX guard
rejects. The gate's own guard is by string shape, and these tests pin that.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import json
from types import SimpleNamespace

import pytest


def _load_smoke():
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "claudexor_platform_smoke.py"
    spec = importlib.util.spec_from_file_location("claudexor_platform_smoke", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


smoke = _load_smoke()


def test_managed_fixture_main_rebinds_before_runtime_imports(tmp_path, monkeypatch):
    monkeypatch.setattr(smoke.os, "environ", dict(os.environ))
    monkeypatch.setenv("HOME", str(tmp_path / "operator"))
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path / "live-data"))
    monkeypatch.setenv("CLAUDEXOR_CONFIG_DIR", str(tmp_path / "live-engine"))
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-secret")
    monkeypatch.setenv("OUROBOROS_BUNDLE_DIR", str(tmp_path / "bundle"))
    isolated = tmp_path / "isolated"
    isolated.mkdir()
    monkeypatch.setattr(smoke.tempfile, "mkdtemp", lambda **_kw: str(isolated))
    def run(args):
        assert args.lane == "fixture" and args.managed_runtime
        assert os.environ["OUROBOROS_DATA_DIR"] == str(isolated / "data")
        assert os.environ["HOME"] == str(isolated / "home")
        assert os.environ["USERPROFILE"] == str(isolated / "home")
        assert "CLAUDEXOR_CONFIG_DIR" not in os.environ and "OPENAI_API_KEY" not in os.environ
        assert os.environ["OUROBOROS_BUNDLE_DIR"] == str(tmp_path / "bundle")
        return {}
    monkeypatch.setattr(smoke, "run_smoke", run)
    monkeypatch.setattr(smoke, "emit_summary", lambda *_a: None)
    assert smoke.main(["--managed-runtime", "--lane", "fixture"]) == 0


def test_managed_fixture_uses_short_posix_temp_parent(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(smoke.tempfile, "mkdtemp", lambda **kwargs: calls.append(kwargs) or str(tmp_path))
    root = smoke.isolated_fixture_root()
    assert root == tmp_path.resolve()
    if os.name != "nt" and pathlib.Path("/tmp").is_dir():
        assert calls == [{"prefix": "cx-", "dir": "/tmp"}]
    else:
        assert calls == [{"prefix": "cx-"}]


@pytest.mark.parametrize("platform", ["darwin-arm64", "linux-x64", "win32-x64"])
def test_operator_cli_resolves_exact_bundled_node_without_npm_or_install(tmp_path, monkeypatch, platform):
    from ouroboros import claudexor_runtime as rt, config, platform_layer as pl

    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    manager = rt.ClaudexorRuntimeManager()
    pin = manager.pin
    assert pin is not None
    cli = rt.managed_runtime_dir(pin) / pin.cli_entrypoint
    cli.parent.mkdir(parents=True)
    cli.write_text("fixture CLI")
    node = tmp_path / "bundle" / "node-standalone" / ("node.exe" if platform == "win32-x64" else "bin/node")
    node.parent.mkdir(parents=True)
    node.write_text("exact fixture Node")
    monkeypatch.setattr(manager, "_managed_metadata", lambda: {"verified": True})
    monkeypatch.setattr(pl, "bundled_resource_bases", lambda: [tmp_path / "bundle"])
    monkeypatch.setattr(pl, "embedded_node_candidates", lambda _base: [node])
    monkeypatch.setattr(pl, "probe_node_version", lambda path: pin.node_version if path == str(node) else "")
    monkeypatch.setattr(pl, "node_distribution_platform", lambda: platform)
    monkeypatch.setattr(manager, "_install", lambda *_a, **_kw: pytest.fail("read-only resolution installed runtime"))
    monkeypatch.setattr(manager, "_probe", lambda *_a, **_kw: pytest.fail("resolution probed the daemon"))
    monkeypatch.setattr(manager, "ensure", lambda *_a, **_kw: pytest.fail("resolution provisioned runtime"))
    assert manager.resolve_cli_command(require_npm=False) == [str(node), str(cli)]
    assert manager.resolve_cli_command() == [] and manager.resolve_cli_command(require_npm=True) == []


@pytest.mark.parametrize("barrier,receipt,rc,passes", [
    (False, {"ok": True, "stopped": True, "outcome": "exited"}, 0, True),
    (True, {"ok": True, "stopped": True, "outcome": "exited"}, 0, True),
    (True, {"ok": True, "stopped": True, "outcome": "killed"}, 0, False),
    (True, {"ok": True}, 0, False),
    (True, {"ok": True, "stopped": True, "outcome": "exited"}, 1, False),
])
def test_operator_fixture_observes_receipt_and_always_cleans_up(tmp_path, monkeypatch, barrier, receipt, rc, passes):
    from ouroboros import claudexor_daemon as d, claudexor_runtime as rt, config, process_custody as c, platform_layer as pl

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    home = tmp_path / "claudexor" / "daemon"
    lease_dir = home / "fixture.writer"
    if barrier:
        lease_dir = lease_dir / "active.writer"
    lease_dir.mkdir(parents=True)
    lease = lease_dir / "owner.json"
    lease.write_text(json.dumps({"pid": 42, "token": "fixture-token", "identity": {"present": True}}))
    c.append_jsonl(c.ledger_path(tmp_path), {"pid": 42, "purpose": d.CUSTODY_PURPOSE,
                                           "fingerprint": {"start_time": "new", "start_time_boot": "new"}})
    monkeypatch.setattr(c, "_fingerprint_matches", lambda *_a, **_kw: False)
    run_dir = tmp_path / "run"
    events = run_dir / "events.jsonl"
    events.parent.mkdir(parents=True)
    events.write_text('{"seq":6,"type":"harness.event","payload":{"harness_id":"fake-hang","type":"thinking"}}\n{"partial":')
    cleanup, cancelled, calls = [], [], []
    old = SimpleNamespace(wait=lambda **_kw: 0)
    proc = SimpleNamespace(pid=42, poll=lambda: None if not cleanup else 0, wait=lambda **_kw: 0)
    owner = SimpleNamespace(_proc=old, stop=lambda: cleanup.append(True))
    gateway = SimpleNamespace(
        engine_version="3.10.1", handshake=lambda: {"engine": {"sha": "a" * 40}},
        register_project=lambda _r: "project", start_run=lambda *_a, **_kw: {"runId": "run"},
        get_run=lambda *_a, **_kw: {"summary": {"state": "running", "runDir": str(run_dir)}},
        cancel_run=lambda *_a, **_kw: cancelled.append(True), close=lambda: None,
    )
    def start():
        owner._proc = proc
        return gateway
    monkeypatch.setattr(d, "ensure_owned_gateway", start)
    monkeypatch.setattr(d, "get_owned_daemon", lambda: owner)
    def cli_command(*, require_npm):
        assert require_npm is False
        return ["node", "claudexor.bundle.cjs"]
    monkeypatch.setattr(rt, "get_runtime_manager", lambda: SimpleNamespace(
        pin=SimpleNamespace(version="3.10.1", build_sha="a" * 40),
        resolve_cli_command=cli_command,
    ))
    def run(argv, **kw):
        calls.append(argv)
        assert "identity" not in json.loads(lease.read_text())
        assert c._read_ledger(tmp_path)[0]["fingerprint"] == {"start_time": ""}
        if passes:
            lease.unlink()
            lease_dir.rmdir()
        return smoke.subprocess.CompletedProcess(argv, rc, stdout=json.dumps(receipt))
    monkeypatch.setattr(smoke.subprocess, "run", run)
    def stop():
        result = smoke.subprocess.run(["node", "claudexor.bundle.cjs", "daemon", "stop", "--json"])
        return "stopped" if result.returncode == 0 and receipt.get("stopped") else "unconfirmed"
    monkeypatch.setattr(d, "OwnedClaudexorDaemon", lambda: SimpleNamespace(_proc=None, stop_outcome=stop))
    monkeypatch.setattr(pl, "pid_is_alive", lambda _pid: False)
    monkeypatch.setattr(smoke, "build_request", lambda *_a: {"harnesses": ["fake-hang"]})
    facts = {}
    if passes:
        smoke.verify_operator_stop_fixture(tmp_path, facts)
        assert facts["operator_stop"]["pid_gone"] and facts["operator_stop"]["lease_released"]
    else:
        with pytest.raises(smoke.SmokeFailure, match="cooperative shutdown"):
            smoke.verify_operator_stop_fixture(tmp_path, facts)
    assert cleanup == [True] and bool(cancelled) is not passes
    assert calls == [["node", "claudexor.bundle.cjs", "daemon", "stop", "--json"]]


# -- the path-shape guard ------------------------------------------------------


def test_a_plain_child_resolves_under_the_root(tmp_path):
    assert smoke.confined_child(tmp_path, "README.md") == pathlib.Path(
        os.path.realpath(str(tmp_path / "README.md"))
    )


@pytest.mark.parametrize("part", ["..", ".", "", "a/b", "a\\b", "C:", "x:y"])
def test_a_traversing_or_rooted_component_is_refused(tmp_path, part):
    with pytest.raises(smoke.SmokeFailure) as excinfo:
        smoke.confined_child(tmp_path, part)
    assert excinfo.value.code == "unsafe_path_component"


def test_a_symlink_that_leaves_the_root_is_refused(tmp_path):
    # The guard resolves BEFORE comparing, so a link out of the root is caught by the
    # same string-shape rule that catches `..` — no membership question is asked.
    outside = tmp_path.parent / "outside-the-root"
    outside.mkdir(exist_ok=True)
    root = tmp_path / "root"
    root.mkdir()
    try:
        (root / "escape").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("this platform/user cannot create a directory symlink")
    with pytest.raises(smoke.SmokeFailure) as excinfo:
        smoke.confined_child(root, "escape")
    assert excinfo.value.code == "path_escaped_root"


# -- the mutation evidence -----------------------------------------------------


def test_the_fixture_lane_needs_the_deterministic_file_on_disk(tmp_path):
    ok, why = smoke.mutation_evidence("fixture", tmp_path)
    assert not ok and smoke.FIXTURE_CHANGE_FILE in why
    (tmp_path / smoke.FIXTURE_CHANGE_FILE).write_text("x", encoding="utf-8")
    ok, why = smoke.mutation_evidence("fixture", tmp_path)
    assert ok


def test_an_unedited_readme_is_not_evidence(tmp_path):
    (tmp_path / smoke.README_NAME).write_text(smoke.README_SEED, encoding="utf-8")
    ok, why = smoke.mutation_evidence("live", tmp_path)
    assert not ok and "byte-identical" in why


def test_a_readme_edited_into_something_else_is_not_the_asked_edit(tmp_path):
    (tmp_path / smoke.README_NAME).write_text("totally different\n", encoding="utf-8")
    ok, why = smoke.mutation_evidence("live", tmp_path)
    assert not ok and smoke.LIVE_EXPECT_TOKEN in why


def test_the_asked_edit_is_evidence(tmp_path):
    (tmp_path / smoke.README_NAME).write_text(
        smoke.README_SEED.replace("pending", smoke.LIVE_EXPECT_TOKEN), encoding="utf-8"
    )
    ok, _ = smoke.mutation_evidence("live", tmp_path)
    assert ok


def test_live_retry_seed_comparison_normalizes_windows_crlf(tmp_path):
    (tmp_path / smoke.README_NAME).write_bytes(
        smoke.README_SEED.replace("\n", "\r\n").encode("utf-8")
    )
    assert smoke.live_seed_is_unchanged(tmp_path)


def test_poll_retries_one_windows_git_atomic_object_race(monkeypatch):
    class GitRaceError(Exception):
        code = "ENOENT"

    class Gateway:
        calls = 0

        def get_run(self, _run_id):
            self.calls += 1
            if self.calls == 1:
                raise GitRaceError(
                    r"ENOENT: lstat 'C:\r\reviews\w\.git\objects\e5\tmp_obj_x'"
                )
            return {"summary": {"state": "succeeded"}}

    gateway = Gateway()
    monkeypatch.setattr(smoke.time, "sleep", lambda _seconds: None)
    assert smoke.poll_to_terminal(gateway, "run-1", 30)["summary"]["state"] == "succeeded"
    assert gateway.calls == 2


def test_poll_does_not_retry_an_unrelated_enoent(monkeypatch):
    class OtherError(Exception):
        code = "ENOENT"

    class Gateway:
        def get_run(self, _run_id):
            raise OtherError("ENOENT: missing final/answer.md")

    monkeypatch.setattr(smoke.time, "sleep", lambda _seconds: None)
    with pytest.raises(OtherError):
        smoke.poll_to_terminal(Gateway(), "run-1", 30)


@pytest.mark.parametrize(
    ("lane", "attempt", "mutated", "seed_unchanged"),
    [
        (lane, attempt, mutated, seed_unchanged)
        for lane in ("fixture", "live")
        for attempt in (1, 2)
        for mutated in (False, True)
        for seed_unchanged in (False, True)
    ],
)
def test_live_no_mutation_retry_truth_table(
        lane, attempt, mutated, seed_unchanged):
    expected = (
        lane == "live" and attempt == 1 and not mutated and seed_unchanged
    )
    assert smoke.should_retry_live_no_mutation(
        lane, attempt, mutated, seed_unchanged
    ) is expected


@pytest.mark.parametrize(("lane", "expected_attempts"), [("fixture", 1), ("live", 2)])
def test_smoke_retry_budget_and_final_no_mutation_are_strict(
        tmp_path, monkeypatch, lane, expected_attempts):
    root = tmp_path / lane
    root.mkdir()
    (root / smoke.README_NAME).write_text(smoke.README_SEED, encoding="utf-8")

    class FakeGateway:
        engine_version = "9.9.9"

        def __init__(self):
            self.requests = []
            self.keys = []
            self.removed = []
            self.cancelled = []
            self.closed = 0

        def handshake(self):
            return {"protocolMajor": 3, "engine": {"sha": "a" * 40}}

        def register_project(self, path):
            assert path == str(root)
            return "project-1"

        def start_run(self, request, idempotency_key):
            self.requests.append(request)
            self.keys.append(idempotency_key)
            return {"runId": f"run-{len(self.requests)}"}

        def get_run(self, run_id):
            return {
                "summary": {"state": "succeeded", "runDir": f"/runs/{run_id}"},
                "primaryOutput": {
                    "kind": "answer", "path": "final/answer.md",
                    "text": "ok", "bytes": 2,
                },
            }

        def get_run_artifact(self, run_id, path):
            return b"ok"

        def cancel_run(self, run_id, reason):
            self.cancelled.append((run_id, reason))

        def remove_project(self, project_id):
            self.removed.append(project_id)

        def close(self):
            self.closed += 1

    gateway = FakeGateway()

    class FakeUnavailable(Exception):
        pass

    class FakeCx:
        ClaudexorUnavailable = FakeUnavailable

        @staticmethod
        def discover_daemon():
            return smoke.argparse.Namespace(host="127.0.0.1", port=7777)

        @staticmethod
        def ClaudexorGateway(endpoint):
            return gateway

        @staticmethod
        def engine_at_least(actual, floor):
            return True

    request = {"prompt": "same request"}
    monkeypatch.setattr(smoke, "load_seam", lambda: FakeCx)
    monkeypatch.setattr(smoke, "seed_fixture_repo", lambda: root)
    monkeypatch.setattr(smoke, "build_request", lambda *args: request)
    monkeypatch.setattr(smoke, "containment_report", lambda *args: {})
    args = smoke.argparse.Namespace(
        lane=lane,
        harness="fake-implement" if lane == "fixture" else "codex",
        model="" if lane == "fixture" else "gpt-test",
        effort="" if lane == "fixture" else "low",
        max_seconds=10,
        grace_seconds=1,
        secret_env="",
        secret_name="",
        managed_runtime=False,
    )

    with pytest.raises(smoke.SmokeFailure) as excinfo:
        smoke.run_smoke(args)

    assert excinfo.value.code == "no_mutation"
    assert len(gateway.requests) == expected_attempts
    assert all(sent is request for sent in gateway.requests)
    assert len(gateway.keys) == expected_attempts
    assert len(set(gateway.keys)) == expected_attempts
    assert excinfo.value.facts["run_ids"] == [
        f"run-{attempt}" for attempt in range(1, expected_attempts + 1)
    ]
    attempts = excinfo.value.facts["attempts"]
    assert [item["run_id"] for item in attempts] == excinfo.value.facts["run_ids"]
    assert [item["outcome"] for item in attempts] == (
        ["no_mutation"] if lane == "fixture"
        else ["retryable_no_mutation", "no_mutation"]
    )
    assert gateway.removed == ["project-1"]
    assert gateway.closed == 1


def test_managed_smoke_stops_the_serving_identity_with_the_same_home(
    tmp_path, monkeypatch
):
    captured = {}
    monkeypatch.setenv("CLAUDEXOR_DAEMON_SOCK", "foreign.sock")
    monkeypatch.setenv("CLAUDEXOR_CONTROL_PORT", "9999")

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        return smoke.subprocess.CompletedProcess(
            command,
            0,
            stdout='{"stopped":true,"outcome":"exited","detail":"fixture"}\n',
            stderr="",
        )

    monkeypatch.setattr(smoke.subprocess, "run", fake_run)
    receipt = smoke.graceful_stop_managed_runtime(
        ["/exact/node", "/exact/claudexord.bundle.cjs"],
        tmp_path / "owned-home",
        "3.3.7",
        "a" * 40,
    )

    assert captured["command"][-3:] == ["--stop", "3.3.7", "a" * 40]
    assert captured["env"]["CLAUDEXOR_CONFIG_DIR"] == str(tmp_path / "owned-home")
    assert "CLAUDEXOR_DAEMON_SOCK" not in captured["env"]
    assert "CLAUDEXOR_CONTROL_PORT" not in captured["env"]
    assert receipt == {"stopped": True, "already_stopped": False, "outcome": "exited"}


def test_serving_build_identity_comes_from_the_frozen_handshake_shape():
    build_sha = "a" * 40
    body = {
        "protocolMajor": 3,
        "engine": {"version": "3.3.7", "sha": build_sha, "entry": "claudexord.bundle.cjs"},
    }

    assert smoke.handshake_engine_sha(body) == build_sha


# -- the honesty block ---------------------------------------------------------


def test_every_lane_states_that_subscriptions_are_not_covered():
    # The limit the owner's own open question is about must be present on BOTH lanes,
    # in the text a reader of a green check sees — not only in a source comment.
    for lane in ("fixture", "live"):
        text = "\n".join(smoke._limits_block(lane)).lower()
        assert "subscription" in text
        assert "not covered" in text or "cannot be authenticated" in text


def test_the_live_lane_names_the_one_field_that_differs_from_production():
    text = "\n".join(smoke._limits_block("live"))
    assert "authPreference" in text and "api_key" in text and "subscription" in text


def test_the_live_lane_discloses_the_bounded_no_edit_retry():
    text = "\n".join(smoke._limits_block("live")).lower()
    assert "one bounded second attempt" in text
    assert "second no-edit result is still a hard failure" in text
    assert "never retried" in text


def test_the_fixture_lane_says_no_model_and_no_child_process():
    text = "\n".join(smoke._limits_block("fixture")).lower()
    assert "no model" in text and "no child process was spawned" in text


def test_the_fixture_lane_discloses_the_dropped_delegated_marker():
    # The fixture request cannot carry `execution.delegated` (a fake maps no
    # `external_sandbox_full`), and that divergence must be readable on the green
    # check itself, not only in a source comment.
    text = "\n".join(smoke._limits_block("fixture"))
    assert "execution.delegated" in text and "external_sandbox_full" in text


def test_both_lanes_state_that_the_route_is_pinned():
    # `primaryHarness` alone cannot start a lane the auto-pool excludes; the summary
    # must say the route was pinned so nobody reads the check as auto-pool coverage.
    for lane in ("fixture", "live"):
        text = "\n".join(smoke._limits_block(lane))
        assert "harnesses" in text and "primaryHarness" in text


def test_the_summary_reaches_the_step_summary_file(tmp_path, monkeypatch):
    target = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(target))
    smoke.emit_summary("live", {"engine_version": "9.9.9"}, "PASSED", "ok")
    written = target.read_text(encoding="utf-8")
    assert "9.9.9" in written and "subscription" in written.lower()


def test_three_os_gate_installs_the_reviewed_runtime_instead_of_floating_npm():
    workflow = (
        pathlib.Path(__file__).resolve().parents[1]
        / ".github" / "workflows" / "claudexor-platform-gate.yml"
    ).read_text(encoding="utf-8")
    assert "os: [ubuntu-latest, windows-latest, macos-latest]" in workflow
    assert workflow.count("--managed-runtime") == 2
    assert "node-version: '24.16.0'" in workflow
    assert workflow.count("Stage the host-owned bundled Node layout") == 1
    assert "claudexor@next" not in workflow
    assert "CLAUDEXOR_NPM_SPEC" not in workflow
    assert "npm install -g claudexor" not in workflow
    assert "CLAUDEXOR_CLAUDE_BIN" in workflow
    assert "CLAUDEXOR_CODEX_BIN" in workflow
    assert "cygpath -w" in workflow
    assert "gpt-5.4-mini" not in workflow
    assert "model: gpt-5.4" in workflow
    assert "effort: medium" in workflow
    assert '--effort "${{ matrix.effort }}"' in workflow
    assert "- os: ubuntu-latest\n            harness: codex" in workflow
    assert "- os: windows-latest\n            harness: codex" in workflow
    for dependency_surface in (
        ".github/actions/setup-python-env/**",
        "pyproject.toml",
        "uv.lock",
        "requirements-runtime.lock",
    ):
        assert workflow.count(f"- '{dependency_surface}'") == 2


# -- the request shape (needs the seam) ----------------------------------------


def test_the_request_uses_the_seams_own_authority_shape(tmp_path):
    pytest.importorskip(
        "ouroboros.subagents",
        reason="the delegated-transport seam is not in this tree",
    )
    subagents = pytest.importorskip("ouroboros.subagents")
    if not hasattr(subagents, "delegated_run_shape"):
        pytest.skip("this tree predates delegated_run_shape")

    live = smoke.build_request("live", "claude", "m", "low", tmp_path, "p", 300)
    fixture = smoke.build_request("fixture", "fake-implement", "", "", tmp_path, "p", 300)

    shape = subagents.delegated_run_shape(acting=True)
    for request in (live, fixture):
        assert request["mode"] == shape.mode
        assert request["access"] == shape.access
        assert request["scope"] == {"kind": "project", "root": str(tmp_path)}
    # The delegated marker rides the live lane only: a fake maps no
    # `external_sandbox_full`, so the engine refuses the delegated mutating shape
    # on a pinned fake lane wherever the host has a kernel boundary.
    assert live["execution"] == {
        "isolation": shape.isolation,
        "delegated": shape.delegated,
        "workspaceRoot": str(tmp_path),
    }
    assert fixture["execution"] == {"isolation": shape.isolation, "delegated": False}
    # Both lanes pin the pool: `primaryHarness` alone only reorders the auto-pool,
    # and the auto-pool excludes `fake-*` routes entirely.
    assert live["harnesses"] == ["claude"] and live["primaryHarness"] == "claude"
    assert fixture["harnesses"] == ["fake-implement"]
    assert fixture["primaryHarness"] == "fake-implement"
    # The one deliberate auth divergence from production, and its absence where it
    # would be a fabricated claim.
    assert live["authPreference"] == "api_key"
    assert "authPreference" not in fixture
    assert "model" not in fixture and "effort" not in fixture


# -- the request shape (stub seam, runs on every tree) ---------------------------
#
# The assertions above only run on a tree that carries the delegated-transport
# seam. The pool-pinning and delegated-marker rules are THIS repository's own
# logic, so they are also pinned here against a stub authority shape — otherwise a
# regression on a seamless tree would only surface in CI on the integration branch.


class _StubShape:
    access = "workspace_write"
    mode = "agent"
    isolation = "live"
    delegated = True


def _stub_seam(monkeypatch):
    import sys as _sys
    import types as _types

    stub = _types.ModuleType("ouroboros.subagents")
    stub.delegated_run_shape = lambda acting: _StubShape()
    monkeypatch.setitem(_sys.modules, "ouroboros.subagents", stub)


def test_the_fixture_request_pins_the_fake_pool_and_drops_the_delegated_marker(
        tmp_path, monkeypatch):
    _stub_seam(monkeypatch)
    request = smoke.build_request("fixture", "fake-implement", "", "", tmp_path, "p", 300)
    assert request["harnesses"] == ["fake-implement"]
    assert request["primaryHarness"] == "fake-implement"
    assert request["execution"] == {"isolation": "live", "delegated": False}


def test_the_live_request_pins_the_pool_and_keeps_the_delegated_marker(
        tmp_path, monkeypatch):
    _stub_seam(monkeypatch)
    request = smoke.build_request("live", "claude", "m", "low", tmp_path, "p", 300)
    assert request["harnesses"] == ["claude"]
    assert request["primaryHarness"] == "claude"
    assert request["execution"] == {
        "isolation": "live",
        "delegated": True,
        "workspaceRoot": str(tmp_path),
    }
    assert request["authPreference"] == "api_key"


# -- the primary-artifact EOF read -----------------------------------------------


class _StubGateway:
    def __init__(self, body: bytes = b"", error: Exception | None = None):
        self._body = body
        self._error = error
        self.requested: list = []

    def get_run_artifact(self, run_id: str, path: str) -> bytes:
        self.requested.append((run_id, path))
        if self._error is not None:
            raise self._error
        return self._body


def test_the_full_artifact_body_matching_the_reported_size_is_the_result():
    body = b"Implemented by the fake harness.\n"
    gateway = _StubGateway(body)
    detail = {"primaryOutput": {"kind": "answer", "path": "final/answer.md",
                                "text": body.decode(), "bytes": len(body),
                                "truncated": False}}
    facts = smoke.read_primary_artifact_to_eof(gateway, "run-1", detail)
    assert gateway.requested == [("run-1", "final/answer.md")]
    assert facts["read_bytes"] == len(body) and facts["verified"] == "size"


def test_a_redaction_length_change_is_verified_by_the_preview_prefix():
    # The artifact route serves text through redactSecrets, so the served length
    # may legally differ from the reported on-disk bytes; the preview being a
    # prefix of the body (up to the engine's own redaction-overlap slack at the
    # boundary) is then the verification the contract offers. The preview must be
    # longer than the slack window for a prefix to remain — same as production.
    preview = "stable preview line\n" * 200  # 4000 chars > _PREVIEW_PREFIX_SLACK
    body = (preview + "a redacted tail that changed length").encode()
    detail = {"primaryOutput": {"kind": "answer", "path": "final/answer.md",
                                "text": preview, "bytes": 7,
                                "truncated": True}}
    facts = smoke.read_primary_artifact_to_eof(_StubGateway(body), "run-1", detail)
    assert facts["verified"] == "preview_prefix"


def test_a_body_tied_to_neither_size_nor_preview_is_refused():
    detail = {"primaryOutput": {"kind": "answer", "path": "final/answer.md",
                                "text": "the run claimed this text", "bytes": 999,
                                "truncated": False}}
    with pytest.raises(smoke.SmokeFailure) as excinfo:
        smoke.read_primary_artifact_to_eof(_StubGateway(b"something else"), "run-1", detail)
    assert excinfo.value.code == "primary_artifact_unverified"


def test_a_terminal_run_with_no_primary_artifact_is_refused():
    with pytest.raises(smoke.SmokeFailure) as excinfo:
        smoke.read_primary_artifact_to_eof(_StubGateway(), "run-1", {"primaryOutput": None})
    assert excinfo.value.code == "primary_artifact_missing"


def test_an_unreadable_artifact_is_refused_with_a_named_reason():
    gateway = _StubGateway(error=RuntimeError("boom"))
    detail = {"primaryOutput": {"kind": "answer", "path": "final/answer.md",
                                "text": "x", "bytes": 1, "truncated": False}}
    with pytest.raises(smoke.SmokeFailure) as excinfo:
        smoke.read_primary_artifact_to_eof(gateway, "run-1", detail)
    assert excinfo.value.code == "primary_artifact_unread"
