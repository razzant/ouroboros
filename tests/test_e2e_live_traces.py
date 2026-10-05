"""The live stand's per-lane trace bundle (devtools/e2e_live/traces.py).

A lane's data root is lost with the CI runner (issue #1501: two SM1 deadline failures nobody could read), so the
stand copies the lane server's journals into ``lanes/<id>/traces/`` for the run artifact. The artifact of a public
repository is downloadable by any signed-in user: every credential value is replaced by a fingerprint marker, a
bundle in which one survives is withheld with a typed fact, and ``settings.json`` is never part of it.
"""
from __future__ import annotations

import dataclasses
import json
import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from devtools.benchmarks.common.manifests import repo_provenance  # noqa: E402
from devtools.benchmarks.common.secrets import credential_fingerprint  # noqa: E402
from devtools.e2e_live import run_live_lanes, scenarios, traces  # noqa: E402

FAKE_KEY = "sk-or-v1-e2e-live-traces-test-key-never-printed-0123456789"
STUB_KEY = "stub-key-not-a-credential"
FORK = "state/headless_tasks/t1/data"
TRACED = ["logs/events.jsonl", "logs/server.log", "logs/server.log.1", "logs/tools.jsonl", "task_results/t1.json",
          "state/advisory_review.json", "state/usage_attempts.jsonl", "state/queue_snapshot.json",
          "observability/calls/t1/llm_1.json", f"{FORK}/logs/events.jsonl", f"{FORK}/state/advisory_review.json",
          f"{FORK}/task_results/t1.json"]
NOT_TRACED = ["settings.json", f"{FORK}/settings.json", "memory/identity.md", "observability/blobs/abc.prompt.gz",
              "state/auth_secret.key", "state/state.json", "logs/nested/old.jsonl"]


def _write(path: pathlib.Path, text: str) -> None:
    """LF bytes on every platform: write_text would turn each "\\n" into "\\r\\n" on Windows, and the
    tests compare exact bytes and sizes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))


def _data_root(tmp_path: pathlib.Path, *, planted: str = "") -> pathlib.Path:
    """A lane data root shaped like a real one; ``planted`` is written into two journals and the settings."""
    root = tmp_path / "lane" / "data"
    for rel in TRACED + NOT_TRACED:
        _write(root / rel, json.dumps({"file": rel, "note": f"call with {planted}" if planted else "ok"}) + "\n")
    _write(root / "settings.json", json.dumps({
        "OPENROUTER_API_KEY": planted, "OPENAI_COMPATIBLE_API_KEY": STUB_KEY,
        "OPENAI_COMPATIBLE_BASE_URL": "http://127.0.0.1:4242/v1", "OUROBOROS_MODEL_MAX_TOKENS": "16384"}))
    _write(root / "logs" / "server.log", f"listening on http://127.0.0.1:4242/v1 max 16384 auth {STUB_KEY}\n")
    return root


def _bundle(lane: pathlib.Path) -> dict[str, bytes]:
    base = lane / traces.TRACES_DIR
    return {p.relative_to(base).as_posix(): p.read_bytes() for p in sorted(base.rglob("*")) if p.is_file()}


def test_the_bundle_carries_the_journals_of_the_lane_and_its_forks_and_never_settings(tmp_path):
    root = _data_root(tmp_path)
    fact = traces.publish_lane_traces(root.parent, root, {})
    bundle = _bundle(root.parent)
    assert sorted(bundle) == sorted(TRACED), sorted(bundle)
    for rel in NOT_TRACED:
        assert rel not in bundle, rel
    assert not any(rel.endswith("settings.json") for rel in bundle)
    # Byte-exact copies when nothing is secret and nothing is over the bound.
    assert all(bundle[rel] == (root / rel).read_bytes() for rel in TRACED)
    assert fact == {"published": True, "files": len(TRACED), "bytes": sum(len(b) for b in bundle.values()),
                    "redacted": 0, "limit_bytes": traces.BUNDLE_LIMIT_BYTES, "truncated": []}
    # A lane that never got a data root records that, typed.
    assert traces.publish_lane_traces(tmp_path / "x", tmp_path / "x" / "data", {}) == {
        "published": False, "reason": "no_data_root"}


def test_every_credential_value_is_replaced_by_its_fingerprint_and_nothing_else_is(tmp_path):
    root = _data_root(tmp_path, planted=FAKE_KEY)
    secrets = traces.lane_secrets(root / "settings.json", {"OUROBOROS_E2E_LIVE_OPENROUTER_KEY": FAKE_KEY})
    # Secret-shaped settings keys and the named env value; never a URL, a number or a missing file's keys.
    assert secrets == {FAKE_KEY: "OUROBOROS_E2E_LIVE_OPENROUTER_KEY", STUB_KEY: "OPENAI_COMPATIBLE_API_KEY"}
    assert traces.lane_secrets(tmp_path / "absent.json", {"K": ""}) == {}
    fact = traces.publish_lane_traces(root.parent, root, secrets)
    bundle = _bundle(root.parent)
    assert fact["published"] is True and fact["files"] == len(TRACED)
    for rel, data in bundle.items():
        assert FAKE_KEY.encode() not in data and STUB_KEY.encode() not in data, rel
    marker = f"<redacted:OUROBOROS_E2E_LIVE_OPENROUTER_KEY {credential_fingerprint(FAKE_KEY)}>"
    assert marker in bundle["logs/events.jsonl"].decode() and marker in bundle[f"{FORK}/logs/events.jsonl"].decode()
    server_log = bundle["logs/server.log"].decode()
    assert f"<redacted:OPENAI_COMPATIBLE_API_KEY {credential_fingerprint(STUB_KEY)}>" in server_log
    assert "http://127.0.0.1:4242/v1" in server_log and "16384" in server_log   # non-secret values stay
    assert fact["redacted"] == len(TRACED)   # the key once in every file but server.log, which carries the stub key


def test_a_value_that_survives_redaction_withholds_the_whole_bundle(tmp_path, monkeypatch):
    root = _data_root(tmp_path, planted=FAKE_KEY)
    secrets = {FAKE_KEY: "OPENROUTER_API_KEY"}
    assert traces.publish_lane_traces(root.parent, root, secrets)["published"] is True   # the guard is quiet here
    monkeypatch.setattr(traces, "_redact", lambda data, replacements: (data, 0))       # a redaction that misses
    fact = traces.publish_lane_traces(root.parent, root, secrets)
    assert fact["published"] is False and fact["reason"] == "secret_residue"
    assert "logs/events.jsonl" in fact["files_with_residue"] and "logs/server.log" not in fact["files_with_residue"]
    assert not (root.parent / traces.TRACES_DIR).exists()
    assert FAKE_KEY not in json.dumps(fact)


def test_an_oversized_bundle_keeps_the_newest_tail_of_each_journal_and_says_so(tmp_path):
    root = tmp_path / "lane" / "data"
    big = "".join(json.dumps({"seq": i, "pad": "x" * 40}) + "\n" for i in range(400))
    _write(root / "logs" / "events.jsonl", big)
    _write(root / "logs" / "chat.jsonl", '{"small": true}\n')
    _write(root / "task_results" / "t1.json", json.dumps({"status": "failed", "pad": "y" * 300}))
    limit = 4000
    fact = traces.publish_lane_traces(root.parent, root, {}, limit=limit)
    bundle = _bundle(root.parent)
    assert fact["published"] is True and fact["limit_bytes"] == limit
    assert bundle["logs/chat.jsonl"] == b'{"small": true}\n'                          # under its share: whole
    assert bundle["task_results/t1.json"] == (root / "task_results" / "t1.json").read_bytes()   # JSON never cut
    lines = bundle["logs/events.jsonl"].decode().splitlines()
    head = json.loads(lines[0])["trace_truncated"]
    assert head["path"] == "logs/events.jsonl" and head["original_bytes"] == len(big) and head["limit_bytes"] == limit
    assert lines[-1] == big.splitlines()[-1] and json.loads(lines[1])["seq"] > 0       # the newest lines, whole
    assert fact["truncated"] == [{"path": "logs/events.jsonl", "original_bytes": len(big),
                                  "kept_tail_bytes": head["kept_tail_bytes"]}]
    assert sum(len(b) for b in bundle.values()) <= limit + len(lines[0]) + 1


def test_a_rotated_server_log_backup_travels_and_is_cut_like_a_journal(tmp_path):
    """server.py rotates server.log at 2 MiB into server.log.1..3: the older history of a long lane lives there.
    Over the bound a backup keeps its newest tail like any journal; a JSON file stays whole."""
    root = tmp_path / "lane" / "data"
    lines = "".join(f"line {i:04d} " + "x" * 40 + "\n" for i in range(200))
    _write(root / "logs" / "server.log", lines)
    _write(root / "logs" / "server.log.1", lines)
    _write(root / "task_results" / "t1.json", json.dumps({"pad": "y" * 3000}))
    fact = traces.publish_lane_traces(root.parent, root, {}, limit=8000)
    bundle = _bundle(root.parent)
    assert sorted(bundle) == ["logs/server.log", "logs/server.log.1", "task_results/t1.json"], sorted(bundle)
    assert sorted(cut["path"] for cut in fact["truncated"]) == ["logs/server.log", "logs/server.log.1"], fact
    assert bundle["logs/server.log.1"].decode().splitlines()[-1] == lines.splitlines()[-1]
    assert bundle["task_results/t1.json"] == (root / "task_results" / "t1.json").read_bytes()


class _NoopServer:
    base_url, attestation = "http://127.0.0.1:1", {}
    __init__ = start = stop = lambda self, *a, **k: None


PRIOR_ROW_FIELDS = ["schema", "scenario", "attempt", "title", "status", "stub", "profile", "self_mod",
                    "preflight_test_workers", "started_at", "checks", "facts", "error", "screenshots", "ui", "budget",
                    "self_mod_absorb", "attestation", "digests", "model_slots", "grants", "runtime_outcome",
                    "reason_code", "no_orphans_after_stop", "orphan_scan", "ended_at", "duration_sec"]


def test_a_lane_row_records_the_traces_fact_beside_every_prior_field(tmp_path, monkeypatch):
    """The real ``run_attempt`` path with a no-op server: the scenario writes a journal line carrying the key; the
    stored ``result.json`` gains ``traces`` and keeps every field it had, and the bundle holds no key and no
    settings file. A budget-refused attempt never started a lane: its row says so in the same field."""
    monkeypatch.setattr(run_live_lanes.tempfile, "gettempdir", lambda: "/tmp/short")
    monkeypatch.setattr(run_live_lanes, "IsolatedServer", _NoopServer)
    monkeypatch.setattr(run_live_lanes, "PROCFS_AVAILABLE", False)
    monkeypatch.setattr(run_live_lanes, "resolve_ui_client", lambda base_url: (None, "ui_unavailable:test"))

    def acceptance(ctx):
        _write(ctx.data_root / "logs" / "events.jsonl", json.dumps({"type": "llm_round", "auth": FAKE_KEY}) + "\n")
        ctx.check("scenario_ok", True)

    monkeypatch.setitem(run_live_lanes.SCENARIOS, "SK1", dataclasses.replace(scenarios.SCENARIOS["SK1"], acceptance=acceptance))
    seed = tmp_path / "source"
    seed.mkdir()
    (seed / "VERSION").write_text("7.0.0-test\n", encoding="utf-8")
    for cmd in (["init", "-q"], ["add", "-A"], ["-c", "user.name=t", "-c", "user.email=t@e.invalid", "commit", "-q", "-m", "seed"]):
        subprocess.run(["git", *cmd], cwd=str(seed), check=True)
    out = tmp_path / "out"
    args = run_live_lanes.parse_args(["--scenarios", "SK1", "--out", str(out), "--watch-interval", "600"])
    budget = run_live_lanes.RunBudget(100.0, 8.0, reader=lambda root: (0.0, 0))
    row = run_live_lanes.run_attempt(("SK1", 1), args, out, run_live_lanes.effective_settings(args, FAKE_KEY),
                                     run_live_lanes.Stagger(0.0), {}, seed, budget, dispatch_index=0, key=FAKE_KEY,
                                     seed_sha=repo_provenance(seed)["head"])
    lane = out / "lanes" / "SK1_a1"
    stored = json.loads((lane / "result.json").read_text(encoding="utf-8"))
    assert row["status"] == "pass" and stored["status"] == "pass", stored
    assert sorted(stored) == sorted([*PRIOR_ROW_FIELDS, "traces"]), sorted(stored)
    assert stored["traces"]["published"] is True and stored["traces"]["redacted"] == 1, stored["traces"]
    bundle = _bundle(lane)
    assert "logs/events.jsonl" in bundle and not any(rel.endswith("settings.json") for rel in bundle)
    assert all(FAKE_KEY.encode() not in data for data in bundle.values())
    assert FAKE_KEY in (lane / "data" / "settings.json").read_text(encoding="utf-8")   # the lane file keeps it
    index = [json.loads(line) for line in (out / "result_index.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(index) == 1 and FAKE_KEY not in json.dumps(index)

    refused = run_live_lanes.run_attempt(("SK1", 2), args, out, {}, run_live_lanes.Stagger(0.0), {}, seed,
                                         run_live_lanes.RunBudget(1.0, 8.0, reader=lambda root: (0.0, 0)),
                                         dispatch_index=1)
    assert refused["status"] == "not_run" and refused["traces"] == {"published": False, "reason": "lane_not_started"}
