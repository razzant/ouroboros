"""The live stand's per-lane trace bundle (devtools/e2e_live/traces.py).

A lane's data root is lost with the CI runner (issue #1501: two SM1 deadline failures nobody could read), so the
stand copies the lane server's journals into ``lanes/<id>/traces/`` for the run artifact. The artifact of a public
repository is downloadable by any signed-in user: every credential value is replaced by a fingerprint marker, a
bundle in which one survives is withheld with a typed fact, and ``settings.json`` is never part of it.
"""
from __future__ import annotations

import base64
import dataclasses
import json
import pathlib
import shutil
import sqlite3
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
TRACED = ["logs/events.jsonl", "logs/server.log", "logs/server.log.1", "logs/tools.jsonl", "logs/supervisor.jsonl",
          "state/advisory_review.json", "state/usage_attempts.jsonl", "state/queue_snapshot.json",
          "state/review_ledger/rl-1.json", "state/private_copies/body_t1.json", "state/preflight_receipts/k1.json",
          "state/body_adoption/handoff.json", "state/pending_restart_verify.json",
          "observability/calls/t1/llm_1.json", f"{FORK}/logs/events.jsonl", f"{FORK}/state/advisory_review.json",
          f"{FORK}/state/review_ledger/rl-2.json"]
NOT_TRACED = ["settings.json", f"{FORK}/settings.json", "memory/identity.md", "observability/blobs/abc.prompt.gz",
              "state/auth_secret.key", "state/state.json", "logs/nested/old.jsonl", "state/body_adoption/switch.py",
              "state/current.sqlite-wal", "state/dialog.sqlite-shm"]


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
                    "redacted": 0, "limit_bytes": traces.BUNDLE_LIMIT_BYTES, "truncated": [], "sqlite_exports": [], "gaps": []}
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
    _write(root / "state" / "advisory_review.json", json.dumps({"status": "failed", "pad": "y" * 300}))
    limit = 4000
    fact = traces.publish_lane_traces(root.parent, root, {}, limit=limit)
    bundle = _bundle(root.parent)
    assert fact["published"] is True and fact["limit_bytes"] == limit
    assert bundle["logs/chat.jsonl"] == b'{"small": true}\n'                          # under its share: whole
    assert bundle["state/advisory_review.json"] == (root / "state" / "advisory_review.json").read_bytes()   # JSON is never cut
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
    _write(root / "state" / "advisory_review.json", json.dumps({"pad": "y" * 3000}))
    fact = traces.publish_lane_traces(root.parent, root, {}, limit=8000)
    bundle = _bundle(root.parent)
    assert sorted(bundle) == ["logs/server.log", "logs/server.log.1", "state/advisory_review.json"], sorted(bundle)
    assert sorted(cut["path"] for cut in fact["truncated"]) == ["logs/server.log", "logs/server.log.1"], fact
    assert bundle["logs/server.log.1"].decode().splitlines()[-1] == lines.splitlines()[-1]
    assert bundle["state/advisory_review.json"] == (root / "state" / "advisory_review.json").read_bytes()


SECRET = "sk-or-v1-e2e-live-sqlite-secret-never-exported-0123456789"
MARKER = f"<redacted:OPENROUTER_API_KEY {credential_fingerprint(SECRET)}>".encode()


def _wal_store(path: pathlib.Path, scratch: pathlib.Path) -> bytes:
    """A real store at ``path`` whose committed rows live only in its WAL, as a stopped server can leave one: a TEXT and
    a BLOB value carrying the secret, a REAL, a NULL, and a virtual table with its shadow pages. Returns the raw BLOB."""
    work = scratch / f"build-{path.name}"
    work.mkdir(parents=True)
    conn = sqlite3.connect(work / path.name)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA wal_autocheckpoint=0")
    conn.execute("CREATE TABLE tasks (id TEXT PRIMARY KEY, body TEXT, raw BLOB, score REAL, absent TEXT) WITHOUT ROWID")
    conn.execute("CREATE VIRTUAL TABLE notes USING fts5(body)")
    blob = b"\x00\xfe head " + SECRET.encode() + b" tail \xff"
    conn.execute("INSERT INTO tasks VALUES (?, ?, ?, ?, ?)", ("t1", f"call with {SECRET}", blob, 1.5, None))
    conn.execute("INSERT INTO notes VALUES (?)", (f"note carrying {SECRET}",))
    conn.commit()
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "-wal"):     # copied while the writer holds it open: its close would checkpoint them away
        shutil.copyfile(work / (path.name + suffix), path.with_name(path.name + suffix))
    conn.close()
    return blob


def _export_lines(data: bytes) -> tuple[dict, list[dict]]:
    lines = [json.loads(line) for line in data.decode("ascii").splitlines()]
    return lines[0]["sqlite_logical_export"], lines[1:]


def test_a_store_travels_as_redacted_rows_with_its_wal_and_never_as_pages(tmp_path):
    """The committed rows live only in the WAL, which the raw copy this replaces never carried. The bundle holds typed
    rows with the secret gone from the TEXT and from inside the BLOB (scanned before base64 could hide it), the virtual
    table through its own SELECT and none of its shadow pages; the source is left as it was found."""
    root = tmp_path / "lane" / "data"
    source = root / "state" / "current.sqlite"
    blob = _wal_store(source, tmp_path / "scratch")
    _write(root / "logs" / "events.jsonl", '{"ok": true}\n')
    assert (source.with_name("current.sqlite-wal")).stat().st_size > 0
    found = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in source.parent.iterdir()}
    fact = traces.publish_lane_traces(root.parent, root, {SECRET: "OPENROUTER_API_KEY"})
    bundle = _bundle(root.parent)

    assert {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in source.parent.iterdir()} == found, "no -shm, no checkpoint"
    assert fact["published"] is True and fact["gaps"] == []
    assert sorted(bundle) == ["logs/events.jsonl", "state/current.sqlite.logical.jsonl"], "never the database or WAL files"
    data = bundle["state/current.sqlite.logical.jsonl"]
    assert SECRET.encode() not in data and base64.b64encode(SECRET.encode()) not in data
    header, rows = _export_lines(data)
    assert header["source"] == "state/current.sqlite" and header["quick_check"] == "ok" and header["wal_bytes"] > 0
    assert "not atomic" in header["consistency"] and SECRET not in json.dumps(header)
    assert sorted((t["name"], t["kind"], t["rows"]) for t in header["tables"]) == [("notes", "virtual", 1), ("tasks", "table", 1)]
    declared = next(t for t in header["tables"] if t["name"] == "tasks")
    assert {c["name"]: c["type"] for c in declared["columns"]}["raw"] == "BLOB" and "WITHOUT ROWID" in declared["sql"]
    assert {row["table"] for row in rows} == {"tasks", "notes"}, "no sqlite_* internal and no notes_data/_idx/_content shadow"
    task = next(row["row"] for row in rows if row["table"] == "tasks")
    assert task["id"] == "t1" and task["score"] == 1.5 and task["absent"] is None
    assert task["body"] == f"call with {MARKER.decode()}"
    assert task["raw"]["$type"] == "blob"
    assert base64.b64decode(task["raw"]["base64"]) == blob.replace(SECRET.encode(), MARKER) and task["raw"]["bytes"] == len(blob.replace(SECRET.encode(), MARKER))
    assert next(row["row"] for row in rows if row["table"] == "notes") == {"body": f"note carrying {MARKER.decode()}"}
    assert fact["sqlite_exports"] == [{"path": "state/current.sqlite", "file": "state/current.sqlite.logical.jsonl",
                                       "tables": 2, "rows": 2, "bytes": len(data), "redacted": 3}]


def test_a_bad_store_is_a_named_gap_and_its_neighbours_stay(tmp_path):
    """Garbage and a damaged real database each withhold only their own export, with reason and counts; the good store
    of the same root, a fork's store and the journals are published, and no partial export file appears."""
    root = tmp_path / "lane" / "data"
    _wal_store(root / "state" / "current.sqlite", tmp_path / "scratch")
    _wal_store(root / FORK / "state" / "current.sqlite", tmp_path / "scratch-fork")
    _write(root / "state" / "usage.sqlite", "this is not a database, it merely has the name\n")
    damaged = root / "state" / "dialog.sqlite"
    conn = sqlite3.connect(damaged)
    conn.execute("CREATE TABLE rows_ (id INTEGER PRIMARY KEY, body TEXT)")
    conn.executemany("INSERT INTO rows_ (body) VALUES (?)", [("x" * 900,) for _ in range(200)])
    conn.commit()
    conn.close()
    pages = bytearray(damaged.read_bytes())
    pages[4096 * 3:4096 * 12] = b"\xee" * (4096 * 9)
    damaged.write_bytes(bytes(pages))
    _write(root / "logs" / "events.jsonl", '{"ok": true}\n')
    fact = traces.publish_lane_traces(root.parent, root, {SECRET: "OPENROUTER_API_KEY"})
    bundle = _bundle(root.parent)

    assert fact["published"] is True and "logs/events.jsonl" in bundle
    assert sorted(export["path"] for export in fact["sqlite_exports"]) == ["state/current.sqlite", f"{FORK}/state/current.sqlite"]
    gaps = {gap["path"]: gap for gap in fact["gaps"]}
    assert sorted(gaps) == ["state/dialog.sqlite", "state/usage.sqlite"]
    assert gaps["state/usage.sqlite"]["reason"] == "unreadable:DatabaseError"
    assert gaps["state/dialog.sqlite"]["reason"] == "quick_check_failed" or gaps["state/dialog.sqlite"]["reason"].startswith("unreadable:")
    assert all({"tables_total", "tables_exported", "rows_total", "rows_exported"} <= set(gap) for gap in gaps.values())
    assert not any("usage.sqlite" in rel or "dialog.sqlite" in rel for rel in bundle), "a withheld store leaves no file"
    assert SECRET not in json.dumps(fact)


def test_a_secret_surviving_in_a_finished_export_withholds_only_that_export(tmp_path):
    """The final scan is the last word on a finished file: with the redaction missing, the export never reaches the
    bundle's list and its gap counts what had been exported."""
    root = tmp_path / "data"
    _wal_store(root / "state" / "current.sqlite", tmp_path / "scratch")
    scratch = tmp_path / "work"
    scratch.mkdir()
    exports, gaps = traces._logical_exports(root, scratch, [], {SECRET: "OPENROUTER_API_KEY"})
    assert exports == [] and [(g["path"], g["reason"], g["rows_exported"], g["rows_total"]) for g in gaps] == [
        ("state/current.sqlite", "secret_residue", 2, 2)]


UNICODE_SECRET = "ключ-sk-проба-e2e-live-метаданные-0123456789"


def test_a_secret_in_a_table_column_or_type_name_is_redacted_before_json_escapes_it(tmp_path):
    """``ensure_ascii`` writes a non-ASCII name as ``\\uXXXX``, which no byte scan after it can match: a name is redacted
    where a value is, before encoding. Decoded, the export holds the secret nowhere (the raw scans alone missed it)."""
    root = tmp_path / "lane" / "data"
    store = root / "state" / "dialog.sqlite"
    store.parent.mkdir(parents=True)
    conn = sqlite3.connect(store)
    named = f'"t {UNICODE_SECRET}"'
    conn.execute(f'CREATE TABLE {named} ("c {UNICODE_SECRET}" "TEXT {UNICODE_SECRET}", plain TEXT)')
    conn.execute(f"INSERT INTO {named} VALUES (?, ?)", (f"v {UNICODE_SECRET}", "ok"))
    conn.commit()
    conn.close()
    fact = traces.publish_lane_traces(root.parent, root, {UNICODE_SECRET: "OUROBOROS_TEST_SECRET"})
    header, rows = _export_lines(_bundle(root.parent)["state/dialog.sqlite.logical.jsonl"])

    assert UNICODE_SECRET not in json.dumps([header, rows], ensure_ascii=False), "decoded, the export must not carry it"
    marker = f"<redacted:OUROBOROS_TEST_SECRET {credential_fingerprint(UNICODE_SECRET)}>"
    table = header["tables"][0]
    assert table["name"] == f"t {marker}" and table["columns"][0] == {"name": f"c {marker}", "type": f"TEXT {marker}"}
    assert rows == [{"table": f"t {marker}", "row": {f"c {marker}": f"v {marker}", "plain": "ok"}}]
    assert fact["published"] is True and fact["gaps"] == []


def test_a_store_copy_that_keeps_changing_is_refused_not_torn(tmp_path, monkeypatch):
    """A writer still running (a stop that could not be proved) must not yield a main file and WAL from different moments."""
    root = tmp_path / "data"
    _wal_store(root / "state" / "current.sqlite", tmp_path / "scratch")
    ticks = iter(range(1000))
    monkeypatch.setattr(traces, "_stamp", lambda path: next(ticks))
    scratch = tmp_path / "work"
    scratch.mkdir()
    exports, gaps = traces._logical_exports(root, scratch, [], {})
    assert exports == [] and gaps[0]["reason"] == "source_changing" and gaps[0]["rows_exported"] == 0


def test_an_export_is_never_cut_and_counts_against_the_bound(tmp_path):
    root = tmp_path / "lane" / "data"
    _wal_store(root / "state" / "current.sqlite", tmp_path / "scratch")
    lines = "".join(json.dumps({"seq": i, "pad": "x" * 40}) + "\n" for i in range(400))
    _write(root / "logs" / "events.jsonl", lines)
    whole = traces.publish_lane_traces(root.parent, root, {})
    exported = whole["sqlite_exports"][0]["bytes"]
    cut = traces.publish_lane_traces(root.parent, root, {}, limit=exported + 2000)
    bundle = _bundle(root.parent)
    assert [c["path"] for c in cut["truncated"]] == ["logs/events.jsonl"]
    assert len(bundle["state/current.sqlite.logical.jsonl"]) == exported
    assert json.loads(bundle["logs/events.jsonl"].decode().splitlines()[-1])["seq"] == 399


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
