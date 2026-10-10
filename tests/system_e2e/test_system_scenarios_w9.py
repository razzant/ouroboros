"""S36 — own-body candidates (#1539) on a real isolated server, keyless.

The unit suites (``tests/test_body_candidate.py`` and the adoption suites) drive the
dispatcher in-process against a miniature body. THIS scenario is the smallest real
consumer: a ``server.py`` started on a private checkout laid out like an installation
(the checkout is named ``repo``), a pooled worker running one scripted root task, the
real registry dispatch, and only durable artifacts read back. The checkout is the
byte-faithful copy of the tree under test (``candidate_checkout`` with origin proof),
so an uncommitted repair is what the server imports, and the server proves it.

WHAT S36 ASSERTS:
1. the root's first body write, spelled as the installation names it
   (``repo/notes/qualification.txt``), prepares and binds a candidate, and later writes
   in both serving spellings (``repo/…`` and the absolute serving path) reach the SAME
   candidate files — never ``<candidate>/repo/…``;
2. the serving clone is byte-identical (HEAD, porcelain status, no ``notes/``);
3. a refused resume is the typed ``CANDIDATE_MISSING`` in the durable tool log, not a
   generic tool error;
4. a process the root starts runs in the candidate with the candidate's isolated data
   root, never the server's.

Adoption (a restart that switches the serving tree) is S37's subject.

S37 — the restart-bound adoption on the same kind of real server. A direct (launcher-less)
server re-executes itself in place on POSIX, so ONE server process carries both
generations: generation A's root authors a note in its candidate, lands it through the
blocking review organ (stub verdicts) and calls ``request_restart(adopt_commit=…)``; the
exit tail arms the handoff only on its own stop evidence, the re-executed process's
package-init hook switches the serving checkout before its imports, and generation B
settles the adoption on boot and verifies the restart on the serving SHA. The serving
checkout is installation-shaped and CLEAN — the tree under test committed as the
``ouroboros`` line — because a restart resets a real install to that line.

S38 — the server dies between an evolution cycle's candidate commit and its receipts.
The supervisor mints the cycle, which commits in its candidate through the blocking
review organ; its next model round is held, the whole tree is SIGKILLed, and the two
receipts written after ``git commit`` (the transaction's SHA and the candidate row's
reviewed provenance) are removed — the durable state that crash leaves, shaped the way
S22 shapes its window. A fresh server boot on the same install and data root must
recover the exact commit from the candidate at its worker boot, with its reviewed
provenance, and must not absorb it: the serving checkout never held it.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import urllib.request

import pytest

from devtools.benchmarks.common.server_runner import seed_owner_state
from tests.candidate_checkout import SENTINEL_ROUTE, candidate_checkout
from tests.system_e2e.harness import (
    LANE_MOCK,
    REPO_ROOT,
    ArtifactOracle,
    ModelGate,
    ScriptedStubModel,
    classify_call,
    keyless_settings,
    repo_tree_fingerprint,
    require_lane,
    start_server,
    submit_running,
    wait_durable_result,
    wait_until,
)


def _s36_script(clone: pathlib.Path) -> list:
    return [
        {"tool": "write_file", "arguments": {"path": "repo/notes/qualification.txt", "content": "first\n"}},
        {"tool": "edit_text", "arguments": {"path": str(clone / "notes" / "qualification.txt"),
                                            "old_str": "first", "new_str": "second"}},
        {"tool": "write_file", "arguments": {"path": "repo/notes/later.txt", "content": "later\n"}},
        {"tool": "prepare_self_change", "arguments": {"resume": "no-such-candidate"}},
        {"tool": "run_command", "arguments": {"cmd": [
            sys.executable, "-c", "import os; print('CANDIDATE_DATA=' + os.environ['OUROBOROS_DATA_DIR'])"]}},
    ]


def _tool_rows(oracle: ArtifactOracle, name: str) -> list:
    return [row for row in oracle.tools_rows()
            if row.get("type") == "tool_call" and str(row.get("tool") or "") == name]


@pytest.mark.integration
@pytest.mark.serial
def test_s36_body_candidate_on_a_real_server_keeps_the_serving_clone_and_types_its_refusals(tmp_path_factory):
    require_lane(LANE_MOCK)
    root = tmp_path_factory.mktemp("s36")
    worktrees = root / "worktrees"
    with candidate_checkout(REPO_ROOT, root / "install" / "repo", origin_proof=True) as checkout, \
            ScriptedStubModel(_s36_script(checkout.path)) as stub:
        clone = checkout.path  # private: the server adds the candidate worktree to it
        settings = keyless_settings(stub, OUROBOROS_RUNTIME_MODE="pro",
                                    OUROBOROS_SUBAGENT_WORKTREE_ROOT=str(worktrees))
        server = start_server(checkout, root, settings)
        try:
            before = repo_tree_fingerprint(clone, ("server.py", "BIBLE.md"))
            task_id = submit_running(server, "Note the qualification in the body, then finish.")
            result = server.wait_task(task_id, timeout=300)
            assert result.get("status") == "completed", result
            oracle = ArtifactOracle(server.data_root)
            wait_durable_result(oracle, task_id)
            assert stub.script_consumed(), "S36 script was not fully consumed"
            task_log = oracle.task_drive(task_id)

            # 1. One candidate, every spelling of the body path inside it.
            registries = sorted(server.data_root.rglob("state/subagent_worktrees.json"))
            rows = [row for path in registries for row in json.loads(path.read_text())["worktrees"]
                    if row.get("kind") == "body_candidate"]
            assert len(rows) == 1, registries
            candidate = pathlib.Path(rows[0]["path"])
            assert candidate.resolve().is_relative_to(worktrees.resolve()), candidate
            assert (candidate / "notes" / "qualification.txt").read_text() == "second\n"
            assert (candidate / "notes" / "later.txt").read_text() == "later\n"
            assert not (candidate / "repo").exists()
            writes = _tool_rows(task_log, "write_file") + _tool_rows(task_log, "edit_text")
            assert [row.get("status") for row in writes] == ["ok"] * 3, writes

            # 2. The serving clone never saw the unfinished work.
            assert repo_tree_fingerprint(clone, ("server.py", "BIBLE.md")) == before
            assert not (clone / "notes").exists()

            # 3. The typed refusal reached the durable tool log.
            refused = _tool_rows(task_log, "prepare_self_change")
            assert [row.get("status") for row in refused] == ["blocked"], refused
            assert "CANDIDATE_MISSING" in json.dumps(refused) and "TOOL_ERROR" not in json.dumps(refused), refused

            # 4. A process inside the candidate saw the candidate's isolated data root.
            ran = json.dumps(_tool_rows(task_log, "run_command"), ensure_ascii=False)
            isolated = str(candidate.with_name(candidate.name + ".env") / "data")
            assert f"CANDIDATE_DATA={isolated}" in ran, ran
            assert str(server.data_root) not in ran.split("CANDIDATE_DATA=")[1].split("\\n")[0]
        finally:
            server.stop()


# ===========================================================================
# S37 — restart-bound adoption of a reviewed candidate commit, one real server.
# ===========================================================================

S37_DOC_PATH = "docs/notes/system_e2e_s37_adopted.md"
S37_DOC = "# system_e2e S37\n\nAdopted through a restart.\n"
S37_COMMIT_MESSAGE = "docs: system_e2e S37 adopted candidate note"
S37_REASON = "S37 adopt the reviewed candidate"


def _git(repo: pathlib.Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=str(repo), check=True, capture_output=True, text=True).stdout.strip()


def _candidate_rows(data_root: pathlib.Path) -> list:
    return [row for path in sorted(pathlib.Path(data_root).rglob("state/subagent_worktrees.json"))
            for row in json.loads(path.read_text(encoding="utf-8"))["worktrees"] if row.get("kind") == "body_candidate"]


def _adopt_the_reviewed_commit(data_root: pathlib.Path):
    """Dynamic step: the model names the exact commit its gate recorded on the candidate row."""
    def step(_body):
        rows = _candidate_rows(data_root)
        reviewed = rows[0].get("reviewed_commits") if len(rows) == 1 else []
        if not reviewed:
            return {"final": f"S37: no reviewed candidate commit to adopt ({rows})"}
        return {"tool": "request_restart", "arguments": {"reason": S37_REASON, "adopt_commit": reviewed[-1]}}
    return step


def _s37_script(data_root: pathlib.Path) -> list:
    return [
        {"tool": "write_file", "arguments": {"root": "system_repo", "path": S37_DOC_PATH, "content": S37_DOC}},
        {"tool": "commit_reviewed", "arguments": {
            "commit_message": S37_COMMIT_MESSAGE, "paths": [S37_DOC_PATH],
            "skip_advisory_review": True, "skip_tests": True,
            "goal": "Land the S37 note through the blocking review organ.", "scope": f"{S37_DOC_PATH} only."}},
        _adopt_the_reviewed_commit(data_root),
    ]


def _committed_install(source: pathlib.Path, destination: pathlib.Path) -> pathlib.Path:
    """The selected bytes as the clean ``ouroboros`` line of an installation-shaped checkout."""
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns(".git"), symlinks=True)
    for args in (("init", "-q", "-b", "ouroboros"), ("config", "user.name", "SystemHarness"),
                 ("config", "user.email", "system-harness@e2e.invalid"), ("add", "-A"),
                 ("commit", "-q", "--no-verify", "-m", "system_e2e S37: the tree under test")):
        _git(destination, *args)
    assert _git(destination, "status", "--porcelain") == ""
    return destination


def _jsonl(path: pathlib.Path, row_type: str) -> list:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines() if path.exists() else []:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get("type") == row_type:
            rows.append(row)
    return rows


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=10) as response:  # noqa: S310 - the scenario's loopback server
        return response.read()


@pytest.mark.integration
@pytest.mark.serial
@pytest.mark.skipif(os.name == "nt", reason="a direct server re-executes in place only on POSIX; Windows "
                                           "starts a replacement process this harness does not own")
def test_s37_restart_adopts_the_reviewed_candidate_commit_on_one_real_server(tmp_path_factory):
    require_lane(LANE_MOCK)
    root = tmp_path_factory.mktemp("s37")
    data_root = root / "data"
    with candidate_checkout(REPO_ROOT, root / "source" / "repo", origin_proof=True) as source, \
            ScriptedStubModel(_s37_script(data_root)) as stub:
        install = _committed_install(source.path, root / "install" / "repo")
        old_head = _git(install, "rev-parse", "HEAD")
        settings = keyless_settings(stub, OUROBOROS_RUNTIME_MODE="advanced", OUROBOROS_REVIEW_ENFORCEMENT="blocking")
        server = start_server(install, root, settings)
        try:
            # Generation A imported the tree under test: its static sentinel and its VERSION.
            assert _fetch(server.base_url + SENTINEL_ROUTE) == source.sentinel_bytes
            assert json.loads(_fetch(server.base_url + "/api/health"))["runtime_version"] == source.version_text
            pid = server.proc.pid
            task_id = submit_running(server, "Write the S37 note, land it reviewed, then adopt it with a restart.")
            supervisor_log, events_log = data_root / "logs" / "supervisor.jsonl", data_root / "logs" / "events.jsonl"
            closed = wait_until(lambda: _jsonl(supervisor_log, "body_adoption_closed"), 600)
            verified = wait_until(lambda: [row for row in _jsonl(events_log, "restart_verify")
                                           if row.get("expected_sha") != old_head], 300)
            if not (closed and verified):
                for name in ("body_adoption_not_armed", "body_adoption_armed", "body_adoption_unconfirmed"):
                    print("S37 supervisor rows: " + json.dumps({name: _jsonl(supervisor_log, name)}))
                print("S37 restart_verify: " + json.dumps(_jsonl(events_log, "restart_verify")))
                print("S37 candidates: " + json.dumps(_candidate_rows(data_root)))
                print("S37 model calls: " + json.dumps(stub.kinds()))
            assert closed and verified, (closed, verified)

            # The adoption of exactly the gate-recorded commit, armed by generation A's own stop
            # evidence and switched before generation B's imports, in the same server process.
            rows = _candidate_rows(data_root)
            assert len(rows) == 1 and rows[0]["task_id"] == task_id, rows
            cand = rows[0]["reviewed_commits"][-1]
            assert _git(install, "rev-parse", f"{cand}^") == old_head  # built on the serving HEAD
            assert [(row["outcome"], str(row.get("adoption_id") or "")) for row in closed] == [
                ("adopted", str(closed[0].get("adoption_id") or ""))], closed
            phases = [event["phase"] for event in closed[0]["events"]]
            assert phases[0] == "authorized" and phases[-1] == "adopted", phases
            assert phases.index("armed") < phases.index("switched") < phases.index("adopted"), phases
            # The tool authorized it in a generation-A worker; the server process bound and armed it,
            # switched it (re-executed, before its imports) and settled it.
            boots = _jsonl(events_log, "worker_boot")
            old_workers = {row.get("pid") for row in boots if row.get("git_sha") == old_head}
            assert closed[0]["events"][0]["pid"] in old_workers, (closed[0]["events"], boots)
            assert {event["pid"] for event in closed[0]["events"][1:]} == {pid}, closed[0]["events"]
            assert server.proc.poll() is None, "the server re-executes in place; it did not exit"

            # Generation B serves the adopted commit: the checkout is AT it, clean, the note in place,
            # and the restart is verified on the serving SHA.
            assert _git(install, "rev-parse", "HEAD") == cand
            assert _git(install, "symbolic-ref", "--short", "HEAD") == "ouroboros"
            assert _git(install, "status", "--porcelain") == ""
            assert (install / S37_DOC_PATH).read_text(encoding="utf-8") == S37_DOC
            assert [(row["ok"], row["expected_sha"], row["observed_sha"]) for row in verified] == [
                (True, cand, cand)], verified
            assert json.loads(_fetch(server.base_url + "/api/health"))["runtime_version"] == source.version_text
            assert any(row.get("git_sha") == cand for row in _jsonl(events_log, "worker_boot")), boots
            assert "triad_review" in stub.kinds() and "two_part_review" in stub.kinds(), stub.kinds()
            assert ArtifactOracle(data_root).task_drive(task_id).tools_rows(), "generation A's tool log is empty"
        finally:
            server.stop()


# ===========================================================================
# S38 — a server crash between the evolution cycle's candidate commit and its receipts.
# ===========================================================================

S38_DOC_PATH = "docs/notes/system_e2e_s38_evolution.md"
S38_COMMIT_MESSAGE = "docs: system_e2e S38 evolution cycle note"
S38_SCRIPT = [
    {"tool": "write_file", "arguments": {"root": "system_repo", "path": S38_DOC_PATH,
                                         "content": "# system_e2e S38\n\nSelf-evolution cycle payload.\n"}},
    {"tool": "commit_reviewed", "arguments": {
        "commit_message": S38_COMMIT_MESSAGE, "paths": [S38_DOC_PATH],
        "skip_advisory_review": True, "skip_tests": True,
        "goal": "Land the S38 evolution note through the blocking review organ.", "scope": f"{S38_DOC_PATH} only."}},
]


def _after_the_commit(body: dict) -> bool:
    """The cycle's first model round after commit_reviewed returned."""
    return classify_call(body) == "agent" and any(
        (call.get("function") or {}).get("name") == "commit_reviewed"
        for message in body.get("messages") or [] if isinstance(message, dict)
        for call in message.get("tool_calls") or [] if isinstance(call, dict))


def _campaign(data_root: pathlib.Path) -> dict:
    try:
        return json.loads((data_root / "state" / "evolution_campaign.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _sigkill_server_group(server) -> None:
    group = os.getpgid(server.proc.pid)
    os.killpg(group, signal.SIGKILL)
    server.proc.wait(timeout=30)
    assert wait_until(lambda: subprocess.run(["pgrep", "-g", str(group)], capture_output=True).returncode == 1, 30)


@pytest.mark.integration
@pytest.mark.serial
@pytest.mark.skipif(os.name == "nt", reason="the crash is a POSIX process-group SIGKILL")
def test_s38_boot_recovers_a_crashed_evolution_candidate_commit_without_absorbing_it(
        tmp_path_factory, monkeypatch):
    require_lane(LANE_MOCK)
    root = tmp_path_factory.mktemp("s38")
    data_root, state_dir = root / "data", root / "data" / "state"
    seed_owner_state(data_root, evolution_enabled=True)
    monkeypatch.setenv("OUROBOROS_EVOLUTION_AUTO_RESTART", "false")
    settings_kwargs = dict(OUROBOROS_RUNTIME_MODE="advanced", OUROBOROS_REVIEW_ENFORCEMENT="blocking")
    gate = ModelGate(_after_the_commit, timeout=600)
    with candidate_checkout(REPO_ROOT, root / "source" / "repo", origin_proof=True) as source:
        install = _committed_install(source.path, root / "install" / "repo")
        old_head = _git(install, "rev-parse", "HEAD")

        # ---- Generation A: the supervisor's cycle commits in its candidate; the server dies.
        with ScriptedStubModel(S38_SCRIPT, gate=gate) as stub:
            server = start_server(install, root, keyless_settings(stub, **settings_kwargs))
            try:
                assert gate.arrived.wait(600), ("the cycle never returned from commit_reviewed", stub.kinds())
                tx = _campaign(data_root)["active_transaction"]
                cand, task_id = str(tx["commit_sha"]), str(tx["task_id"])
                rows = _candidate_rows(data_root)
                assert len(rows) == 1 and rows[0]["task_id"] == task_id and rows[0]["reviewed_commits"] == [cand]
                assert tx["commit_intent"]["parents"] == [old_head] and _git(install, "rev-parse", "HEAD") == old_head
                assert "triad_review" in stub.kinds() and "two_part_review" in stub.kinds(), stub.kinds()
                _sigkill_server_group(server)
            finally:
                gate.release.set()
                if server.proc.poll() is None:
                    server.stop()

        # Shape the crash window: the receipts written after `git commit` never landed.
        campaign_path = state_dir / "evolution_campaign.json"
        campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
        for key in ("commit_sha", "commit_receipt", "triad_scope_status", "author_disposition"):
            campaign["active_transaction"].pop(key, None)
        campaign["active_transaction"]["restart_required"] = False
        campaign_path.write_text(json.dumps(campaign), encoding="utf-8")
        for registry in sorted(data_root.rglob("state/subagent_worktrees.json")):
            stored = json.loads(registry.read_text(encoding="utf-8"))
            for row in stored["worktrees"]:
                if row.get("kind") == "body_candidate":
                    row["reviewed_commits"] = []
            registry.write_text(json.dumps(stored), encoding="utf-8")
        assert not list(state_dir.glob("pending_restart_verify*"))
        state_path = state_dir / "state.json"
        state_blob = json.loads(state_path.read_text(encoding="utf-8"))
        state_blob["evolution_mode_enabled"] = False  # no cycle 2 mid-scenario (S22's shape)
        state_path.write_text(json.dumps(state_blob), encoding="utf-8")

        # ---- Generation B: a fresh boot on the same install and data root.
        with ScriptedStubModel([]) as stub_b:
            server_b = start_server(install, root, keyless_settings(stub_b, **settings_kwargs))
            try:
                recovered = wait_until(lambda: str((_campaign(data_root).get("active_transaction") or {})
                                                   .get("commit_sha") or "") == cand, 300)
                campaign_b = _campaign(data_root)
                events = ArtifactOracle(data_root)
                print("S38 campaign: " + json.dumps(campaign_b, ensure_ascii=False)[:6000])
                print("S38 worker_boot: " + json.dumps(events.events("worker_boot")))
                assert recovered, campaign_b
                open_tx = campaign_b["active_transaction"]
                assert open_tx["commit_receipt"]["reason"] == "recovered_from_commit_intent", open_tx
                assert (open_tx["restart_required"], open_tx["restart_verified"]) == (True, False), open_tx
                assert open_tx.get("cycle_outcome") != "absorbed" and not campaign_b.get("absorbed_cycles_done")
                assert not [row for row in campaign_b.get("transaction_history") or []
                            if row.get("commit_sha") == cand], campaign_b.get("transaction_history")
                assert [row["reviewed_commits"] for row in _candidate_rows(data_root)] == [[cand]]
                assert campaign_b.get("last_boot_reconcile_gen"), campaign_b
                for kind in ("evolution_tx_reconciled", "evolution_tx_abandoned", "evolution_tx_reconcile_blocked"):
                    assert not events.events(kind), (kind, events.events(kind))
                # Only the serving SHA proves a restart: the serving checkout never held the commit.
                assert _git(install, "rev-parse", "HEAD") == old_head and _git(install, "status", "--porcelain") == ""
                assert any(row.get("git_sha") == old_head for row in events.events("worker_boot"))
            finally:
                server_b.stop()
