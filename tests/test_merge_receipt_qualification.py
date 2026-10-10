"""Exercise merge uncertainty and publication through registered tool/delivery consumers."""
from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import pathlib
import shutil
import subprocess
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Event
from types import SimpleNamespace

import pytest

from ouroboros import merge_receipts, review_ledger
from ouroboros.tools import github
from tests import test_pr_merge_receipts as merge_fixture
from tests.test_pr_merge_receipts import BASE, HEAD, MERGE, TREE, _receipts

world = merge_fixture.world

REAL_GH = github._gh_run
PRE_EFFECT = (
    "X Pull request octo/demo#7 is not mergeable: the base branch policy prohibits the merge.\n"
    "To have the pull request merged after all the requirements have been met, add the `--auto` flag.\n"
    "To use administrator privileges to immediately merge the pull request, add the `--admin` flag.\n"
)


def registered_merge(world):
    handler = next(entry.handler for entry in github.get_tools() if entry.name == "pr_merge")
    return handler(world.ctx, number=7, expected_head_sha=HEAD, method="squash",
                   reviewed_head_sha=HEAD, reviewed_base_sha=BASE, review_task_ids=["review-1"])


@pytest.mark.parametrize("hostname", ["github.com", "github.example.test"])
def test_body_patch_uses_selected_repo_scoped_identity_without_org_access(world, monkeypatch, hostname):
    """The real gh binding needs no GraphQL organization metadata to edit a body."""
    selected_env = {"GH_TOKEN": "synthetic-repo-token-without-read-org"}
    repo = f"{hostname}/chosen/repository"
    world.gh.pr["url"] = f"https://{repo}/pull/7"
    prefix, suffix = 'Résumé "quoted" \\ path.\r\n\n', '\n\nOwner footer.  \n'
    note = "\nAdded during merge: @literal-file-name.\n"
    world.gh.pr["body"] = prefix + "<!-- ouroboros:merge-receipt ab -->\nold\n<!-- /ouroboros:merge-receipt -->" + suffix
    writes = []

    def process(argv, **kw):
        assert kw["env"] == selected_env  # every call retains the same selected identity
        args = argv[1:]
        if args[:1] == ["pr"]:
            assert args[-2:] == ["--repo", repo]
        if args[:1] == ["api"] and "PATCH" in args:
            assert args == ["api", "repos/chosen/repository/pulls/7", "--hostname", hostname,
                            "--method", "PATCH", "--input", "-"]
            assert kw["timeout"] == 60
            assert _receipts(world)[0]["outcome"]["status"] == "merged"
            writes.append(json.loads(kw["input"]))
        result = world.gh(args, world.ctx, input_data=kw.get("input"))
        if args[:2] == ["pr", "merge"]:
            world.gh.pr["body"] += note
        return SimpleNamespace(returncode=result.exit_code, stdout=result.text if result.ok else "",
                               stderr="" if result.ok else result.text)

    monkeypatch.setattr(github, "_gh_run", REAL_GH)
    monkeypatch.setattr(github.subprocess, "run", process)
    monkeypatch.setattr(github, "_gh_env", lambda ctx: selected_env)
    monkeypatch.setattr(github, "github_token_from_env_or_settings", lambda: selected_env["GH_TOKEN"])
    handler = next(entry.handler for entry in github.get_tools() if entry.name == "pr_merge")
    text = handler(world.ctx, number=7, expected_head_sha=HEAD, method="squash", repo=repo)
    (receipt,) = _receipts(world)
    assert receipt["publication"]["body"] == {"status": "published"}, text
    expected = prefix + merge_receipts.public_block(receipt) + suffix + note
    assert writes == [{"body": expected}] and world.gh.pr["body"] == expected
    assert world.gh.pr["title"] == "demo"
    assert receipt["publication"]["card"]["status"] == "owed"
    assert not any(c[:2] == ["pr", "edit"] or c[:2] == ["api", "graphql"] or c[0] == "auth"
                   for c in world.gh.calls)


@pytest.mark.parametrize("failure", ["EOF", "HTTP 502: Bad Gateway (https://api.github.com/graphql)",
                                     "HTTP 409: Conflict (https://api.github.com/graphql)",
                                     PRE_EFFECT + "HTTP 503: lost reply", "timeout"])
def test_real_transport_unknown_exit_never_releases_intent(world, monkeypatch, failure):
    effects = []

    def process(argv, **kw):
        args = argv[1:]
        if args[:2] == ["pr", "merge"]:
            effects.append(args)
            if failure == "timeout":
                raise github.subprocess.TimeoutExpired(argv, 120)
            return SimpleNamespace(returncode=1, stdout="", stderr=failure)
        result = world.gh(args, world.ctx, input_data=kw.get("input"))
        return SimpleNamespace(returncode=0, stdout=result.text, stderr="")

    monkeypatch.setattr(github, "_gh_run", REAL_GH)
    monkeypatch.setattr(github.subprocess, "run", process)
    monkeypatch.setattr(github, "_gh_env", lambda ctx: {})
    monkeypatch.setattr(github, "github_token_from_env_or_settings", lambda: "")
    assert "PR_MERGE_UNKNOWN" in registered_merge(world)
    rid = _receipts(world)[0]["receipt_id"]
    assert "sent no new merge request" in registered_merge(world)
    assert len(effects) == 1 and _receipts(world)[0]["outcome"]["status"] == "unknown"
    world.gh.pr.update(state="MERGED", mergeCommit={"oid": MERGE})
    assert "merge: merged" in registered_merge(world)
    assert _receipts(world)[0]["receipt_id"] == rid and len(effects) == 1
    assert _receipts(world)[0]["outcome"]["attribution"] == "unproven"


@pytest.mark.parametrize("failure", ["cli_missing", "rejection"])
def test_proven_pre_effect_failure_can_recover(world, monkeypatch, failure):
    attempts = []

    def process(argv, **kw):
        args = argv[1:]
        if args[:2] == ["pr", "merge"]:
            attempts.append(args)
            if len(attempts) == 1:
                if failure == "cli_missing":
                    raise FileNotFoundError(2, "missing", "gh")
                return SimpleNamespace(returncode=1, stdout="", stderr=PRE_EFFECT)
        result = world.gh(args, world.ctx, input_data=kw.get("input"))
        return SimpleNamespace(returncode=0, stdout=result.text, stderr="")

    monkeypatch.setattr(github, "_gh_run", REAL_GH)
    monkeypatch.setattr(github.subprocess, "run", process)
    monkeypatch.setattr(github, "_gh_env", lambda ctx: {})
    monkeypatch.setattr(github, "github_token_from_env_or_settings", lambda: "")
    assert "PR_MERGE_REFUSED" in registered_merge(world)
    assert "merge: merged" in registered_merge(world)
    assert len(attempts) == 2
    assert [r["outcome"]["status"] for r in _receipts(world)] == ["refused", "merged"]


@pytest.mark.parametrize("gap", ["settlement", "publication_receipt", "card", "body"])
def test_post_effect_gaps_reach_the_registered_tool(world, monkeypatch, gap):
    original = merge_receipts.write_receipt

    def fail_write(*args, **kw):
        if not kw.get("claim") and (gap == "settlement" or (gap == "publication_receipt" and "publication" in kw)):
            raise OSError("injected write failure")
        return original(*args, **kw)

    with monkeypatch.context() as patch:
        if gap in ("settlement", "publication_receipt"):
            patch.setattr(merge_receipts, "write_receipt", fail_write)
        elif gap == "card":
            from supervisor import terminal_delivery
            patch.setattr(terminal_delivery, "update_json_locked",
                          lambda *a, **k: (_ for _ in ()).throw(OSError("injected outbox failure")))
        else:
            world.gh.fail_edit = True
        text = registered_merge(world)
    assert "merge: merged" in text  # confirmed effect survives the gap
    if gap in ("settlement", "publication_receipt"):
        assert "receipt persistence" in text and "unknown" in text
    if gap == "settlement":
        assert "PR-body" in text and "card" in text
    if gap == "card":
        assert "card" in text and "outbox_unwritable" in text
    if gap == "body":
        assert "PR-body" in text
    world.gh.fail_edit = False
    assert "sent no new merge request" in registered_merge(world)
    assert sum(c[:2] == ["pr", "merge"] for c in world.gh.calls) == 1


@pytest.mark.serial
@pytest.mark.parametrize("delivery_order", ["buffered", "delivered_first", "replay_reverse"])
def test_concurrent_publication_through_outbox_dedup_and_history(world, monkeypatch, delivery_order):
    from ouroboros.gateway.history import make_chat_history_endpoint
    from supervisor import events_chat_delivery as delivery
    from supervisor import message_bus
    from supervisor import terminal_delivery as td

    monkeypatch.setattr(delivery, "_DELIVERED_MESSAGE_IDS", deque(maxlen=1000))
    monkeypatch.setattr(message_bus, "DATA_DIR", world.root)
    monkeypatch.setattr(message_bus, "publish_event", lambda *a, **k: None)
    bridge = message_bus.LocalChatBridge({})
    monkeypatch.setattr(message_bus, "get_bridge", lambda: bridge)
    monkeypatch.setattr(message_bus, "load_state", lambda: {"owner_id": 7})
    frames = []
    bridge._broadcast_fn = frames.append
    errors = []
    ctx = SimpleNamespace(DRIVE_ROOT=world.root, task_registry={}, send_with_budget=message_bus.send_with_budget,
                          append_jsonl=lambda *a: errors.append(a))
    world.gh.merge = "accepted_open"
    blocked, release = Event(), Event()
    gh = world.gh

    def delayed(args, *a, **kw):
        if args[:1] == ["api"] and "PATCH" in args and "**queued**" in str(kw.get("input_data")):
            blocked.set()
            assert release.wait(5)
        return gh(args, *a, **kw)

    monkeypatch.setattr(github, "_gh_run", delayed)
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(registered_merge, world)
        try:
            assert blocked.wait(5)
            world.gh.pr.update(state="MERGED", mergeCommit={"oid": MERGE})
            assert "merge: merged" in registered_merge(world)
            if delivery_order == "delivered_first":
                for event in world.ctx.pending_events:
                    delivery._handle_send_message(event, ctx)
                world.ctx.pending_events.clear()
        finally:
            release.set()
        assert "merge: merged" in first.result(5)
    if delivery_order == "replay_reverse":
        world.ctx.pending_events.clear()
        monkeypatch.setattr(td, "_REPLAY_MIN_AGE_SEC", 0)
        queue = Queue()
        td.replay_pending_deliveries(world.root, event_queue=queue)
        events = []
        while not queue.empty():
            events.append(queue.get_nowait())
        events.reverse()
    else:
        events = world.ctx.pending_events
    for event in events:
        delivery._handle_send_message(event, ctx)
    assert not errors
    assert td.pending_deliveries(world.root) == []
    chats = [row for row in frames if row.get("type") == "chat"]
    assert chats and all("card_row_revision" in row for row in chats)
    response = asyncio.run(make_chat_history_endpoint(world.root)(SimpleNamespace(query_params={"chat_id": "7"})))
    history = json.loads(response.body)["messages"]
    rows = [row for row in history if row.get("card_row_id")]
    assert rows and all("card_row_revision" in row for row in rows)
    assert max(rows, key=lambda row: row["card_row_revision"])["text"].find("merge: merged") >= 0
    fixture = world.root / "delivery-projection.json"
    fixture.write_text(json.dumps({"live": chats, "history": rows}), encoding="utf-8")
    node = shutil.which("node")
    assert node, "node is required for the real live/replay projection qualification"
    result = subprocess.run([node, "--test", "--test-name-pattern=canonical receipt revisions",
                             "web/tests/chat_card_row_placement.test.js"],
                            cwd=pathlib.Path(__file__).resolve().parents[1],
                            env={**os.environ, "MERGE_RECEIPT_PROJECTION": str(fixture)},
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert sum(c[:2] == ["pr", "merge"] for c in world.gh.calls) == 1


def test_legacy_ambiguous_refusal_keeps_observation_custody(world):
    world.gh.merge = "timeout_open"
    registered_merge(world)
    receipt = _receipts(world)[0]
    receipt["effect"]["failure"] = "exit"
    receipt["outcome"] = {"status": "refused"}
    # Reproduce an old persisted receipt, bypassing today's canonical join.
    from ouroboros.task_results import task_result_path
    path = task_result_path(world.root, "merge-task")
    record = json.loads(path.read_text(encoding="utf-8"))
    record["merge_receipts"] = [receipt]
    path.write_text(json.dumps(record), encoding="utf-8")
    assert "PR_MERGE_UNKNOWN" in registered_merge(world)
    assert sum(c[:2] == ["pr", "merge"] for c in world.gh.calls) == 1


def test_proven_rejection_survives_unavailable_readback(world):
    gh = world.gh
    world.gh.merge = "refused"
    world.gh.fail_view_after_merge = True
    assert "PR_MERGE_UNKNOWN" in registered_merge(world)
    world.gh.fail_view_after_merge = False
    assert "PR_MERGE_REFUSED" in registered_merge(world)
    assert sum(c[:2] == ["pr", "merge"] for c in gh.calls) == 1
    world.gh.merge = "ok"
    assert "merge: merged" in registered_merge(world)
    assert sum(c[:2] == ["pr", "merge"] for c in gh.calls) == 2


def test_receipt_revision_is_canonical_and_idempotent_across_fact_enrichment(world, monkeypatch):
    gh = world.gh
    registered_merge(world)
    first = _receipts(world)[0]
    registered_merge(world)
    unchanged = _receipts(world)[0]
    assert unchanged["revision"] == first["revision"]
    assert unchanged["publication"]["card"]["delivery_id"] == first["publication"]["card"]["delivery_id"]

    def missing_head_tree(args, *a, **kw):
        if args[:1] == ["api"] and args[1].endswith(HEAD):
            return github.GhResult(False, "lost read", 1, 502, "exit")
        return gh(args, *a, **kw)

    monkeypatch.setattr(github, "_gh_run", missing_head_tree)
    registered_merge(world)
    gap = _receipts(world)[0]
    monkeypatch.setattr(github, "_gh_run", gh)
    registered_merge(world)
    recovered = _receipts(world)[0]
    assert first["revision"] < gap["revision"] < recovered["revision"]
    assert merge_receipts.card_row_text(recovered) == merge_receipts.card_row_text(first)
    assert recovered["publication"]["card"]["delivery_id"] != first["publication"]["card"]["delivery_id"]
    assert recovered["outcome"]["attribution"] == "this_call" and recovered["outcome"]["merge_sha"] == MERGE


def bounded_receipt_history(world, monkeypatch):
    """Actual r3 delivery, late r2, deduped catch-up, then a default-quota read."""
    from ouroboros.gateway import history as history_api
    from ouroboros.utils import append_jsonl

    test_concurrent_publication_through_outbox_dedup_and_history(world, monkeypatch, "delivered_first")
    progress_path = world.root / "logs/progress.jsonl"
    raw = [json.loads(line) for line in progress_path.read_text(encoding="utf-8").splitlines()]
    receipt_events = [r for r in raw if r.get("card_row_id")]
    assert [r["card_row_revision"] for r in receipt_events] == [3, 2]
    for i in range(59):
        append_jsonl(world.root / "logs/progress.jsonl", {
            "type": "task_progress", "chat_id": 7, "task_id": "merge-task",
            "text": f"Ordinary activity {i}", "ts": f"2099-01-01T00:00:{i:02d}Z",
        })
    sources = {progress_path: progress_path.read_bytes()}
    selected = []
    annotate = history_api._annotate_terminal_task_truth

    def capture(messages, *args, **kwargs):
        selected.append(json.loads(json.dumps(messages)))
        return annotate(messages, *args, **kwargs)

    from ouroboros import task_status
    load_result = task_status.load_effective_task_result
    reads = []

    def read_result(data_dir, task_id, **kwargs):
        assert kwargs == {"materialize_artifacts": False}
        reads.append(task_id)
        return load_result(data_dir, task_id, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(task_status, "load_effective_task_result", read_result)
        patch.setattr(history_api, "_annotate_terminal_task_truth", capture)
        response = asyncio.run(history_api.make_chat_history_endpoint(world.root)(
            SimpleNamespace(query_params={"chat_id": "7"})))
    payload = json.loads(response.body)
    assert reads == ["merge-task"]  # one existing result read, reused by every selected row
    assert len(selected[0]) == 60  # no quota override; r3 really is outside the selected window
    (stale,) = [r for r in selected[0] if r.get("card_row_id")]
    assert stale["card_row_revision"] == 2 and "merge: queued" in stale["text"]
    (projected,) = [r for r in payload["messages"] if r.get("card_row_id")]
    assert all(projected[k] == stale[k] for k in ("history_id", "history_position", "ts", "card_row_id"))
    assert {path: path.read_bytes() for path in sources} == sources
    evidence = json.dumps({"payload": payload, "selected": selected[0]})
    (world.root / "cold-history.json").write_text(evidence, encoding="utf-8")
    if output := os.environ.get("OUROBOROS_UI_EVIDENCE_DIR"):
        output = pathlib.Path(output)
        output.mkdir(parents=True, exist_ok=True)
        (output / "cold-history.json").write_text(evidence, encoding="utf-8")
        (output / "cold-progress.jsonl").write_bytes(sources[progress_path])
        (output / "canonical-receipts.json").write_text(json.dumps(_receipts(world)), encoding="utf-8")
    return payload


@pytest.mark.serial
def test_cold_bounded_history_supplies_current_receipt_to_new_chat(world, monkeypatch):
    payload = bounded_receipt_history(world, monkeypatch)
    fixture = world.root / "cold-history.json"
    result = subprocess.run([shutil.which("node"), "--test", "--test-name-pattern=cold bounded receipt",
                             "web/tests/chat_card_row_placement.test.js"],
                            env={**os.environ, "COLD_RECEIPT_PROJECTION": str(fixture)},
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    (receipt,) = [r for r in payload["messages"] if r.get("card_row_id")]
    assert receipt["card_row_revision"] == 3 and "merge: merged" in receipt["text"]


@pytest.mark.serial
@pytest.mark.parametrize("gap", ["missing_result", "unreadable_result", "missing_receipt", "older_revision", "malformed_receipt", "other_task"])
def test_bounded_history_discloses_unavailable_canonical_receipt(world, monkeypatch, gap):
    from ouroboros.gateway.history import make_chat_history_endpoint
    from ouroboros.task_results import task_result_path

    bounded_receipt_history(world, monkeypatch)
    result_path = task_result_path(world.root, "merge-task", create=False)
    record = json.loads(result_path.read_text(encoding="utf-8"))
    if gap == "missing_result":
        result_path.unlink()
    elif gap == "unreadable_result":
        result_path.write_text("{torn", encoding="utf-8")
    else:
        if gap == "missing_receipt":
            record["merge_receipts"] = []
        elif gap == "older_revision":
            record["merge_receipts"][0]["revision"] = 1
        elif gap == "malformed_receipt":
            record["merge_receipts"][0]["outcome"] = "malformed"
        else:
            # A same-id receipt from a different task must not supply truth.
            other_path = task_result_path(world.root, "other-task")
            other_path.write_text(json.dumps(record), encoding="utf-8")
            record["merge_receipts"] = []
        result_path.write_text(json.dumps(record), encoding="utf-8")
    source = world.root / "logs/progress.jsonl"
    before = source.read_bytes()
    response = asyncio.run(make_chat_history_endpoint(world.root)(SimpleNamespace(query_params={"chat_id": "7"})))
    (row,) = [r for r in json.loads(response.body)["messages"] if r.get("card_row_id")]
    assert "merge: queued" in row["text"] and "merge: merged" not in row["text"]
    assert "Current merge receipt unavailable; showing recorded event." in row["text"]
    assert row["card_row_revision"] == 2 and source.read_bytes() == before


@pytest.mark.serial
def test_frozen_history_page_refreshes_receipt_without_changing_positions(world, monkeypatch):
    from ouroboros.gateway.history import make_chat_history_endpoint

    before = bounded_receipt_history(world, monkeypatch)
    endpoint = make_chat_history_endpoint(world.root)
    (canonical,) = _receipts(world)
    canonical["coverage"]["gaps"] = ["head_tree_unavailable"]
    current = merge_receipts.write_receipt(world.root, "merge-task", canonical)
    assert current["revision"] == 4
    # No new delivery: re-reading a physically frozen page must still obtain current truth.
    after = json.loads(asyncio.run(endpoint(SimpleNamespace(query_params={
        "chat_id": "7", "cursor": before["page_cursor"],
    }))).body)
    assert [(r["history_id"], r["history_position"], r["ts"]) for r in after["messages"]] == [
        (r["history_id"], r["history_position"], r["ts"]) for r in before["messages"]]
    assert all(after[key] == before[key] for key in ("page_cursor", "next_cursor", "has_more"))
    (row,) = [r for r in after["messages"] if r.get("card_row_id")]
    assert row["card_row_revision"] == 4 and "head_tree_unavailable" in row["text"]


# --- a named host review record (review_record_id) supplies the reviewed subject ---------

RECORD_ID = "rec-merge-1"
PRIVATE_ROOT = "/private/review-root"


@dataclasses.dataclass
class LedgerSubject:
    root_kind: str
    root: str
    kind: str
    base: str
    head: str
    tree_sha: str


@dataclasses.dataclass
class LedgerRecord:
    """The dataclass shape ``review_ledger.load_record`` may return instead of a mapping."""
    record_id: str
    revision: int
    state: str
    subject: LedgerSubject
    verdict: dict
    panel: dict


def ledger_record(kind="base..head", *, head=HEAD, base=BASE, tree=TREE, aggregate="PASS", state="settled"):
    return {"record_id": RECORD_ID, "revision": 1, "state": state,
            "subject": {"root_kind": "active_workspace", "root": PRIVATE_ROOT, "kind": kind,
                        "base": base, "head": head, "tree_sha": tree},
            "verdict": {"aggregate": aggregate, "per_question": {"change": aggregate, "coupling": "PASS"}},
            "panel": {"seats": 3, "distinct_models": 2}}


def install_ledger(monkeypatch, records=(), *, error=None, answer=None):
    """Stand in for the ``review_ledger`` reader ``merge_receipts`` binds; returns what it was asked."""
    asked = []

    def load_record(drive_root, record_id):
        asked.append((pathlib.Path(drive_root).resolve(), record_id))
        if error is not None:
            raise error
        return answer if answer is not None else dict(records).get(record_id)

    monkeypatch.setattr(merge_receipts, "review_ledger", SimpleNamespace(load_record=load_record))
    return asked


def record_merge(world, **kw):
    handler = next(entry.handler for entry in github.get_tools() if entry.name == "pr_merge")
    return handler(world.ctx, number=7, expected_head_sha=HEAD, method="squash", **kw)


def test_review_record_id_is_its_own_optional_argument():
    schema = next(e.schema for e in github.get_tools() if e.name == "pr_merge")["parameters"]
    assert schema["properties"]["review_record_id"]["type"] == "string"
    assert "review_record_id" not in schema["required"] and "review_reference" not in schema["properties"]


@pytest.mark.parametrize("shape", ["mapping", "dataclass"])
def test_a_matching_review_record_covers_the_head_and_names_its_source(world, monkeypatch, shape):
    record = ledger_record()
    if shape == "dataclass":
        record = LedgerRecord(**{**record, "subject": LedgerSubject(**record["subject"])})
    asked = install_ledger(monkeypatch, {RECORD_ID: record})
    out = record_merge(world, review_record_id=RECORD_ID, review_task_ids=["review-1"])
    assert asked == [(world.root.resolve(), RECORD_ID)]
    assert ["pr", "merge", "7", "--squash", "--match-head-commit", HEAD] in world.gh.calls
    (receipt,) = _receipts(world)
    assert receipt["coverage"]["status"] == "covers_head" and receipt["coverage"]["gaps"] == []
    review = receipt["review"]
    # Named review tasks stay observations; beside a record they declare nothing.
    assert review["declared"] is None and review["declared_only"] is False
    assert [row["status"] for row in review["host_observed"]] == ["completed"]
    assert review["record"]["subject"] == {"root_kind": "active_workspace", "root": PRIVATE_ROOT,
                                           "kind": "base..head", "base": BASE, "head": HEAD, "tree_sha": TREE}
    assert review["record"]["verdict"] == {"aggregate": "PASS", "per_question": {"change": "PASS", "coupling": "PASS"}}
    assert review["record"]["panel"] == {"seats": 3, "distinct_models": 2}
    assert out.startswith("✅ PR #7 merge: merged") and "review source: record, verdict PASS" in out
    body = world.gh.pr["body"]
    assert "- Review record (host review ledger): `base..head` subject" in body
    assert "verdict **PASS** (change PASS, coupling PASS); panel 3 seats, 2 distinct models" in body
    assert "- Host-observed review tasks: 1 completed of 1 named" in body
    assert "Declared review" not in body and "No review was declared" not in body
    assert RECORD_ID not in body and PRIVATE_ROOT not in body and "active_workspace" not in body
    assert "review-1" not in body and "merge-task" not in body


def test_a_record_written_by_the_review_ledger_covers_the_head_through_the_real_reader(world):
    drive_root = review_ledger.ledger_root(world.ctx)
    record = review_ledger.ReviewLedgerRecord(
        record_id=RECORD_ID, task_id="merge-task", subject=ledger_record()["subject"],
        verdict={"aggregate": "PASS", "per_question": {"change": "PASS", "coupling": "PASS"}},
        panel={"seats": [{"seat_id": f"s{i}"} for i in range(3)], "distinct_models": ["a", "b"]})
    review_ledger.write_record(drive_root, record)
    out = record_merge(world, review_record_id=RECORD_ID)
    (receipt,) = _receipts(world)
    assert receipt["coverage"]["status"] == "covers_head" and receipt["coverage"]["gaps"] == []
    assert receipt["review"]["record"]["panel"] == {"seats": 3, "distinct_models": 2}
    assert receipt["review"]["record"]["subject"]["kind"] == "base..head" and receipt["review"]["declared_only"] is False
    assert out.startswith("✅ PR #7 merge: merged") and "review source: record, verdict PASS" in out


@pytest.mark.parametrize("kind", ["index", "worktree"])
def test_an_uncommitted_record_at_the_same_head_is_a_named_gap_and_still_merges(world, monkeypatch, kind):
    install_ledger(monkeypatch, {RECORD_ID: ledger_record(kind)})
    out = record_merge(world, review_record_id=RECORD_ID)
    (receipt,) = _receipts(world)
    assert receipt["outcome"]["status"] == "merged"  # loud, never a lock
    assert receipt["coverage"] == {**receipt["coverage"], "status": "unknown", "gaps": ["subject_kind_not_committed"]}
    assert out.startswith("⚠️ PR #7 merge: merged") and "subject_kind_not_committed" in out
    assert f"`{kind}` subject" in world.gh.pr["body"] and "gaps: subject_kind_not_committed" in world.gh.pr["body"]


@pytest.mark.parametrize(("facts", "status", "gaps"), [
    ({"base": "e" * 40}, "unknown", ["reviewed_base_differs"]),
    ({"base": ""}, "unknown", ["reviewed_base_not_recorded"]),
    ({"tree": "9" * 40}, "unknown", ["merged_tree_differs_from_reviewed_head"]),
    ({"tree": ""}, "unknown", ["tree_comparison_unavailable"]),
    ({"head": "e" * 40}, "changes_after_review", []),
    ({"head": ""}, "unknown", ["reviewed_head_not_recorded"]),
])
def test_covers_head_needs_the_record_head_base_and_tree(world, monkeypatch, facts, status, gaps):
    install_ledger(monkeypatch, {RECORD_ID: ledger_record(**facts)})
    out = record_merge(world, review_record_id=RECORD_ID)
    (receipt,) = _receipts(world)
    assert receipt["outcome"]["status"] == "merged"
    assert (receipt["coverage"]["status"], receipt["coverage"]["gaps"]) == (status, gaps)
    assert out.startswith("⚠️ PR #7 merge: merged")


@pytest.mark.parametrize("absent", [{}, {"error": KeyError(RECORD_ID)}, {"error": FileNotFoundError(RECORD_ID)}])
def test_a_nonexistent_record_id_is_an_argument_refusal_before_any_effect(world, monkeypatch, absent):
    asked = install_ledger(monkeypatch, **absent)
    out = record_merge(world, review_record_id=RECORD_ID, reviewed_head_sha=HEAD)
    assert out.startswith("⚠️ PR_MERGE_REFUSED: arguments") and f"review_record_id={RECORD_ID!r}" in out
    assert [record_id for _, record_id in asked] == [RECORD_ID]
    assert world.gh.calls == [] and _receipts(world) == []  # nothing read, merged or recorded


@pytest.mark.parametrize("failure", ["store", "shape", "corrupt_file"])
def test_an_unreadable_record_is_not_reported_as_an_absent_one(world, monkeypatch, failure):
    if failure == "corrupt_file":  # the real reader over a damaged record file
        path = review_ledger.record_path(review_ledger.ledger_root(world.ctx), RECORD_ID)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not a record", encoding="utf-8")
    else:
        install_ledger(monkeypatch, **({"error": OSError("disk")} if failure == "store" else {"answer": ["x"]}))
    out = record_merge(world, review_record_id=RECORD_ID)
    assert out.startswith("⚠️ PR_MERGE_REFUSED: review_record_unreadable") and "nothing was merged" in out
    assert world.gh.calls == [] and _receipts(world) == []


def test_a_declaration_without_a_record_is_declared_only_and_says_so(world, monkeypatch):
    asked = install_ledger(monkeypatch, error=AssertionError("an empty id is the omitted path"))
    handler = next(entry.handler for entry in github.get_tools() if entry.name == "pr_merge")
    out = handler(world.ctx, number=7, expected_head_sha=HEAD, method="squash", reviewed_head_sha=HEAD,
                  reviewed_base_sha=BASE, review_task_ids=["review-1"], review_verdict="PASS", review_record_id="  ")
    assert asked == []
    (receipt,) = _receipts(world)
    assert receipt["review"]["declared_only"] is True and receipt["review"]["record"] is None
    assert receipt["coverage"] == {**receipt["coverage"], "status": "covers_head", "gaps": []}
    assert out.startswith("✅ PR #7 merge: merged") and out.split("\n")[0].endswith("; review source: declaration")
    assert "(declared by the merging agent; no host review record)" in world.gh.pr["body"]
    assert "Review record (host review ledger)" not in world.gh.pr["body"]
    unreviewed = {**receipt, "review": {**receipt["review"], "declared": None, "declared_only": False}}
    assert "review source" not in merge_receipts.card_row_text(unreviewed)


@pytest.mark.parametrize(("state", "aggregate"), [("settled", "FAIL"), ("pending", "PASS"), ("settled", "QUORUM_FAILED")])
def test_a_record_reads_green_only_when_settled_with_pass(world, monkeypatch, state, aggregate):
    install_ledger(monkeypatch, {RECORD_ID: ledger_record(aggregate=aggregate, state=state)})
    out = record_merge(world, review_record_id=RECORD_ID)
    (receipt,) = _receipts(world)
    assert receipt["outcome"]["status"] == "merged"  # a failed review is a fact, never a veto
    assert receipt["coverage"]["status"] == "covers_head" and receipt["coverage"]["gaps"] == []
    assert out.startswith("⚠️ PR #7 merge: merged") and f"review source: record, verdict {aggregate}" in out
    assert (f"({state})" in out) is (state != "settled")
    assert f"verdict **{aggregate}**" in world.gh.pr["body"]


@pytest.mark.parametrize("bind_record", [True, False])
def test_a_shell_merged_pr_stays_receiptless_with_or_without_a_record(world, monkeypatch, bind_record):
    install_ledger(monkeypatch, {RECORD_ID: ledger_record()})
    world.gh.pr.update(state="MERGED", mergeCommit={"oid": MERGE})  # merged from a shell or the web page
    out = record_merge(world, **({"review_record_id": RECORD_ID} if bind_record else {}))
    assert out.startswith("⚠️ PR_MERGE_REFUSED: pr_not_open")
    assert not any(call[:2] == ["pr", "merge"] for call in world.gh.calls) and _receipts(world) == []
