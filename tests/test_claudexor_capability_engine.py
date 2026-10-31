"""Offline real-engine qualification through LLMClient and the owned Accounts proxy.

Run explicitly with CAPABILITY_ENGINE_ROOT and CAPABILITY_NODE_BINARY. The
engine driver substitutes provider HTTP/auth only; it cannot use real accounts.
CAPABILITY_CAPTURE_DIR retains exact JSON for the ordinary replay tests.
"""
from __future__ import annotations

import asyncio
from collections import deque
import json
import os
from pathlib import Path
import queue
import select
import subprocess
from types import SimpleNamespace

import pytest

from ouroboros import claudexor_daemon as owned, llm_claudexor as transport, model_wait, usage_accounting as ua
from ouroboros.gateways import claudexor as wire
from ouroboros.llm import LLMClient
from tests._usage_store_testing import ledger_rows
from tests.claudexor_capability_wire import write_capture

TREE = "94eaa9ef8cb232b3d3a4b31628c9e1a9c1b5c666"
OLDER_TREE = "572efe9bf9beac5ba7f697939c476783505afbca"
MODEL = "claudexor::codex=gpt-6.1-sol"
DRIVER = Path(__file__).parent / "fixtures/llm_wire/claudexor_capability/engine.mjs"
pytestmark = pytest.mark.integration
# A helper bound to a Project room: the account notice still goes to the owner's chat.
TASK = {"id": "capability-test", "chat_id": 777, "title": "Helper call"}
NOTICE = "model_capability_notice"


@pytest.fixture
def real_engine(request, tmp_path, monkeypatch):
    engine, node = os.environ.get("CAPABILITY_ENGINE_ROOT"), os.environ.get("CAPABILITY_NODE_BINARY")
    if not engine or not node:
        pytest.skip("Set CAPABILITY_ENGINE_ROOT and CAPABILITY_NODE_BINARY for the offline engine")
    scenario = request.param
    root = tmp_path / "data"
    # AF_UNIX paths must stay short, even inside a safe_test pytest root.
    import tempfile
    home = Path(tempfile.mkdtemp(prefix="cx-", dir=os.environ["OUROBOROS_TEST_TEMP_ROOT"]))
    stderr = (home / "stderr.log").open("w+", encoding="utf-8")
    process = subprocess.Popen([node, str(DRIVER.resolve()), engine, str(home), scenario],
                               cwd=engine, stdout=subprocess.PIPE, stderr=stderr, text=True)
    records, uploads, ids = [], [], []
    try:
        assert select.select([process.stdout], [], [], 15)[0], "engine startup timed out"
        line = process.stdout.readline()
        if not line:
            stderr.seek(0)
            pytest.fail(f"engine startup failed: {stderr.read()}")
        endpoint = wire.DaemonEndpoint(**json.loads(line))
        original = wire.ClaudexorGateway.__init__

        def record(response):
            response.read()
            request = response.request
            records.append({"method": request.method, "path": request.url.raw_path.decode(),
                            "status": response.status_code, "body_utf8": response.text})

        def initialize(gateway, endpoint):
            original(gateway, endpoint)
            gateway._client.event_hooks["response"] = [record]
        monkeypatch.setattr(wire.ClaudexorGateway, "__init__", initialize)
        upload = wire.ClaudexorGateway.upload_model_request

        def upload_record(gateway, payload, *, idempotency_key):
            uploads.append(json.loads(json.dumps(payload)))
            ids.append(idempotency_key)
            return upload(gateway, payload, idempotency_key=idempotency_key)
        monkeypatch.setattr(wire.ClaudexorGateway, "upload_model_request", upload_record)

        def gateway(*_args, **_kwargs):
            client = wire.ClaudexorGateway(endpoint)
            client.handshake()
            return client
        monkeypatch.setattr(transport, "ensure_owned_gateway", gateway)
        monkeypatch.setattr(transport, "read_owned_gateway", gateway)
        monkeypatch.setattr(owned, "owned_config_dir", lambda: home)
        monkeypatch.setattr(owned, "get_owned_daemon", lambda: SimpleNamespace(status_dict=lambda: {"state": "running"}))
        monkeypatch.setattr(wire, "discover_daemon_at", lambda path: endpoint if path == home else pytest.fail("foreign engine root"))
        monkeypatch.setattr(owned.OwnedClaudexorDaemon, "ensure_running", lambda *_a, **_k: pytest.fail("no daemon lifecycle action"))
        monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
        monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(root / "settings.json"))
        monkeypatch.setenv("OUROBOROS_MODEL_ACCOUNTS", "{}")
        monkeypatch.setenv("TOTAL_BUDGET", "100")
        monkeypatch.setattr(transport.config, "CLAUDEXOR_MODEL_POLL_INTERVAL_SEC", 0.005)
        monkeypatch.setattr(transport, "owned_engine_version", lambda: "")
        events = queue.Queue()
        from ouroboros.task_results import write_task_result
        write_task_result(root, "capability-test", "running")
        with ua.usage_scope(ua.UsageScope(drive_root=root, task_id="capability-test", root_task_id="capability-test")), \
                model_wait.task_model_wait_scope(task=TASK, drive_root=root,
                                                event_queue=events, worker_slot_held=False):
            yield SimpleNamespace(root=root, home=home, scenario=scenario, client=LLMClient(), gateway=gateway,
                                  events=events, records=records, uploads=uploads, ids=ids)
    finally:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
            pytest.fail("fixture engine required a forced stop")
        stderr.close()
        destination = os.environ.get("CAPABILITY_CAPTURE_DIR")
        if destination:
            folder = Path(destination)
            name = request.node.name.replace("[", "-").replace("]", "")
            write_capture(folder, f"{name}.json", {
                "engine_tree": os.environ.get("CAPABILITY_ENGINE_TREE", TREE),
                "scenario": scenario, "provider": "offline Codex HTTP fixture",
                "uploads": uploads, "idempotency_keys": ids, "wire": records,
                "provider_posts": json.loads((home / "provider-posts.json").read_text(encoding="utf-8")) if (home / "provider-posts.json").exists() else [],
            }, prefixes=[(str(DRIVER.resolve().parent), "/tmp/cx-capability"),
                         (str(home.resolve()), f"/tmp/cx-capability/{scenario}"),
                         (str(home), f"/tmp/cx-capability/{scenario}")])


def call(fixture, asynchronous=False, **kwargs):
    values = {"messages": [{"role": "user", "content": "hello"}], "model": MODEL, **kwargs}
    return asyncio.run(fixture.client.chat_async(**values)) if asynchronous else fixture.client.chat(**values)


def notices(fixture, monkeypatch):
    """Deliver through the real supervisor and read the owner's durable history."""
    from supervisor import events_chat_delivery as delivery, message_bus
    from ouroboros.utils import append_jsonl
    from tests.test_main_notice import _history
    bridge = message_bus.LocalChatBridge({})
    monkeypatch.setattr(message_bus, "DATA_DIR", fixture.root)
    monkeypatch.setattr(message_bus, "get_bridge", lambda: bridge)
    monkeypatch.setattr(message_bus, "load_state", lambda: {"owner_id": 1})
    monkeypatch.setattr(message_bus, "publish_event", lambda *_a, **_k: None)
    monkeypatch.setattr(delivery, "_DELIVERED_MESSAGE_IDS", deque(maxlen=256))
    ctx = SimpleNamespace(DRIVE_ROOT=fixture.root, RUNNING={}, append_jsonl=append_jsonl,
                          send_with_budget=message_bus.send_with_budget)
    while not fixture.events.empty():
        event = fixture.events.get_nowait()
        if event.get("type") == "send_message":
            # The owner's chat, never the helper's own room.
            assert event.get("system_type") != NOTICE or (event["chat_id"], event["task_id"]) == (1, "")
            delivery._handle_send_message(event, ctx)
    return [row for row in _history(fixture.root, 1) if row.get("system_type") == NOTICE]


def marks(fixture):
    from ouroboros.gateway.claudexor_accounts import _read_status_payload
    status = _read_status_payload(False)
    assert status["reads"]["accounts"] == "ok"
    evidence = status["profiles"]["routing_evidence"]
    return [row["profile_id"] for row in evidence["capabilities"]], evidence


def results(fixture):
    return [json.loads(row["body_utf8"]) for row in fixture.records
            if row["path"].endswith("/result") and row["status"] == 200]


def correlated(fixture, first=0):
    one, two = fixture.uploads[first], fixture.uploads[first + 1]
    assert {**one, "account": {"mode": "auto"}, "retryOfOperationId": two["retryOfOperationId"]} == two


@pytest.mark.parametrize("real_engine", ["serve", "both", "pin", "no_alternative"], indirect=True)
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
def test_model_capability(real_engine, asynchronous, monkeypatch):
    f = real_engine
    scenario = f.scenario
    kwargs = {"model_account_override": "a"} if scenario == "pin" else {}
    if scenario == "serve":
        answer, usage = call(f, asynchronous, **kwargs)
        assert answer["content"] == "served by b"
        assert len(usage["ledger_attempt_ids"]) == 2
    else:
        with pytest.raises(transport.ClaudexorModelError) as raised:
            call(f, asynchronous, **kwargs)
        error = raised.value
        assert error.code == "model_capability_refused"
        assert "gpt-6.1-sol" in error.display_message and "a:" in error.display_message
        if scenario == "both":
            assert "b:" in error.display_message
            # The same terminal refusal as a burst of helper completions: one row.
            for _ in range(16):
                model_wait.announce_capability_refusal(error, drive_root=f.root)
    expected = 2 if scenario in {"serve", "both"} else 1
    assert len(f.uploads) == len(set(f.ids)) == expected
    if expected == 2:
        correlated(f)
    rows = ledger_rows(f.root)
    assert len(rows) == expected and all(row["state"] == "settled" for row in rows)
    assert all(row["cost_usd"] is None and row["cost_final"] is False for row in rows)
    marked, evidence = marks(f)
    assert marked == (["a", "b"] if scenario == "both" else [] if scenario == "pin" else ["a"])
    if scenario == "serve":
        assert evidence["capabilities"][0]["state"] == "confirmed"
    delivered = notices(f, monkeypatch)
    assert len(delivered) == (0 if scenario == "serve" else 1)
    if delivered:
        assert delivered[0]["role"] == "system" and delivered[0]["text"].endswith("First seen in Helper call.")
        assert ("pinned account a" in delivered[0]["text"]) is (scenario == "pin")


@pytest.mark.parametrize("real_engine", ["generic_serve", "generic_both"], indirect=True)
def test_any_refusal_before_generation(real_engine, monkeypatch):
    """An unrecognized refusal retries once; only service elsewhere marks the account."""
    f = real_engine
    if f.scenario == "generic_serve":
        answer, _ = call(f)
        assert answer["content"] == "served by b"
    else:
        with pytest.raises(transport.ClaudexorModelError) as raised:
            call(f)
        context = raised.value.problem["context"]
        assert raised.value.code == "invalid_request" and context["capabilityKind"] == "unclassified"
        assert context["retryOnOtherAccount"] is False and context["differentialOutcome"] == "inconclusive"
    assert len(f.uploads) == len(set(f.ids)) == 2
    correlated(f)
    marked, evidence = marks(f)
    assert marked == (["a"] if f.scenario == "generic_serve" else [])
    assert sorted(row["profile_id"] for row in evidence["refusals"]) == (
        ["a"] if f.scenario == "generic_serve" else ["a", "b"])
    assert not notices(f, monkeypatch)


@pytest.mark.parametrize("real_engine", ["third"], indirect=True)
def test_double_refusal_leaves_a_third_account(real_engine, monkeypatch):
    f = real_engine
    with pytest.raises(transport.ClaudexorModelError) as raised:
        call(f)
    context = raised.value.problem["context"]
    assert context["differentialOutcome"] == "equivalent_refusal" and context["alternativeRemains"] is True
    assert not notices(f, monkeypatch)
    answer, _ = call(f)
    assert answer["content"] == "served by c"
    assert marks(f)[0] == ["a", "b"]
    assert not notices(f, monkeypatch)


@pytest.mark.parametrize("real_engine", ["returns"], indirect=True)
def test_one_notice_per_episode(real_engine, monkeypatch):
    f = real_engine
    for _ in range(3):  # the same accounts keep refusing: one case, one row
        with pytest.raises(transport.ClaudexorModelError):
            call(f)
    assert len(notices(f, monkeypatch)) == 1
    (f.home / "provider-mode.json").write_text(json.dumps({"serving": ["a"]}), encoding="utf-8")
    answer, _ = call(f)
    assert answer["content"] == "served by a"
    (f.home / "provider-mode.json").write_text(json.dumps({"serving": []}), encoding="utf-8")
    with pytest.raises(transport.ClaudexorModelError):
        call(f)
    # History is cumulative: the returning failure is a new episode, delivered once more.
    assert len(notices(f, monkeypatch)) == 2


@pytest.mark.parametrize("real_engine", ["continuation"], indirect=True)
def test_bound_continuation_and_next_call(real_engine, monkeypatch):
    f = real_engine
    seed, _ = call(f)
    assert seed["nativeContinuation"]["route"]["credentialProfileId"] == "a"
    conversation = [{"role": "user", "content": "hello"}, seed, {"role": "user", "content": "continue"}]
    with pytest.raises(transport.ClaudexorModelError) as raised:
        call(f, messages=conversation)
    assert len(f.uploads) == 2  # seed + one refused operation, no capability resend
    context = raised.value.problem["context"]
    assert context["retryBlockedBy"] == "native_continuation" and context["alternativeRemains"] is True
    assert not notices(f, monkeypatch)
    answer, _ = call(f, messages=conversation)
    assert answer["content"] == "served by b"
    # Mark chooses B. Its exact no-start invalid_continuation triggers the
    # host's EXISTING reset, then B serves the portable text in one new operation.
    assert len(f.uploads) == 4
    assert [row.get("problem", {}).get("code") if row.get("problem") else None for row in results(f)] == [
        None, "model_capability_refused", "invalid_continuation", None]
    assert all("retryOfOperationId" not in request for request in f.uploads)
    assert seed["nativeContinuation"], "the caller's history stays intact"


@pytest.mark.parametrize("real_engine", ["preferred"], indirect=True)
def test_preferred_account_yields_to_mark(real_engine, monkeypatch):
    f = real_engine
    prepare = transport._request

    def preferred(*args, **kwargs):
        # Isolate a SOFT preference from account-bound native history, which
        # deliberately withholds the retry grant (covered above).
        payload = prepare(*args, **kwargs)
        payload["account"] = {"mode": "auto", "preferredProfileId": "a"}
        return payload

    monkeypatch.setattr(transport, "_request", preferred)
    answer, usage = call(f)
    assert answer["content"] == "served by b" and len(usage["ledger_attempt_ids"]) == 2
    first, retry = f.uploads
    assert first["account"] == {"mode": "auto", "preferredProfileId": "a"}
    assert retry["account"] == {"mode": "auto"}
    correlated(f)
    later, _ = call(f)
    assert later["content"] == "served by b"
    assert f.uploads[2]["account"] == first["account"] and "retryOfOperationId" not in f.uploads[2]
    assert len(f.uploads) == len(set(f.ids)) == 3
    facts = results(f)
    assert [row["route"]["credentialProfileId"] for row in facts] == ["a", "b", "b"]
    assert facts[0]["problem"]["context"]["retryOnOtherAccount"] is True
    assert not notices(f, monkeypatch)


@pytest.mark.parametrize("real_engine", ["retry_invalid"], indirect=True)
def test_expired_correlation_keeps_original(real_engine, monkeypatch):
    f = real_engine
    with pytest.raises(transport.ClaudexorModelError) as raised:
        call(f)
    assert len(f.uploads) == 2
    assert raised.value.code == "model_capability_refused"
    assert raised.value.operation_id == f.uploads[1]["retryOfOperationId"]
    terminal = [json.loads(row["body_utf8"]) for row in f.records
                if row["path"].startswith("/v2/model-operations")]
    assert any(row.get("problem", {}).get("code") == "model_retry_invalid"
               and row["dispatch"]["state"] == "not_started" for row in terminal if row.get("problem"))
    assert len(json.loads((f.home / "provider-posts.json").read_text(encoding="utf-8"))) == 1
    context = raised.value.problem["context"]
    assert context["retryOnOtherAccount"] is True
    assert context["nextStep"] not in raised.value.display_message
    assert not notices(f, monkeypatch)


@pytest.mark.parametrize("real_engine", ["older"], indirect=True)
def test_older_engine_never_sends_correlation(real_engine):
    if os.environ.get("CAPABILITY_ENGINE_TREE") != OLDER_TREE:
        pytest.skip("Run separately against the published 4.4.0 engine")
    f = real_engine
    with pytest.raises(transport.ClaudexorModelError) as raised:
        call(f)
    assert raised.value.code == "invalid_request"
    assert "retryOnOtherAccount" not in raised.value.problem["context"]
    assert len(f.uploads) == 1 and "retryOfOperationId" not in f.uploads[0]
    from ouroboros.gateway.claudexor_accounts import _read_status_payload
    status = _read_status_payload(False)
    assert status["reads"]["accounts"] == "ok"
    # The capability query is not advertised: the host does not ask, and shows nothing.
    assert "routing_evidence" not in status["profiles"] and "routing_evidence_read" not in status["profiles"]
