"""Paired Telegram/Places contract. Run with the two frozen checkouts in an isolated test runner."""

import asyncio
import json
import os
from pathlib import Path
import runpy
import shutil
import socket
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

import pytest


@pytest.fixture(scope="module")
def paired(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("paired")
    core = Path(os.environ["OUROBOROS_CORE_QA"]).resolve(strict=True)
    hub = Path(os.environ["OUROBOROS_HUB_QA"]).resolve(strict=True)
    for directory in ("home", "app", "data", "repo"):
        (tmp_path / directory).mkdir()
    env = {
        "PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path / "home"), "PYTHONPATH": str(core),
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
        "OUROBOROS_APP_ROOT": str(tmp_path / "app"), "OUROBOROS_DATA_DIR": str(tmp_path / "data"),
        "OUROBOROS_REPO_DIR": str(tmp_path / "repo"), "OUROBOROS_SETTINGS_PATH": str(tmp_path / "data/settings.json"),
        "OUROBOROS_RUNTIME_MODE": "advanced", "OUROBOROS_HOST_SERVICE_PORT": "1",
    }
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), str(hub)], cwd=tmp_path,
                            env=env, text=True, capture_output=True, timeout=180)
    return SimpleNamespace(root=tmp_path, env=env, result=result)


def test_private_recipient(paired):
    assert paired.result.returncode == 0, paired.result.stdout + paired.result.stderr
    previous = Path(os.environ["OUROBOROS_CORE_PREVIOUS_QA"]).resolve(strict=True)
    check = """import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from ouroboros import presence_runner as runner
assert Path(runner.__file__).resolve().parent.parent == Path(sys.argv[1])
record = json.loads(Path('rollback-observation.json').read_text(encoding='utf-8'))
assert record['result'] and record['metadata']['presence_conditional_reply']['text'] == record['result']
result = runner.presence_result_from_stored(record, 'fixture-promoted-24')
assert (result.outcome, result.text) == ('silent', '')
"""
    subprocess.run([sys.executable, "-I", "-B", "-c", check, str(previous)], check=True, cwd=paired.root,
                   env={**paired.env, "PYTHONPATH": str(previous)}, timeout=30)


def test_disconnected_outbound_place_ends_only_its_unsent_parts(paired):
    # The scenario asserts every step; its bounded record shows that this case ran to its end.
    assert paired.result.returncode == 0, paired.result.stdout + paired.result.stderr
    record = json.loads((paired.root / "disconnect-observation.json").read_text(encoding="utf-8"))
    assert record == {"refusals": [[409, "place_disconnected", "blocked"]] * 2, "confirmed_parts": 1,
                      "same_authorization": "failed", "fresh": "delivered", "confirmation_source_kept": True}


def published_digest(hub, name):
    published = hub / "skills" / name
    reducer = runpy.run_path(str(hub / "scripts" / "release_digest.py"))
    front = reducer["read_manifest"](published)
    hub_digest = reducer["payload_digest"](published, manifest_entry=front["entry"], manifest_scripts=front.get("scripts"))
    entry_digest = json.loads((hub / "entries" / f"{name}.json").read_text(encoding="utf-8"))["digest"]
    assert hub_digest == entry_digest, (hub_digest, entry_digest)
    return hub_digest


def reject_network(event, args):
    if event == "socket.getaddrinfo" or (event == "socket.connect" and args[0].family in
                                          {socket.AF_INET, socket.AF_INET6}):
        raise AssertionError(f"Unexpected external I/O: {event}")


def exercise(hub):
    sys.addaudithook(reject_network)
    import httpx
    from ouroboros import extension_loader, places, presence_admission, safety
    from ouroboros.config import load_settings
    from ouroboros.dialogue_provenance import presence_root_carrier
    from ouroboros.extension_surface_names import extension_surface_name
    from ouroboros.gateway.host_service import create_host_service_app
    from ouroboros.place_reply import prepare_focus_reply, records_only_now
    from ouroboros.presence_runner import PresenceTurnResult, _build_task, build_presence_result_event, presence_result_from_stored
    from ouroboros.task_results import write_task_result
    from ouroboros.skill_loader import SkillReviewState, find_skill, save_enabled, save_review_state, save_skill_grants
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    assert Path(places.__file__).resolve() == Path(os.environ["PYTHONPATH"], "ouroboros", "places.py")
    root = Path.cwd()
    data = root / "data"
    name = "telegram-bot"
    key = "TELEGRAM_PUBLIC_BOT_TOKEN"

    def select(account_key):
        (data / "settings.json").write_text(json.dumps({key: account_key}), encoding="utf-8")

    select("fixture-first")
    source = data / "skills" / "ouroboroshub" / name
    published = hub / "skills" / name
    hub_digest = published_digest(hub, name)
    shutil.copytree(published, source, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    loaded = find_skill(data, name)
    assert loaded and loaded.skill_dir.resolve() == source.resolve()
    assert loaded.content_hash == hub_digest, (loaded.content_hash, hub_digest, loaded.load_error)
    save_enabled(data, name, True)
    save_review_state(data, name, SkillReviewState(status="pass", content_hash=loaded.content_hash))
    save_skill_grants(data, name, [key], content_hash=loaded.content_hash, requested_keys=[key],
                      granted_permissions=["presence"], requested_permissions=["presence"])
    assert extension_loader.reload_all(data, load_settings)[name] is None
    door = extension_surface_name(name, "telegram_send")
    initial = extension_loader.get_tool(door)
    assert initial and initial["content_hash"] == loaded.content_hash and not initial.get("out_of_process")
    assert "default" not in initial["schema"]["properties"]["direct_account_id"]
    plugin = sys.modules[initial["handler"].__module__]
    assert Path(plugin.__file__).read_bytes() == (source / "plugin.py").read_bytes()
    rt = plugin._RUNTIME
    package = type(rt).__module__.rsplit(".", 1)[0]
    events = sys.modules[package + ".events"]
    api = sys.modules[package + ".api"]
    sys.path[:0] = [str(source / "tests"), str(source)]
    from fake_bot_api import Clock, FakeBotApi
    bots = {"fixture-first": FakeBotApi(bot_id=9), "fixture-second": FakeBotApi(bot_id=77)}
    for bot in bots.values():
        for user_id in (55, 57, 58, 59):
            bot.chats[str(user_id)] = {"id": user_id, "type": "private"}
    # The outbox preserves FIFO within a chat and topic. Move only its synthetic clock when a retry is due.
    clock = SimpleNamespace(time=Clock(time.time()), gmtime=time.gmtime, strftime=time.strftime, sleep=time.sleep)
    sys.modules[package + ".custody"].time = sys.modules[package + ".runtime"].time = clock
    rt.client_factory = lambda fixture: api.TelegramClient(fixture, transport=bots[fixture])
    assert not rt.recorded_bot() and all(not b.calls for b in bots.values())

    judgments, next_verdict = [], []
    def model(**kwargs):
        prompt = kwargs["user_prompt"]
        evidence, proposal = prompt.split("Host evidence (provenance is not consent):\n", 1)[1].split("\nProposed tool call:", 1)
        args = json.loads(proposal.split("\nArguments:\n", 1)[1].split("\nIs this safe?", 1)[0])
        judgments.append((json.loads(evidence), args))
        return {"content": json.dumps({"status": next_verdict.pop() if next_verdict else "SAFE",
                                       "reason": "Synthetic assessment"})}, {}
    safety.get_safety_mode = lambda *args, **kwargs: "full"
    safety.get_safety_model = lambda: "fixture/checker"
    safety._resolve_safety_routing = lambda: (False, False, None)
    safety.runtime_catalog_settings = lambda: {}
    safety._safety_model_call = model
    presence_admission.focus_enabled = lambda _: True
    presence_admission.focus_allowance = lambda _: {"status": "available", "remaining_usd": 2, "settled_usd": 0}

    replies, delayed = [], {}
    def runner(*, admission, event, staged_files, **kwargs):
        # A promoted root's actual automatic-reply path. The focus normally speaks with its send tool.
        task = _build_task(admission, event, drive_root=data, staged_files=staged_files)
        ctx = ToolContext(repo_dir=root / "repo", drive_root=data, task_id=task["id"], task_metadata=task["metadata"])
        assert records_only_now(ctx)
        message_id = str(event.message["message_id"])
        if message_id == "24":
            focus = ToolRegistry(repo_dir=root / "repo", drive_root=data)
            focus._ctx.task_id, focus._ctx.task_metadata = task["id"], task["metadata"]
            result = focus.execute_result(door, {"chat_id": "55", "text": "focus-same-chat", "request_id": "focus-same-chat"})
            assert result.status == "ok" and judged("77") == b.place_id
            assert queued("focus-same-chat")["owner_confirmation"]["account_id"] == "77"
        parent = task["id"]
        if message_id in {"24", "25"}:
            ctx.task_id = task["id"] = f"fixture-promoted-{message_id}"
        ctx.task_metadata = presence_root_carrier(task["metadata"])
        task["metadata"], task["_presence_turn"] = ctx.task_metadata, False
        assert not records_only_now(ctx)
        words = f"Reply to fixture {event.source_event_id}"
        allowed, reason = prepare_focus_reply(ctx, words)
        if not allowed:
            assert reason
            assert prepare_focus_reply(ctx, "", quiet=True) == (True, "")
        outcome = ctx._presence_completion["outcome"]
        replies.append((event, judgments[-1], outcome))
        row = build_presence_result_event(task, words if allowed else "", ctx, terminal_origin="model_final", silent=not allowed)
        stored = write_task_result(data, task["id"], "completed", metadata=task["metadata"], chat_id=task["chat_id"],
                                   result=words, terminal_origin="model_final")
        assert stored["result"] == words
        if message_id in {"21", "24", "25"}:  # the three accepted, explicitly confirmed private replies
            assert outcome == "message" and event.direct_reply_version == 2
            frozen = row["conditional_reply"]
            assert frozen == {"outcome": "message", "text": words,
                              "owner_confirmation": ctx._presence_completion["owner_confirmation"]}
            assert frozen == stored["metadata"]["presence_conditional_reply"]
            if message_id == "24":
                # Preserve this accepted synthetic record for the actual previous host reader.
                (root / "rollback-observation.json").write_text(json.dumps({
                    field: stored[field] for field in ("metadata", "status", "terminal_origin", "result")
                }), encoding="utf-8")
            assert (row["outcome"], row["text"], row["message"]) == ("silent", "", "")
            assert (stored["metadata"]["presence_outcome"], stored["metadata"]["presence_result_text"]) == ("silent", "")
            assert "owner_confirmation" not in row and "presence_owner_confirmation" not in stored["metadata"]
        else:
            assert "conditional_reply" not in row and "presence_conditional_reply" not in stored["metadata"]
        if task["id"] != parent:
            assert row["conditional_reply"]["owner_confirmation"]["tool"] == "place_reply"
            delayed[message_id] = (task["id"], row)
            return PresenceTurnResult("deferred", "", parent, work_ref=task["id"],
                                      delivery_reporting_version=event.delivery_reporting_version)
        return presence_result_from_stored(stored, parent, data)

    asgi = httpx.ASGITransport(app=create_host_service_app(data, presence_runner=runner))
    address_reads, work_reads = [], []
    class HostHTTP:
        async def request_json(self, method, url, *, headers, payload, timeout_sec):
            parsed = urlsplit(url)
            target = parsed.path + ("?" + parsed.query if parsed.query else "")
            async with httpx.AsyncClient(transport=asgi, base_url="http://fixture") as client:
                response = await client.request(method, target, headers=headers, json=payload)
            result = response.json()
            if parsed.path == "/presence/place":
                address_reads.append((parse_qs(parsed.query), result, response.status_code))
            if parsed.path.startswith("/presence/work/") and method == "GET":
                work_reads.append((parsed.path.rsplit("/", 1)[-1], response.status_code, result))
            if response.is_error:
                host_module = sys.modules[type(rt.submitter).__module__]
                raise host_module.PresenceHostHTTPError(response.status_code, str(result.get("error")), result)
            return result
    rt.submitter.transport = HostHTTP()
    registry = ToolRegistry(repo_dir=root / "repo", drive_root=data)
    registry._ctx.task_id = "synthetic-owner-work"
    registry._ctx.task_metadata = {"work_kind": "owner"}

    def room(account, chat, state):
        note = "Words for " + account if account else "Shared room note"
        place = places.connect_place(data, connector=name, provider="telegram", account_id=account, conversation_id=chat,
                                     label="Same fixture label", first={"note": note}, wakes=["any_message"])
        places.edit_place(data, place.place_id, "set_state", author="owner", state=state)
        return place
    a, b = room("9", "55", "active"), room("77", "55", "paused")
    group = room("77", "-505", "active")
    def confirm(place, enable=True, *, user="55"):
        places.edit_place(data, place.place_id, "confirm_owner_account" if enable else "revoke_owner_account",
                          author="owner", platform_actor_id=user)
    def propose(label, chat="55", **extra):
        args = {"kind": "message", "chat_id": chat, "text": label, "request_id": label, **extra}
        before = dict(args)
        result = registry.execute_result(door, args)
        assert args == before
        return result
    def queued(label):
        return rt.store.outbox_payload("telegram-send:" + label)
    def receipt(label):
        return rt.store.delivery_receipt("telegram-send:" + label)["state"]
    def judged(account, chat="55", *, owner=True):
        evidence, args = judgments[-1]
        subject = evidence["subject"]["place"]
        if owner:
            assert subject["confirmed_owner_dm"] == {"connector": name, "provider": "telegram",
                                                    "account_id": account, "user_id": chat}
        else:
            assert "confirmed_owner_dm" not in subject
        row, = subject["places"]
        assert row.get("account_id", "") == account and "sends_as" not in row
        assert row["place_state"] == "active" and row["conversation_id"] == args["chat_id"] == chat
        expected = "Words for " + account if account else "Shared room note"
        assert [n["text"] for n in evidence["standing_delegation"]["place_notes"]] == [expected]
        if account:
            assert args["direct_account_id"] == account
        else:
            assert "direct_account_id" not in args
        return row["place_id"]

    async def ordinary_non_owner(first, second):
        # This chat has a connected Place on any bot. An ordinary allowed call needs no private-chat lookup.
        shared = room("", "58", "active")
        assert propose("ordinary-other", "58").status == "ok"
        assert judged("", "58", owner=False) == shared.place_id
        payload = queued("ordinary-other")
        assert payload["kind"] == "message" and "direct_account_id" not in payload
        assert "owner_confirmation" not in payload
        lookups = len(first.methods("getChat"))
        assert await rt.process_one_outbox() and receipt("ordinary-other") == "delivered"
        assert len(first.methods("getChat")) == lookups and first.methods("sendMessage")[-1]["chat_id"] == "58"
        observed = address_reads[-1][1]
        assert (observed["place_id"], observed["state"]) == (shared.place_id, "active")
        assert "owner_confirmation" not in observed

        # Only bot 9's Place names this dialog. Removing its owner confirmation leaves its delivery condition.
        sole = room("9", "57", "active")
        confirm(sole, user="57")
        confirm(sole, False, user="57")
        assert propose("ordinary-sole", "57").status == "ok"
        assert judged("9", "57", owner=False) == sole.place_id
        payload = queued("ordinary-sole")
        assert (payload["kind"], payload["direct_account_id"], payload["direct_reply_user_id"]) == ("direct_send", "9", "57")
        select("fixture-second")
        reads = len(address_reads)
        assert await rt.process_one_outbox() and receipt("ordinary-sole") == "held"
        assert len(address_reads) == reads and not second.methods("getChat") and not second.methods("sendMessage")

        select("fixture-first")
        places.edit_place(data, sole.place_id, "set_state", author="owner", state="paused")
        before = len(judgments)
        assert propose("sole-paused", "57").status == "blocked" and not queued("sole-paused") and len(judgments) == before
        clock.time.advance(6)
        assert await rt.process_one_outbox() and receipt("ordinary-sole") == "held"
        query, place, status = address_reads[-1]
        assert status == 200
        assert (query["account_id"], query["conversation_id"], place["place_id"], place["state"]) == (
            ["9"], ["57"], sole.place_id, "paused")
        assert len(first.methods("getChat")) == lookups and first.methods("sendMessage")[-1]["chat_id"] == "58"
        places.edit_place(data, sole.place_id, "set_state", author="owner", state="active")
        clock.time.advance(6)
        assert await rt.process_one_outbox() and receipt("ordinary-sole") == "delivered"
        assert address_reads[-1][1] == {**place, "state": "active"}
        assert first.methods("getChat")[-1] == {"chat_id": "57"}
        sent = first.methods("sendMessage")[-1]
        assert (sent["chat_id"], sent["text"]) == ("57", "ordinary-sole") and not second.methods("sendMessage")

    def switch_publication(manifest_bytes):
        # Distinct reviewed bytes in the disposable skill store produce distinct authenticated clients.
        nonlocal rt
        (source / "SKILL.md").write_bytes(manifest_bytes)
        candidate = find_skill(data, name)
        digest = candidate.content_hash
        save_review_state(data, name, SkillReviewState(status="pass", content_hash=digest))
        save_skill_grants(data, name, [key], content_hash=digest, requested_keys=[key],
                          granted_permissions=["presence"], requested_permissions=["presence"])
        assert extension_loader.reload_all(data, load_settings)[name] is None
        tool = extension_loader.get_tool(door)
        assert tool["content_hash"] == digest
        rt = sys.modules[tool["handler"].__module__]._RUNTIME
        modules = type(rt).__module__.rsplit(".", 1)[0]
        sys.modules[modules + ".custody"].time = sys.modules[modules + ".runtime"].time = clock
        client = sys.modules[modules + ".api"].TelegramClient
        rt.client_factory = lambda token: client(token, transport=bots[token])
        rt.submitter.transport = HostHTTP()
        return digest

    def recreate_older_registry(binding):
        snapshot = places.place_records(data)
        assert places.places_path(data).is_file()
        assert snapshot.revision == 1 < binding["observed_revision"]
        assert binding["place_id"] not in snapshot.places

    async def deferred_replies(second):
        async def submit(update_id):
            update = {"update_id": update_id, "message": {"message_id": update_id, "date": 1_700_000_000 + update_id,
                      "text": "Deferred fixture", "chat": second.chats["55"],
                      "from": {"id": 55, "is_bot": False, "first_name": "Fixture"}}}
            event = events.parse_telegram_update(update, bot_account_id="77", places=True)
            rt.store.commit_update(update_id, event)
            assert await rt.process_one_inbox()
            ref, row = delayed[str(update_id)]
            frozen = row["conditional_reply"]
            delivery = f"presence:{event.source_event_id}:output:{ref}:{row['output_ref']}"
            return ref, row["output_ref"], frozen["text"], frozen["owner_confirmation"], delivery

        confirm(a)  # The same user is confirmed on both bots when the focus speaks without a pin.
        ref, output_ref, words, binding, delivery = await submit(24)
        assert await rt.process_one_outbox() and receipt("focus-same-chat") == "delivered"
        assert second.methods("sendMessage")[-1]["text"] == "focus-same-chat"
        count, checks = len(second.methods("sendMessage")), len(judgments)
        latest = (published / "SKILL.md").read_bytes()
        declaration = b"direct_recipient: {provider: telegram, confirmation: owner_confirmation}"
        assert latest.count(declaration) == 1
        earlier = latest.replace(declaration, b"direct_recipient: {provider: telegram}")
        assert switch_publication(earlier) != hub_digest
        assert await rt.process_one_work()
        work, status, rejected = work_reads[-1]
        assert (work, status, rejected["code"]) == (ref, 409, "presence_recipient_capability_required")
        assert words not in json.dumps(rejected) and not rt.store.outbox_payload(delivery)
        assert not await rt.process_one_outbox() and len(second.methods("sendMessage")) == count

        assert switch_publication(latest) == hub_digest
        path = places.places_path(data)
        saved = path.rename(path.with_suffix(".saved"))

        async def assert_work_held():
            clock.time.advance(60)
            assert await rt.process_one_work()
            work, status, unavailable = work_reads[-1]
            assert (work, status, unavailable["code"], unavailable["disposition"]) == (ref, 503, "places_unavailable", "retry")
            assert words not in json.dumps(unavailable) and not rt.store.outbox_payload(delivery)
            assert len(second.methods("sendMessage")) == count

        try:
            await assert_work_held()
            assert not path.exists()
            recreate_older_registry(binding)
            await assert_work_held()
        finally:
            saved.replace(path)
        clock.time.advance(60)
        assert await rt.process_one_work()
        work, status, reply = work_reads[-1]
        assert (work, status, reply["delivery_reporting_version"]) == (ref, 200, 1)
        queued_reply = rt.store.outbox_payload(delivery)
        assert reply["owner_confirmation"] == binding == queued_reply["owner_confirmation"]
        assert reply["output_ref"] == output_ref and reply["text"] == words
        assert queued_reply["direct_account_id"] == "77" and queued_reply["text"] == words

        # Both bots have an Active Place and the same confirmed user. The selected reply still belongs to bot 77.
        first = bots["fixture-first"]
        first_count = len(first.methods("sendMessage"))
        select("fixture-first")
        assert await rt.process_one_outbox() and rt.store.delivery_receipt(delivery)["state"] == "held"
        held = rt.store.outbox_payload(delivery)
        assert (held["direct_account_id"], held["text"], held["owner_confirmation"]) == ("77", words, binding)
        assert len(first.methods("sendMessage")) == first_count and len(second.methods("sendMessage")) == count

        select("fixture-second")
        clock.time.advance(6)
        assert await rt.process_one_outbox() and rt.store.delivery_receipt(delivery)["state"] == "delivered"
        sent = second.methods("sendMessage")[-1]
        assert (sent["chat_id"], sent["text"], sent["reply_parameters"]) == ("55", words, {"message_id": 24})
        assert address_reads[-1][1]["owner_confirmation"] == binding
        assert address_reads[-1][1]["place_id"] == binding["delivery_place"]["place_id"] == b.place_id
        clock.time.advance(120)
        assert not await rt.process_one_work() and not await rt.process_one_outbox()
        assert len(judgments) == checks and len(second.methods("sendMessage")) == count + 1
        assert len(first.methods("sendMessage")) == first_count

        ref, _, _, _, delivery = await submit(25)
        confirm(b, False)
        confirm(b)
        checks = len(judgments)
        assert await rt.process_one_work()
        work, status, reply = work_reads[-1]
        assert (work, status, reply["outcome"], reply["text"]) == (ref, 200, "silent", "")
        assert not rt.store.outbox_payload(delivery) and not await rt.process_one_outbox()
        assert len(second.methods("sendMessage")) == count + 1 and len(judgments) == checks
        confirm(a, False)

    def assert_owner_read(binding, status, code="place_owner_confirmation_changed", disposition="rejected"):
        query, answer, actual_status = address_reads[-1]
        assert actual_status == status and json.loads(query["owner_confirmation"][0]) == binding
        assert (query["account_id"], query["conversation_id"]) == ([binding["account_id"]], [binding["conversation_id"]])
        if status == 200:
            assert answer["state"] == "active" and answer["owner_confirmation"] == binding
        elif status == 409:
            assert (answer["ok"], answer["code"], answer["disposition"]) == (False, code, disposition)
        else:
            assert (answer["ok"], answer["code"], answer["disposition"]) == (False, "places_unavailable", "retry")

    def failed_reports(label, confirmed):
        row = rt.store.delivery_receipt("telegram-send:" + label)
        assert row["state"] == "failed" and row["provider_receipt"] == {}
        assert row.get("confirmed_messages", []) == confirmed
        reports = [entry["payload"] for entry in row["history_reports"]]
        *parts, final = reports
        assert [(part["part_id"], part["state"], part["message"]["message_id"]) for part in parts] == [
            (str(index), "delivered", message["message_id"]) for index, message in enumerate(confirmed)]
        assert (final["part_id"], final["state"], final["text"], final["message"]["confirmed_parts"]) == (
            "status", "failed", "", len(confirmed))
        return reports

    async def revoked_after_enqueue(second, target):
        confirm(target, user="59")
        for label in ("revoked-before-read", "reconfirmed-before-read"):
            assert propose(label, "59").status == "ok"
            assert judged("77", "59") == target.place_id
        binding = queued("revoked-before-read")["owner_confirmation"]
        assert binding == queued("reconfirmed-before-read")["owner_confirmation"] == judgments[-1][1]["owner_confirmation"]
        count, checks = len(second.methods("sendMessage")), len(judgments)
        lookups = len(second.methods("getChat"))
        confirm(target, False, user="59")
        assert await rt.process_one_outbox()
        assert_owner_read(binding, 409)
        failed_reports("revoked-before-read", [])

        confirm(target, user="59")
        assert propose("fresh-after-revoke", "59").status == "ok"
        assert judged("77", "59") == target.place_id
        fresh = queued("fresh-after-revoke")["owner_confirmation"]
        assert fresh["confirmed_at"] != binding["confirmed_at"]
        assert fresh["observed_revision"] > binding["observed_revision"]
        assert {**fresh, "confirmed_at": binding["confirmed_at"],
                "observed_revision": binding["observed_revision"]} == binding
        assert await rt.process_one_outbox()
        assert_owner_read(binding, 409)
        failed_reports("reconfirmed-before-read", [])
        assert len(second.methods("sendMessage")) == count and receipt("fresh-after-revoke") == "pending"
        assert len(second.methods("getChat")) == lookups
        assert await rt.process_one_outbox() and receipt("fresh-after-revoke") == "delivered"
        assert_owner_read(fresh, 200)
        assert [call["text"] for call in second.methods("sendMessage")[count:]] == ["fresh-after-revoke"]
        assert len(judgments) == checks + 1

    async def unread_then_restored(second):
        label = "queued-during-outage"
        assert propose(label, "59").status == "ok"
        binding = queued(label)["owner_confirmation"]
        assert propose("following-outage", "59").status == "ok"
        count = len(second.methods("sendMessage"))
        path = places.places_path(data)
        saved = path.rename(path.with_suffix(".saved"))

        async def assert_outbox_held():
            clock.time.advance(6)
            assert await rt.process_one_outbox() and receipt(label) == "held"
            assert_owner_read(binding, 503)
            assert not await rt.process_one_outbox() and receipt("following-outage") == "pending"
            assert len(second.methods("sendMessage")) == count

        try:
            await assert_outbox_held()
            assert not path.exists()
            path.write_bytes(b"{")
            await assert_outbox_held()
            path.unlink()
            recreate_older_registry(binding)
            await assert_outbox_held()
        finally:
            saved.replace(path)
        clock.time.advance(6)
        for ready in (label, "following-outage"):
            assert await rt.process_one_outbox() and receipt(ready) == "delivered"
            assert_owner_read(binding, 200)
        assert [call["text"] for call in second.methods("sendMessage")[count:]] == [label, "following-outage"]

    async def revoked_between_parts(second, target):
        label = "multipart-before-revoke"
        assert propose(label, "59", text="a" * 4100, markdown=False).status == "ok"
        row = queued(label)
        first, remaining = row["_rendered_chunks"]
        count, reads = len(second.methods("sendMessage")), len(address_reads)
        post, confirmed = second.post_json, []

        async def revoke_after_first(url, payload, timeout_sec):
            answer = await post(url, payload, timeout_sec)
            if url.endswith("/sendMessage"):
                confirmed.append(answer["result"])
                if len(confirmed) == 1:
                    confirm(target, False, user="59")
                    confirm(target, user="59")
            return answer

        with patch.object(second, "post_json", revoke_after_first):
            assert await rt.process_one_outbox()
        assert len(confirmed) == 1 and first["text"] != remaining["text"]
        assert [call["text"] for call in second.methods("sendMessage")[count:]] == [first["text"]]
        assert [status for _, _, status in address_reads[reads:]] == [200, 200, 409]
        assert_owner_read(row["owner_confirmation"], 409)
        part, _ = failed_reports(label, confirmed)
        assert part["text"] == confirmed[0]["text"] == first["text"]
        assert propose("fresh-after-part", "59").status == "ok"
        assert await rt.process_one_outbox() and receipt("fresh-after-part") == "delivered"
        assert_owner_read(queued("fresh-after-part")["owner_confirmation"], 200)
        while await rt.process_one_delivery_report():
            pass
        entries = rt.store.delivery_receipt("telegram-send:" + label)["history_reports"]
        assert len(entries) == 2 and all(entry["acked"] for entry in entries)
        clock.time.advance(120)
        assert not await rt.process_one_outbox()
        assert [call["text"] for call in second.methods("sendMessage")[count:]] == [first["text"], "fresh-after-part"]
        failed_reports(label, confirmed)

    async def disconnected_outbound(second, keeper):
        # The owner confirmation stays on another connected Place; only the send's exact outbound Place goes away.
        confirm(keeper, user="57")
        outbound = room("77", "57", "active")
        label = "outbound-parts"
        assert propose(label, "57", text="b" * 4100, markdown=False).status == "ok"
        assert judged("77", "57") == outbound.place_id
        row = queued(label)
        binding = row["owner_confirmation"]
        assert (binding["place_id"], binding["connected_revision"]) == (keeper.place_id, keeper.connected_revision)
        assert (binding["delivery_place"]["place_id"], binding["delivery_place"]["connected_revision"]) == (
            outbound.place_id, outbound.connected_revision)
        assert propose("outbound-next", "57").status == "ok"  # queued under the same, soon invalid, authorization
        assert queued("outbound-next")["owner_confirmation"] == binding
        first, remaining = row["_rendered_chunks"]
        count, reads = len(second.methods("sendMessage")), len(address_reads)
        post, confirmed = second.post_json, []

        async def disconnect_after_first(url, payload, timeout_sec):
            answer = await post(url, payload, timeout_sec)
            if url.endswith("/sendMessage"):
                confirmed.append(answer["result"])
                if len(confirmed) == 1:
                    places.edit_place(data, outbound.place_id, "disconnect", author="owner")
            return answer

        with patch.object(second, "post_json", disconnect_after_first):
            assert await rt.process_one_outbox()
        assert len(confirmed) == 1 and first["text"] != remaining["text"]
        assert [call["text"] for call in second.methods("sendMessage")[count:]] == [first["text"]]
        assert [status for _, _, status in address_reads[reads:]] == [200, 200, 409]
        assert_owner_read(binding, 409, "place_disconnected", "blocked")
        part, _ = failed_reports(label, confirmed)
        assert part["text"] == confirmed[0]["text"] == first["text"]
        assert "original outbound Place was disconnected" in rt.store.delivery_receipt("telegram-send:" + label)["error"]
        assert await rt.process_one_outbox()  # the row queued under the same authorization ends unsent
        assert_owner_read(binding, 409, "place_disconnected", "blocked")
        failed_reports("outbound-next", [])
        assert len(second.methods("sendMessage")) == count + 1

        # Fresh work is admitted for the reconnected Place under its own then-active binding.
        again = room("77", "57", "active")
        assert again.place_id == outbound.place_id and again.connected_revision > outbound.connected_revision
        assert propose("outbound-fresh", "57").status == "ok"
        assert judged("77", "57") == again.place_id
        fresh = queued("outbound-fresh")["owner_confirmation"]
        assert fresh["delivery_place"] == {**binding["delivery_place"], "connected_revision": again.connected_revision}
        assert fresh["observed_revision"] > binding["observed_revision"]
        assert {**fresh, "observed_revision": binding["observed_revision"],
                "delivery_place": binding["delivery_place"]} == binding
        assert await rt.process_one_outbox() and receipt("outbound-fresh") == "delivered"
        assert_owner_read(fresh, 200)
        assert address_reads[-1][1]["place_id"] == again.place_id
        assert [call["text"] for call in second.methods("sendMessage")[count:]] == [first["text"], "outbound-fresh"]
        while await rt.process_one_delivery_report():
            pass
        entries = rt.store.delivery_receipt("telegram-send:" + label)["history_reports"]
        assert len(entries) == 2 and all(entry["acked"] for entry in entries)
        clock.time.advance(120)
        assert not await rt.process_one_outbox()
        failed_reports(label, confirmed)
        (root / "disconnect-observation.json").write_text(json.dumps({
            "refusals": [[status, answer["code"], answer["disposition"]]
                         for _, answer, status in address_reads[reads:] if status != 200],
            "confirmed_parts": len(rt.store.delivery_receipt("telegram-send:" + label)["confirmed_messages"]),
            "same_authorization": receipt("outbound-next"), "fresh": receipt("outbound-fresh"),
            "confirmation_source_kept": fresh["place_id"] == keeper.place_id != again.place_id,
        }), encoding="utf-8")

    async def scenario():
        first, second = bots.values()
        confirm(a)
        assert propose("cold").status == "ok"
        judged("9")
        assert queued("cold")["direct_account_id"] == "9" and not first.calls
        assert await rt.process_one_outbox() and receipt("cold") == "delivered"
        assert address_reads[-1][1]["owner_confirmation"] == queued("cold")["owner_confirmation"]
        assert [method for method, _ in first.calls] == ["deleteWebhook", "getMe", "getChat", "sendMessage"]
        assert address_reads[-1][0]["account_id"] == ["9"] and first.methods("sendMessage")[0]["chat_id"] == "55"
        await ordinary_non_owner(first, second)
        places.edit_place(data, b.place_id, "set_state", author="owner", state="active")
        places.edit_place(data, a.place_id, "set_state", author="owner", state="paused")
        before = len(judgments)
        assert propose("paused").status == "blocked" and not queued("paused") and len(judgments) == before
        places.edit_place(data, a.place_id, "set_state", author="owner", state="active")
        assert propose("in-flight").status == "ok"
        select("fixture-second")
        assert await rt.process_one_outbox() and receipt("in-flight") == "held" and not second.methods("sendMessage")
        confirm(b)
        before = len(judgments)
        result = propose("ambiguous-ordinary")
        assert result.status == "blocked" and "exact account pin" in result.text
        assert len(judgments) == before and not queued("ambiguous-ordinary")
        assert propose("explicit-second", direct_account_id="77").status == "ok"
        assert judged("77") == b.place_id and queued("explicit-second")["owner_confirmation"]["account_id"] == "77"
        assert await rt.process_one_outbox() and receipt("explicit-second") == "delivered"
        assert second.methods("sendMessage")[-1]["text"] == "explicit-second" and receipt("in-flight") == "held"
        select("fixture-first")
        clock.time.advance(6)
        assert await rt.process_one_outbox() and receipt("in-flight") == "delivered"
        assert first.methods("sendMessage")[-1]["text"] == "in-flight"
        select("fixture-second")
        confirm(a, False)
        assert propose("rotated").status == "ok"
        judged("77")
        assert await rt.process_one_outbox() and receipt("rotated") == "delivered"
        assert address_reads[-1][0]["account_id"] == ["77"] and second.methods("sendMessage")[-1]["chat_id"] == "55"
        assert extension_loader.get_tool(door)["handler"] is initial["handler"]
        next_verdict.append("DANGEROUS")
        assert propose("refused").status != "ok" and not queued("refused")
        judged("77")
        # An explicit private pin with no private Chat answer never reaches a sending API.
        for label, answer in (("group-type", {"id": 55, "type": "group"}), ("other-id", {"id": 56, "type": "private"})):
            count = len(second.methods("sendMessage"))
            second.chats["55"] = answer
            assert propose(label).status == "ok" and await rt.process_one_outbox()
            assert receipt(label) == "held" and len(second.methods("sendMessage")) == count
            second.chats["55"] = {"id": 55, "type": "private"}
            clock.time.advance(6)
            assert await rt.process_one_outbox() and receipt(label) == "delivered"
            assert second.methods("sendMessage")[-1]["text"] == label
        confirm(b, False)
        for label, chat in (("revoked", "55"), ("other", "56")):
            next_verdict.append("SUSPICIOUS")
            assert propose(label, chat, direct_account_id="77").status != "ok" and not queued(label)
            assert "confirmed_owner_dm" not in judgments[-1][0]["subject"]["place"]
        confirm(b)
        second.chats["-505"] = {"id": -505, "type": "supergroup", "title": group.label}
        for update_id, chat, author in ((21, "55", 55), (22, "-505", 55), (23, "55", 55)):
            refused = update_id == 23
            if refused:
                next_verdict.append("DANGEROUS")
            message = {"message_id": update_id, "date": 1_700_000_000 + update_id, "text": "Fixture",
                       "chat": second.chats[chat], "from": {"id": author, "is_bot": False, "first_name": "Fixture"}}
            event = events.parse_telegram_update({"update_id": update_id, "message": message}, bot_account_id="77", places=True)
            rt.store.commit_update(update_id, event)
            count = len(second.methods("sendMessage"))
            assert await rt.process_one_inbox()
            for _ in range(200):
                await rt.process_one_work()
                await rt.process_one_outbox()
                state = rt.store.status_snapshot()
                if (len(replies) == update_id - 20 and state["inbox_submitted"] == len(replies)
                        and not state["inbox_retained"] and not state["outbox_waiting"]):
                    break
                await asyncio.sleep(0.05)
                clock.time.advance(6)
            else:
                raise AssertionError(f"Telegram did not settle fixture {update_id}: {state}")
            assert len(second.methods("sendMessage")) == count + int(not refused)
            captured, (evidence, args), outcome = replies[-1]
            assert captured.source_event_id == event.source_event_id and outcome == ("silent" if refused else "message")
            assert evidence["subject"]["tool"] == "place_reply"
            fact = evidence["subject"]["place"].get("confirmed_owner_dm")
            assert (fact == {"connector": name, "provider": "telegram", "account_id": "77", "user_id": "55"}
                    if chat == "55" else fact is None)
            assert args["destination"] == {"transport": "telegram", "account_id": "77", "conversation_id": chat, "thread_id": ""}
            if fact:
                assert args["reply_anchor"]["reply_targets"] == []
                assert all("sends_as" not in row for row in evidence["subject"]["place"]["places"])
            if refused:
                assert not next_verdict
                continue
            sent = second.methods("sendMessage")[-1]
            assert sent["chat_id"] == chat and sent["text"] == args["text"]
            assert str(sent["reply_parameters"]["message_id"]) == str(update_id)
        await deferred_replies(second)
        target = room("77", "59", "active")
        await revoked_after_enqueue(second, target)
        await unread_then_restored(second)
        await revoked_between_parts(second, target)
        await disconnected_outbound(second, target)
        count = len(second.methods("sendMessage"))
        del second.chats["55"]
        assert propose("chat-unavailable").status == "ok" and await rt.process_one_outbox()
        assert receipt("chat-unavailable") != "delivered" and len(second.methods("sendMessage")) == count
        assert second.methods("getChat")[-1] == {"chat_id": "55"}
    asyncio.run(scenario())


if __name__ == "__main__":
    exercise(Path(sys.argv[1]))
