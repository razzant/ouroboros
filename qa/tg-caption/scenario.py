"""One synthetic private Telegram conversation on the candidate's real Main page."""

from contextlib import ExitStack
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile
import threading
import time
import traceback


A, B, REPLY, AGAIN = 601, 602, 603, 604
MIB = 1024 * 1024
EXPECTED_LIMIT = 10 * MIB
LARGE_SIZES = (58 * MIB, 64 * MIB)
SAME, SHORT = "same-clip.mp4", "short-marker.webm"
CAPTION = "QA-F14: Is the copper marker visible?"
FOLLOW = "QA-R29: This short clip shows the green mark."
DRAFT = "QA draft: the short clip has not arrived yet."
FINAL = "The short clip arrived. The original clip exceeded the download limit."
WEBM_SHA = "62cd50ac01969eaa61c2bd78c9c35a125293c28554df9441a72ac512a379431f"
HOST = "[Host fact: attachment delivery; not the owner's words]"
OWNER = "[Message from my human]: "
BUBBLES = "#chat-messages .chat-bubble.user.has-attachments"


def check_guest(candidate):
    assert re.fullmatch(r"[0-9a-f]{40}", candidate)
    assert sys.platform == "linux" and os.getuid() != 0
    assert sorted(p.name for p in Path("/sys/class/net").iterdir()) == ["lo"]
    assert not Path("/proc/net/route").read_text().splitlines()[1:]
    status = dict(line.split(":\t", 1) for line in Path("/proc/self/status").read_text().splitlines() if ":\t" in line)
    assert all(int(status[k], 16) == 0 for k in ("CapPrm", "CapEff", "CapBnd", "CapAmb"))
    assert status["NoNewPrivs"] == "1" and status["Seccomp"] == "2"
    assert not Path("/var/run/docker.sock").exists()
    check_candidate(candidate)


def check_candidate(candidate):
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    assert head == candidate
    subprocess.run(["git", "diff", "--quiet", "HEAD", "--"], check=True)


def sha(blob):
    return sha256(blob).hexdigest()


def inputs(media):
    def update(mid, **fields):
        return media._update(mid, **{"from": {"id": media.OWNER_CHAT, "first_name": "Fixture Sender", "is_bot": False}},
                             date=int(time.time()), **fields)

    def large(tag, size):
        return {"file_id": "qa-fetch-" + tag, "file_unique_id": "qa-source-" + tag,
                "file_name": SAME, "mime_type": "video/mp4", "file_size": size, "width": 320, "height": 240, "duration": 3}

    first = update(A, caption=CAPTION, video=large("copper", LARGE_SIZES[0]))
    bare = update(B, video=large("granite", LARGE_SIZES[1]))
    short = {"file_id": "qa-short", "file_unique_id": "qa-source-short", "file_name": SHORT,
             "mime_type": "video/webm", "file_size": len(media._VP8_CLIP),
             "width": 16, "height": 16, "duration": 3}
    reply = update(REPLY, caption=FOLLOW, reply_to_message=first["message"], video=short)
    replay = {"update_id": AGAIN, "message": bare["message"]}
    return first, bare, reply, replay


def check_rows(rows, keys, chat, note):
    from ouroboros.chat_chain import format_address, row_address

    assert [r["client_message_id"] for r in rows] == keys
    first, bare, reply = rows
    facts = []
    for index, (row, words) in enumerate(zip(rows, (CAPTION, "", FOLLOW))):
        assert (row["chat_id"], row["user_id"], row["source"]) == (chat, chat, "skill:telegram")
        placeholder = row.get("text_placeholder") is True
        assert ("" if placeholder else row.get("text") or "") == words
        assert not placeholder or row is bare
        assert bool(row.get("reply_to")) == (row is reply)
        if row is reply:
            continue
        fact, = row["attachments"]
        assert fact["name"] == SAME and fact["size"] == LARGE_SIZES[index]
        assert fact.get("unavailable") == "too_large" and not fact.get("upload")
        assert fact.get("limit_bytes", EXPECTED_LIMIT) == EXPECTED_LIMIT
        facts.append(fact)
        note(f"ROW {keys[index]}  {format_address(row_address(row))}  words={bool(words)}  refusal=too_large  size={fact['size']}")
    ref = reply["reply_to"]
    assert ref["address"] == format_address(row_address(first)) and ref["snippet"] == CAPTION
    assert not ref.get("unresolved") and (ref["role"], ref["kind"]) == ("user", "message")
    upload, = reply["attachments"]
    assert (upload["name"], upload["kind"], upload["mime"], upload["sha256"]) == (SHORT, "video", "video/webm", WEBM_SHA)
    note(f"REPLY {REPLY} -> {A}  {ref['address']}  exact caption; WebM sha256={WEBM_SHA}")
    return ref, facts, upload


def check_staged_reply(oracle, turn, key, ref, upload, stored, size, bare_key, note):
    from ouroboros.artifacts import task_artifact_dir_path
    from ouroboros.owner_mailbox import KIND_OWNER_TEXT, drain_owner_entries
    from supervisor.queue_timeouts import _task_drive_for_task

    data = oracle.data_root.resolve()
    task = oracle.running_row(turn).get("task") or {}
    assert task.get("_is_direct_chat"), "no published direct turn for the Reply"
    drive = _task_drive_for_task(task, turn, canonical=data).resolve()
    assert drive.is_relative_to(data), "direct turn drive is outside the isolated guest data root"
    status = {}
    pending = drain_owner_entries(drive, turn, _read_status=status, kinds={KIND_OWNER_TEXT})
    assert status.get("complete"), "mailbox observation incomplete"
    entry, = [row for row in pending if row.get("client_message_id") == key]
    for row in pending:
        if row.get("client_message_id") == bare_key:
            assert not row.get("reply_reference") and not row.get("text")
    assert entry["text"].splitlines()[0] == FOLLOW
    assert entry["reply_reference"]["address"] == ref["address"]
    entry_file, = entry["attachment_manifest"]
    assert (entry_file["status"], entry_file["root"], entry_file["label"]) == ("staged", "artifact_store", SHORT)
    assert (entry_file["mime"], entry_file["size"], entry_file["sha256"]) == ("video/webm", size, WEBM_SHA)
    rel = PurePosixPath(entry_file["relpath"])
    assert len(rel.parts) == 2 and rel.parts[0] == "attachments" and rel.name != ".."
    artifact_root = task_artifact_dir_path(drive, turn).resolve()
    staged = artifact_root.joinpath(*rel.parts)
    assert artifact_root.is_relative_to(drive) and staged.resolve(strict=True) == staged
    assert Path(entry_file["abs_path"]) == staged and not staged.samefile(stored)
    assert staged.stat().st_size == size and sha(staged.read_bytes()) == upload["sha256"] == WEBM_SHA
    note(f"MAILBOX  Reply staged as a separate artifact of held turn={turn}  size={size}  sha256={WEBM_SHA}")
    return rel.as_posix()


def user_messages(body):
    from tests.system_e2e.harness import message_text

    return [message_text(m) for m in body.get("messages", []) if m.get("role") == "user"]


def refusal_block(fact):
    return (f"{HOST}\n[ATTACHMENTS]\n- {SAME}: not received (reason=too_large, "
            f"mime=video/mp4, size={fact['size']}, limit_bytes={EXPECTED_LIMIT}, ordinal=0)\n[END_ATTACHMENTS]")


def has_staged_reply(msg, relpath):
    lines = msg.splitlines()
    return (bool(lines) and lines[0] == OWNER + FOLLOW
            and any(line.startswith(f"- {SHORT} (video/webm): ") and relpath in line
                    and "status=staged" in line for line in lines))


def quote_matches(msg, ref, fact):
    header, sep, content = msg.partition("\n")
    return (sep and header.startswith("[Reply source from ") and header.endswith(f" at {ref['address']}]")
            and content == CAPTION + "\n" + refusal_block(fact))


def reply_evidence(messages, ref, fact, relpath):
    replies = [msg for msg in messages if msg.startswith(OWNER + FOLLOW)]
    quotes = [msg for msg in messages if msg.startswith("[Reply source from ")]
    return (len(replies) == len(quotes) == 1 and has_staged_reply(replies[0], relpath)
            and HOST not in replies[0] and quote_matches(quotes[0], ref, fact))


def check_posts(posts, owner, released, ref, facts, relpath, note):
    rounds = [p for p in posts if p["body"].get("tools")]
    assert len(rounds) == 2 and [p["body"] for p in rounds] == owner.turn_calls and not owner.other_calls
    assert all(p["local"] and p["path"] == "/v1/chat/completions" for p in posts)
    first, later = rounds
    assert first["at"] < released <= later["at"]
    opening = user_messages(first["body"])
    source, = [u for u in opening if u.startswith((CAPTION, OWNER + CAPTION))]
    a_fact, b_fact = map(refusal_block, facts)
    assert source.splitlines()[0] in (CAPTION, OWNER + CAPTION)
    assert ("\n\n" + a_fact) in source and source.count(a_fact) == 1
    assert all(FOLLOW not in u and b_fact not in u for u in opening)
    arrived = user_messages(later["body"])
    assert reply_evidence(arrived, ref, facts[0], relpath)
    assert not any(re.match(r"^\[Message from my human\]: *\(attachment", u, re.I) for u in arrived)
    bare, = [u for u in arrived if b_fact in u]
    assert bare == b_fact
    note(f"POST 1  sha256={first['sha256']}  A caption and attributed refusal")
    note(f"POST 2  sha256={later['sha256']}  B host-only frame, exact A quote and staged WebM")


def await_send_ack(bot, chat, turn, oracle, note):
    from tests.system_e2e.harness import wait_until
    from tests.test_main_followup_browser import _turn_ids

    def confirmed():
        sent = bot.seen("sendMessage")
        return sent[0]["provider_id"] if (len(sent) == 1 and sent[0].get("status") == 200
                and sent[0].get("chat") == str(chat) and sent[0].get("text") == FINAL
                and not _turn_ids(oracle)) else None

    provider_id = wait_until(confirmed, 45)
    assert provider_id, "no single acknowledged ordinary answer after direct turn settlement"
    note(f"ACK  Bot API accepted message_id={provider_id} for settled turn={turn}")
    return provider_id


def capture(page, out, stage, note):
    from tests import test_attachment_consumer_composition as media

    boxes = page.locator(BUBBLES)
    video = boxes.nth(2).locator("video")
    video.wait_for(state="attached", timeout=30000)
    page.wait_for_function("v => v.error || v.readyState >= 1", arg=video.element_handle(), timeout=20000)
    assert boxes.count() == 3
    shapes = []
    for index, (tag, words) in enumerate(zip(("a", "b", "reply"), (CAPTION, "", FOLLOW))):
        box = boxes.nth(index)
        shape = box.evaluate("""b => ({ caption: b.querySelector(':scope > .message')?.textContent || '',
            files: [...b.querySelectorAll('.chat-file-card')].map(card => ({
                name: card.querySelector('.chat-file-name').textContent,
                reason: card.querySelector('.chat-file-meta').textContent, disabled: card.disabled,
            })), players: b.querySelectorAll('video').length,
            quote: b.querySelector('.chat-quote[data-mode="reply"] .chat-quote-main')?.getAttribute('aria-label'),
            tail: b.textContent.includes('[Attached file:') })""")
        box.screenshot(path=out / f"{stage}-{tag}.png", timeout=6000, animations="disabled")
        assert shape["caption"] == words and not shape["tail"]
        if index < 2:
            card, = shape["files"]
            assert card["name"] == SAME and card["disabled"] and shape["players"] == 0
            assert re.search(r"limit|too large|exceeds", card["reason"], re.I), "the visible card must show the refusal reason"
        else:
            assert shape["players"] == 1 and not shape["files"] and CAPTION in (shape["quote"] or "")
            assert video.evaluate("v => !v.error")
            media._play_video_and_audio_split(page, box, stage)
        shapes.append(shape)
        note(f"UI {stage} {tag}  authored caption={bool(words)}  disabled refusal={index < 2}  video={shape['players']}")
    return shapes


def capture_answer(page, out, stage):
    answer = page.locator("#chat-messages .chat-bubble.assistant").filter(has=page.get_by_text(FINAL, exact=True))
    answer.wait_for(timeout=20000)
    assert answer.count() == 1 and not page.get_by_text(DRAFT, exact=True).count()
    answer.screenshot(path=out / f"{stage}-answer.png", animations="disabled")


def conversation(out, data, candidate, note):
    from playwright.sync_api import sync_playwright
    from ouroboros.install_identity import acting_as
    from ouroboros.subagent_messages import is_task_card_message
    from tests import test_attachment_consumer_composition as media
    from tests.system_e2e.harness import ArtifactOracle, ModelGate, ScriptedStubModel, wait_durable_result, wait_until
    from tests.test_main_followup_browser import _Owner, _turn_ids
    from loopback import BOT_ID, CREATE_OR_EDIT, Bot, observe_posts, telegram_server

    assert sha(media._VP8_CLIP) == WEBM_SHA
    chat = media.OWNER_CHAT
    updates = inputs(media)
    first, bare, reply, replay = updates
    keys = [f"tg:{BOT_ID}:{chat}:{mid}" for mid in (A, B, REPLY)]
    gate = ModelGate(lambda b: b.get("tools") and any(CAPTION in u for u in user_messages(b)), timeout=240)
    ref, facts, relpath = None, None, None

    def after_mail(body):
        users = user_messages(body)
        complete = (ref is not None and relpath is not None
                    and reply_evidence(users, ref, facts[0], relpath)
                    and [u for u in users if refusal_block(facts[1]) in u] == [refusal_block(facts[1])])
        return {"final": FINAL if complete else "QA: missing input"}

    owner = _Owner(CAPTION, [lambda _: {"final": DRAFT}, after_mail])
    model = ScriptedStubModel([owner] * 8, gate=gate, final_answer="QA clip")
    posts = observe_posts(model, note)
    state = data / "state" / "skills" / "telegram"
    bot = Bot(A, {"qa-short": ("videos/" + SHORT, media._VP8_CLIP)}, state, note)
    oracle = ArtifactOracle(data)

    def inbound():
        return [r for r in media._rows(data) if r.get("direction") == "in"]

    def deliveries():
        return [r for r in oracle._jsonl("logs/chat_annotations.jsonl") if r.get("action") == "mailbox_delivery"]

    with ExitStack() as stack:
        stack.enter_context(bot.listening())
        stack.enter_context(model)
        server = stack.enter_context(telegram_server(data, bot, model, chat, candidate, note))
        browser = stack.enter_context(sync_playwright()).chromium.launch()
        stack.callback(browser.close)
        stack.callback(gate.release.set)
        page = browser.new_page(viewport={"width": 1120, "height": 780}, locale="en-US", reduced_motion="reduce")
        page.set_default_timeout(12000)
        media._ready(page, server.base_url)
        assert not inbound()
        bot.offer(first)
        assert gate.arrived.wait(120)
        turn, = wait_until(lambda: _turn_ids(oracle), 25) or []
        assert [r["client_message_id"] for r in inbound()] == keys[:1]
        assert not bot.seen(*CREATE_OR_EDIT)
        note("HELD  socket POST for A is in flight; one direct turn, no reply")
        bot.offer(bare, reply)
        assert wait_until(lambda: len(inbound()) >= 3 and bot.acknowledged(REPLY)
                          and any(r.get("client_message_id") == keys[2] for r in deliveries()), 60)
        rows, acks = inbound(), deliveries()
        ref, facts, upload = check_rows(rows, keys, chat, note)
        reply_ack, = [r for r in acks if r.get("client_message_id") == keys[2]]
        assert reply_ack["target"] == turn and reply_ack["status"] == "delivered"
        assert media._uploads(data) == [upload["upload"]]
        stored = data / "uploads" / upload["upload"]
        assert sha(stored.read_bytes()) == WEBM_SHA
        assert gate.held == 1 and not gate.release.is_set() and not gate.timed_out and not owner.turn_calls
        relpath = check_staged_reply(oracle, turn, keys[2], ref, upload, stored, len(media._VP8_CLIP), keys[1], note)
        bot.offer(replay)
        assert wait_until(lambda: bot.acknowledged(AGAIN), 25)
        assert inbound() == rows and deliveries() == acks and sha(stored.read_bytes()) == WEBM_SHA
        note(f"REPLAY  peer served an additional update carrying accepted message {B}, then saw it acknowledged; "
             "canonical rows, deliveries and uploaded bytes did not change")
        live = capture(page, out, "live", note)
        assert _turn_ids(oracle) == [turn] and gate.held == 1 and not gate.timed_out
        assert not bot.seen(*CREATE_OR_EDIT) and not owner.turn_calls
        released = time.monotonic_ns()
        gate.release.set()
        with acting_as(data):
            result = wait_durable_result(oracle, turn, timeout=150)
        assert result["status"] == "completed" and not gate.timed_out
        note("DURABLE  status=completed")
        check_posts(posts, owner, released, ref, facts, relpath, note)
        provider_id = await_send_ack(bot, chat, turn, oracle, note)
        capture_answer(page, out, "live")
        page.reload(wait_until="domcontentloaded", timeout=30000)
        page.wait_for_function(media.WS_OPEN, timeout=30000)
        assert capture(page, out, "reload", note) == live
        capture_answer(page, out, "reload")
        served = page.request.get(server.base_url + "/api/files/download", params={"upload": upload["upload"]}, timeout=10000)
        assert served.status == 200 and served.body() == media._VP8_CLIP
        after = len(bot.seen("getUpdates"))
        assert wait_until(lambda: len(bot.seen("getUpdates")) > after, 15)

    # Inspect the final ledger after server shutdown, when no late delivery can join it.
    assert bot.acknowledged(AGAIN)
    assert not gate.timed_out and gate.matched == 2 and len(owner.turn_calls) == 2 and not owner.other_calls
    assert len([p for p in posts if p["body"].get("tools")]) == 2
    sent, = bot.seen(*CREATE_OR_EDIT)
    assert (sent["method"], sent["verb"], sent["chat"], sent["text"]) == ("sendMessage", "POST", str(chat), FINAL)
    assert sent["at"] >= released and sent["status"] == 200 and sent["provider_id"] == provider_id > 0
    assert [e["file_id"] for e in bot.seen("getFile")] == ["qa-short"]
    download, = bot.seen("file")
    assert (download["verb"], download["status"], download["sha256"]) == ("GET", 200, WEBM_SHA)
    assert download["path"] == "videos/" + SHORT
    assert all(e["status"] == 200 for e in bot.seen())
    assert not {e["method"] for e in bot.seen()} - {
        "getMe", "getChat", "setMyCommands", "getChatMenuButton", "setChatMenuButton",
        "getUpdates", "getFile", "file", "sendChatAction", "sendMessage"}
    assert inbound() == rows and deliveries() == acks and media._uploads(data) == [upload["upload"]]
    ordinary = [r for r in media._rows(data) if r.get("direction") == "out" and r.get("chat_id") == chat
                and not is_task_card_message(r) and r.get("text", "").strip() not in ("", "\u200b")]
    assert [(r.get("task_id"), r["text"]) for r in ordinary] == [(turn, FINAL)]
    note(f"BOT  1 sendMessage after release, turn={turn}, HTTP 200, provider receipt={sent['provider_id']}")
    note(f"ANSWER  {FINAL}")
    note(f"BYTES  one getFile and one GET; SPA download sha256={WEBM_SHA}; WebM played live and after reload")


def main():
    if not __debug__ or len(sys.argv) != 3:
        raise SystemExit("Expected assertions enabled, output directory and full candidate SHA")
    out, candidate = Path(sys.argv[1]).resolve(), sys.argv[2]
    if not out.is_dir() or any(out.iterdir()):
        raise SystemExit("Expected an empty output directory")
    with (out / "observations.txt").open("x", encoding="utf-8", buffering=1) as report:
        report_lock = threading.Lock()

        def note(line):
            with report_lock:
                print(line, file=report, flush=True)

        try:
            check_guest(candidate)
            note(f"BEGIN  product={candidate}  network=loopback  Chromium  scripted model")
            sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path.cwd())]
            with tempfile.TemporaryDirectory(prefix="caption-") as temp:
                conversation(out, Path(temp) / "data", candidate, note)
            check_candidate(candidate)
            note("PASS")
        except Exception as exc:
            # Only reviewed source locations leave the guest. Exception messages and
            # pytest locals can contain a full model request or the server environment.
            frames = [f for f in traceback.extract_tb(exc.__traceback__)
                      if Path(f.filename).resolve().parent == Path(__file__).resolve().parent]
            where = " > ".join(f"{Path(f.filename).name}:{f.lineno}" for f in frames)
            note(f"FAIL  {type(exc).__name__}  {where}")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
