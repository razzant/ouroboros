"""Settings › Telegram › Connection as the owner sees it: the candidate's server, renderer and bundled skill form.

Real outcomes are first made by the shipped journal against the loopback Bot API origin (tests/test_telegram_outbox.py)
on the fixture skill's own state; the server then loads that state, and the page is read, measured and used at a phone
and a desktop width in both themes. A press on Delete copies sends exactly the ids the form showed: a copy kept after
the form was drawn stays. No bot token, Telegram process or network use.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import shutil
import textwrap
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from tests.test_telegram_outbox import _Bridge, _closed_base, _Origin
from tests.test_telegram_outbox_copies import _blob, _Producer
from tests.test_ui_smoke_playwright import direct_server_with_data as direct_server_with_data  # noqa: F401

pytestmark = [pytest.mark.ui_browser]

_NAME = "telegram_connection_form"
_LABEL = "Ремонт доставки Telegram"
_MEASURE = Path(__file__).resolve().parents[1] / "skills" / "ui_design" / "references" / "measure.js"


def _outcomes(state: Path, homes: Path, monkeypatch) -> str:
    """Delivery issues, made by the real journal: refused, unknown, expired, bounded, telemetry, strays. Twelve
    copies are kept, so the oldest (the expired question) is outside the ten the form lists; a file name spells it
    on lines of its own. Returns that oldest copy's id."""
    async def scenario():
        produce = _Producer(monkeypatch, homes)  # the homes the classification reads live on their own data root
        long_reply = produce.reply(" ".join(f"Шаг {index}: проверка доставки." for index in range(330)))
        project = {"chat_id": 1023, "message_origin": {"label": _LABEL}, "transport": {}}
        question = {**project, "task_id": "work-7", "quiz_id": "q7", "question": "Оставить старый кэш?",
                    "options": [{"label": "Да"}, {"label": "Нет"}]}
        report = {**project, "filename": "Квартальный отчёт по доставке сообщений в Telegram — финальная версия (2).pdf",
                  "file_base64": base64.b64encode(_blob(2_400_000, 50)).decode(), "mime": "application/pdf"}
        tricky = {**project, "file_base64": base64.b64encode(b"notes").decode(), "mime": "text/plain"}
        bitmap = produce.photo(b"BM" + _blob(180_000, 51), mime="image/bmp")
        async with _Origin(["ok", "close", "ok"]) as first:
            bridge = _Bridge(state, first.base, TELEGRAM_MIRROR_PROGRESS="on")
            bridge.api.get_runtime_info = lambda: {"data_dir": str(homes)}
            await bridge.publish(*long_reply)  # part 2's answer is lost: 2 of 3 arrived, the app holds it
        async with _Origin() as later:
            bridge.serve(_closed_base())  # nothing listens: the question waits, then a day passes
            await bridge.publish("chat.quiz", question)
            bridge.now += 24 * 3600 + 5
            bridge.serve(later.base)
            await bridge.drain()
            assert later.requests == []
        oldest = next(r["id"] for r in bridge.records() if r.get("held"))
        tricky["filename"] = f"notes\n{oldest}\nstray-0123456789abcdef\n000000000099-0bad0bad.event.json"
        async with _Origin(["403"] * 12 + ["close"]) as refusing:
            bridge.serve(refusing.base)
            for index in range(9):
                await bridge.publish("chat.outbound", {**project, "text": f"Ответ {index + 1} без строки в истории."})
            await bridge.publish("chat.document", tricky)
            await bridge.publish("chat.outbound", {"text": "Проверяю логи.", "is_progress": True, "transport": {}})
            await bridge.publish(*bitmap)
            await bridge.publish("chat.document", report)
            # Kept copies have no bound of their own: the next answer still goes.
            await bridge.publish("chat.outbound", {**project, "text": "Итог проверки без дома в истории."})
            assert refusing.methods()[-1] == "sendMessage" and refusing.answered == 1
        await bridge.stop()
        assert sum(1 for r in bridge.records() if r.get("held")) == 13
        for name in ("000000000090-0bad0bad.event.json", "черновик отчёта 1.event.json"):
            (state / "outbox" / name).write_text(json.dumps({"text": "saved before its record", "transport": {}}),
                                                 encoding="utf-8")
        return oldest

    return asyncio.run(scenario())


def _install(data: Path, homes: Path, monkeypatch) -> str:
    """The fixture skill over a journal of real outcomes; returns the id of the oldest kept copy, which is not listed."""
    from ouroboros.skill_loader import SkillReviewState, compute_content_hash, save_review_state

    state = data / "state" / "skills" / _NAME
    state.parent.mkdir(parents=True, exist_ok=True)
    oldest = _outcomes(state, homes, monkeypatch)
    target = data / "skills" / "external" / _NAME
    target.mkdir(parents=True)
    shutil.copytree(Path(__file__).resolve().parents[1] / "skills" / "telegram", target / "bundled",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (target / "SKILL.md").write_text(textwrap.dedent(f"""\
        ---
        name: {_NAME}
        description: Actual Telegram Connection form over a journal of real outcomes.
        version: 0.1.0
        type: extension
        entry: plugin.py
        permissions: [route, widget, read_settings]
        ---
        # Telegram Connection fixture
        """), encoding="utf-8")
    # The actual register() and routes; no supervised task (so no sender) and no subscription. The journal is read
    # once, as a sender's start reads it; one test-only route keeps a copy after the form was drawn.
    (target / "plugin.py").write_text(textwrap.dedent("""\
        import json
        from starlette.responses import JSONResponse
        from .bundled.plugin import register as register_telegram
        from .bundled.lib.telegram_outbox import outbox_for
        class FormOnlyAPI:
            def __init__(self, api): self.api = api
            def __getattr__(self, name): return getattr(self.api, name)
            def register_supervised_task(self, *args, **kwargs): pass
            def subscribe_event(self, *args, **kwargs): pass
            def register_companion_process(self, *args, **kwargs): pass
        def register(api):
            form = FormOnlyAPI(api)
            register_telegram(form)
            outbox = outbox_for(form)
            outbox._load()
            async def keep_one_more(_request):
                event = {"text": "kept after the form was drawn", "transport": {}, "chat_id": 1}
                record = outbox._new_record("outbound", event, outbox._next_seq(), "refused")
                record.update(copy="only", held=True, payload_bytes=64, finished_at=record["created_at"])
                (outbox.root / f"{record['id']}.event.json").write_text(json.dumps(event))
                (outbox.root / f"{record['id']}.json").write_text(json.dumps(record))
                outbox._ops[record["id"]] = record
                outbox._refresh()
                return JSONResponse({"id": record["id"]})
            async def lose_record(_request):
                from .bundled.lib import telegram_outbox as journal
                record = outbox._new_record("outbound", {}, outbox._next_seq(), "pending")
                record.update(copy="only")
                outbox._write_payload(record, b'{"text":"Unique reply","transport":{}}')
                unlink, write = journal._unlink, outbox._write
                def unavailable_unlink(path):
                    if path == outbox._payload_path(record):
                        raise PermissionError("fixture payload held")
                    unlink(path)
                async def unavailable_write(_record):
                    raise PermissionError("fixture metadata unavailable")
                journal._unlink, outbox._write = unavailable_unlink, unavailable_write
                try:
                    await outbox._refuse(record, "journal PermissionError")
                finally:
                    journal._unlink, outbox._write = unlink, write
                return JSONResponse({"id": outbox._strays[-1]["id"]})
            api.register_route("test/keep-one-more", handler=keep_one_more, methods=("POST",))
            api.register_route("test/lose-record", handler=lose_record, methods=("POST",))
        """), encoding="utf-8")
    save_review_state(data, _NAME, SkillReviewState(
        status="pass", content_hash=compute_content_hash(target, manifest_entry="plugin.py")))
    return oldest


def test_connection_lists_what_did_not_arrive_and_deletes_exactly_the_copies_it_showed(direct_server_with_data,
                                                                                        monkeypatch, tmp_path):
    from playwright.sync_api import expect, sync_playwright

    fixture = direct_server_with_data
    data = fixture["data_dir"]
    oldest = _install(data, tmp_path / "homes", monkeypatch)
    outbox = data / "state" / "skills" / _NAME / "outbox"
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(data.parent)))
    evidence.mkdir(parents=True, exist_ok=True)
    measure = _MEASURE.read_text(encoding="utf-8")
    report = {}
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)

        def open_form(width, height, scheme, *, initial=True):
            context = browser.new_context(viewport={"width": width, "height": height}, device_scale_factor=2,
                                          color_scheme=scheme)
            page = context.new_page()
            page.set_default_timeout(20000)
            errors = []
            page.on("pageerror", lambda err: errors.append(str(err)))
            page.route("**/*", lambda route: route.continue_() if urlsplit(route.request.url).hostname in
                       ("127.0.0.1", "localhost") else route.abort())
            page.goto(fixture["url"], wait_until="domcontentloaded")
            page.wait_for_function("() => Boolean(window.__ouroSettings)")
            assert page.evaluate("name => window.__ouroSettings.open(`skill/${name}`)", _NAME) is not False
            connection = page.locator(f'[data-extension-settings-form][data-skill="{_NAME}"][data-route="connection"]')
            copies = page.locator(f'[data-extension-settings-form][data-skill="{_NAME}"][data-route="connection/copies"]')
            expect(connection.locator('[name="delivery_status"]')).to_contain_text("recent delivery issues (see Delivery issues)")
            expect(copies.locator('[name="not_delivered"]')).to_contain_text("Refused by Telegram")
            # What they take leads the list: thirteen kept copies and two stray files, mostly the report's megabytes.
            took = copies.locator('[name="kept_copies"]').text_content().splitlines()[0]
            if initial:
                expect(copies.locator('[name="kept_copies"]')).to_contain_text("0bad0bad.event.json")
                assert re.fullmatch(r"15 kept copies take \d+\.\d MB on this computer\.", took), took
            return context, page, connection, copies, errors

        page = browser.new_page()
        page.route("**/*", lambda route: route.continue_() if urlsplit(route.request.url).hostname in
                   ("127.0.0.1", "localhost") else route.abort())
        page.goto(fixture["url"], wait_until="domcontentloaded")
        toggled = page.evaluate("""async name => {
            const r = await fetch(`/api/skills/${name}/toggle`, {method:'POST',
                headers:{'Content-Type':'application/json'}, body:JSON.stringify({enabled:true})});
            return {status:r.status, body:await r.json()};
        }""", _NAME)
        assert toggled["status"] == 200 and toggled["body"].get("enabled") is True, toggled
        page.close()
        for width, height, scheme in ((390, 844, "light"), (390, 844, "dark"), (1440, 1000, "light"),
                                      (1440, 1000, "dark")):
            context, page, connection, copies, errors = open_form(width, height, scheme)
            tag = f"{width}-{scheme}"
            page.screenshot(path=str(evidence / f"connection-{tag}-screen.png"))
            connection.screenshot(path=str(evidence / f"connection-{tag}-status.png"))
            copies.screenshot(path=str(evidence / f"connection-{tag}-copies.png"))
            facts = page.evaluate(f"({measure})({{subject: '[data-route=\"connection/copies\"]'}})")
            geometry = page.evaluate("""() => [...document.querySelectorAll('[data-skill] [name="delivery_status"],'
                + '[data-skill] [name="not_delivered"], [data-skill] [name="kept_copies"], [data-skill] [name="delete_ids"],'
                + '[data-skill] [name="confirm_delete"],'
                + '[data-skill] button[type=submit]')]
                .map(el => { const b = el.getBoundingClientRect(); return {name: el.getAttribute('name') || el.textContent.trim(),
                  x: Math.round(b.left), w: Math.round(b.width), h: Math.round(b.height), scrollW: el.scrollWidth,
                  clientW: el.clientWidth, scrollH: el.scrollHeight, clientH: el.clientHeight}; })""")
            report[tag] = {"findings": facts["findings"], "counts": facts["facts"].get("counts"), "geometry": geometry,
                           "page_errors": errors}
            assert not errors, errors
            assert all(row["scrollW"] <= row["clientW"] + 1 for row in geometry if row["name"] in
                       {"delivery_status", "not_delivered", "kept_copies", "delete_ids"}), geometry  # never sideways
            context.close()
        # The owner deletes on a phone: first without the tick, then with it, after another copy was kept.
        context, page, connection, copies, errors = open_form(390, 844, "light")
        shown = copies.locator('[name="delete_ids"]').text_content()
        listed = shown.splitlines()
        assert len(listed) == 12 and oldest not in listed, shown  # the ten newest kept copies and two stray files
        display = copies.locator('[name="kept_copies"]').text_content()
        assert f"notes\\u000a{oldest}\\u000a" in display and oldest not in display.splitlines()  # one line
        before = sorted(path.name for path in outbox.iterdir())
        copies.get_by_role("button", name="Delete copies").click()
        expect(copies.locator("[data-extension-settings-status]")).to_contain_text("Tick “Delete the kept copies")
        assert sorted(path.name for path in outbox.iterdir()) == before
        newer = page.evaluate(f"""async () => (await (await fetch('/api/extensions/{_NAME}/test/keep-one-more',
            {{method: 'POST'}})).json()).id""")
        sent = []
        page.on("request", lambda request: sent.append(request.post_data) if request.method == "POST"
                and request.url.endswith("/connection/copies") else None)
        copies.locator('[name="confirm_delete"]').check()
        copies.screenshot(path=str(evidence / "connection-390-light-copies-ticked.png"))
        copies.get_by_role("button", name="Delete copies").click()
        expect(copies.locator("[data-extension-settings-status]")).to_contain_text("Deleted")
        body = json.loads(sent[-1])
        assert body["delete_ids"] == shown and body["confirm_delete"] is True  # exactly the shown <output>
        result = copies.locator("[data-extension-settings-status]").text_content()
        assert result == (f"Deleted {len(listed)} kept copies. 4 other kept copies stay; open this page again to "
                          "delete them."), result
        expect(copies.locator('[name="kept_copies"]')).to_contain_text(newer)
        expect(copies.locator('[name="delete_ids"]')).to_have_text("")  # disarmed: a second press deletes nothing
        after = sorted(path.name for path in outbox.iterdir())
        assert f"{newer}.event.json" in after and f"{oldest}.event.json" in after  # never listed: both stay
        assert not set(after) & {f"{item}.event.json" for item in listed if not item.startswith("stray-")}
        assert "000000000090-0bad0bad.event.json" not in after and "черновик отчёта 1.event.json" not in after
        expect(copies.locator('[name="not_delivered"]')).to_contain_text("Its kept copy was deleted.")
        copies.get_by_role("button", name="Delete copies").click()  # the tick stayed
        expect(copies.locator("[data-extension-settings-status]")).to_contain_text("Nothing to delete.")
        assert sorted(path.name for path in outbox.iterdir()) == after
        page.screenshot(path=str(evidence / "connection-390-light-after-delete.png"))
        copies.screenshot(path=str(evidence / "connection-390-light-copies-after-delete.png"))
        # A refusal whose record could not be saved lives only in this load's outcome list. Delete its orphan
        # through the same form: the visible outcome must no longer claim that its only copy is kept.
        lost = page.evaluate(f"""async () => (await (await fetch('/api/extensions/{_NAME}/test/lose-record',
            {{method: 'POST'}})).json()).id""")
        context.close()
        context, page, connection, copies, errors = open_form(390, 844, "light", initial=False)
        expect(copies.locator('[name="delete_ids"]')).to_contain_text(lost)
        expect(copies.locator('[name="not_delivered"]')).to_contain_text("Only copy kept on this computer")
        copies.locator('[name="confirm_delete"]').check()
        copies.get_by_role("button", name="Delete copies").click()
        expect(copies.locator("[data-extension-settings-status]")).to_contain_text("Deleted")
        issues = copies.locator('[name="not_delivered"]')
        expect(issues).to_contain_text("Not sent: the outbox couldn't save it. Its kept copy was deleted.")
        expect(issues).not_to_contain_text("Only copy kept on this computer")
        expect(copies.locator('[name="kept_copies"]')).to_contain_text("No copies are kept.")
        assert not errors
        copies.screenshot(path=str(evidence / "connection-390-light-lost-deleted.png"))
        context.close()
        browser.close()
    (evidence / "connection-geometry.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
