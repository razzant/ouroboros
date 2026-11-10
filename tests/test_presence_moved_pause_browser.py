"""A moved Presence turn's owner Pause, Resume and Restart, pressed in the real SPA (#2566).

The same isolated host, scripted model and Chromium page as ``test_presence_project_activity_browser`` (its
``Consumer``): a real Host event starts the inline author, the owner moves it to a Project over the real route,
then pauses and resumes it from the Project's own work list. Unmoved, the same Pause is the negative control. The
owner's real Restart of this isolated server returns the turn it finds working and keeps the one it finds paused;
only the owner's own Resume moves the work off its Place allowance.
"""
from __future__ import annotations

import contextlib
import http.client
import json
import threading

import pytest

from tests.system_e2e.harness import body_text, wait_until
from tests.test_owner_wait_integration import wait_clone as clone_fixture
from tests.test_presence_contact_browser import TOKEN, _event, _http
from tests.test_presence_project_activity_browser import (
    ACTOR_WORDS, FIRST_ANSWER, PROJECT, RESTART_INLINE, SECOND_ANSWER, Consumer, _actions, _row, _sent_to, _work, kind_of,
)

pytestmark = [pytest.mark.browser]
wait_clone = clone_fixture
_MENU_ITEMS = "body > .task-control-menu .task-control-item"


def _press(c, task_id, label):
    """Press ``label`` in the Project row's task actions, opening the menu again if a screenshot closed it."""
    if not c.page.locator(_MENU_ITEMS).filter(has_text=label).count():
        c.close_menu()
        c.row_menu("project", task_id)
    c.page.locator(_MENU_ITEMS).filter(has_text=label).first.click()


def test_a_moved_turn_pauses_from_its_project_keeps_its_saved_work_and_resumes_in_the_real_spa(wait_clone, tmp_path):
    """The owner's Pause of a moved text turn, pressed in the Project's work list, saves its work under the same id: no
    model call while it waits, Resume offered, Restart not warning it away, the Host's poll pending (never failed).
    The owner's Resume, pressed on the paused row, continues the same author under the same id, which answers once in
    the Project; that Resume, through the owner's own door, moves the work off its Place allowance (the owner's choice of
    2026-11-02). Unmoved, the same Pause is refused and changes nothing (the negative control). The owner's Project words
    to a lane-carried author are B1's delivery, not this walk's."""
    from ouroboros.budget_pause import budget_pause_row
    from ouroboros.owner_pause import fence_closed, read_fence

    with Consumer(wait_clone, tmp_path, "presence-project-pause-browser") as c:
        page, problems, stage, tid = c.page, c.problems, c.facts["stages"], c.tid
        project_chat = int(c.project["chat_id"])
        try:
            c.start_turn()
            assert c.first.arrived.wait(120), "1: the author never reached its first model round"
            refused = _http(c.server.port, "POST", f"/api/tasks/{tid}/pause", {"request_id": "pause-unmoved", "actor": "owner"})
            stage["1-unmoved-pause"] = {"status": refused["status"], "body": refused["body"]}
            problems.check((refused["status"], refused["body"].get("reason_code")) == (409, "presence_pause_unsupported"),
                           f"1: the unmoved turn's Pause was not refused as unsupported: {refused}")
            problems.check(_actions(_work(c.state(), tid)) == ["stop"], "1: a refused Pause changed the unmoved turn's controls")

            # --- 2. Moved: Pause is offered beside Stop, and Restart does not say it is lost.
            moved = _http(c.server.port, "POST", "/api/projects/from-task", {"task_id": tid, "id": PROJECT, "name": "Activity room"})
            problems.check(moved["status"] == 200, f"2: the move answered {moved}")
            c.states["2-moved"] = state = c.state()
            problems.check(_actions(_work(state, tid)) == ["pause", "stop"], f"2: the moved turn's controls are {_actions(_work(state, tid))}")
            warning = c.restart_dialog("2-restart-moved")
            stage["2-restart"] = warning
            problems.check(bool(warning) and RESTART_INLINE not in warning["body"],
                           f"2: Restart says the moved text turn is interrupted, though a restart saves it: {warning}")
            c.open_project(PROJECT)
            c.wait_row("project", tid, "the moved turn in the Project's work list")
            menu = c.row_menu("project", tid)
            stage["2-menu"] = menu
            problems.check(set(menu) == {"Pause", "Stop now"}, f"2: the Project's task actions offer {menu}")
            c.shot("2-project-menu-moved")

            # --- 3. The owner's Pause, pressed in that menu: the author abandons its held round and parks at its last
            # ready point; that round's late answer, released only now, is retained and never adopted.
            _press(c, tid, "Pause")
            # ``wait_for_browser``: a blocking poll would starve the Playwright route the press's request crosses.
            closed = c.wait_for_browser(lambda: fence_closed(read_fence(c.root, tid)), 20)
            problems.check(bool(closed), "3: the Pause pressed in the Project left no closed fence")
            paused = c.wait_for_browser(lambda: "resume" in _actions(_work(c.state(), tid)), 90)
            calls = len(c.stub.calls)
            c.first.release.set()
            c.states["3-paused"] = state = c.state()
            stage["3-paused"] = {"controls": _actions(_work(state, tid)), "row": _row(state, tid),
                                 "pause": {k: (budget_pause_row(c.root, tid) or {}).get(k) for k in ("state", "reason", "rail")}}
            problems.check(bool(paused), f"3: the paused turn offers no Resume: {stage['3-paused']}")
            page.wait_for_timeout(4000)
            after = [kind_of(body) for _kind, body in list(c.stub.calls)[calls:]]
            stage["3-model-calls-after-release"] = after
            # The stub records a held round when it is released: exactly that one, and no later call of the author.
            problems.check([k for k in after if k.startswith("author")] in ([], ["author-1"]),
                           f"3: the author called its model while paused: {after}")
            problems.check(not _sent_to(c.dialogue(), FIRST_ANSWER, project_chat), "3: the paused round's answer was sent")
            reply = (c.post.get("reply") or {}).get("body") or {}
            stage["3-host-reply"] = {k: reply.get(k) for k in ("status", "outcome", "continuation_ref")}
            ref = str(reply.get("continuation_ref") or tid)
            poll = _http(c.server.host_service_port, "GET", f"/presence/work/{ref}?binding_id={c.place.place_id}", host=True)
            stage["3-poll"] = {"status": poll["status"], "status_field": poll["body"].get("status")}
            problems.check(poll["status"] == 202 and poll["body"].get("status") == "pending",
                           f"3: the Host's poll of the saved turn says {poll['status']} {poll['body'].get('status')!r}")
            row = c.wait_row("project", tid, "the paused turn in the Project's work list", lambda r: r["word"] != "Working", timeout=30)
            stage["3-project-row"] = row
            c.shot("3-project-paused")
            warning = c.restart_dialog("3-restart-paused")
            stage["3-restart"] = warning
            problems.check(bool(warning) and RESTART_INLINE not in warning["body"],
                           f"3: Restart says the saved turn is interrupted: {warning}")

            # --- 4. The owner's Resume, pressed on the paused row: the same author continues and answers once.
            c.open_project(PROJECT)
            c.open_list("project")
            pill = page.locator(f'#project-panel .rw-row[data-id="{tid}"] .rw-pill-resume')
            pill.wait_for(timeout=15000)
            c.shot("4-project-resume-offered")
            calls = len(c.stub.calls)
            pill.click()
            grant = c.wait_for_browser(lambda: (budget_pause_row(c.root, tid) or {}).get("grant"), 30) or {}
            stage["4-grant"] = {k: grant.get(k) for k in ("selected_by", "authority", "producer_allowance_lifted")}
            lift = grant.get("producer_allowance_lifted") or {}
            problems.check((lift.get("by") or {}).get("door") == "api_task_resume" and lift.get("ceiling_usd"),
                           f"4: the owner's Resume did not record the allowance decision: {stage['4-grant']}")
            ended = c.wait_for_browser(lambda: _row(c.state(), tid) is None, 120)
            problems.check(bool(ended), "4: the census still lists the author after Resume")
            resumed = [body for _kind, body in list(c.stub.calls)[calls:] if kind_of(body).startswith("author")]
            stage["4-model-calls-after-resume"] = [kind_of(body) for body in resumed]
            problems.check(len(resumed) == 1, f"4: the resumed author called its model {len(resumed)} times")
            if resumed:
                text = body_text(resumed[0])
                first_role = (resumed[0].get("messages") or [{}])[0].get("role")
                stage["4-resumed-request"] = {"first_role": first_role, "messages": len(resumed[0].get("messages") or []),
                                              "reentry": "[PRESENCE CONVERSATION RESUMED]" in text,
                                              "lift_said": "off its $" in text}
                problems.check(first_role == "system", f"4: the resumed request lost its system prompt: {first_role}")
                problems.check(ACTOR_WORDS in text and "[PRESENCE CONVERSATION RESUMED]" in text
                               and "While the Pause kept your work parked" in text,
                               f"4: the resumed request does not carry its saved turn and reentry facts: {stage['4-resumed-request']}")
                problems.check("off its $" in text and "ordinary project budget" in text,
                               "4: the resumed author was not told the owner's Resume moved it off its allowance")
            dialogue = c.dialogue()
            answers = {text: _sent_to(dialogue, text, project_chat) for text in (FIRST_ANSWER, SECOND_ANSWER)}
            stage["4-answers"] = {text: len(rows) for text, rows in answers.items()}
            problems.check(sum(len(rows) for rows in answers.values()) == 1, f"4: the Project holds {stage['4-answers']}, not one answer")
            problems.check(not any(_sent_to(dialogue, text, c.place_chat) for text in answers), "4: an answer was sent to the place")
            feed = f"#pchat-{PROJECT}-messages"
            spoken = next((text for text, rows in answers.items() if rows), FIRST_ANSWER)
            page.locator(f"{feed} .chat-bubble").filter(has_text=spoken).first.wait_for(timeout=30000)
            settled = c.wait_row("project", tid, "the answered turn's row", lambda r: r["word"] == "Done", timeout=60)
            stage["4-project-row"] = settled and {k: settled.get(k) for k in ("word", "resume", "more")}
            problems.check(bool(settled) and not settled.get("resume"), f"4: the answered row says {stage['4-project-row']}")
            c.shot("4-project-answered")
            final = _http(c.server.host_service_port, "GET", f"/presence/work/{ref}?binding_id={c.place.place_id}", host=True)
            stage["4-poll"] = {"status": final["status"], "status_field": final["body"].get("status")}
            problems.check(final["status"] == 200 and final["body"].get("status") == "completed",
                           f"4: the Host's poll after the answer says {final['status']} {final['body'].get('status')!r}")
        finally:
            c.facts["problems"] = list(problems)
        assert not c.blocked and not c.errors, (c.errors, c.blocked)
        assert not problems, "\n".join(problems)


@pytest.fixture
def stand(monkeypatch):
    """The isolated server keeps its checkout across a real Restart (``safe_restart``'s stand lever)."""
    import tests.test_presence_project_activity_browser as activity

    class _Stand(activity._Server):
        def _env(self):
            return {**super()._env(), "OUROBOROS_DISABLE_MANAGED_UPDATES": "1"}

    monkeypatch.setattr(activity, "_Server", _Stand)


class _WorkingConsumer(Consumer):
    """The author's first round reads once, so its working state is saved; its second round is the one held."""

    def _dispatch(self, body):
        if kind_of(body) == "author-1":
            return {"tool": "get_task_result", "arguments": {"task_id": self.tid}}
        return super()._dispatch(body)


def _owner_restart(c):
    """The owner's real Restart of this isolated server (the command the header's button sends), to its next generation."""
    boots = len(c.oracle.events("startup_verification"))
    reply = _http(c.server.port, "POST", "/api/command", {"cmd": "/restart"})
    assert reply["status"] == 200, reply
    assert wait_until(lambda: len(c.oracle.events("startup_verification")) > boots or None, 180), "no next generation"

    def ready():
        try:
            return _http(c.server.port, "GET", "/api/state")["body"].get("supervisor_ready") is True or None
        except OSError:
            return None

    assert wait_until(ready, 90), "the next generation never became ready"
    c.page.reload(wait_until="domcontentloaded")
    c.page.locator("#chat-status").filter(has_text="Online").wait_for(timeout=60000)


def _quick(port, method, path, body=None, *, host=False):
    """One short read: a server between generations must show as down, not hold the reader through its boot."""
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
    try:
        connection.request(method, path, body=None if body is None else json.dumps(body),
                           headers={"Content-Type": "application/json", **({"X-Skill-Token": TOKEN} if host else {})})
        response = connection.getresponse()
        return {"status": response.status, "body": json.loads(response.read())}
    finally:
        connection.close()


@contextlib.contextmanager
def _host_watch(c, task_id):
    """What the Host answers about the turn through a Restart, its boot window included (the Host serves before the
    queue is restored): ``(poll status, poll word, version 0 resend status, its code, its custody, supervisor ready)``."""
    heard, stop = [], threading.Event()
    payload = _event(c.place, "live-inline")
    payload["event"]["text"] = ACTOR_WORDS
    payload.pop("continuation_version", None)

    def watch():
        while not stop.is_set():
            try:
                ready = _quick(c.server.port, "GET", "/api/state")["body"].get("supervisor_ready") is True
                polled = _quick(c.server.host_service_port, "GET", f"/presence/work/{task_id}?binding_id={c.place.place_id}", host=True)
                resent = _quick(c.server.host_service_port, "POST", "/presence/turn", payload, host=True)
            except (OSError, ValueError, http.client.HTTPException):
                heard.append(("down",))
            else:
                heard.append((polled["status"], polled["body"].get("status"), resent["status"], resent["body"].get("code"),
                               resent["body"].get("custody"), ready))
            stop.wait(0.1)

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        yield heard
    finally:
        stop.set()
        watcher.join(15)


def _heard(heard):
    answers = [row for row in heard if len(row) > 1]
    return {"down": len(heard) - len(answers), "answers": len(answers), "in_boot_window": len([row for row in answers if not row[-1]]),
            "distinct": sorted({row for row in answers}, key=str)}, answers


def _settled_row(c, task_id):
    from ouroboros.task_results import load_task_result

    row = load_task_result(c.root, task_id) or {}
    return row if row.get("status") in {"completed", "failed", "cancelled"} else None


def test_the_owners_restart_returns_a_moved_turn_it_finds_working_and_lifts_nothing(wait_clone, tmp_path, stand):
    """The Restart stops the moved author inside its held round; the next generation returns its saved work under the
    same id, in a worker that retakes the conversation first, and its answer stands once in the Project. No Resume was
    pressed, so its Place allowance stays (an automatic return is never the owner's choice)."""
    from ouroboros.working_checkpoint import _working_dir

    with _WorkingConsumer(wait_clone, tmp_path, "presence-project-restart-working-browser") as c:
        problems, stage, tid = c.problems, c.facts["stages"], c.tid
        project_chat = int(c.project["chat_id"])
        try:
            c.first.release.set()
            c.start_turn()
            assert c.second.arrived.wait(180), "1: the author never reached its second model round"
            moved = _http(c.server.port, "POST", "/api/projects/from-task", {"task_id": tid, "id": PROJECT, "name": "Activity room"})
            problems.check(moved["status"] == 200, f"1: the move answered {moved}")
            saved = wait_until(lambda: sorted(p.name for p in _working_dir(c.root, tid).glob("*.json")) or None, 60)
            stage["1-working-saved"] = saved
            problems.check(bool(saved), "1: the author saved no working state before the Restart")
            warning = c.restart_dialog("1-restart-working")
            stage["1-restart"] = warning
            problems.check(bool(warning) and RESTART_INLINE not in warning["body"], f"1: Restart says it interrupts the moved turn: {warning}")
            before = c.oracle.task_result(tid)
            stage["1-before"] = {k: before.get(k) for k in ("status", "task_attempt", "chat_id")}
            ceiling = (before.get("metadata") or {}).get("root_cost_ceiling_usd")
            problems.check(bool(ceiling), "1: the Place turn carries no allowance to keep")
            calls = len(c.stub.calls)
            with _host_watch(c, tid) as heard:
                _owner_restart(c)  # the held round stays held through the teardown: the Restart catches the author inside it
                # While it is still held the row is returned or running: the Host keeps answering about it.
                c.wait_for_browser(lambda: len(heard) > 3, 10)
            stage["2-through-restart"], answers = _heard(heard)
            problems.check(bool(answers), "2: the Host answered nothing through the Restart")
            problems.check(all((row[0], row[1]) == (202, "pending") for row in answers), f"2: the poll through the Restart: {stage['2-through-restart']}")
            problems.check(all(row[2] == 409 and row[3] in {"presence_turn_parked", "presence_turn_custody_unknown"} for row in answers),
                           f"2: the resend through the Restart: {stage['2-through-restart']}")
            c.second.release.set()  # only the round the next generation asks may answer
            c.states["2-after-restart"] = c.state()
            stage["2-returned-row"] = next(({k: t.get(k) for k in ("id", "_attempt", "_presence_turn", "_presence_run")}
                                            for t in c.oracle.queue_snapshot().get("pending", []) if t.get("id") == tid), None)
            done = wait_until(lambda: _settled_row(c, tid), 180)
            row = done or c.oracle.task_result(tid)
            stage["2-settled"] = {k: row.get(k) for k in ("status", "task_attempt", "reason_code", "terminal_origin", "chat_id")}
            stage["2-queue"] = {"pending": [t.get("id") for t in c.oracle.queue_snapshot().get("pending", [])],
                                "running": [t.get("id") for t in c.oracle.queue_snapshot().get("running", [])]}
            problems.check(row.get("status") == "completed", f"2: the returned turn ended {stage['2-settled']}")
            after = [kind_of(body) for _kind, body in list(c.stub.calls)[calls:]]
            stage["2-model-calls-after-restart"] = after
            wait_until(lambda: _sent_to(c.dialogue(), SECOND_ANSWER, project_chat) or None, 30)
            dialogue = c.dialogue()
            answers = {text: len(_sent_to(dialogue, text, project_chat)) for text in (FIRST_ANSWER, SECOND_ANSWER)}
            stage["2-answers"] = answers
            attempts = row.get("task_attempt"), (row.get("execution_owner") or {}).get("kind")
            stage["2-attempt-owner"] = attempts
            problems.check(sum(answers.values()) == 1, f"2: the Project holds {answers}, not one answer")
            problems.check(not any(_sent_to(dialogue, text, c.place_chat) for text in answers), "2: an answer went to the place")
            # It ran as the turn it was: its built record, its tools, its saved loop; never the lane's sparse seed.
            returned = [body for _kind, body in list(c.stub.calls)[calls:] if kind_of(body).startswith("author")][-1:]
            first = next(body for _kind, body in list(c.stub.calls) if kind_of(body) == "author-1")
            tools = lambda body: sorted(t.get("function", {}).get("name", "") for t in body.get("tools") or [])  # noqa: E731
            stage["2-returned-tools-same"] = bool(returned) and tools(returned[0]) == tools(first)
            problems.check(stage["2-returned-tools-same"], "2: the returned turn's tools differ from the ones it was admitted with")
            if returned:
                text = body_text(returned[0])
                stage["2-returned-request"] = {"first_role": (returned[0].get("messages") or [{}])[0].get("role"),
                                               "reentry": "[PRESENCE CONVERSATION RESUMED]" in text, "restart_said": "a restart" in text}
                problems.check(stage["2-returned-request"]["first_role"] == "system" and ACTOR_WORDS in text
                               and "[PRESENCE CONVERSATION RESUMED]" in text,
                               f"2: the returned request lacks its saved turn or reentry facts: {stage['2-returned-request']}")
            metadata = row.get("metadata") or {}
            stage["2-record"] = {"focus": bool(metadata.get("focus")), "wake_cause": metadata.get("wake_cause"),
                                 "attempt": row.get("task_attempt")}
            problems.check(bool(metadata.get("focus")) and metadata.get("wake_cause") == "focus",
                           f"2: the returned turn ran from its seed, not its built record: {sorted(metadata)}")
            stage["2-allowance"] = {"ceiling": metadata.get("root_cost_ceiling_usd"), "lift": metadata.get("producer_allowance_lifted")}
            problems.check(metadata.get("root_cost_ceiling_usd") == ceiling and not metadata.get("producer_allowance_lifted"),
                           f"2: an automatic return changed the allowance: {stage['2-allowance']}")
            final = _http(c.server.host_service_port, "GET", f"/presence/work/{tid}?binding_id={c.place.place_id}", host=True)
            stage["2-poll"] = {"status": final["status"], "status_field": final["body"].get("status")}
            problems.check(final["status"] == 200 and final["body"].get("status") == "completed", f"2: the poll says {stage['2-poll']}")
            ended = wait_until(lambda: _row(c.state(), tid) is None or None, 120)
            problems.check(bool(ended), "2: the census still lists the returned turn after its answer")
            c.open_project(PROJECT)
            settled = c.wait_row("project", tid, "the returned turn's row", lambda r: r["word"] == "Done", timeout=60)
            stage["2-project-row"] = settled and {k: settled.get(k) for k in ("word", "resume", "more")}
            problems.check(bool(settled) and not settled.get("resume"), f"2: the returned row says {stage['2-project-row']}")
            c.shot("2-project-answered-after-restart")
        finally:
            c.facts["problems"] = list(problems)
        assert not c.blocked and not c.errors, (c.errors, c.blocked)
        assert not problems, "\n".join(problems)


def test_the_owners_restart_keeps_a_paused_moved_turn_and_a_stop_from_the_spa_reaches_it(wait_clone, tmp_path, stand):
    """Saved by the owner's Pause and past its carrier's release, the turn stays paused through the owner's Restart: its row
    says Paused with Resume, the Host's poll stays pending, a resend of its event at version 0 hears the queue keeps it and
    starts nothing. Stop, pressed in the Project's row, reaches that queued work (never a false 404) and ends it."""
    with Consumer(wait_clone, tmp_path, "presence-project-restart-paused-browser") as c:
        page, problems, stage, tid = c.page, c.problems, c.facts["stages"], c.tid
        try:
            c.start_turn()
            assert c.first.arrived.wait(120), "1: the author never reached its first model round"
            moved = _http(c.server.port, "POST", "/api/projects/from-task", {"task_id": tid, "id": PROJECT, "name": "Activity room"})
            problems.check(moved["status"] == 200, f"1: the move answered {moved}")
            pause = _http(c.server.port, "POST", f"/api/tasks/{tid}/pause", {"request_id": "pause-moved", "actor": "owner"})
            problems.check(pause["status"] == 200, f"1: the Pause answered {pause}")
            # The author abandons its held round and parks; that round stays held to the end of this walk.
            problems.check(bool(wait_until(lambda: "resume" in _actions(_work(c.state(), tid)) or None, 90)), "1: no Resume offered")
            carried = wait_until(lambda: not [r for r in c.oracle.queue_snapshot().get("running", []) if r.get("id") == tid] or None, 30)
            problems.check(bool(carried), "1: the parked turn still has a RUNNING row")
            page.wait_for_timeout(3000)  # its carrier's hold ends with its settlement: nothing of it holds the slot now
            calls = len(c.stub.calls)
            # Through the Restart and the window before its queue restore: pending at every read, the resend retained at
            # every read, never failed, never a new author.
            with _host_watch(c, tid) as heard:
                _owner_restart(c)
                c.wait_for_browser(lambda: len(heard) > 3, 10)
            stage["2-through-restart"], answers = _heard(heard)
            problems.check(bool(answers), "2: the Host answered nothing through the Restart")
            problems.check(all((row[0], row[1]) == (202, "pending") for row in answers), f"2: the poll through the Restart: {stage['2-through-restart']}")
            problems.check(all(row[2] == 409 and row[3] in {"presence_turn_parked", "presence_turn_custody_unknown"} for row in answers),
                           f"2: the resend through the Restart: {stage['2-through-restart']}")
            payload = _event(c.place, "live-inline")
            payload["event"]["text"] = ACTOR_WORDS
            payload.pop("continuation_version", None)
            c.states["2-after-restart"] = state = c.state()
            stage["2-controls"] = _actions(_work(state, tid))
            stage["2-after-restart-row"] = {k: (_row(state, tid) or {}).get(k) for k in ("kind", "phase", "pause_cause", "can_resume")}
            problems.check("resume" in stage["2-controls"], f"2: after the Restart its controls are {stage['2-controls']}")
            poll = _http(c.server.host_service_port, "GET", f"/presence/work/{tid}?binding_id={c.place.place_id}", host=True)
            stage["2-poll"] = {"status": poll["status"], "status_field": poll["body"].get("status")}
            problems.check(poll["status"] == 202 and poll["body"].get("status") == "pending", f"2: the poll says {stage['2-poll']}")
            resend = _http(c.server.host_service_port, "POST", "/presence/turn", payload, host=True)
            stage["2-v0-resend"] = {"status": resend["status"], **{k: resend["body"].get(k) for k in ("code", "disposition", "custody", "turn_ref")}}
            problems.check((resend["status"], resend["body"].get("code"), resend["body"].get("custody")) == (409, "presence_turn_parked", "queued"),
                           f"2: the resend answered {stage['2-v0-resend']}")
            page.wait_for_timeout(3000)
            after = [kind_of(body) for _kind, body in list(c.stub.calls)[calls:]]
            stage["2-model-calls"] = after
            problems.check(not [k for k in after if k.startswith("author")], f"2: the author called its model while paused: {after}")
            c.open_project(PROJECT)
            row = c.wait_row("project", tid, "the paused turn after the Restart", lambda r: r["word"] == "Paused", timeout=45)
            stage["2-project-row"] = row and {k: row.get(k) for k in ("word", "secondary", "resume", "more")}
            problems.check(bool(row) and row.get("resume") is True, f"2: the row offers no Resume: {stage['2-project-row']}")
            c.shot("2-project-paused-after-restart")
            menu = c.row_menu("project", tid)
            stage["3-menu"] = menu
            problems.check("Stop now" in menu, f"3: the paused row's actions are {menu}")
            answers = c.stop_pending(tid)
            c.wait_for_browser(lambda: answers, 30)
            stage["3-cancel-answers"] = answers
            problems.check(bool(answers) and all(status in (200, 202) for status, _url in answers), f"3: Stop answered {answers}")
            done = wait_until(lambda: _settled_row(c, tid), 60)
            stage["3-settled"] = {k: (done or {}).get(k) for k in ("status", "reason_code")}
            problems.check(bool(done) and done.get("status") == "cancelled", f"3: the stopped work ended {stage['3-settled']}")
            final = [kind_of(body) for _kind, body in list(c.stub.calls)[calls:]]
            problems.check(not [k for k in final if k.startswith("author")], f"3: the author called its model after Stop: {final}")
            settled = c.wait_row("project", tid, "the stopped work's row", lambda r: r["word"] == "Stopped", timeout=60)
            stage["3-project-row"] = settled and {k: settled.get(k) for k in ("word", "resume", "more")}
            problems.check(bool(settled) and not settled.get("resume"), f"3: the stopped row says {stage['3-project-row']}")
            c.shot("3-project-stopped")
        finally:
            c.facts["problems"] = list(problems)
        assert not c.blocked and not c.errors, (c.errors, c.blocked)
        assert not problems, "\n".join(problems)
