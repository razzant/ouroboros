"""S47-S50: a returned work whose first request no longer fits the local window parks under its own id.

A keyless real server runs the working tree (``candidate_checkout``) with Main on the built-in local lane. A
managed root writes a file, the owner presses Pause, and the application stops (SIGTERM to the whole tree). After
the next boot, while it is still paused, another task lengthens ``identity`` through its ordinary writer: that
write is admitted, because every fresh Main keeps its room, but this saved work carries a long first message of its
own, so its request is now larger than the window.

The owner's Resume is then the real attempt: a worker restores the saved work, the serving instance's own count
refuses the request before dispatch, and the same id parks again under the technical rail with that count on its
durable row. No completion is asked for, no terminal is written, nothing is offered as Continue.

- S47: after a third task shortens ``identity``, a second Resume continues the same saved work to its answer, and
  the file it wrote before the Pause was written once.
- S48: the owner instead ends that attempt (``context_choice.end_attempt``, naming the hold the task shows): the
  ordinary size terminal, nothing sent, and the existing Continue starts one successor under a new id. A choice
  about an ordinary Pause, an earlier hold or an overview nobody offered is refused and changes nothing.
- S49: the work holds an earlier, longer overview. While neither that one nor the current one fits, the park offers
  nothing. Once the current one would fit, the park shows both counts; the owner's choice of that revision is the
  real attempt on the held overview again, and only the park that follows saves the work holding the current one.
  The next Resume continues there; the earlier overview stays readable.
- S50: the owner's choice of the current overview outlives an ordinary stop of the application. The serving count
  of the chosen Resume's own attempt is held open, so the grant is spent and nothing is parked yet; the application
  stops there (SIGTERM to the whole tree) and boots again. The saved work returns from its checkpoint with the
  choice it carried: the held overview is refused again and that park saves the work holding the chosen revision,
  as if nothing had stopped. Nothing was asked of the model in between.

Doubled, and only these: the model server (the loopback stub also answers the built-in server's count route with
a deterministic price: a fixed background plus 500 tokens per distinct marker) and the built-in server's process
identity (a sleeping child whose arguments name the window, published as the ``local_model`` service binding the
way ``LocalModelManager`` publishes its own). The Pause, the stop, the restore, every Resume, the size recovery, the
park, the choices, the Continue and the results are the shipped ones.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import queue
import re
import subprocess
import sys
import threading
import time

import pytest

from devtools.benchmarks.common.server_runner import _api_status
from tests.system_e2e.harness import (
    LANE_MOCK,
    ArtifactOracle,
    ScriptedStubModel,
    keyless_settings,
    message_text,
    require_lane,
    start_server,
    submit_running,
    wait_durable_result,
    wait_until,
)
from tests.system_e2e.test_presence_resilience import s33_candidate  # noqa: F401 - pytest fixture

MARK = "COLD_RETURN_SAVED_WORK"
FINAL = "COLD_RETURN_CONTINUED_AFTER_ITS_RESUME"
WINDOW = 80_000
BACKGROUND = 36_808  # what the fixture counter prices everything unmarked at
PRICE = 500  # tokens per distinct marker
REPLY_FLOOR = 8_192
WORK, IDENTITY, OVERVIEW = "CRTW", "CRTR", "CRTO"  # marker families: the work's words, identity, the overview
WORDS = 37  # markers in the work's own first message: 18,500 fixture tokens no recovery may cut
EFFECT = {"tool": "write_file", "arguments": {
    "root": "task_drive", "path": "cold-return-effect.txt", "content": "written once, before the Pause"}}
KEEPALIVE = {"tool": "list_files", "arguments": {"path": "."}}
NOT_OFFERED = "context_choice_not_offered"
# S48 and S49 boot twice and run more tasks than the lane's ceiling was measured for (``conftest``).
LONG_SCENARIO_TIMEOUT_SEC = 1800


def _markers(prefix: str, count: int) -> str:
    return " ".join(f"{prefix}{index:03d}" for index in range(count))


def _marked(text: str) -> set:
    return {word[:7] for word in text.replace('"', " ").replace("\\n", " ").split()
            if word[:4] in {WORK, IDENTITY, OVERVIEW} and word[4:7].isdigit()}


def _asked(body: dict, memory=()) -> str:
    """Whose request this is: the work's when its user rows carry the work's whole first message, else the latest
    memory task named there. A later task's words show the paused work's title, so the work's mark names nothing."""
    text = "\n".join(message_text(message) for message in body.get("messages") or []
                     if isinstance(message, dict) and message.get("role") == "user")
    if sum(word.startswith(WORK) for word in _marked(text)) == WORDS:
        return MARK
    return next((mark for mark in reversed(list(memory)) if mark in text), "")


def _system(body: dict) -> str:
    return "\n".join(message_text(message) for message in body.get("messages") or []
                     if isinstance(message, dict) and message.get("role") == "system")


class _LocalLaneStub(ScriptedStubModel):
    """The loopback model, also the built-in server's count route (``/extras/measure_chat``).

    The count is exact in the product's sense: it answers for the very payload it was sent (``native_input_sha256``)
    under one process id and window. ``finish`` turns the keepalive rounds into the final answer."""

    def __init__(self) -> None:
        super().__init__(latency_sec=0.2, select_retained=True)
        self.pid, self.finish, self.counts, self.served = 0, False, [], []
        self.measured: list = []  # every counted request, beside its count
        self.memory: dict = {}  # a memory task's mark -> its remaining steps
        # While ``holding``, each count waits at its own gate before it answers: the scenario opens a gate, or leaves
        # it closed and stops the application with that count unanswered (``held_counts`` hands the gates out).
        self.holding, self.held_counts = False, queue.Queue()
        handler, outer = self._server.RequestHandlerClass, self
        post = handler.do_POST

        def do_post(request):
            if not request.path.rstrip("/").endswith("/extras/measure_chat"):
                return post(request)
            raw = request.rfile.read(int(request.headers.get("Content-Length") or 0)) or b"{}"
            return request._send(outer._measure(json.loads(raw.decode("utf-8"))))

        handler.do_POST = do_post

    def _measure(self, payload: dict) -> dict:
        from ouroboros.local_model_server import input_fingerprint
        from ouroboros.utils import machine_json

        # Mirror measure_chat_input: this is the formatter's input, not the HTTP body.
        rendered = machine_json({key: payload.get(key) for key in
                                 ("messages", "functions", "function_call", "tools", "tool_choice")})
        tokens = BACKGROUND + PRICE * len(_marked(rendered))
        if self.holding:
            gate = threading.Event()
            self.held_counts.put((tokens, gate))
            gate.wait(LONG_SCENARIO_TIMEOUT_SEC)
        with self._lock:
            self.counts.append(tokens)
            self.measured.append(payload)
        return {"supported": True, "input_is_exact": True, "output_limit_enforced": True,
                "reasoning_included_in_limit": True, "input_tokens": tokens, "context_window": WINDOW,
                "raw_context_window": WINDOW + 128, "native_input_sha256": input_fingerprint(payload),
                "process_id": self.pid, "tokenizer_template_provenance": {
                    "source": "serving_llama_formatter", "formatter": "fixture", "library_version": "fixture",
                    "chat_format": "fixture", "launched_n_ctx": WINDOW,
                    "rendered_prompt_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest()}}

    def _next_step(self, body):
        asked = _asked(body, self.memory)
        if asked in self.memory:
            return self.memory[asked].pop(0) if self.memory[asked] else None
        if asked != MARK:
            return None
        step = {"final": FINAL} if self.finish else dict(KEEPALIVE if self.served else EFFECT)
        self.served.append("final" if "final" in step else step["tool"])
        return step

    def work_rounds(self) -> list:
        """The completions asked for this work (agent rounds and its final), in order."""
        with self._lock:
            return [body for kind, body in self.calls if kind in {"agent", "final"} and _asked(body) == MARK]

    def posts(self) -> int:
        """Every completion POST the model received so far, whatever it was for (the count route is not one)."""
        with self._lock:
            return len(self.calls)

    def posts_since(self, index: int) -> list:
        with self._lock:
            return [(kind, _asked(body, self.memory) or message_text((body.get("messages") or [{}])[-1])[:120])
                    for kind, body in self.calls[index:]]

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])


def _without_reply_cap(payload: dict) -> dict:
    """The refused count and its recount may differ only in the output cap, not the input or its clock."""
    return {key: value for key, value in payload.items() if key != "max_tokens"}


def _difference(sent: dict, counted: dict) -> dict:
    """Where two requests differ, for a failed comparison: their keys, and the first row that is not the same."""
    keys = sorted(key for key in set(sent) | set(counted) if key != "messages" and sent.get(key) != counted.get(key))
    left, right = sent.get("messages") or [], counted.get("messages") or []
    row = next((index for index, pair in enumerate(zip(left, right)) if pair[0] != pair[1]), None)
    at = None
    if row is not None:
        a, b = json.dumps(left[row], sort_keys=True), json.dumps(right[row], sort_keys=True)
        cut = next((index for index, pair in enumerate(zip(a, b)) if pair[0] != pair[1]), min(len(a), len(b)))
        at = {"row": row, "role": left[row].get("role"), "sent": a[max(0, cut - 80):cut + 160], "counted": b[max(0, cut - 80):cut + 160]}
    return {"keys": {key: (str(sent.get(key))[:200], str(counted.get(key))[:200]) for key in keys},
            "rows": (len(left), len(right)), "first_different_row": at}


def _post(server, path: str, body: dict | None = None) -> dict:
    return _api_status(server.base_url, "POST", path, body or {}, timeout=120)


def _resume(server, task_id: str, choice: dict | None = None) -> dict:
    return _post(server, f"/api/tasks/{task_id}/resume", {"context_choice": choice} if choice is not None else None)


def _pause_row(data, task_id: str) -> dict:
    from ouroboros.budget_pause import budget_pause_row

    try:
        return budget_pause_row(data, task_id) or {}
    except Exception as exc:  # an unreadable row is a fact for the forensics, not a crash of the reader
        return {"unreadable": repr(exc)}


def _pending(oracle: ArtifactOracle, task_id: str) -> list:
    return [dict(row.get("task") or {}) for row in oracle.pending_rows(task_id)]


def _rows(oracle: ArtifactOracle) -> list:
    snapshot = oracle.queue_snapshot()
    return [str(row.get("id")) for state in ("pending", "running") for row in snapshot.get(state) or []
            if isinstance(row, dict)]


def _task_done(oracle: ArtifactOracle, task_id: str) -> list:
    return [row for row in oracle.events("task_done") if str(row.get("task_id") or "") == task_id]


def _facts(oracle, data, task_id: str, stub) -> dict:
    row = _pause_row(data, task_id)
    return {"pause_row": {key: row.get(key) for key in (
                "state", "reason", "rail", "cause", "pause_id", "pause_generation", "context_nonfit", "unreadable")},
            "grant": {key: (row.get("grant") or {}).get(key) for key in (
                "grant_id", "consumed_at", "revoked_at", "context_choice")},
            "queue": _rows(oracle), "result": {key: oracle.task_result(task_id).get(key) for key in (
                "status", "reason_code")}, "work_rounds": len(stub.work_rounds()), "served": stub.served,
            "counts": stub.counts[-8:]}


def _settled_posts(stub, oracle: ArtifactOracle, *, quiet: float = 4.0, timeout: float = 120.0) -> int:
    """The model's POST count once no task runs and none arrived for ``quiet`` seconds: the start of an interval in
    which every completion POST would be the next act's own."""
    deadline, seen, since = time.time() + timeout, -1, time.time()
    while time.time() < deadline:
        count = stub.posts()
        if count != seen or oracle.running_ids():
            seen, since = count, time.time()
        elif time.time() - since >= quiet:
            return count
        time.sleep(0.25)
    raise AssertionError(f"the model's traffic never settled: {stub.posts_since(max(0, seen - 6))}")


def _waits_in_queue(oracle: ArtifactOracle, task_id: str) -> bool:
    return bool(wait_until(lambda: task_id not in oracle.running_ids()
                           and any(isinstance(row.get("_budget_pause"), dict) for row in _pending(oracle, task_id)), 180))


def _parked_after(data, task_id: str, earlier: str) -> dict:
    """The durable row of the task's next waiting window park after pause ``earlier`` ({} when none comes)."""
    from ouroboros.budget_pause import STATE_PAUSED
    from ouroboros.cold_return import RAIL_CONTEXT_WINDOW

    def parked():
        row = _pause_row(data, task_id)
        return row if (row.get("rail") == RAIL_CONTEXT_WINDOW and row.get("state") == STATE_PAUSED
                       and row.get("pause_id") != earlier) else None

    return wait_until(parked, 420) or {}


def _stopped_after_the_grant_was_spent(stub, server, data, task_id: str, pause_id: str) -> dict:
    """Stop the application while the worker that took the grant of pause ``pause_id`` waits for a serving count.

    Every count is held at its gate (``stub.holding``); one that arrives before the grant is consumed is let through.
    The first one held after it belongs to the returned work's own attempt: nothing can be parked before it answers.
    Returns the consumed row as it stood at the stop."""
    from ouroboros.budget_pause import STATE_RESUMED

    deadline, consumed, gate = time.time() + 420, {}, None
    try:
        while not consumed and time.time() < deadline:
            try:
                _tokens, gate = stub.held_counts.get(timeout=5)
            except queue.Empty:
                continue
            row = _pause_row(data, task_id)
            if row.get("pause_id") == pause_id and row.get("state") == STATE_RESUMED:
                consumed = row
            else:
                gate.set()
        assert consumed, ("no serving count was asked after the grant was spent", _standing(data, task_id))
        server.stop()
    finally:
        stub.holding = False
        if gate is not None:
            gate.set()
        while not stub.held_counts.empty():  # a count that queued meanwhile answers nobody now
            stub.held_counts.get_nowait()[1].set()
    return consumed


def _next_window_park(data, task_id: str, earlier: str):
    """The task's waiting window park after pause ``earlier``, or ``None`` while there is none."""
    from ouroboros.budget_pause import STATE_PAUSED
    from ouroboros.cold_return import RAIL_CONTEXT_WINDOW

    row = _pause_row(data, task_id)
    return row if (row.get("rail") == RAIL_CONTEXT_WINDOW and row.get("state") == STATE_PAUSED
                   and row.get("pause_id") != earlier) else None


def _shown_hold(server, task_id: str) -> dict:
    """What a client reads of the task's hold (``GET /api/tasks/{id}``: ``current_stop.window``)."""
    read = _api_status(server.base_url, "GET", f"/api/tasks/{task_id}", timeout=120)
    assert read["status"] == 200, read
    stop = read["body"].get("current_stop") or {}
    return {**stop["window"], "stop_identity": stop["identity"]} if stop.get("window") else {}


def _named(hold: dict, **action) -> dict:
    return {"pause_id": hold["pause_id"], "pause_generation": hold["pause_generation"],
            "stop_identity": hold.get("stop_identity", "another-stop"), **action}


def _standing(data, task_id: str) -> dict:
    """What makes the task's durable pause the hold it is: its identity, state, cause, numbers, source and grant."""
    row = _pause_row(data, task_id)
    return {key: row.get(key) for key in (
        "pause_id", "pause_generation", "state", "reason", "rail", "cause", "context_nonfit", "source_ref",
        "resume_generation", "grant", "unreadable")}


def _refused_and_unchanged(server, data, task_id: str, choice: dict) -> None:
    """A choice the task's current hold does not offer: refused typed, and its durable pause stands as it was."""
    before = _standing(data, task_id)
    answer = _resume(server, task_id, choice)
    assert answer["status"] == 409 and answer["body"].get("error") == NOT_OFFERED, (choice, answer)
    assert _standing(data, task_id) == before, (choice, before, _standing(data, task_id))


def _identity(data, count: int) -> tuple:
    """``(tool, step, check)``: ``identity`` rewritten with ``count`` markers through its ordinary tool."""
    path = pathlib.Path(data) / "memory" / "identity.md"
    text = f"A synthetic identity. {_markers(IDENTITY, count)}"

    def write(_body):
        return {"tool": "update_identity", "arguments": {
            "content": text, "expected_revision": hashlib.sha256(path.read_bytes()).hexdigest()}}

    def check(told):
        assert path.read_text(encoding="utf-8") == text, told[-3:]

    return "update_identity", write, check


def _overview(data, count: int, edition: str) -> tuple:
    """``(tool, step, check)``: the global overview rewritten with ``count`` markers through ``knowledge_write``,
    claiming Main's room, so every work there holds the revision it started with."""
    path = pathlib.Path(data) / "memory" / "knowledge" / "overview.md"
    body = f"A synthetic orientation, {edition} edition. {_markers(OVERVIEW, count)} end of the orientation."
    content = (f"---\ntype: note\ntitle: Synthetic orientation ({edition})\n"
               f"summary: A synthetic orientation for this scenario, {edition} edition.\nclaims:\n- room:1\n---\n\n{body}\n")

    def write(_body):
        revision = {"expected_revision": hashlib.sha256(path.read_bytes()).hexdigest()} if path.exists() else {}
        return {"tool": "knowledge_write", "arguments": {
            "topic": "overview", "scope": "global", "mode": "overwrite", "content": content, **revision}}

    def check(told):
        assert path.exists() and body in path.read_text(encoding="utf-8"), told[-3:]

    return "knowledge_write", write, check


def _overview_revision(data) -> str:
    return hashlib.sha256((pathlib.Path(data) / "memory" / "knowledge" / "overview.md").read_bytes()).hexdigest()


def _write_memory(server, stub, oracle: ArtifactOracle, name: str, *writes) -> None:
    """Another managed task changes always-loaded memory through the ordinary tools (each admission counts a fresh
    Main on the same serving counter), then answers."""
    mark = f"COLD_RETURN_MEMORY_{name}"
    stub.memory[mark] = [{"tool": "enable_tools", "arguments": {"tools": ",".join(tool for tool, _s, _c in writes)}},
                         *(step for _tool, step, _check in writes)]
    task_id = submit_running(server, f"Rewrite memory as instructed. [{mark}]")
    result = wait_durable_result(oracle, task_id, timeout=420)
    assert result.get("status") == "completed", result
    with stub._lock:
        told = [str(message.get("content"))[:700] for _kind, body in stub.calls if _asked(body, stub.memory) == mark
                for message in body.get("messages") or [] if isinstance(message, dict) and message.get("role") == "tool"]
    for _tool, _step, check in writes:
        check(told)


@pytest.fixture
def serving_identity():
    """A live process whose arguments name the built-in server's window, as ``start_server`` launches it."""
    started = []

    def start(port: int) -> int:
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(7200)", "--model",
                                  "cold-return-fixture.gguf", "--port", str(port), "--n_ctx", str(WINDOW)],
                                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        started.append(child)
        return child.pid

    try:
        yield start
    finally:
        for child in started:  # the exact children this fixture started, never a pattern
            child.kill()
            child.wait(timeout=30)


def _settings(stub) -> dict:
    return keyless_settings(
        stub, USE_LOCAL_MAIN="true", LOCAL_MODEL_PORT=stub.port, LOCAL_MODEL_CONTEXT_LENGTH=WINDOW,
        LOCAL_MODEL_SOURCE="", OUROBOROS_CONTEXT_MODE="nano", OUROBOROS_MAX_WORKERS=1)


def _pause_the_work(candidate, root, stub, settings, *, first=None, first_count: int) -> tuple:
    """Generation A: the work writes its file, the owner pauses it, the application stops.

    ``first`` runs before the work starts. Returns ``(task id, the owner Pause's durable row, the completions the
    work had asked for)``."""
    from ouroboros.budget_pause import STATE_PAUSED
    from ouroboros.server_process import record_service_binding

    data_root = pathlib.Path(root) / "data"
    (data_root / "state").mkdir(parents=True, exist_ok=True)
    record_service_binding(data_root, "local_model", "127.0.0.1", stub.port, pid=stub.pid)
    server = start_server(candidate, root, settings)
    data, oracle = server.data_root, ArtifactOracle(server.data_root)
    try:
        if first is not None:
            first(server, oracle)
        task_id = submit_running(server, f"Keep listing the repository files. [{MARK}] {_markers(WORK, WORDS)}")
        assert wait_until(lambda: sorted(str(path) for path in data.rglob(EFFECT["arguments"]["path"])), 240), \
            _facts(oracle, data, task_id, stub)
        assert first_count in stub.counts, stub.counts  # the serving count priced the sent request
        assert _post(server, f"/api/tasks/{task_id}/pause", {"request_id": "cold-return-pause"})["status"] in (200, 202)
        assert wait_until(lambda: _pause_row(data, task_id).get("state") == STATE_PAUSED
                          and task_id not in oracle.running_ids(), 240), _facts(oracle, data, task_id, stub)
        owner_pause = _pause_row(data, task_id)
        assert owner_pause.get("reason") == "owner", owner_pause
    finally:
        server.stop()
    rounds_at_pause = len(stub.work_rounds())
    assert rounds_at_pause >= 1 and stub.served.count("write_file") == 1, stub.served
    return task_id, owner_pause, rounds_at_pause


def _restored_queue(oracle: ArtifactOracle, task_id: str) -> None:
    assert wait_until(lambda: any(isinstance(row.get("_budget_pause"), dict)
                                  for row in _pending(oracle, task_id)), 180), _rows(oracle)


def _one_effect(data, stub) -> None:
    """The effect of the round before the Pause was asked for and written once."""
    assert stub.served.count("write_file") == 1, stub.served
    effects = sorted(data.rglob(EFFECT["arguments"]["path"]))
    assert len(effects) == 1 and effects[0].read_text(encoding="utf-8") == EFFECT["arguments"]["content"]


@pytest.mark.integration
def test_s47_a_returned_work_that_no_longer_fits_parks_then_resumes_once_it_fits(  # noqa: PLR0915
        s33_candidate, tmp_path_factory, serving_identity):  # noqa: F811
    require_lane(LANE_MOCK)
    from ouroboros.budget_pause import STATE_PAUSED, STATE_RESUMED
    from ouroboros.cold_return import CAUSE, FACTS_KEY, RAIL_CONTEXT_WINDOW, WAIT_TEXT

    root = tmp_path_factory.mktemp("s47")
    grown, shortened = 34, 30  # identity's markers, twice
    with _LocalLaneStub() as stub:
        stub.pid = serving_identity(stub.port)
        settings = _settings(stub)
        task_id, owner_pause, rounds_at_pause = _pause_the_work(
            s33_candidate, root, stub, settings, first_count=BACKGROUND + PRICE * WORDS)

        # ---------------------------------------------------------------
        # Generation B: identity grew; Resume is refused by the exact count and the same id parks.
        # ---------------------------------------------------------------
        server = start_server(s33_candidate, root, settings)
        try:
            data, oracle = server.data_root, ArtifactOracle(server.data_root)
            _restored_queue(oracle, task_id)
            # A fresh Main keeps its room under this identity (17,000 of the 20,000 its memory may take, and the
            # 15,000 of work room besides), so the write is admitted; the saved work carries 18,500 of its own
            # first message, which no recovery may cut, and is 500 over.
            _write_memory(server, stub, oracle, "GROWN", _identity(data, grown))
            # From here to the next memory task the refused return is the only act: every completion POST counts.
            posts_before, counts_before = _settled_posts(stub, oracle), len(stub.counts)
            granted = _resume(server, task_id)
            assert granted["status"] == 200 and granted["body"].get("exact_continuation"), granted
            parked = _parked_after(data, task_id, owner_pause["pause_id"])
            assert parked, _facts(oracle, data, task_id, stub)
            assert parked.get("rail") == RAIL_CONTEXT_WINDOW and parked.get("cause") == CAUSE, parked
            assert parked.get("reason") == "authority" and not parked.get("auto_resume"), parked
            needed = BACKGROUND + PRICE * (WORDS + grown)
            assert parked.get(FACTS_KEY) == {
                "input_tokens": needed, "window_tokens": WINDOW, "reply_floor_tokens": REPLY_FLOOR,
                "over_tokens": needed + REPLY_FLOOR - WINDOW, "model": parked[FACTS_KEY]["model"],
                "basis": "exact_local_count"}, parked.get(FACTS_KEY)
            assert parked[FACTS_KEY]["over_tokens"] == 500
            # Counted, never generated: the refused return asked the model for nothing.
            assert len(stub.counts) > counts_before and needed in stub.counts[counts_before:], stub.counts
            assert _waits_in_queue(oracle, task_id), _facts(oracle, data, task_id, stub)
            stored = oracle.task_result(task_id)
            assert stored.get("status") == "scheduled" and stored.get("reason_code") == "authority_paused", stored
            assert _task_done(oracle, task_id) == [], _task_done(oracle, task_id)
            assert wait_until(lambda: _rows(oracle) == [task_id], 60), _facts(oracle, data, task_id, stub)
            # The hold a client reads is this park, with its numbers and no overview to choose.
            assert {key: value for key, value in _shown_hold(server, task_id).items() if key != "stop_identity"} == {
                "pause_id": parked["pause_id"], "pause_generation": parked["pause_generation"],
                "input_tokens": needed, "window_tokens": WINDOW, "reply_floor_tokens": REPLY_FLOOR, "over_tokens": 500}
            # The census every client lays its controls on offers this park's Resume, with the step as its reason.
            state = _api_status(server.base_url, "GET", "/api/state", timeout=120)["body"]
            census = [row for row in state.get("active_chat_activities") or [] if row.get("activity_id") == task_id]
            assert len(census) == 1 and census[0].get("can_resume") is True, (census, state.get("active_chat_activities"))
            assert (census[0].get("phase"), census[0].get("pause_cause"), census[0].get("pause_reason")) == (
                "budget_paused", "authority", WAIT_TEXT), census
            assert (census[0].get("stop") or {}).get("rail") == RAIL_CONTEXT_WINDOW, census
            # Nothing wakes it by itself: the park stands until the owner's Resume.
            assert not wait_until(lambda: _pause_row(data, task_id).get("state") != STATE_PAUSED, 15), \
                _facts(oracle, data, task_id, stub)
            assert stub.posts() == posts_before, stub.posts_since(posts_before)
            assert len(stub.work_rounds()) == rounds_at_pause

            # -----------------------------------------------------------
            # Identity is shortened; the next Resume continues the same saved work.
            # -----------------------------------------------------------
            _write_memory(server, stub, oracle, "SHORTENED", _identity(data, shortened))
            stub.finish = True
            resumed = _resume(server, task_id)
            assert resumed["status"] == 200 and resumed["body"].get("exact_continuation"), resumed
            result = wait_durable_result(oracle, task_id, timeout=600)
            assert result.get("status") == "completed" and FINAL in str(result.get("result") or ""), result
            consumed = _pause_row(data, task_id)
            assert consumed.get("state") == STATE_RESUMED and consumed.get("pause_id") == parked["pause_id"], consumed
            fitted = BACKGROUND + PRICE * (WORDS + shortened)
            assert fitted in stub.counts and fitted + REPLY_FLOOR <= WINDOW
            continued = stub.work_rounds()[rounds_at_pause:]
            assert continued, "the second Resume asked the model for nothing"
            # A restart would ask with the task's words alone; the continued request carries the rounds before it.
            assert any(message.get("role") in {"assistant", "tool"}
                       for message in continued[0].get("messages") or [] if isinstance(message, dict)), \
                "the Resume restarted instead of continuing the saved work"
            done = wait_until(lambda: _task_done(oracle, task_id) or None, 60)
            assert done and len(done) == 1 and done[0].get("status") == "completed", done
            _one_effect(data, stub)
            assert wait_until(lambda: _rows(oracle) == [], 60), _rows(oracle)
        finally:
            server.stop()


@pytest.mark.integration
@pytest.mark.timeout(LONG_SCENARIO_TIMEOUT_SEC)
def test_s48_the_owner_ends_a_window_parks_attempt_and_the_ordinary_continue_starts_one_successor(  # noqa: PLR0915
        s33_candidate, tmp_path_factory, serving_identity):  # noqa: F811
    require_lane(LANE_MOCK)
    from ouroboros.budget_pause import STATE_PAUSED, STATE_RESUMED
    from ouroboros.cold_return import FACTS_KEY

    root = tmp_path_factory.mktemp("s48")
    grown, shortened = 34, 30
    with _LocalLaneStub() as stub:
        stub.pid = serving_identity(stub.port)
        settings = _settings(stub)
        task_id, owner_pause, rounds_at_pause = _pause_the_work(
            s33_candidate, root, stub, settings, first_count=BACKGROUND + PRICE * WORDS)

        server = start_server(s33_candidate, root, settings)
        try:
            data, oracle = server.data_root, ArtifactOracle(server.data_root)
            _restored_queue(oracle, task_id)
            # -----------------------------------------------------------
            # An ordinary Pause offers neither choice, and a choice is read whole or not at all.
            # -----------------------------------------------------------
            paused = _pause_row(data, task_id)
            assert paused.get("pause_id") == owner_pause["pause_id"] and paused.get("state") == STATE_PAUSED, paused
            assert {key: value for key, value in _shown_hold(server, task_id).items() if key != "stop_identity"} == {}, "an ordinary Pause shows no window hold"
            _refused_and_unchanged(server, data, task_id, _named(paused, end_attempt=True))
            _refused_and_unchanged(server, data, task_id, _named(paused, overview_revision="0" * 64))
            for malformed in ({"end_attempt": True}, _named(paused), _named(paused, end_attempt=False),
                              _named(paused, end_attempt=True, overview_revision="0" * 64)):
                answer = _resume(server, task_id, malformed)
                assert answer["status"] == 400 and answer["body"].get("reason_code") == "invalid_context_choice", answer
            mixed = _post(server, f"/api/tasks/{task_id}/resume",
                          {"context_choice": _named(paused, end_attempt=True), "reason": "with another field"})
            assert mixed["status"] == 400 and mixed["body"].get("reason_code") == "invalid_context_choice", mixed
            assert _pause_row(data, task_id).get("pause_id") == paused["pause_id"] and _pause_row(data, task_id).get(
                "state") == STATE_PAUSED and not _pause_row(data, task_id).get("grant"), \
                "a refused request touched the ordinary Pause"

            # -----------------------------------------------------------
            # Identity grew; the plain Resume parks the same id on the window.
            # -----------------------------------------------------------
            _write_memory(server, stub, oracle, "GROWN", _identity(data, grown))
            posts_before = _settled_posts(stub, oracle)
            granted = _resume(server, task_id)
            assert granted["status"] == 200 and granted["body"].get("exact_continuation"), granted
            parked = _parked_after(data, task_id, owner_pause["pause_id"])
            assert parked and _waits_in_queue(oracle, task_id), _facts(oracle, data, task_id, stub)
            shown = _shown_hold(server, task_id)
            assert (shown.get("pause_id"), shown.get("pause_generation")) == (
                parked["pause_id"], parked["pause_generation"]) and "overview" not in shown, shown
            assert shown["over_tokens"] == parked[FACTS_KEY]["over_tokens"] == 500

            # A choice names the hold it was shown for: an earlier hold, another generation of this one, or an
            # overview this park never offered is refused, and the park stands as it was.
            _refused_and_unchanged(server, data, task_id, _named(owner_pause, end_attempt=True))
            _refused_and_unchanged(server, data, task_id, {**_named(shown, end_attempt=True),
                                                           "pause_generation": shown["pause_generation"] - 1})
            _refused_and_unchanged(server, data, task_id, {**_named(shown, end_attempt=True), "pause_id": "0" * 32})
            _refused_and_unchanged(server, data, task_id, _named(shown, overview_revision="0" * 64))
            _refused_and_unchanged(server, data, task_id, {**_named(shown, try_again=True), "stop_identity": "another-stop"})
            assert _waits_in_queue(oracle, task_id) and _task_done(oracle, task_id) == []

            # -----------------------------------------------------------
            # The owner ends this attempt: the size terminal with its recorded cause, nothing sent.
            # -----------------------------------------------------------
            ended = _resume(server, task_id, _named(shown, end_attempt=True))
            assert ended["status"] == 200 and ended["body"].get("exact_continuation"), ended
            result = wait_durable_result(oracle, task_id, timeout=420)
            assert (result.get("status"), result.get("reason_code")) == ("failed", "context_overflow"), result
            consumed = _pause_row(data, task_id)
            assert consumed.get("state") == STATE_RESUMED and consumed.get("pause_id") == parked["pause_id"], consumed
            done = wait_until(lambda: _task_done(oracle, task_id) or None, 60)
            assert done and len(done) == 1 and done[0].get("status") == "failed", done
            assert wait_until(lambda: _rows(oracle) == [], 60), _rows(oracle)
            # From the grown identity to here the refused return, four refused choices and the chosen end were the
            # only acts: the model was asked for nothing.
            assert _settled_posts(stub, oracle) == posts_before, stub.posts_since(posts_before)
            assert len(stub.work_rounds()) == rounds_at_pause
            read = _api_status(server.base_url, "GET", f"/api/tasks/{task_id}", timeout=120)["body"]
            offer = read.get("continuation_offer") or {}
            assert offer.get("eligible") is True and offer.get("cause") == "context_overflow", offer
            late = _resume(server, task_id, _named(shown, end_attempt=True))
            assert late["status"] in (404, 409), late  # an ended attempt has no hold left to choose about

            # -----------------------------------------------------------
            # The existing Continue: one successor under a new id, after room was made.
            # -----------------------------------------------------------
            _write_memory(server, stub, oracle, "SHORTENED", _identity(data, shortened))
            stub.finish = True
            pressed = _post(server, f"/api/tasks/{task_id}/continue", {"action_nonce": "cold-return-continue-1"})
            assert pressed["status"] == 200 and pressed["body"].get("ok"), pressed
            successor = str(pressed["body"].get("successor_task_id") or "")
            assert successor and successor != task_id, pressed
            again = _post(server, f"/api/tasks/{task_id}/continue", {"action_nonce": "cold-return-continue-1"})
            assert again["status"] == 200 and again["body"].get("successor_task_id") == successor, again
            other = _post(server, f"/api/tasks/{task_id}/continue", {"action_nonce": "cold-return-continue-2"})
            assert other["status"] == 409 and other["body"].get("successor_task_id") == successor, other
            continued = wait_durable_result(oracle, successor, timeout=600)
            assert continued.get("status") == "completed", continued
            kept = oracle.task_result(task_id)
            assert (kept.get("status"), kept.get("reason_code")) == ("failed", "context_overflow"), kept
            assert len(_task_done(oracle, task_id)) == 1
            _one_effect(data, stub)
        finally:
            server.stop()


@pytest.mark.integration
@pytest.mark.timeout(LONG_SCENARIO_TIMEOUT_SEC)
def test_s49_a_window_park_offers_the_current_overview_and_only_the_owners_choice_takes_it(  # noqa: PLR0915
        s33_candidate, tmp_path_factory, serving_identity):  # noqa: F811
    require_lane(LANE_MOCK)
    from ouroboros.budget_pause import STATE_RESUMED
    from ouroboros.cold_return import FACTS_KEY, OVERVIEW_KEY

    root = tmp_path_factory.mktemp("s49")
    held_markers, current_markers = 6, 1  # the overview this work starts with, and the one written while it is paused
    tight, roomy = 33, 30  # identity's markers: neither overview fits, then only the current one
    held = {}
    with _LocalLaneStub() as stub:
        stub.pid = serving_identity(stub.port)
        settings = _settings(stub)

        def first(server, oracle):
            _write_memory(server, stub, oracle, "OVERVIEW_HELD", _overview(server.data_root, held_markers, "first"))
            held["revision"] = _overview_revision(server.data_root)

        task_id, owner_pause, rounds_at_pause = _pause_the_work(
            s33_candidate, root, stub, settings, first=first,
            first_count=BACKGROUND + PRICE * (WORDS + held_markers))
        earlier = held["revision"]
        started = _system(stub.work_rounds()[0])
        assert "## Held knowledge notes" in started and f"Revision: {earlier}" in started, \
            "the work does not hold the overview it started with"

        server = start_server(s33_candidate, root, settings)
        try:
            data, oracle = server.data_root, ArtifactOracle(server.data_root)
            _restored_queue(oracle, task_id)
            # -----------------------------------------------------------
            # The overview is shortened and identity grows: with the held overview the saved work is 3,000 over,
            # and with the current one in its place still 500 over. The park offers nothing.
            # -----------------------------------------------------------
            _write_memory(server, stub, oracle, "OVERVIEW_CURRENT_AND_TIGHT",
                          _overview(data, current_markers, "second"), _identity(data, tight))
            current = _overview_revision(data)
            assert current != earlier
            posts_before, counts_before = _settled_posts(stub, oracle), len(stub.counts)
            assert _resume(server, task_id)["status"] == 200
            first_park = _parked_after(data, task_id, owner_pause["pause_id"])
            assert first_park and _waits_in_queue(oracle, task_id), _facts(oracle, data, task_id, stub)
            with_held = BACKGROUND + PRICE * (WORDS + held_markers + tight)
            assert first_park[FACTS_KEY]["input_tokens"] == with_held and first_park[FACTS_KEY]["over_tokens"] == 3_000
            assert OVERVIEW_KEY not in first_park[FACTS_KEY], first_park[FACTS_KEY]
            # Both forms were counted on the serving instance; the current one does not fit either.
            assert BACKGROUND + PRICE * (WORDS + current_markers + tight) in stub.counts[counts_before:], \
                stub.counts[counts_before:]
            shown = _shown_hold(server, task_id)
            assert shown.get("pause_id") == first_park["pause_id"] and OVERVIEW_KEY not in shown, shown
            _refused_and_unchanged(server, data, task_id, _named(shown, overview_revision=current))
            assert stub.posts() == posts_before, stub.posts_since(posts_before)

            # -----------------------------------------------------------
            # Identity is shortened a little: the held overview is still 1,500 over, the current one fits. The
            # park shows both revisions and both counts, and takes nothing by itself.
            # -----------------------------------------------------------
            _write_memory(server, stub, oracle, "ROOMY", _identity(data, roomy))
            posts_before, counts_before = _settled_posts(stub, oracle), len(stub.counts)
            assert _resume(server, task_id)["status"] == 200
            offering = _parked_after(data, task_id, first_park["pause_id"])
            assert offering and _waits_in_queue(oracle, task_id), _facts(oracle, data, task_id, stub)
            with_held = BACKGROUND + PRICE * (WORDS + held_markers + roomy)
            with_current = BACKGROUND + PRICE * (WORDS + current_markers + roomy)
            assert with_held + REPLY_FLOOR > WINDOW >= with_current + REPLY_FLOOR
            offer = {"held_revision": earlier, "current_revision": current, "input_tokens": with_current}
            assert offering[FACTS_KEY]["input_tokens"] == with_held and offering[FACTS_KEY]["over_tokens"] == 1_500
            assert offering[FACTS_KEY][OVERVIEW_KEY] == offer, offering[FACTS_KEY]
            # The recount may use a different reply cap, but must replay the refused input, including its clock.
            # Between the two recounts the overview's own texts in the system row change, and the host adds exactly
            # one row: its notice of the owner's choice, naming the earlier revision, immediately before the clock.
            counts, measured = stub.counts[counts_before:], stub.measured[counts_before:]
            assert counts.count(with_current) == 1, counts
            exchanged = counts.index(with_current)
            assert exchanged >= 2 and counts[exchanged - 1] == with_held, counts
            recounted, swapped = measured[exchanged - 1], measured[exchanged]
            refused = [_without_reply_cap(payload) for tokens, payload in zip(counts[:exchanged - 1], measured)
                       if tokens == with_held]
            replayed = _without_reply_cap(recounted)
            assert replayed in refused, _difference(refused[-1] if refused else {}, replayed)
            from ouroboros.send_clock import CLOCK_NOTE_PREFIX

            refused_rows, swapped_rows = recounted["messages"], swapped["messages"]
            # Every payload field but the rows is equal, an absent key never standing for a null one.
            assert {key: value for key, value in recounted.items() if key != "messages"} == {
                key: value for key, value in swapped.items() if key != "messages"}, _difference(recounted, swapped)
            assert message_text(refused_rows[-1]).startswith(CLOCK_NOTE_PREFIX), "the refused input ends with its clock"
            assert len(swapped_rows) == len(refused_rows) + 1, (len(refused_rows), len(swapped_rows))
            notice = swapped_rows[-2]
            assert notice == {"role": "user", "content": (
                "[SYSTEM NOTICE]\nThe owner chose the current overview for this work. The earlier overview remains "
                f"readable: knowledge_read(topic='overview', scope='global', version='{earlier}').")}, notice
            assert notice not in refused_rows, "the notice is the host's one new row"
            assert swapped_rows[-1] == refused_rows[-1], "the clock line is the same row, last"
            assert swapped_rows[1:-2] == refused_rows[1:-1], "every other row is exactly the refused input's"
            assert recounted["messages"][0] != swapped["messages"][0] and swapped["messages"][0].get("role") == "system"
            # The exchanged row is a request of a work holding the current overview: it names that revision and the
            # earlier one nowhere, an index entry or the address of an outlined index's whole text included.
            exchanged_row, refused_row = message_text(swapped["messages"][0]), message_text(recounted["messages"][0])
            assert f"Revision: {current}" in exchanged_row and earlier not in exchanged_row, "the exchange left a held text"
            assert f"Revision: {earlier}" in refused_row and current not in refused_row
            shown = _shown_hold(server, task_id)
            assert {key: value for key, value in shown.items() if key != "stop_identity"} == {
                "pause_id": offering["pause_id"], "pause_generation": offering["pause_generation"],
                "input_tokens": with_held, "window_tokens": WINDOW, "reply_floor_tokens": REPLY_FLOOR,
                "over_tokens": 1_500, OVERVIEW_KEY: offer,
            }, shown
            assert isinstance(shown["stop_identity"], str) and shown["stop_identity"], shown
            # The choice names this hold and the revision it counted: the first park, the held revision itself and
            # an unknown one are refused, and the park stands as it was.
            _refused_and_unchanged(server, data, task_id, _named(first_park, overview_revision=current))
            _refused_and_unchanged(server, data, task_id, _named(shown, overview_revision=earlier))
            _refused_and_unchanged(server, data, task_id, _named(shown, overview_revision="0" * 64))

            # -----------------------------------------------------------
            # The owner chooses the current overview. Its Resume is the real attempt on the held one again, which
            # is refused again; the park that follows saves the work holding the chosen revision.
            # -----------------------------------------------------------
            chosen = _resume(server, task_id, _named(shown, overview_revision=current))
            assert chosen["status"] == 200 and chosen["body"].get("exact_continuation"), chosen
            taken = _parked_after(data, task_id, offering["pause_id"])
            assert taken and _waits_in_queue(oracle, task_id), _facts(oracle, data, task_id, stub)
            assert taken[FACTS_KEY]["input_tokens"] == with_held, taken[FACTS_KEY]  # what was refused: the held one
            assert taken[FACTS_KEY][OVERVIEW_KEY] == {**offer, "taken": True}, taken[FACTS_KEY]
            after_choice = _shown_hold(server, task_id)
            assert after_choice[OVERVIEW_KEY] == {**offer, "taken": True}, after_choice
            assert isinstance(after_choice["stop_identity"], str) and after_choice["stop_identity"], after_choice
            assert after_choice["stop_identity"] != shown["stop_identity"], after_choice
            _refused_and_unchanged(server, data, task_id, _named(after_choice, overview_revision=current))
            assert _task_done(oracle, task_id) == []
            # Two refused returns and the refused choices asked the model for nothing.
            assert stub.posts() == posts_before, stub.posts_since(posts_before)
            assert len(stub.work_rounds()) == rounds_at_pause

            # -----------------------------------------------------------
            # The plain Resume continues the same saved work, now holding the current overview.
            # -----------------------------------------------------------
            stub.finish, counts_before = True, len(stub.counts)
            resumed = _resume(server, task_id)
            assert resumed["status"] == 200 and resumed["body"].get("exact_continuation"), resumed
            result = wait_durable_result(oracle, task_id, timeout=600)
            assert result.get("status") == "completed" and FINAL in str(result.get("result") or ""), result
            consumed = _pause_row(data, task_id)
            assert consumed.get("state") == STATE_RESUMED and consumed.get("pause_id") == taken["pause_id"], consumed
            assert with_current in stub.counts[counts_before:] and with_held not in stub.counts[counts_before:], \
                stub.counts[counts_before:]
            continued = stub.work_rounds()[rounds_at_pause:]
            assert continued, "the Resume after the choice asked the model for nothing"
            sent = _system(continued[0])
            assert f"Revision: {current}" in sent and f"Revision: {earlier}" not in sent, "the held overview was not exchanged"
            assert _marked(sent) & {f"{OVERVIEW}{index:03d}" for index in range(held_markers)} == {f"{OVERVIEW}000"}
            assert "second edition" in sent and "first edition" not in sent
            assert any(message.get("role") in {"assistant", "tool"}
                       for message in continued[0].get("messages") or [] if isinstance(message, dict)), \
                "the Resume restarted instead of continuing the saved work"
            done = wait_until(lambda: _task_done(oracle, task_id) or None, 60)
            assert done and len(done) == 1 and done[0].get("status") == "completed", done
            _one_effect(data, stub)
        finally:
            server.stop()
        # The replaced revision is still read exactly at its versioned address.
        from ouroboros.tools.registry import ToolRegistry

        # The notice's own reference is the genuine address of the earlier revision: read it as the notice names it.
        named = re.search(r"version='([^']+)'", message_text(notice))
        assert named and named.group(1) == earlier, message_text(notice)
        read = ToolRegistry(repo_dir=pathlib.Path(str(s33_candidate)), drive_root=data).execute(
            "knowledge_read", {"topic": "overview", "scope": "global", "version": named.group(1)})
        assert f"{OVERVIEW}{held_markers - 1:03d}" in read and "first edition" in read, read[-400:]


@pytest.mark.integration
@pytest.mark.timeout(LONG_SCENARIO_TIMEOUT_SEC)
def test_s50_the_chosen_overview_outlives_an_ordinary_stop_after_its_grant_was_spent(  # noqa: PLR0915
        s33_candidate, tmp_path_factory, serving_identity):  # noqa: F811
    require_lane(LANE_MOCK)
    from ouroboros.budget_pause import STATE_RESUMED
    from ouroboros.cold_return import CHOICE_KEY, FACTS_KEY, OVERVIEW_KEY

    root = tmp_path_factory.mktemp("s50")
    held_markers, current_markers, roomy = 6, 1, 30  # S49's numbers: the held overview is 1,500 over, the current fits
    held = {}
    with _LocalLaneStub() as stub:
        stub.pid = serving_identity(stub.port)
        settings = _settings(stub)

        def first(server, oracle):
            _write_memory(server, stub, oracle, "OVERVIEW_HELD", _overview(server.data_root, held_markers, "first"))
            held["revision"] = _overview_revision(server.data_root)

        task_id, owner_pause, rounds_at_pause = _pause_the_work(
            s33_candidate, root, stub, settings, first=first,
            first_count=BACKGROUND + PRICE * (WORDS + held_markers))
        earlier = held["revision"]
        with_held = BACKGROUND + PRICE * (WORDS + held_markers + roomy)
        with_current = BACKGROUND + PRICE * (WORDS + current_markers + roomy)

        # ---------------------------------------------------------------
        # Generation B: the park offers the current overview; the owner chooses it.
        # ---------------------------------------------------------------
        server = start_server(s33_candidate, root, settings)
        stopped = False
        try:
            data, oracle = server.data_root, ArtifactOracle(server.data_root)
            _restored_queue(oracle, task_id)
            _write_memory(server, stub, oracle, "OVERVIEW_CURRENT_AND_ROOMY",
                          _overview(data, current_markers, "second"), _identity(data, roomy))
            current = _overview_revision(data)
            posts_before = _settled_posts(stub, oracle)
            assert _resume(server, task_id)["status"] == 200
            offering = _parked_after(data, task_id, owner_pause["pause_id"])
            assert offering and _waits_in_queue(oracle, task_id), _facts(oracle, data, task_id, stub)
            offer = {"held_revision": earlier, "current_revision": current, "input_tokens": with_current}
            assert offering[FACTS_KEY][OVERVIEW_KEY] == offer, offering[FACTS_KEY]
            shown = _shown_hold(server, task_id)
            choice = _named(shown, overview_revision=current)

            # The chosen Resume's worker takes the saved work over and spends the grant; its own attempt then asks
            # the serving instance for a count, which is held open. The application stops there.
            stub.holding = True
            chosen = _resume(server, task_id, choice)
            assert chosen["status"] == 200 and chosen["body"].get("exact_continuation"), chosen
            consumed = _stopped_after_the_grant_was_spent(stub, server, data, task_id, offering["pause_id"])
            stopped = True
            assert consumed.get("state") == STATE_RESUMED and (consumed.get("grant") or {}).get("consumed_at"), consumed
            assert (consumed.get("grant") or {}).get(CHOICE_KEY) == choice, consumed.get("grant")
            assert _task_done(ArtifactOracle(data), task_id) == [], "the stop ended the work it should have kept"
        finally:
            if not stopped:
                stub.holding = False
                server.stop()
        assert len(stub.work_rounds()) == rounds_at_pause, "the model was asked before the stop"

        # ---------------------------------------------------------------
        # Generation C: the saved work returns from its checkpoint, not from that pause, and still carries the
        # choice. The held overview is refused again; the park that follows holds the chosen revision.
        # ---------------------------------------------------------------
        server = start_server(s33_candidate, root, settings)
        try:
            data, oracle = server.data_root, ArtifactOracle(server.data_root)
            counts_before = len(stub.counts)
            taken = wait_until(lambda: _next_window_park(data, task_id, offering["pause_id"]), 45)
            returned_by = "itself"
            if not taken:
                # A stop keeps saved work for its Resume where nothing returns it by itself (as S42's stop hold
                # does): the plain one, with no choice in it. What it restores is the checkpoint, which carries
                # the owner's.
                returned_by = _resume(server, task_id)
                taken = _parked_after(data, task_id, offering["pause_id"])
            assert taken, (returned_by, _facts(oracle, data, task_id, stub), _standing(data, task_id))
            assert taken[FACTS_KEY]["input_tokens"] == with_held, taken[FACTS_KEY]  # what was refused: the held one
            assert taken[FACTS_KEY][OVERVIEW_KEY] == {**offer, "taken": True}, (returned_by, taken[FACTS_KEY])
            assert with_held in stub.counts[counts_before:] and with_current in stub.counts[counts_before:], \
                stub.counts[counts_before:]
            assert _waits_in_queue(oracle, task_id) and _task_done(oracle, task_id) == []
            assert _shown_hold(server, task_id)[OVERVIEW_KEY] == {**offer, "taken": True}
            # The refused returns on both sides of the stop asked the model for nothing on this work's behalf
            # (a boot may ask for its own; this work's rounds are told apart by its words).
            _settled_posts(stub, oracle)
            assert [kind for kind, whose in stub.posts_since(posts_before) if whose == MARK] == [], \
                stub.posts_since(posts_before)
            assert len(stub.work_rounds()) == rounds_at_pause

            # The plain Resume continues the same saved work, now holding the current overview.
            stub.finish = True
            assert _resume(server, task_id)["status"] == 200
            result = wait_durable_result(oracle, task_id, timeout=600)
            assert result.get("status") == "completed" and FINAL in str(result.get("result") or ""), result
            continued = stub.work_rounds()[rounds_at_pause:]
            assert continued, "the Resume after the choice asked the model for nothing"
            sent = _system(continued[0])
            assert f"Revision: {current}" in sent and f"Revision: {earlier}" not in sent, "the held overview was not exchanged"
            assert any(message.get("role") in {"assistant", "tool"}
                       for message in continued[0].get("messages") or [] if isinstance(message, dict)), \
                "the Resume restarted instead of continuing the saved work"
            done = wait_until(lambda: _task_done(oracle, task_id) or None, 60)
            assert done and len(done) == 1 and done[0].get("status") == "completed", done
            _one_effect(data, stub)
        finally:
            server.stop()
