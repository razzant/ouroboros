"""Owner-facing host notices (#1369): the provider terminal sentence and the fallback
notice (#2154) through their real producers, and the readers of old late-review rows.

The record keeps its tokens — the exception repr, the provider's typed fields — in the
``llm_api_error`` event and Logs; the row a person reads says the same facts in words and
points at the exact record. The retired task acceptance wrote late-review rows and their
applied records; nothing writes one now, but the stored ones still render: the projection
bounds an old late note in words, history replays its evidence pointer, and the task
artifact route serves exactly this task's own old record.
"""

from __future__ import annotations

from tests import _dialogue as dialogue_test

import asyncio
import json
from types import SimpleNamespace

import pytest

from ouroboros.loop_transport import provider_failure_hint, provider_terminal_fallback_text


LONG_SUMMARY = ("The budget section is complete and every claim is backed by the attached ledger, "
                "including the two contested rows. ") * 8
REPR = "PermissionDeniedError(\"Error code: 403 - {'error': {'message': 'The API key was revoked.', 'code': 403}}\")"


def _run(**overrides):
    """An old host task-acceptance run, as the retired machine stored it."""
    run = {
        "authority": "host_root", "aggregate_signal": "PASS", "panel_id": "panel_1",
        "request": {"surface": "task_acceptance", "subject": "answer A", "retry_key": "rk"},
        "slot_roster": [{"slot_id": "triad_w45a8z", "model": "openai/gpt-5.5", "route": "agent_session"},
                        {"slot_id": "triad_bkydwq", "model": "anthropic/claude-opus-5-5", "route": "api_chat"},
                        {"slot_id": "triad_qq11zz", "model": "claudexor::codex=gpt-6-astra", "route": "agent_session"},
                        {"slot_id": "triad_late01", "model": "x/late", "route": "api_chat"}],
        "actors": [
            {"slot_id": "triad_w45a8z", "model": "openai/gpt-5.5", "status": "ok", "operation_state": "settled",
             # A delegated engine run reports its final attempt's model: an observation.
             "usage": {"resolved_model": "openai/gpt-5.5-2026-09",
                       "observed_attempt": {"harness_id": "codex", "model": "openai/gpt-5.5-2026-09"}},
             "semantic_verdict": "PASS",
             "parsed": {"verdict": "PASS", "summary": LONG_SUMMARY, "findings": []}},
            {"slot_id": "triad_bkydwq", "model": "anthropic/claude-opus-5-5", "status": "ok", "operation_state": "settled",
             "semantic_verdict": "DEGRADED",
             "parsed": {"verdict": "DEGRADED", "summary": "The ledger could not be opened from the packet.", "findings": []}},
            {"slot_id": "triad_qq11zz", "model": "claudexor::codex=gpt-6-astra", "status": "error", "operation_state": "settled",
             "error": "ClaudexorModelError: engine_busy (HTTP 429)", "reported_cause": "The engine is rate limited"},
            {"slot_id": "triad_late01", "model": "x/late", "operation_state": "pending_dispatch"},
        ],
    }
    run.update(overrides)
    return run


# ---------------------------------------------------------------------------
# An old late note on its stored panel
# ---------------------------------------------------------------------------

def test_an_old_late_note_is_cut_in_words_and_names_the_record_only_when_stored():
    """Five reviewers at the per-seat note bound composed a note past the projection's
    2000-char display bound. The card and the owner's row print the projected note: its
    cut is said in words, naming the review record only when one is stored, and the raw
    run keeps every byte."""
    from ouroboros import review_projection

    note = "\n".join(["On the delivered version of this answer, reviewers later passed it.", *(
        f"- provider/reviewer-{i} (requested): passed it — {LONG_SUMMARY[:400].rstrip()}… (shortened)"
        for i in range(5))])
    marker = f"\n⚠️ OMISSION NOTE: truncated at 2000 chars; original length {len(note)}"
    assert len(note) - 2000 > len(marker), "past the 2000-char bound and its anti-waste floor"
    stored = _run(late_settlement={"note": note}, applied_source_status="available")
    panel = review_projection.compact_review_projection([stored])["panels"][0]
    assert panel["applied_source_status"] == "available"
    assert panel["late_settlement"]["note"] == (
        note[:2000].rstrip() + "… (shortened; the complete text is in the review record)")
    assert stored["late_settlement"]["note"] == note, "the raw run keeps every byte"
    unstored = {**stored, "applied_source_status": "unavailable"}
    assert (review_projection.compact_review_projection([unstored])["panels"][0]["late_settlement"]["note"]
            == note[:2000].rstrip() + "… (shortened)")


# ---------------------------------------------------------------------------
# The provider terminal sentence
# ---------------------------------------------------------------------------

class _BodyError(Exception):
    def __init__(self, message, *, status_code, body=None):
        super().__init__(message)
        self.status_code = status_code
        if body is not None:
            self.body = body


class _Raising:
    def __init__(self, error):
        self.error, self.calls = error, 0

    def chat(self, **_kwargs):
        self.calls += 1
        raise self.error


class _Events:
    def __init__(self):
        self.events = []

    def put(self, event):
        self.events.append(event)

    put_nowait = put


def _call(tmp_path, llm, usage):
    from ouroboros.loop_llm_call import call_llm_with_retry

    return call_llm_with_retry(llm, [{"role": "user", "content": "hi"}], "openai/fixture", None, "high", 3,
                               tmp_path, "task-403", 1, _Events(), usage, "task", False)


def test_the_exception_owner_stamps_only_the_providers_own_sentence(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.loop_llm_call.time.sleep", lambda _s: pytest.fail("a 403 must not retry"))
    body = {"error": {"message": "The API key was revoked.", "code": 403, "metadata": {"provider_name": None}}}
    llm = _Raising(_BodyError(f"Error code: 403 - {body}", status_code=403, body=body))
    usage = {}
    msg, _cost = _call(tmp_path, llm, usage)
    assert msg is None and llm.calls == 1
    assert usage["_last_llm_error_kind"] == "auth_error"
    assert usage["_last_llm_provider_message"] == "The API key was revoked."
    assert "Error code" in usage["_last_llm_error"], "the diagnostic keeps the exception text"
    hint = provider_failure_hint(usage)
    assert hint == ' The provider said: "The API key was revoked."'
    text = provider_terminal_fallback_text(usage, is_context_overflow=False, is_transport_wait=False,
                                           waited_sec=0.0, is_deadline_exhausted=False)
    assert text.startswith("⚠️ The model provider returned no usable response. The provider said: \"The API key was revoked.\"")
    assert "Error code" not in text and "{" not in text and "PermissionDenied" not in text
    # The durable event keeps the diagnostic message for Logs.
    rows = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    event, = [row for row in rows if row["type"] == "llm_api_error"]
    assert event["provider_message"] == "The API key was revoked." and "Error code" in event["error"]


def test_without_a_provider_sentence_the_host_classifies_and_points_at_logs(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.loop_llm_call.time.sleep", lambda _s: pytest.fail("a 401 must not retry"))
    llm = _Raising(_BodyError("AuthenticationError('401 invalid_api_key')", status_code=401))
    usage = {"_last_llm_provider_message": "stale sentence from an earlier round"}
    msg, _cost = _call(tmp_path, llm, usage)
    assert msg is None and "_last_llm_provider_message" not in usage
    assert usage["_last_llm_error_kind"] == "auth_error"
    hint = provider_failure_hint(usage)
    assert hint == (" No readable provider message is available; the host classified the failure as "
                    "an authentication or authorization refusal. The task's Logs keep the host's record of this failure.")
    assert "AuthenticationError" not in hint and "stale sentence" not in hint


def test_a_body_error_inside_an_empty_response_carries_the_providers_sentence(tmp_path):
    class _Empty:
        calls = 0

        def chat(self, **_kwargs):
            self.calls += 1
            return ({"content": "", "finish_reason": "stop"},
                    {"provider_error": {"code": "401", "type": "invalid_request_error",
                                        "message": "Incorrect API key provided.", "kind": "provider_error"},
                     "prompt_tokens": 3, "completion_tokens": 0})

    llm, usage = _Empty(), {}
    msg, _cost = _call(tmp_path, llm, usage)
    assert msg is None and llm.calls == 1
    assert usage["_last_llm_provider_message"] == "Incorrect API key provided."
    assert provider_failure_hint(usage) == ' The provider said: "Incorrect API key provided."'


def test_a_body_error_sentence_its_producer_cut_is_quoted_as_shortened(tmp_path):
    """The HTTP-200 body-error producer keeps a bounded ``message`` and says when it
    cut it; the owner's row then never closes that prefix as the provider's whole
    sentence. Classification and retry read code/type/kind and do not change."""
    from ouroboros.llm import LLMClient

    client = LLMClient()
    target = client._resolve_remote_target("deepseek/deepseek-v4-flash-0731")
    sentence = "Your organisation has used its whole monthly request allowance for this model family. " * 6

    def produced(message):
        body = {"id": "gen-1", "choices": [{"message": {"content": ""}, "finish_reason": "stop"}],
                "error": {"code": 402, "message": message}}
        return client._normalize_remote_response(body, target, skip_cost_fetch=True)[1]

    short = produced("Insufficient credits.")
    assert short["provider_error"]["message"] == "Insufficient credits." and "message_truncated" not in short["provider_error"]
    cut = produced(sentence)
    assert cut["provider_error"]["message"] == sentence[:300] and cut["provider_error"]["message_truncated"] is True

    class _Empty:
        calls = 0

        def chat(self, **_kwargs):
            self.calls += 1
            return {"content": "", "finish_reason": "stop"}, {**cut, "prompt_tokens": 3, "completion_tokens": 0}

    llm, usage = _Empty(), {}
    msg, _cost = _call(tmp_path, llm, usage)
    assert msg is None and llm.calls == 1 and usage["_last_llm_error_kind"] == "provider_body_error"
    assert usage["_last_llm_provider_message_cut"] is True
    assert provider_failure_hint(usage) == (f' The provider said: "{" ".join(sentence[:300].split())}…" (shortened). '
                                            "The task's Logs keep the host's record of this failure.")


def test_a_usable_response_clears_the_stale_provider_sentence(tmp_path):
    class _Ok:
        def chat(self, **_kwargs):
            return {"content": "fine"}, {"prompt_tokens": 3, "completion_tokens": 1}

    usage = {"_last_llm_provider_message": "The API key was revoked.", "_last_llm_provider_message_cut": True,
             "_last_llm_error": REPR, "_last_llm_error_kind": "auth_error"}
    msg, _cost = _call(tmp_path, _Ok(), usage)
    assert msg["content"] == "fine"
    assert not any(key in usage for key in ("_last_llm_provider_message", "_last_llm_provider_message_cut",
                                            "_last_llm_error", "_last_llm_error_kind"))
    assert provider_failure_hint(usage) == ""


def test_the_hint_quotes_the_providers_sentence_and_never_a_repr():
    """A stored sentence is quoted whitespace-normalised; a repr beside a typed kind,
    or a legacy record with a repr and no kind, reads as the host's classification."""
    typed = {"_last_llm_error": f"  {REPR}  ", "_last_llm_error_kind": "auth_error"}
    spoken = provider_failure_hint({**typed, "_last_llm_provider_message": "  Your API key was  revoked.  "})
    assert spoken == ' The provider said: "Your API key was revoked."'
    assert provider_failure_hint(typed) == (" No readable provider message is available; the host classified the failure as "
                                            "an authentication or authorization refusal. The task's Logs keep the "
                                            "host's record of this failure.")
    legacy = provider_failure_hint({"_last_llm_error": "AuthenticationError('401 invalid_api_key')"})
    assert legacy == (" No readable provider message is available; the host classified the failure as a provider failure. "
                      "The task's Logs keep the host's record of this failure.")
    for text in (spoken, provider_failure_hint(typed), legacy):
        assert "PermissionDenied" not in text and "AuthenticationError" not in text and "{" not in text


def test_a_long_provider_sentence_is_shortened_in_words_beside_the_logs_pointer(tmp_path, monkeypatch):
    """No silent cut on the owner's row: the sentence is stamped whole, the SSOT
    display bound decides the cut (its anti-waste floor lets a small overflow
    through whole), and a cut is said in words with the pointer to Logs."""
    monkeypatch.setattr("ouroboros.loop_llm_call.time.sleep", lambda _s: pytest.fail("a 403 must not retry"))
    sentence = "Access to this model was denied for the configured project. " * 14  # ~840 chars
    body = {"error": {"message": sentence, "code": 403}}
    usage = {}
    msg, _cost = _call(tmp_path, _Raising(_BodyError(f"Error code: 403 - {body}", status_code=403, body=body)), usage)
    assert msg is None and usage["_last_llm_provider_message"] == sentence.strip(), "stamped whole, never sliced"
    hint = provider_failure_hint(usage)
    quoted = sentence.strip()[:600].rstrip()
    assert hint == (f' The provider said: "{quoted}…" (shortened). '
                    "The task's Logs keep the host's record of this failure.")
    rows = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    event, = [row for row in rows if row["type"] == "llm_api_error"]
    assert sentence.strip() in " ".join(event["error"].split()), "this body error's text carries the whole sentence"
    # Barely over the bound: cheaper than a disclosure, so it passes whole.
    barely = "x" * 630
    assert provider_failure_hint({"_last_llm_provider_message": barely}) == f' The provider said: "{barely}"'


def test_a_shortened_stream_sentence_promises_logs_no_bytes_they_do_not_hold(tmp_path, monkeypatch):
    """An SSE error frame's sentence lives only in its body: the exception text never
    carries it and the durable event keeps the bounded excerpt, so the owner's row
    points at the host's record without saying the rest of the sentence is there."""
    from ouroboros.llm_stream import ProviderStreamError

    monkeypatch.setattr("ouroboros.loop_llm_call.time.sleep", lambda _s: pytest.fail("a 403 must not retry"))
    sentence = "The upstream stream was closed for this project by the provider's access guard. " * 12
    usage = {}
    msg, _cost = _call(tmp_path, _Raising(ProviderStreamError({"error": {"message": sentence, "code": 403}})), usage)
    assert msg is None and usage["_last_llm_provider_message"] == sentence.strip()
    rows = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    event, = [row for row in rows if row["type"] == "llm_api_error"]
    kept = " ".join([event["error"], event["provider_message"]])
    assert sentence.strip() not in kept and len(event["provider_message"]) <= 600, "Logs hold only an excerpt"
    quoted = sentence.strip()[:600].rstrip()
    assert provider_failure_hint(usage) == (f' The provider said: "{quoted}…" (shortened). '
                                            "The task's Logs keep the host's record of this failure.")


def test_owner_provider_message_reads_body_and_stream_shapes_never_engine_display():
    from ouroboros.llm_claudexor import ClaudexorModelError
    from ouroboros.loop_transport import owner_provider_fields, owner_provider_message

    problem = {"code": "provider_failed", "message": "Codex model request was refused (HTTP 400).",
               "context": {"httpStatus": 400, "stage": "response", "errorCode": "vendor_refusal",
                           "vendorCode": "string_above_max_length", "parameter": "instructions",
                           "providerMessage": "private-body"}}
    engine = ClaudexorModelError(problem)
    assert "stage=response" in engine.display_message, "the fixture's display text is host-composed"
    assert owner_provider_message(engine) == "", "an engine's composed display text is no provider sentence"
    assert owner_provider_fields(engine) == {"code": "string_above_max_length", "parameter": "instructions"}
    assert owner_provider_fields(ClaudexorModelError(problem, unknown=True)) == {}, "an unknown outcome exposes none"
    assert owner_provider_fields(_BodyError("x", status_code=400, body={"message": "m"})) == {}
    assert owner_provider_message(_BodyError("x", status_code=400, body={"message": "  Flat  body  message. "})) == "Flat body message."
    stream = RuntimeError("stream died")
    stream.provider_message = "Upstream closed the stream."
    assert owner_provider_message(stream) == "Upstream closed the stream."
    assert owner_provider_message(RuntimeError(REPR)) == "", "a bare exception has no provider sentence"
    # An empty response's body-error dict: its own message, or nothing.
    assert owner_provider_message({"code": "401", "message": "  Incorrect  API key. "}) == "Incorrect API key."
    assert owner_provider_message({"code": "401", "message": {"nested": "not a sentence"}}) == ""


@pytest.mark.parametrize("unknown", [False, True])
def test_a_claudexor_refusal_never_reads_as_the_providers_sentence(tmp_path, monkeypatch, unknown):
    """The engine's display text (stage, cause, engine code and message) is the
    host's record: it stays in Logs. The owner's row names the provider's own typed
    fields as the provider's, the host's classification as the host's, and a stale
    sentence or field from an earlier round never speaks for this failure."""
    from ouroboros.llm_claudexor import ClaudexorModelError

    monkeypatch.setattr("ouroboros.loop_llm_call.time.sleep", lambda _s: None)
    error = ClaudexorModelError({"code": "provider_failed", "message": "Codex model request was refused (HTTP 400).",
                                 "context": {"httpStatus": 400, "stage": "response", "errorCode": "vendor_refusal",
                                             "vendorCode": "string_above_max_length", "parameter": "instructions",
                                             "providerMessage": "private-body"}}, unknown=unknown)
    usage = {"_last_llm_provider_message": "stale sentence", "_last_llm_provider_fields": {"code": "stale_code"}}
    msg, _cost = _call(tmp_path, _Raising(error), usage)
    assert msg is None and "_last_llm_provider_message" not in usage
    hint = provider_failure_hint(usage)
    for text in ("The provider said", "stage=", "cause=", "provider_failed", "model_outcome_unknown",
                 "Codex model request was refused", "private-body", "stale"):
        assert text not in hint, text
    assert hint.endswith("The task's Logs keep the host's record of this failure.")
    if unknown:
        assert "_last_llm_provider_fields" not in usage
        assert hint.startswith(" No readable provider message is available; the host classified the failure as ")
    else:
        assert usage["_last_llm_provider_fields"] == {"code": "string_above_max_length", "parameter": "instructions"}
        assert hint == (' The provider reported error code "string_above_max_length" and parameter "instructions" '
                        "but no sentence the host can quote; the host classified the failure as a rejected request. "
                        "The task's Logs keep the host's record of this failure.")
    rows = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert all(row["error"] == error.display_message for row in rows if row["type"] == "llm_api_error"), \
        "Logs keep the host's composed diagnostic"


def test_a_usable_response_clears_stale_provider_fields(tmp_path):
    class _Ok:
        def chat(self, **_kwargs):
            return {"content": "fine"}, {"prompt_tokens": 3, "completion_tokens": 1}

    usage = {"_last_llm_provider_fields": {"code": "string_above_max_length"}, "_last_llm_error_kind": "bad_request"}
    msg, _cost = _call(tmp_path, _Ok(), usage)
    assert msg["content"] == "fine" and "_last_llm_provider_fields" not in usage
    assert provider_failure_hint(usage) == ""


# ---------------------------------------------------------------------------
# The fallback notice: chain → task emitter → chat delivery → card and toast
# ---------------------------------------------------------------------------

# The card's line and the toast, painted by their own readers from the delivered rows.
_CARD_READERS = """
import { readFileSync } from 'node:fs';
import { buildTimelineItemHtml } from './modules/chat_activity.js';
import { showTaskIncidentToast } from './modules/chat_media.js';
import { summarizeChatLiveEvent } from './modules/log_events.js';
import { installDom, restoreDom } from './tests/chat_dom_fixture.js';
const { live, replay } = JSON.parse(readFileSync(process.env.OURO_TEST_FALLBACK_ROWS, 'utf8'));
const env = installDom();
try {
    const painted = live.map((row) => {
        const view = summarizeChatLiveEvent(row);
        const toast = showTaskIncidentToast(row);
        const shown = toast && { text: toast.textContent, children: toast.children.length };
        toast?.dispose();
        return { full: view.fullHeadline, promote: view.promote, toast: shown,
            line: buildTimelineItemHtml({ ...view, lineKey: 'k' }, { expandedLineKeys: new Set(['k']), groupId: 'g' }) };
    });
    process.stdout.write(JSON.stringify({ live: painted, replay: replay.map((row) => summarizeChatLiveEvent(row).fullHeadline) }));
} finally { restoreDom(env.prior); }
"""


def test_each_fallback_notice_reaches_the_card_in_its_own_routes_words(tmp_path, monkeypatch):
    """The primary and two candidates refuse, the third answers. Through the real loop, chain
    and call owner, the task's emitter, chat delivery, history and the card's and toast's own
    readers, each notice names the route that just failed with only that route's class and
    sentence, before the next route is dialed: a refused route is never announced as the
    answer. A provider's markup reaches the owner as inert text, its secret redacted; the
    record keeps the typed kind; each switch keeps its own toast key."""
    import os
    import pathlib
    import queue
    import shutil
    from functools import partial

    from ouroboros import fallback_cooldown, loop
    from ouroboros.agent import OuroborosAgent
    from ouroboros.dialogue_import import export_for_older_body
    from ouroboros.gateway.history import make_chat_history_endpoint
    from ouroboros.tools.registry import ToolRegistry
    from supervisor import events_chat_delivery, message_bus
    from tests.group_run import run_in_group

    fallback_cooldown.reset_for_tests()
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "alt/a,alt/b,alt/c")
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    # A bare loop has no agent-built context core, as in the chain's own tests.
    monkeypatch.setattr(loop, "_rebind_context_fit_plan", lambda *a, **k: (None, "max"))
    monkeypatch.setattr("ouroboros.loop_llm_call.time.sleep", lambda _s: pytest.fail("a refusal is not retried"))
    secret = "sk-or-" + "fixture" * 6
    refusals = {
        "test-model": _BodyError("Error code: 400", status_code=400,
                                 body={"error": {"message": "Upstream request failed."}}),
        "alt/a": _BodyError("AuthenticationError('401 invalid_api_key')", status_code=401),
        "alt/b": _BodyError("Error code: 400", status_code=400, body={"error": {
            "message": f"Field <img src=x onerror=alert(1)> is not allowed for key {secret}."}}),
    }

    class _Routes:
        def __init__(self):
            self.models = []

        def default_model(self):
            return "test-model"

        def chat(self, **kwargs):
            self.models.append(kwargs["model"])
            if kwargs["model"] in refusals:
                raise refusals[kwargs["model"]]
            return {"role": "assistant", "content": "done"}, {"prompt_tokens": 1, "completion_tokens": 1}

    events, llm, work, room = queue.Queue(), _Routes(), tmp_path / "work", tmp_path / "room"
    work.mkdir()
    agent = SimpleNamespace(_last_progress_ts=None, _event_queue=events, _current_chat_id=1,
                            _current_task_id="task-1", tools=SimpleNamespace(_ctx=SimpleNamespace(task_attempt=0)),
                            _subagent_progress_meta=lambda _event: {})
    result, _usage, _trace = loop.run_llm_loop(
        messages=[{"role": "user", "content": "go"}], tools=ToolRegistry(repo_dir=work, drive_root=work), llm=llm,
        drive_logs=work, emit_progress=partial(OuroborosAgent._emit_progress, agent),
        incoming_messages=queue.Queue(), task_id="task-1", drive_root=work)

    assert result == "done" and llm.models == ["test-model", "alt/a", "alt/b", "alt/c"], "one call per route"
    rows = [json.loads(line) for line in (work / "events.jsonl").read_text().splitlines()]
    assert [(row["model"], row["error_kind"]) for row in rows if row["type"] == "llm_api_error"] == [
        ("test-model", "bad_request"), ("alt/a", "auth_error"), ("alt/b", "bad_request")], "the record keeps the kind"
    expected = [
        'Fallback: test-model → alt/a. Trying the next model after a rejected request. '
        'The provider said: "Upstream request failed."',
        "Fallback: alt/a → alt/b. Trying the next model after an authentication or authorization refusal; "
        "sign-in is still needed there.",
        "Fallback: alt/b → alt/c. Trying the next model after a rejected request. "
        'The provider said: "Field <img src=x onerror=alert(1)> is not allowed for key ***REDACTED***."',
    ]
    frames = [events.get_nowait() for _ in range(events.qsize())]
    notes = [frame for frame in frames if frame["text"].startswith("💬 ⚡ Fallback: ")]
    assert [frame["text"] for frame in notes] == [f"💬 ⚡ {text}" for text in expected]

    (room / "logs").mkdir(parents=True)
    (room / "logs" / "chat.jsonl").touch()
    live, bridge = [], message_bus.LocalChatBridge()
    bridge._broadcast_fn = live.append
    monkeypatch.setattr(message_bus, "DATA_DIR", room)
    monkeypatch.setattr(message_bus, "load_state", lambda: {"owner_id": 1})
    monkeypatch.setattr(message_bus, "_BRIDGE", bridge)
    monkeypatch.setattr(message_bus, "publish_event", lambda *_: None)
    monkeypatch.setattr(events_chat_delivery, "_bound_project_chat_id", lambda *_: 0)
    delivery = SimpleNamespace(DRIVE_ROOT=room, RUNNING={"task-1": {"task": {"id": "task-1", "_attempt": 0}}},
                               send_with_budget=message_bus.send_with_budget,
                               append_jsonl=lambda *_: pytest.fail("delivery raised"))
    for frame in notes:
        events_chat_delivery._handle_send_message(frame, delivery)
    assert [(row["task_incident"], row["toast_once"]) for row in live] == [
        ("model_lane_switch", f"task-1:model_lane_switch:1:{name}") for name in ("alt/a", "alt/b", "alt/c")]
    export_for_older_body(room)
    response = asyncio.run(make_chat_history_endpoint(room)(SimpleNamespace(query_params={"limit": "10"})))
    replay = [row for row in json.loads(response.body)["messages"] if row.get("is_progress")]
    assert secret not in json.dumps([live, replay])

    shown = tmp_path / "rows.json"
    shown.write_text(json.dumps({"live": live, "replay": replay}), encoding="utf-8")
    node = shutil.which("node")
    assert node, "Node is required for the real card and toast consumer proof"
    run = run_in_group([node, "--input-type=module", "-e", _CARD_READERS],
                       cwd=pathlib.Path(__file__).resolve().parents[1] / "web",
                       env={**os.environ, "OURO_TEST_FALLBACK_ROWS": str(shown)}, timeout=60)
    assert run.returncode == 0, run.stdout + run.stderr
    painted = json.loads(run.stdout)
    assert [view["full"] for view in painted["live"]] == expected == painted["replay"]
    assert [view["toast"] for view in painted["live"]] == [{"text": text, "children": 0} for text in expected]
    assert not any(view["promote"] for view in painted["live"]), "a host note never claims the card title"
    assert all("<img" not in view["line"] for view in painted["live"])
    assert "Field &lt;img src=x onerror=alert(1)&gt; is not allowed" in painted["live"][2]["line"]


# ---------------------------------------------------------------------------
# The late-evidence carrier: stored row → history, and the record route
# ---------------------------------------------------------------------------

EVIDENCE = {"task_id": "t1", "panel_id": "panel_1", "settled_at": "2026-09-28T00:06:00+00:00",
            "reviewed_revision": "delivered", "reviewed_is_emitted": True,
            "source_ref": {"kind": "task_source", "root": "artifact_store", "sha256": "a" * 64, "size": 75000,
                           "path": f"source_handles/context_checkpoints/acceptance-{'a' * 64}.json"}}


def test_an_old_late_row_replays_its_evidence_pointer_through_history(tmp_path):
    """A late-review row stored before the fold carries its evidence pointer; history replays
    it on that row only, and the bus persists the pointer for no row it writes now."""
    from ouroboros import dialogue_store
    from ouroboros.gateway.history import make_chat_history_endpoint
    from supervisor import message_bus
    from supervisor.message_ingress import _chat_record

    text = "On the delivered version of this answer, reviewers later passed it."
    meta = {"card_row": "reviews", "card_row_id": "acceptance-late:rk", "late_evidence": EVIDENCE}
    _root, old_row = _chat_record("system", 1, 0, text, record_type="acceptance_late_settlement", task_id="t1",
                                  message_meta=meta, drive_root=tmp_path)
    old_row["late_evidence"] = dict(EVIDENCE)  # the field the retired writer stored on this row type
    dialogue_store.append(tmp_path, "chat", old_row)
    message_bus.log_chat("system", 1, 0, "Open delegated execution: run-x.", record_type="custody_notice",
                         task_id="t1", message_meta={**meta, "card_row": "timeline"}, drive_root=tmp_path)
    stored = dialogue_test.log_rows(tmp_path / "logs" / "chat.jsonl")
    assert stored[0]["late_evidence"] == EVIDENCE and "late_evidence" not in stored[1]
    response = asyncio.run(make_chat_history_endpoint(tmp_path)(SimpleNamespace(query_params={"chat_id": "1"})))
    rows = {row["text"]: row for row in json.loads(response.body)["messages"]}
    late = rows[text]
    assert late["card_row"] == "reviews" and late["late_evidence"] == EVIDENCE
    assert "late_evidence" not in rows["Open delegated execution: run-x."], "only the stored late row carries the pointer"


def test_the_gateway_contract_names_the_pointer_in_both_mirrors():
    import pathlib
    from typing import get_type_hints

    from ouroboros.gateway.contracts import ChatOutbound

    assert "late_evidence" in get_type_hints(ChatOutbound, include_extras=True)
    text = (pathlib.Path(__file__).resolve().parents[1] / "web" / "modules" / "api_types.js").read_text(encoding="utf-8")
    assert "@property {Object=} late_evidence" in text


@pytest.mark.serial
def test_the_record_route_serves_only_this_tasks_own_old_acceptance_record(tmp_path):
    """An old late notice keeps linking the applied record it was sent with after a newer
    publication replaced its panel's ref: the task artifact route resolves that record by
    this task's host acceptance membership (``task_finalization.host_acceptance_source``);
    a forged, foreign or altered pointer resolves nothing."""
    import hashlib
    import pathlib

    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros import artifacts, review_projection
    from ouroboros.gateway.tasks import api_task_artifact
    from ouroboros.task_results import write_task_result

    run = _run(actors=[actor for actor in _run()["actors"] if actor["operation_state"] == "settled"],
               request={"surface": "task_acceptance", "subject": "answer A", "retry_key": "rk", "task_id": "applied"})

    def record(value):
        return artifacts.store_actor_source_bytes(tmp_path, "applied", category="context_checkpoints",
                                                  source_id="acceptance", data=json.dumps(value).encode(),
                                                  extension="json")

    def late(settled_at):
        return {"note": "On the delivered version of this answer, reviewers later passed it.",
                "settled_at": settled_at, "reviewed_subject": {"retry_key": "rk", "panel_id": "panel_1"}}

    # Two stored publications of one panel, as the retired writer left them: the notice's
    # record, then a newer one the stored panel now names.
    ref = record({**run, "late_settlement": late("2026-09-28T00:06:00+00:00"), "publication_revision": 1})
    newer = record({**run, "late_settlement": late("2026-09-28T00:06:01+00:00"), "publication_revision": 2})
    assert ref["path"] != newer["path"]
    panel_run = {**run, "late_settlement": late("2026-09-28T00:06:01+00:00"), "publication_revision": 2,
                 "applied_source_ref": newer, "applied_source_status": "available"}
    write_task_result(tmp_path, "applied", "completed", result="The delivered answer.",
                      review_projection=review_projection.compact_review_projection([panel_run]))

    app = Starlette(routes=[Route("/api/tasks/{task_id}/artifacts/{name}", api_task_artifact)])
    app.state.drive_root = tmp_path
    with TestClient(app) as client:
        def get(path, name=None):
            return client.get(f"/api/tasks/applied/artifacts/{name or pathlib.PurePosixPath(path).name}",
                              params={"source": path})

        for published in (ref, newer):
            response = get(published["path"])
            assert response.status_code == 200, published
            assert hashlib.sha256(response.content).hexdigest() == published["sha256"]
            assert len(response.content) == published["size"]
        forged = [
            f"source_handles/context_checkpoints/acceptance-{'0' * 64}.json",  # no such record
            record({"authority": "worker", "request": {"surface": "task_acceptance", "task_id": "applied"}})["path"],
            record({**run, "request": {**run["request"], "task_id": "another-task"}})["path"],
            record({**run, "request": {**run["request"], "surface": "plan_review"}})["path"],
            f"source_handles/context_checkpoints/../context_checkpoints/{pathlib.PurePosixPath(ref['path']).name}",
        ]
        for path in forged:
            assert get(path).status_code == 404, path
        assert get(ref["path"], name="other.json").status_code == 404, "the name must be the record's own"
        stored = artifacts.task_artifact_dir_path(tmp_path, "applied") / ref["path"]
        stored.write_bytes(stored.read_bytes().replace(b"answer A", b"answer B"))
        assert get(ref["path"]).status_code == 404, "altered bytes fail their digest"


def test_an_old_records_debt_pinned_source_still_downloads_and_stays_exact(tmp_path):
    """An older task's acceptance debt pinned the exact subject it owed a historical review of. That
    machine is gone, but its published link keeps serving those exact bytes (read-only history), and
    the model's digest reader keeps projecting them; altered bytes or a foreign task resolve nothing."""
    import hashlib
    import pathlib

    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros import artifacts
    from ouroboros.gateway.tasks import api_task_artifact
    from ouroboros.task_finalization import review_source_projection
    from ouroboros.task_results import write_task_result

    ref = artifacts.store_actor_source_bytes(tmp_path, "indebted", category="context_checkpoints",
                                             source_id="acceptance-subject", data=b"the owed subject text",
                                             extension="txt")
    write_task_result(tmp_path, "indebted", "completed", result="The delivered answer.",
                      acceptance_debt={"source_ref": ref, "status": "owed"})

    app = Starlette(routes=[Route("/api/tasks/{task_id}/artifacts/{name}", api_task_artifact)])
    app.state.drive_root = tmp_path
    with TestClient(app) as client:
        response = client.get(f"/api/tasks/indebted/artifacts/{pathlib.PurePosixPath(ref['path']).name}",
                              params={"source": ref["path"]})
        assert response.status_code == 200 and hashlib.sha256(response.content).hexdigest() == ref["sha256"]
        owed = "the owed subject text"
        projected = review_source_projection(tmp_path, "indebted", ref["sha256"], 0, len(owed))
        assert projected.get("text") == owed and projected["source_ref"] == ref, projected
        assert review_source_projection(tmp_path, "other", ref["sha256"], 0, len(owed))["status"] == "unavailable"
        stored = artifacts.task_artifact_dir_path(tmp_path, "indebted") / ref["path"]
        stored.write_bytes(b"another text")
        assert client.get(f"/api/tasks/indebted/artifacts/{pathlib.PurePosixPath(ref['path']).name}",
                          params={"source": ref["path"]}).status_code == 404, "altered bytes fail their digest"
        assert review_source_projection(tmp_path, "indebted", ref["sha256"], 0, len(owed))["status"] == "unavailable"
