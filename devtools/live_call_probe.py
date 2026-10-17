#!/usr/bin/env python3
"""Exercise a real live turn without starting installation services or audio devices.

Run with the application's Python: ``python devtools/live_call_probe.py --scenarios
all --output-dir /tmp/live-call-unique [--google-key-file F] [--data-copy DIR]``.
Calls use Google Gemini Live (its key from ``--google-key-file`` or the settings
source). ``--data-copy`` clones a disposable copy of a data directory (APFS clone)
into the run, so the call opens on a mature install's real memory instead of a fresh
one; the copy is never written, and a running install's own data directory is refused.
Each invocation owns a fresh external directory; it retains JSON evidence, PCM,
instructions and logs. A fresh run also writes ``recorded_exchange.json``, the
redacted wire a replay test can carry; a data-copy run does not (its wire is memory). The in-process path deliberately avoids
server startup's update/daemon duties. Only the input fixture uses speech synthesis
(OpenAI TTS, or saved fixtures with ``--fixture-dir``); Ouroboros speaks through Gemini.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
from collections import deque
from dataclasses import asdict
import hashlib
import json
import logging
import os
from pathlib import Path
import queue
import runpy
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from types import SimpleNamespace
import uuid

REPO = Path(__file__).resolve().parents[1]
SCENARIOS = ("opening", "russian_reply", "tool", "chat", "interruption", "typed", "hangup", "money")
SPEECH = {
    "russian_reply": "Привет! Ты меня слышишь? Ответь одним коротким предложением.",
    "tool": "Найди в интернете, какая сейчас погода в Москве, и скажи коротко.",
    "chat": "Пришли мне в чат ссылку на прогноз погоды.",
    "long": "Расскажи подробно, минуту или дольше, как образуются облака и дождь.",
    "interrupt": "Стоп, достаточно! Скажи только: хорошо, остановился.",
}
RATE, FRAME_BYTES = 24000, 960  # PCM16 mono, 20 ms at the peer boundary.


def daemon_list() -> list[str]:
    result = subprocess.run(["ps", "-axo", "pid,lstart,command"], capture_output=True, text=True, check=True)
    return [line.strip() for line in result.stdout.splitlines()
            if "claudexord.bundle" in line and "grep" not in line]


def output_root(value: str) -> Path:
    root = Path(value).expanduser().resolve()
    if root == REPO or REPO in root.parents or not root.is_relative_to(Path("/tmp").resolve()):
        raise ValueError("output directory must be a fresh directory under /tmp, outside the checkout")
    if root.exists():
        raise ValueError("output directory already exists; keep earlier evidence and choose a new directory")
    root.mkdir(mode=0o700, parents=True)
    return root


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def json_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_scenarios(value: str) -> set[str]:
    selected = set(SCENARIOS) if value == "all" else set(value.split(","))
    if not selected <= set(SCENARIOS):
        raise ValueError("unknown scenarios: " + ", ".join(sorted(selected - set(SCENARIOS))))
    return selected


class Evidence:
    """A thread-safe observation journal; audio bytes live in separate PCM files."""

    def __init__(self, root: Path):
        self.root, self.rows = root, []
        self.started = time.monotonic()
        self.handles: dict = {}  # resume handle -> label, across every session of the recording
        self.lock = threading.Lock()
        self.stream = (root / "observations.jsonl").open("a", encoding="utf-8", buffering=1)

    def note(self, kind: str, **fields) -> dict:
        row = {"at": time.monotonic() - self.started, "kind": kind, **fields}
        with self.lock:
            self.rows.append(row)
            self.stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        return row

    def of(self, kind: str, since: float = 0) -> list[dict]:
        with self.lock:
            return [r for r in self.rows if r["kind"] == kind and r["at"] >= since]


TINY_PCM = "AAABAA=="  # two PCM16 samples: a replayable stand-in for recorded audio


def recorded_exchange(wire: list, model: str, *, data_copy: bool) -> dict | None:
    """The replay fixture of a fresh run's wire; none for a data copy, whose speech, replayed rows
    and tool results are that install's memory (a fixture is meant to be committed)."""
    if not wire or data_copy:
        return None
    return {"header": {
        "provenance": "recorded_live", "model": model, "recorded_on": time.strftime("%Y-%m-%d"),
        "recorder": "devtools/live_call_probe.py",
        "notes": "Audio is a two-sample stand-in beside its recorded length; resume handles are labels; "
                 "the setup's instruction is its length and digest."}, "events": wire}


def redacted_wire(message: dict, handles: dict) -> dict:
    """One Gemini wire message as a replayable fixture row: audio becomes ``TINY_PCM`` beside its
    recorded length, resume handles become labels (the same label where the setup sends back a
    handle it received), the setup's instruction becomes its length and digest, and grounding's
    rendered HTML becomes its length. Everything else stays as received."""
    def walk(value, key=""):
        if isinstance(value, dict):
            if "inlineData" in value and isinstance(value["inlineData"], dict):
                inline = dict(value["inlineData"])
                inline["recorded_bytes"] = len(base64.b64decode(inline.get("data") or ""))
                inline["data"] = TINY_PCM
                return {**{k: walk(v, k) for k, v in value.items() if k != "inlineData"}, "inlineData": inline}
            out = {}
            for k, v in value.items():
                if v and isinstance(v, str) and (k == "newHandle" or (k == "handle" and key == "sessionResumption")):
                    out[k] = handles.setdefault(v, f"recorded-handle-{len(handles) + 1}")
                elif k == "renderedContent" and isinstance(v, str):
                    out[k] = f"[{len(v)} characters of rendered search chips]"
                elif k == "systemInstruction":
                    text = "".join(part.get("text", "") for part in v.get("parts", []))
                    out[k] = {"recorded_chars": len(text), "sha256": hashlib.sha256(text.encode()).hexdigest()}
                elif k == "data" and key == "audio":
                    out[k] = TINY_PCM
                else:
                    out[k] = walk(v, k)
            return out
        if isinstance(value, list):
            return [walk(item, key) for item in value]
        return value
    return walk(message)


class ObservedEngine:
    """Delegate every operation unchanged and retain evidence at the actual wire."""

    def __init__(self, engine, evidence: Evidence, ctx):
        self.engine, self.evidence, self.ctx = engine, evidence, ctx
        original_write, original_receive = engine._write, engine._handle_message
        handles = evidence.handles

        async def write(payload):
            await original_write(payload)
            if "realtimeInput" not in payload or "audio" not in payload.get("realtimeInput", {}):
                evidence.note("gemini_wire", direction="client", message=redacted_wire(payload, handles))
            if payload.get("type") != "input_audio_buffer.append" and "realtimeInput" not in payload:
                if payload.get("type") == "session.update" or "setup" in payload:
                    evidence.note("wire_sent", type=payload.get("type", "setup"),
                                  body_ref=evidence.of("session_config")[-1]["instructions_ref"])
                else:
                    evidence.note("wire_sent", payload=payload)

        async def receive(payload):
            evidence.note("gemini_wire", direction="server", message=redacted_wire(payload, handles))
            await original_receive(payload)

        engine._write, engine._handle_message = write, receive

    async def open(self, config):
        instructions = config.instructions
        try:
            import tiktoken
            tokens, method = len(tiktoken.get_encoding("o200k_base").encode(instructions)), "o200k_base"
        except ImportError:
            from ouroboros.utils import estimate_tokens
            tokens, method = estimate_tokens(instructions), "host_estimate_chars_div_4"
        name = f"instructions-{len(self.evidence.of('session_config')) + 1}.txt"
        (self.evidence.root / name).write_text(instructions, encoding="utf-8")
        facts = dict(getattr(self.ctx, "memory_view_facts", {}) or {})
        self.evidence.note("session_config", instruction_tokens=tokens, instructions_ref=name,
                           rendered_mode=getattr(self.ctx, "active_context_mode", "unknown"),
                           tokenizer=method, instruction_sha256=hashlib.sha256(instructions.encode()).hexdigest(),
                           memory_view=facts, host_sections_addressed=facts.get("host_sections_addressed", []),
                           tools=[t["function"]["name"] for t in config.tools], route=asdict(config.route))
        await self.engine.open(config)

    async def send(self, command):
        await self.engine.send(command)
        if type(command).__name__ not in {"AudioIn", "PlayedAudio"}:
            fields = asdict(command)
            if "images" in fields:
                fields["images"] = {"omitted_binary_bytes": sum(map(len, fields["images"]))}
            self.evidence.note("command", type=type(command).__name__, **fields)

    async def events(self):
        async for event in self.engine.events():
            fields = asdict(event)
            if "kind" in fields:
                fields["error_kind"] = fields.pop("kind")
            if "pcm" in fields:
                fields["pcm_bytes"] = len(fields.pop("pcm"))
            if type(event).__name__ == "Ready":
                fields = {"effective_model": fields["effective"].get("model"),
                          "instructions_ref": self.evidence.of("session_config")[-1]["instructions_ref"]}
            self.evidence.note(type(event).__name__, **fields)
            yield event

    async def close(self, reason):
        await self.engine.close(reason)


class OwnerPeer:
    """Pace capture and simulated playback independently, including queued flushes."""

    def __init__(self, hub, evidence: Evidence):
        from ouroboros.call_protocol import PeerInfo

        self.hub, self.evidence = hub, evidence
        self.peer_id, self.item, self.seq = "probe-owner", "", 0
        self.pending, self.played = deque(), {}
        self.next_play = time.monotonic()
        self.speaking, self.next_capture = False, time.monotonic()
        hub.join(PeerInfo(self.peer_id, "owner", True, True, "Synthetic owner"), RATE)

    def control(self, **message):
        self.hub.receive_control(self.peer_id, message)

    async def pump(self):
        from ouroboros.call_protocol import decode_control, decode_frame

        with (self.evidence.root / "output.pcm").open("wb") as stream:
            while True:
                if (not self.speaking and not self.hub.closed and self.evidence.of("Ready")
                        and time.monotonic() >= self.next_capture):
                    self.capture(bytes(FRAME_BYTES))
                    self.next_capture = time.monotonic() + .02
                message = self.hub.next_outgoing(self.peer_id)
                if isinstance(message, bytes):
                    frame = decode_frame(message)
                    stream.write(frame.pcm)
                    self.pending.append((self.item, frame))
                    self.evidence.note("peer_audio", item_id=self.item, epoch=frame.epoch, bytes=len(frame.pcm))
                elif isinstance(message, str):
                    message = decode_control(message)
                    self.evidence.note("peer_control", message=message)
                    if message["type"] == "flush":
                        self.pending.clear()
                        self.next_play = time.monotonic()
                    elif message["type"] == "caption" and message["speaker"] == "ouroboros":
                        self.item = message["item_id"]
                if self.pending and time.monotonic() >= self.next_play and not self.hub.closed:
                    item, frame = self.pending.popleft()
                    duration = len(frame.pcm) / (2 * frame.rate)
                    self.next_play = max(self.next_play, time.monotonic()) + duration
                    self.played[item] = self.played.get(item, 0) + round(duration * 1000)
                    if item:
                        self.control(type="played", item_id=item, played_ms=self.played[item])
                if message is None:
                    if self.hub.closed:
                        return
                    await asyncio.sleep(.005)

    def capture(self, pcm: bytes):
        from ouroboros.call_protocol import AudioFrame, FRAME_AUDIO_IN, decode_frame, encode_frame

        frame = AudioFrame(FRAME_AUDIO_IN, self.hub.epoch, self.seq, RATE, 0, pcm)
        self.hub.receive_audio(self.peer_id, decode_frame(encode_frame(frame)))
        self.seq += 1

    async def speak(self, pcm: bytes, label: str):

        self.evidence.note("speech_start", label=label)
        self.speaking = True
        start = time.monotonic()
        # A real microphone also sends quiet frames; VAD needs the trailing silence.
        audio = pcm + bytes(RATE * 2)
        for offset in range(0, len(audio), FRAME_BYTES):
            if self.hub.closed:
                raise RuntimeError("call closed while sending speech")
            frame = audio[offset:offset + FRAME_BYTES]
            self.capture(frame)
            await asyncio.sleep(max(0, start + (offset + len(frame)) / (RATE * 2) - time.monotonic()))
            if offset < len(pcm) <= offset + FRAME_BYTES:
                self.evidence.note("speech_end", label=label)
        self.speaking = False


def synthesize(root: Path, label: str, key: str) -> bytes:
    """SSE retains provider token usage alongside the synthesized input fixture."""
    import requests
    from ouroboros import usage_accounting as accounting
    from ouroboros.live_usage import _settlement
    from ouroboros.net_transport import requests_verify_kwargs
    from ouroboros.pricing_catalog import get_catalog_quote

    model = "gpt-4o-mini-tts"
    quote = get_catalog_quote(model, provider="openai", drive_root=root / "data")
    reservation = accounting.reserve_attempt(accounting.AttemptRequest(
        model=model, provider="openai", drive_root=root / "data", task_id="probe-input",
        source="probe_speech_fixture", force_unknown_reservation=True))
    accounting.mark_dispatched(reservation)
    audio, usage = bytearray(), {}
    try:
        with requests.post("https://api.openai.com/v1/audio/speech", headers={"Authorization": "Bearer " + key},
                           json={"model": model, "voice": "coral", "input": SPEECH[label],
                                 "response_format": "pcm", "stream_format": "sse"},
                           stream=True, timeout=60, **requests_verify_kwargs()) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith(b"data: "):
                    if line[6:].strip() == b"[DONE]":
                        break
                    event = json.loads(line[6:])
                    if event.get("type") == "speech.audio.delta":
                        audio.extend(base64.b64decode(event["audio"], validate=True))
                    if event.get("type") == "speech.audio.done":
                        usage = event.get("usage", {})
        counts = {"text_input_tokens": usage.get("input_tokens", 0),
                  "audio_output_tokens": usage.get("output_tokens", 0)} if usage else {}
        normalized, cost = _settlement(quote, {"fixture": label}, counts, usage, "completed")
        accounting.settle_attempt(reservation, normalized, cost_usd=cost, cost_final=False)
    except Exception:
        accounting.mark_unresolved(reservation, "speech_fixture_failed")
        raise
    if not audio or len(audio) % 2:
        raise ValueError("speech API returned no valid PCM16 fixture")
    (root / f"input-{label}.pcm").write_bytes(audio)
    write_json(root / f"input-{label}.json", {"text": SPEECH[label], "usage": usage, "seconds": len(audio) / (2 * RATE)})
    return bytes(audio)


def speech_fixture(root: Path, label: str, key: str, reuse: str | None) -> bytes:
    """Reuse an exact saved synthetic utterance to avoid paying for repeat diagnostics."""
    if reuse:
        source = Path(reuse).resolve()
        if not source.is_relative_to(Path("/tmp").resolve()):
            raise ValueError("fixture reuse is restricted to the isolated /tmp runs")
        metadata = source / f"input-{label}.json"
        if metadata.exists():
            meta = json.loads(metadata.read_text())
            pcm = (source / f"input-{label}.pcm").read_bytes()
            if meta.get("text") != SPEECH[label] or not pcm or len(pcm) % 2:
                raise ValueError("saved fixture does not match the scenario")
            (root / f"input-{label}.pcm").write_bytes(pcm)
            write_json(root / f"input-{label}.json", {**meta, "reused_from": str(metadata)})
            return pcm
    return synthesize(root, label, key)


# A grounded response carries Gemini's search charge, which the public catalog prices nowhere; the
# ledger keeps such a response unknown-priced by design (``live_usage``), a disclosure, not a gap.
GROUNDING_UNPRICED = {"groundingMetadata", "grounding_metadata"}


def attempts_since(rows: list[dict], started_at: str) -> list[dict]:
    """This run's ledger attempts: a data copy carries its install's whole ledger before them."""
    return [row for row in rows if str(row.get("ts_reserved") or "") >= started_at]


def probe_budget(data: Path) -> str:
    """Ten dollars beyond what the run's data has already spent (a copy's ledger is its install's)."""
    from ouroboros.usage_accounting import usage_projection

    spent = float(usage_projection(data, global_limit_usd=1e9).get("settled_usd") or 0.0)
    return str(round(spent + 10, 2))


def money_evidence(rows: list[dict], events: list[dict]) -> tuple[bool, dict]:
    responses = [r for r in rows if r.get("source") == "live_response"]
    grounded = [r["attempt_id"] for r in responses if r.get("cost_usd") is None
                and set(r.get("cost_evidence", {}).get("unpriced") or ()) and
                set(r.get("cost_evidence", {}).get("unpriced") or ()) <= GROUNDING_UNPRICED]
    starts = [e.get("response_id") or e.get("item_id") for e in events if e["kind"] == "ResponseStarted"]
    done = [e.get("response_id") or e.get("item_id") for e in events if e["kind"] == "ResponseDone"]
    keys = [r.get("cost_evidence", {}).get("response_key") for r in responses]
    errors = []
    if not responses or sorted(keys) != sorted(starts) or sorted(keys) != sorted(done):
        errors.append("response starts, settlements and ledger attempts do not match one-to-one")
    for row in responses:
        evidence = row.get("cost_evidence", {})
        counts, quote = evidence.get("usage", {}), evidence.get("tariff_quote")
        if row.get("state") != "settled" or not evidence.get("raw_usage"):
            errors.append("unsettled or missing raw usage: " + row["attempt_id"])
        if row.get("prompt_tokens", 0) != counts.get("text_input_tokens", 0):
            errors.append("audio included in text counter: " + row["attempt_id"])
        if not quote or not all(quote.get(k) for k in ("url", "fetched_at", "sha256", "entry")):
            errors.append("catalog quote unavailable: " + row["attempt_id"])
        if row.get("cost_usd") is None and row["attempt_id"] not in grounded:
            errors.append("unpriced modalities: " + row["attempt_id"])
        if row in responses and not all(row.get(k) for k in ("ts_reserved", "ts_dispatched", "ts_final")):
            errors.append("missing reservation/dispatch/settlement proof: " + row["attempt_id"])
    unknown = [r["attempt_id"] for r in rows if r.get("cost_usd") is None]
    if set(unknown) - set(grounded):
        errors.append("total includes unknown-priced attempts")
    known = sum(r.get("cost_usd") or 0 for r in rows)
    return not errors, {"errors": errors, "attempts": rows, "known_estimated_usd": known,
                        "total_cost_usd": None if unknown else known, "unknown_attempts": unknown,
                        "grounded_unknown_priced_attempts": grounded,
                        "call_estimated_usd": sum(r.get("cost_usd") or 0 for r in responses)}


class Probe:
    def __init__(self, root: Path, selected: set[str], timeout: float):
        self.root, self.engine, self.selected, self.timeout = root, "google", selected, timeout
        self.evidence = Evidence(root)
        self.result = {"engine": self.engine, "path": "in_process_make_agent", "scenarios": {
            s: {"verdict": "NOT_RUN", "evidence": {"reason": "not selected" if s not in selected else "not reached"}}
            for s in SCENARIOS}, "total_cost_usd": None}
        self.agent = self.peer = self.thread = None
        self.started_at = datetime.now(timezone.utc).isoformat()  # the ledger's own timestamp form

    def verdict(self, scenario, passed, **evidence):
        row = {"verdict": "PASS" if passed else "FAIL", "evidence": evidence}
        if evidence.get("route") == "native_search":
            # Gemini's own Google Search answered: the function-tool path did not run in this scenario.
            row.update(route="native_search", function_path="NOT_RUN")
        self.result["scenarios"][scenario] = row
        write_json(self.root / "result.json", self.result)
        print(scenario + ": " + row["verdict"] + (" (native search; function path NOT_RUN)"
                                                  if row.get("route") else ""), flush=True)

    async def wait(self, predicate, *, timeout=None):
        deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
        while not predicate():
            if time.monotonic() >= deadline:
                raise TimeoutError("probe observation deadline exceeded")
            if self.thread is not None and not self.thread.is_alive():
                raise RuntimeError("live task ended before expected observation")
            await asyncio.sleep(.025)

    async def speech_case(self, name, audio):
        since = self.evidence.note("scenario_start", scenario=name)["at"]
        before_rows = len(json_rows(self.root / "data/logs/chat.jsonl"))
        await self.peer.speak(audio, name)
        if name in ("tool", "chat"):
            tool = "web_search" if name == "tool" else "send_user_message"

            def ran():
                return any(e["type"] == "ToolResult" and any(
                    c["name"] == tool and c["call_id"] == e["call_id"] for c in self.evidence.of("ToolCall", since))
                    for e in self.evidence.of("command", since))

            def grounded():  # with native search on, Gemini's own Google Search may answer instead
                return name == "tool" and any((e.get("raw_usage") or {}).get("groundingMetadata")
                                              for e in self.evidence.of("ResponseDone", since))

            await self.wait(lambda: ran() or grounded())
            answer_after = self.evidence.of("command", since)[-1]["at"] if ran() else since
        else:
            answer_after = since
        await self.wait(lambda: any(e.get("speaker") == "ouroboros" and e.get("final")
                                   for e in self.evidence.of("Caption", answer_after)))
        await self.wait(lambda: not self.peer.pending)
        final_items = {e["item_id"] for e in self.evidence.of("Caption", answer_after)
                       if e.get("speaker") == "ouroboros" and e.get("final")}
        await self.wait(lambda: any((r.get("spoken") or {}).get("item_id") in final_items
                                   for r in json_rows(self.root / "data/logs/chat.jsonl")[before_rows:]))
        captions = self.evidence.of("Caption", since)
        audio_rows = self.evidence.of("peer_audio", since)
        ends = self.evidence.of("speech_end", since)
        end = ends[-1]["at"] if ends else since
        return {"since": since, "captions": [c for c in captions if c.get("final")],
                "output_bytes": sum(r["bytes"] for r in audio_rows),
                "latency_seconds": audio_rows[0]["at"] - end if audio_rows else None,
                "chat_rows": json_rows(self.root / "data/logs/chat.jsonl")[before_rows:], "hub_stats": self.hub.stats()}

    def start(self, settings):
        from ouroboros.agent import make_agent
        from ouroboros.call_peers import open_hub
        from ouroboros.call_protocol import CallSpec
        from ouroboros.contracts.chat_id_policy import WEB_UI_CHAT_ID
        from ouroboros.contracts.task_contract import attach_task_contract
        from ouroboros.utils import append_jsonl
        from supervisor import message_bus, state, workers
        from supervisor.events_chat_delivery import _handle_send_message

        data = self.root / "data"
        self.evidence.note("call_open")
        state.init(data, total_budget_limit=10)
        state.init_state()
        state.update_state(lambda s: {**s, "owner_id": 1, "owner_chat_id": WEB_UI_CHAT_ID})
        bridge = message_bus.LocalChatBridge()
        message_bus.init(data, 10, 0, bridge)
        workers.init(REPO, data, 0)
        delivery = SimpleNamespace(DRIVE_ROOT=data, RUNNING={}, bridge=bridge,
                                   send_with_budget=message_bus.send_with_budget, append_jsonl=append_jsonl)
        probe = self

        class EventQueue(queue.Queue):
            def put(self, event, *args, **kwargs):
                probe.evidence.note("task_event", event=event)
                if event.get("type") == "send_message":
                    _handle_send_message({**event, "_is_direct_chat": True}, delivery)

        self.spec = CallSpec(uuid.uuid4().hex, WEB_UI_CHAT_ID, self.engine,
                             settings["OUROBOROS_CALL_MODEL_GOOGLE"], settings["OUROBOROS_CALL_VOICE_GOOGLE"])
        self.hub = open_hub(self.spec)
        self.peer = OwnerPeer(self.hub, self.evidence)
        self.agent = make_agent(str(REPO), str(data), event_queue=EventQueue())
        open_live = self.agent.llm.open_live
        self.agent.llm.open_live = lambda route: ObservedEngine(open_live(route), self.evidence, self.agent.tools._ctx)
        self.task = {"id": "probe-" + uuid.uuid4().hex[:12], "type": "task", "chat_id": WEB_UI_CHAT_ID,
                     "_is_direct_chat": True, "text": "[Live call] A live voice call is open in this room. "
                     "The conversation is spoken; its transcript is written to this chat.",
                     "metadata": {"live_call": self.spec.as_metadata()}}
        attach_task_contract(self.task)
        self.result.update(call_id=self.spec.call_id, task_id=self.task["id"])

        def run():
            try:
                for event in self.agent.handle_task(self.task):
                    self.agent._event_queue.put(event)
            except Exception as exc:
                self.evidence.note("task_error", error=repr(exc))
                logging.exception("Probe task failed")

        self.thread = threading.Thread(target=run, name="probe-live-turn", daemon=True)
        self.thread.start()

    async def scenarios(self, fixtures):
        await self.wait(lambda: bool(self.evidence.of("Ready")))
        await self.wait(lambda: any(r["message"].get("state") == "live" for r in self.evidence.of("peer_control")))
        if "opening" in self.selected:
            config = self.evidence.of("session_config")[-1]
            self.verdict("opening", True, **config, seconds_to_live=self.evidence.of("Ready")[0]["at"] -
                         self.evidence.of("call_open")[0]["at"], attempts=self.evidence.of("session_config"),
                         recovery=self.evidence.of("EngineError"))
        for name in ("russian_reply", "tool", "chat"):
            if name not in self.selected:
                continue
            try:
                facts = await self.speech_case(name, fixtures[name])
                since = facts["since"]
                spoken = [c for c in facts["captions"] if c["speaker"] == "ouroboros"]
                passed = bool(facts["output_bytes"] and spoken)
                if name == "russian_reply":
                    passed &= any(any("\u0400" <= ch <= "\u04ff" for ch in c["text"]) for c in spoken)
                    passed &= any(r.get("source") == "call" and r.get("direction") == "in" for r in facts["chat_rows"])
                    passed &= any(r.get("spoken") for r in facts["chat_rows"])
                else:
                    tool_name = "web_search" if name == "tool" else "send_user_message"
                    calls = [c for c in self.evidence.of("ToolCall", since) if c["name"] == tool_name]
                    results = [r for r in self.evidence.of("command", since) if r["type"] == "ToolResult"
                               and any(c["call_id"] == r["call_id"] for c in calls)]
                    facts.update(tool_calls=calls, tool_results=results,
                                 tool_latency_seconds=results[-1]["at"] - calls[0]["at"] if calls and results else None)
                    grounded = [e for e in self.evidence.of("ResponseDone", since)
                                if (e.get("raw_usage") or {}).get("groundingMetadata")]
                    if name == "tool" and grounded and not calls:
                        # Native search is on: Gemini's own Google Search answered, and its grounding is the evidence.
                        facts.update(route="native_search", grounded_responses=len(grounded))
                    else:
                        passed &= bool(calls and results)
                    if name == "chat":
                        passed &= any("https://" in str(r.get("text", "")) and r.get("direction") == "out"
                                      for r in facts["chat_rows"])
                self.verdict(name, passed, **facts)
            except Exception as exc:
                self.verdict(name, False, error=str(exc), captions=self.evidence.of("Caption"),
                             tool_calls=self.evidence.of("ToolCall"), commands=self.evidence.of("command"))
        if "interruption" in self.selected:
            try:
                since = self.evidence.note("scenario_start", scenario="interruption")["at"]
                await self.peer.speak(fixtures["long"], "long")
                await self.wait(lambda: bool(self.evidence.of("peer_audio", since)))
                await asyncio.sleep(2)
                cut = self.evidence.note("barge_in")["at"]
                await self.peer.speak(fixtures["interrupt"], "interrupt")
                def continued_items():
                    starts = {e.get("response_id") for e in self.evidence.of("ResponseStarted", cut)}
                    return {e.get("item_id") for e in self.evidence.of("ResponseDone", cut)
                            if e.get("response_id") in starts and e.get("status") == "completed"}
                await self.wait(lambda: bool(continued_items()))
                await self.wait(lambda: not self.peer.pending)
                commands = [e for e in self.evidence.of("command", cut) if e["type"] == "Interrupt"]
                flushes = [e for e in self.evidence.of("peer_control", cut) if e["message"]["type"] == "flush"]
                rows = json_rows(self.root / "data/logs/chat.jsonl")
                interrupted = [r for r in rows if (r.get("spoken") or {}).get("status") == "interrupted"]
                continued = [e for e in self.evidence.of("peer_audio", cut) if e["item_id"] in continued_items()]
                # Gemini truncates server-side: the host's flush, cut and continued speech are the proof.
                self.verdict("interruption", bool(commands and flushes and interrupted and continued
                             and not self.hub.closed),
                             commands=commands, flushes=flushes, rows=interrupted,
                             continued_audio_bytes=sum(e["bytes"] for e in continued))
            except Exception as exc:
                self.verdict("interruption", False, error=str(exc))
        if "typed" in self.selected:
            from ouroboros.owner_mailbox import write_owner_message

            marker = "Typed during the call: remember the blue umbrella."
            msg_id = "typed-" + uuid.uuid4().hex
            written = write_owner_message(self.root / "data", marker, self.task["id"], msg_id=msg_id)
            try:
                await self.wait(lambda: any(marker in e.get("text", "") for e in self.evidence.of("command")))
                self.verdict("typed", written, msg_id=msg_id,
                             context=[e for e in self.evidence.of("command") if marker in e.get("text", "")])
            except Exception as exc:
                self.verdict("typed", False, error=str(exc), msg_id=msg_id)

    async def finish(self):
        from ouroboros.owner_mailbox import KIND_FINALIZE_NOW, write_owner_message
        from ouroboros.call_peers import close_hub
        from ouroboros.task_results import load_task_result
        from ouroboros.usage_store import read

        if self.peer and not self.hub.closed:
            self.peer.control(type="bye", reason="probe_finished")
            self.hub.leave(self.peer.peer_id, "probe_finished")
        if self.thread:
            await asyncio.to_thread(self.thread.join, 30)
            if self.thread.is_alive():
                write_owner_message(self.root / "data", "Probe stop", self.task["id"], kind=KIND_FINALIZE_NOW)
                await asyncio.to_thread(self.thread.join, 15)
            task_result = load_task_result(self.root / "data", self.task["id"]) or {}
            write_json(self.root / "task-result.json", task_result)
            terminals = [e for e in self.evidence.of("task_event") if e["event"].get("terminal_origin") == "host_notice"]
            if "hangup" in self.selected:
                self.verdict("hangup", not self.thread.is_alive() and self.hub.closed and bool(task_result.get("live_call"))
                             and any("utterances" in e["event"].get("text", "") for e in terminals),
                             thread_alive=self.thread.is_alive(), hub_closed=self.hub.closed,
                             checkpoint=task_result.get("live_call"), terminal_events=terminals, hub_stats=self.hub.stats())
            close_hub(self.spec.call_id)
        wire = [{"direction": e["direction"], "message": e["message"]} for e in self.evidence.of("gemini_wire")]
        record = recorded_exchange(wire, getattr(getattr(self, "spec", None), "model", ""),
                                   data_copy=bool(self.result.get("data_copy")))
        if record:  # a replayable fixture of what Gemini actually sent and received, redacted (``redacted_wire``)
            write_json(self.root / "recorded_exchange.json", record)
        if getattr(self, "spec", None) is not None:  # how the route counted the call view's text (the fit's witness)
            from ouroboros.capability_evidence import canonical_evidence_root, resolve_main_token_density, route_fingerprint

            density, basis = resolve_main_token_density(canonical_evidence_root(), route_fingerprint(
                provider=self.spec.engine, model=self.spec.model), self.spec.model)
            self.result["text_density"] = {"tokens_per_estimate": round(density, 3), "source": basis}
        with read(self.root / "data") as txn:
            rows = txn.attempts()
            transitions = {r["attempt_id"]: dict(r) for r in txn.conn.execute(
                "SELECT attempt_id, ts_reserved, ts_dispatched, ts_final FROM attempts")}
        rows = attempts_since([{**r, **transitions[r["attempt_id"]]} for r in rows], self.started_at)
        ok, facts = money_evidence(rows, self.evidence.rows)
        self.result.update(total_cost_usd=facts["total_cost_usd"], known_estimated_usd=facts["known_estimated_usd"])
        if "money" in self.selected:
            self.verdict("money", ok, **facts)
        else:
            write_json(self.root / "ledger.json", facts)


def _default_settings_source() -> str:
    """This install's settings file, found as ``config.resolve_data_dir`` finds it."""
    data = os.environ.get("OUROBOROS_DATA_DIR") or str(
        Path(os.environ.get("OUROBOROS_APP_ROOT") or Path.home() / "Ouroboros") / "data")
    return str(Path(data) / "settings.json")


def prepare_settings(root: Path, source: Path, google_key_file: str | None = None):
    from ouroboros.settings_defaults import SETTINGS_DEFAULTS
    from ouroboros.provider_models import OPENAI_DIRECT_DEFAULTS

    original = json.loads(source.read_text(encoding="utf-8"))
    keys = {k: original.get(k, "") for k in ("OPENAI_API_KEY", "GOOGLE_API_KEY")}
    if google_key_file:
        keys["GOOGLE_API_KEY"] = Path(google_key_file).expanduser().read_text(encoding="utf-8").strip()
    # A cloned data copy keeps its own settings (its memory and context choices); the run only
    # swaps in the keys, direct models and a small budget.
    path = root / "data/settings.json"
    copied = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    settings = {**SETTINGS_DEFAULTS, **copied}
    settings.update(keys)
    settings.update(OUROBOROS_MODEL=OPENAI_DIRECT_DEFAULTS["main"],
                    OUROBOROS_MODEL_LIGHT=OPENAI_DIRECT_DEFAULTS["light"],
                    OUROBOROS_MODEL_FALLBACKS=OPENAI_DIRECT_DEFAULTS["fallback"], TOTAL_BUDGET=probe_budget(root / "data"))
    path.touch(mode=0o600)
    write_json(path, settings)
    return settings, keys


async def run_probe(args):
    from unittest.mock import patch

    root = Path(args.output_dir)
    logging.basicConfig(filename=root / "runtime.log", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    probe = Probe(root, select_scenarios(args.scenarios), args.timeout)
    settings, keys = prepare_settings(root, Path(args.settings_source), args.google_key_file)
    probe.result["google_key_present"] = bool(keys["GOOGLE_API_KEY"])
    probe.result["data_copy"] = bool(args.data_copy)
    pump = None
    try:
        if not keys["GOOGLE_API_KEY"] or not (keys["OPENAI_API_KEY"] or args.fixture_dir):
            for name in probe.selected:
                probe.result["scenarios"][name]["evidence"] = {"reason": "selected engine or speech fixture credential absent"}
            return probe.result
        labels = [s for s in ("russian_reply", "tool", "chat") if s in probe.selected]
        if "interruption" in probe.selected:
            labels += ["long", "interrupt"]
        fixtures = {name: await asyncio.to_thread(speech_fixture, root, name, keys["OPENAI_API_KEY"], args.fixture_dir)
                    for name in labels}
        # This standalone probe must never provision a daemon, even if a future
        # context helper begins requesting an optional delegated capability.
        with patch("ouroboros.claudexor_daemon.ensure_owned_gateway", side_effect=RuntimeError("probe_daemon_forbidden")):
            probe.start(settings)
            pump = asyncio.create_task(probe.peer.pump())
            try:
                await probe.scenarios(fixtures)
            except Exception as exc:
                probe.verdict("opening", False, error=str(exc), events=probe.evidence.of("EngineError"))
            finally:
                if pump:
                    pump.cancel()
                    await asyncio.gather(pump, return_exceptions=True)
                await probe.finish()
    except Exception as exc:
        probe.result["failure"] = str(exc)
        logging.exception("Probe setup or teardown failed")
        if probe.thread is None:
            await probe.finish()
    finally:
        # Keep only the actual key fields out of every retained settings copy.
        settings_path = root / "data/settings.json"
        if settings_path.exists():
            safe = json.loads(settings_path.read_text())
            for key in keys:
                safe[key] = ""
            write_json(settings_path, safe)
        write_json(root / "result.json", probe.result)
    return probe.result


# Never copied: the source's Claudexor home (its daemon's descriptor and token would let the copy reach
# that daemon), the observability store (millions of request dumps a call never reads; copying it took
# longer than everything else together on a mature install), the state's engine runtimes and review
# checkouts, and the task workspaces (headless task files, forked drives and trees): whole trees no call
# reads, which made the copy run past half an hour.
DATA_COPY_SKIPPED = ("claudexor", "observability", "state/cx", "state/review_checkouts", "state/headless_tasks",
                     "task_drives", "task_trees")


def data_copy_source(value: str) -> Path:
    """The disposable copy a run clones from; a running install's own data directory is refused.

    A live install keeps writing while it is read (its locks, temporary files and settings with
    every key it holds), so the probe takes a copy made beforehand, as ``call_context_measure`` does.
    """
    source = Path(value).expanduser().resolve()
    if (source / ".git").exists() or (source.parent / "ouroboros.pid").exists() or (source.parent / "repo/.git").exists():
        raise ValueError("refusing a live install's or a versioned data directory; pass a disposable copy of one")
    return source


def copy_data(source: Path, target: Path, *, skipped=DATA_COPY_SKIPPED, prefix: str = "") -> None:
    """A data copy, cloned file by file (``cp -c``) into the run; the source is only read.

    One entry at a time (descending into a folder only when something inside it is skipped), never
    ``skipped``, and never one whole-tree clonefile(2): that held APFS for minutes and froze the machine.
    """
    target.mkdir(mode=0o700)
    for entry in sorted(source.iterdir()):
        name = prefix + entry.name
        if name in skipped or entry.name.endswith(".lock"):  # a lock file means nothing in a copy
            continue
        if entry.is_dir() and not entry.is_symlink() and any(s.startswith(name + "/") for s in skipped):
            copy_data(entry, target / entry.name, skipped=skipped, prefix=name + "/")
        elif subprocess.run(["cp", "-c", "-R", str(entry), str(target)]).returncode:
            raise RuntimeError(f"could not copy {entry}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", default="all", help=",".join(SCENARIOS))
    parser.add_argument("--google-key-file", help="a file holding the Google AI key for Gemini Live (read, never printed)")
    parser.add_argument("--data-copy", help="a disposable copy of a data directory, cloned into the run (never a "
                        "running install's own): a mature install's memory")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--settings-source", default=_default_settings_source(),
                        help="the install whose provider keys the probe uses (read, never written)")
    parser.add_argument("--fixture-dir", help="reuse exact speech fixtures from a previous isolated /tmp run")
    parser.add_argument("--timeout", type=float, default=60, help="per-scenario observation bound in seconds")
    parser.add_argument("--isolated-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    select_scenarios(args.scenarios)
    if args.isolated_child:
        sys.path.insert(0, str(REPO))
        result = asyncio.run(run_probe(args))
        return int(bool(result.get("failure")) or any(r["verdict"] == "FAIL" for r in result["scenarios"].values()))
    source = data_copy_source(args.data_copy) if args.data_copy else None  # refused before anything is made
    root = output_root(args.output_dir)
    if source:
        copy_data(source, root / "data")
    before = daemon_list()
    write_json(root / "daemons-before.json", before)
    helper = runpy.run_path(str(REPO / "ouroboros/test_environment.py"))
    env = helper["isolated_environment"](root, REPO)
    env = {k: v for k, v in env.items() if not k.startswith("CLAUDEXOR_")}
    # The child runs isolated (its own data root), so the settings source resolved here travels explicitly.
    command = [sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv),
               "--settings-source", str(Path(args.settings_source).resolve()), "--isolated-child"]
    try:
        with (root / "console.log").open("w") as stream:
            code = subprocess.call(command, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT)
    finally:
        after = daemon_list()
        write_json(root / "daemons-after.json", after)
    print(json.dumps({"output_dir": str(root), "exit_code": code, "daemon_list_unchanged": before == after}))
    return code or int(before != after)


if __name__ == "__main__":
    raise SystemExit(main())
