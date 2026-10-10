"""Local capacity is a serving-instance fact, independent of training and health."""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from ouroboros import capability_evidence as ce, config, local_model
from ouroboros.llm import LLMClient
from ouroboros.llm_local import local_context_limits


@pytest.fixture
def manager(monkeypatch, tmp_path):
    monkeypatch.delenv("OUROBOROS_IN_WORKER", raising=False)
    manager = local_model.LocalModelManager()
    manager._proc = SimpleNamespace(pid=123, poll=lambda: None)
    manager._status = "ready"
    monkeypatch.setattr(local_model, "get_manager", lambda: manager)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    return manager


@pytest.fixture
def main_plan(manager, monkeypatch, tmp_path):
    from ouroboros import context_fit

    monkeypatch.setattr(config, "runtime_settings", lambda: {"OUROBOROS_MODEL_CONTEXT_WINDOWS": {}})
    monkeypatch.setenv("OUROBOROS_MODEL_CONTEXT_WINDOWS", "{}")
    monkeypatch.setattr(context_fit, "reference_doc_sections", lambda *a, **k: ([], ""))

    def build(mode="max", text="source", *, use_local=True, resolver=None):
        core = context_fit.ContextCore(
            base_prompt="p", bible_md="b", architecture_md="a", development_md="d",
            semi_stable_text="s", dynamic_text="y", user_content_json=json.dumps(text),
            docs_need_development=False,
        )
        return context_fit.build_context_fit_plan(
            SimpleNamespace(drive_root=tmp_path), core,
            {"id": "local-fit", "model": "local-fixture", "use_local_model": use_local},
            preferred_mode=mode, route_resolver=resolver or context_fit.resolve_context_fit_route,
        )
    return build


@pytest.mark.parametrize("unknown", [{}, {"context_length": None}, {"context_length": 0},
                                      {"context_length": -1}, {"context_length": "unknown"},
                                      ConnectionError("unavailable")])
def test_unknown_metadata_does_not_poison_cache_or_health(manager, monkeypatch, unknown):
    health = Mock(side_effect=[unknown, {"context_length": 131072}])
    monkeypatch.setattr(manager, "health_check", health)
    assert manager.get_context_length() == 0
    assert manager.status_dict()["context_length"] == 0
    assert manager.is_running
    assert manager.get_context_length() == 131072
    assert manager.get_context_length() == 131072
    assert health.call_count == 2


@pytest.mark.parametrize("window", [4096, 16384, "131072"])
def test_positive_reported_metadata_is_retained(manager, monkeypatch, window):
    health = Mock(return_value={"context_length": window})
    monkeypatch.setattr(manager, "health_check", health)
    assert manager.get_context_length() == int(window)
    assert manager.get_context_length() == int(window)
    health.assert_called_once()


def test_healthy_server_with_no_window_still_becomes_ready(manager, monkeypatch):
    import requests
    from ouroboros import server_process

    response = Mock()
    response.json.return_value = {"data": [{"id": "local-fixture"}]}
    session = Mock()
    session.get.return_value = response
    # Mock only HTTP; exercise the actual health parser and readiness consumer.
    session_scope = MagicMock()
    session_scope.__enter__.return_value = session
    monkeypatch.setattr(requests, "Session", lambda: session_scope)
    monkeypatch.setattr(server_process, "record_service_binding", lambda *a, **k: None)
    manager._status = "loading"
    manager._wait_for_healthy(timeout=1)
    assert manager.is_running
    assert manager.get_context_length() == 0
    assert manager.serving_context_evidence()["confirmed"] is False
    assert manager.status_dict()["error"] is None


@pytest.mark.parametrize("allow_fetch", [False, True])
@pytest.mark.parametrize("legacy_window", [None, 4096, 131072])
def test_unknown_local_probe_ignores_legacy_cache_and_cloud_metadata(
    manager, monkeypatch, tmp_path, allow_fetch, legacy_window,
):
    manager._context_length = 131072  # Training metadata is not a served window.
    fp = ce.route_fingerprint(provider="openrouter", model="remote/name")
    if legacy_window is not None:
        ce._store_evidence(tmp_path, "probes", fp, ce.CapabilityEvidence(
            legacy_window, ce.STATUS_CONFIRMED, ce.SOURCE_LOCAL_HEALTH, fp,
            "remote/name", "openrouter", ts=ce.utc_now_iso(),
        ).to_json())
    path = ce._store_path(tmp_path)
    before = path.read_bytes() if path.exists() else None
    monkeypatch.setattr(ce, "_provider_metadata_window", lambda *a, **k: pytest.fail("cloud metadata"))
    monkeypatch.setattr(ce, "_generative_probe_window", lambda *a, **k: pytest.fail("generation probe"))
    evidence = ce.probe(tmp_path, provider="openrouter", model="remote/name", use_local=True,
                        allow_fetch=allow_fetch, allow_generative=True)
    assert evidence.window_tokens == 0
    assert evidence.status == ce.STATUS_UNPROBEABLE
    assert not ce.is_known(evidence)
    assert local_context_limits(65536) == (0, 2048)
    assert manager.is_running
    assert (path.read_bytes() if path.exists() else None) == before


def test_live_serving_capacity_reaches_main_and_review_consumers(manager, monkeypatch, tmp_path):
    from ouroboros.context_fit import resolve_context_fit_route
    from ouroboros.reviewer_window import resolve_reviewer_window

    monkeypatch.setattr(config, "runtime_settings", lambda: {"OUROBOROS_MODEL_CONTEXT_WINDOWS": {}})
    monkeypatch.setattr("ouroboros.reviewer_window._LAZY_ROUTE_LOCKS", {})
    monkeypatch.setenv("OUROBOROS_MODEL_CONTEXT_WINDOWS", "{}")
    manager._context_length = 131072
    for window in (16384, 8192, 4096, 0):
        manager._serving_context_length = window
        route, evidence = resolve_context_fit_route(
            {"model": "local-fixture", "use_local_model": True}, allow_fetch=False)
        assert route["use_local"] is True
        assert evidence.window_tokens == window
        assert ce.is_known(evidence) is (window > 0)
        reviewer = resolve_reviewer_window("local-fixture", use_local=True)
        assert reviewer.window_tokens == window
        assert reviewer.status == evidence.status
        assert local_context_limits(65536) == (window, window // 4 if window else 2048)
    assert not ce._store_path(tmp_path).exists()


def test_local_owner_assertion_remains_separate(manager, tmp_path):
    ce.record_owner_ack(tmp_path, provider="local", model="local-fixture", window_tokens=32768)
    evidence = ce.probe(tmp_path, provider="local", model="local-fixture", use_local=True, allow_fetch=False)
    assert evidence.window_tokens == 32768
    assert evidence.status == ce.STATUS_ASSERTED
    assert evidence.source == ce.SOURCE_OWNER_ACK
    assert manager.serving_context_evidence()["confirmed"] is False


def test_serving_measurement_keeps_input_and_instance_binding(manager, monkeypatch):
    import requests
    from ouroboros.local_model_server import input_fingerprint

    manager._context_length = 131072
    manager._serving_context_length = 16384
    manager._measurement_route = True
    payload = {"model": "local-model", "messages": [{"role": "user", "content": "source"}]}
    measured = {"supported": True, "input_is_exact": True, "input_tokens": 42,
                "context_window": 16384, "process_id": 123,
                "native_input_sha256": input_fingerprint(payload)}
    response = Mock()
    response.json.return_value = measured
    session_scope = MagicMock()
    session_scope.__enter__.return_value.post.return_value = response
    monkeypatch.setattr(requests, "Session", lambda: session_scope)
    assert manager.measure_prepared_input(payload) == measured
    changed_payload = {**payload, "messages": [{"role": "user", "content": "changed"}]}
    assert manager.measure_prepared_input(changed_payload)["supported"] is False
    manager._proc = SimpleNamespace(pid=456, poll=lambda: None)
    assert manager.measure_prepared_input(payload)["supported"] is False


@pytest.mark.parametrize("window", [0, 16384, 65536])
def test_running_local_dispatch_uses_serving_capacity_without_cloud_fallback(
    manager, main_plan, monkeypatch, tmp_path, window,
):
    from ouroboros.context_fit import measure_main_fit

    manager._context_length = 131072 if window else 4096
    manager._serving_context_length = window
    client = LLMClient(api_key="unused")
    sent = []

    def create(**payload):
        sent.append(payload)
        return SimpleNamespace(model_dump=lambda: {
            "choices": [{"message": {"role": "assistant", "content": "local answer"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 6000, "completion_tokens": 2, "total_tokens": 6002},
        })

    monkeypatch.setattr(client, "_get_local_client", lambda: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    monkeypatch.setattr(client, "_resolve_remote_target", lambda *a, **k: pytest.fail("cloud fallback"))
    text = "Keep this source intact. " * 1000  # Exceeds the fictitious 4096 allowance.
    plan = main_plan(text=text)
    messages = plan.messages_for("max")
    fit = measure_main_fit(plan, messages, None, drive_root=tmp_path,
                           profile="owner_max", rendered_mode="max", round_id="local-fit:round:2")
    assert fit.action == "send"
    assert fit.measurement.capacity_total_tokens == (window or None)
    message, usage = client.chat(messages=messages,
                                 model="local-fixture", max_tokens=65536, use_local=True)
    assert message["content"] == "local answer"
    assert usage["provider"] == "local"
    assert len(sent) == 1
    assert sent[0]["messages"][-1]["content"] == text
    assert sent[0]["max_tokens"] == (window // 4 if window else 2048)
    assert plan.output_reserve_tokens == fit.measurement.response_reserve_tokens == sent[0]["max_tokens"]
    assert not ce._load(config.DATA_DIR)["probes"]


@pytest.mark.parametrize("window", [16384, 65536])
@pytest.mark.parametrize("mode", ["max", "low", "nano"])
def test_local_main_forecast_keeps_real_pressure_and_nano_headroom(manager, main_plan, tmp_path, mode, window):
    from ouroboros.context_budget import NANO_MIN_HEADROOM_TOKENS
    from ouroboros.context_fit import measure_main_fit

    manager._serving_context_length = window
    plan = main_plan(mode)
    messages = plan.messages_for(mode)
    fit = measure_main_fit(plan, messages, None, drive_root=tmp_path,
                           profile="owner_" + mode, rendered_mode=mode, round_id="local-fit:round:2")
    assert fit.action == "send"
    assert plan.output_reserve_tokens == local_context_limits(65536)[1]
    # Nano's reserve is the reply floor: 8,192, or the local lane's smaller quarter-window ceiling.
    assert fit.measurement.response_reserve_tokens == (
        min(NANO_MIN_HEADROOM_TOKENS, plan.output_reserve_tokens) if mode == "nano" else plan.output_reserve_tokens)
    assert fit.measurement.reply_allowance_tokens == plan.output_reserve_tokens  # a short input: the whole ceiling
    messages.append({"role": "user", "content": "pressure " * 100000})
    pressured = measure_main_fit(plan, messages, None, drive_root=tmp_path,
                                profile="owner_" + mode, rendered_mode=mode, round_id="local-fit:round:3")
    assert pressured.action == "reclaim_once"
    assert pressured.measurement.capacity_deficit_tokens > 0


@pytest.mark.parametrize("mode", ["max", "low", "nano"])
def test_main_forecast_rebinds_remote_local_restart_and_unknown(manager, main_plan, monkeypatch, tmp_path, mode):
    from ouroboros import context, context_fit, loop
    from ouroboros.context_budget import NANO_MIN_HEADROOM_TOKENS
    from ouroboros.tools.registry import ToolRegistry

    def resolve(task, **kwargs):
        if task["use_local_model"]:
            return context_fit.resolve_context_fit_route(task, allow_fetch=False)
        return ({"model": task["model"], "provider": "openai", "use_local": False},
                SimpleNamespace(route_fp="remote-route", status="confirmed", stale=False, window_tokens=131072))

    monkeypatch.setattr(context, "_context_fit_route", resolve)
    plan = main_plan(mode, use_local=False, resolver=resolve)
    assert plan.output_reserve_tokens == 65536
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_id = "route-switch"
    registry._ctx.event_queue = None
    messages = plan.messages_for(mode)
    for local, window, reserve in [(True, 16384, 4096), (True, 65536, 16384),
                                   (True, 0, 2048), (False, 131072, 65536)]:
        manager._serving_context_length = window
        plan, active = loop._rebind_context_fit_plan(
            plan, registry, messages, model="local-fixture" if local else "remote-fixture",
            use_local=local, preferred_mode=mode, tool_schemas=[], model_role="fallback:0",
        )
        assert active == plan.preferred_mode == plan.initial_mode == mode
        assert plan.output_reserve_tokens == reserve
        assert plan.window_tokens == window
        assert ce.is_known(plan) is bool(window)
        fit = context_fit.measure_main_fit(plan, messages, None, drive_root=tmp_path,
            profile="owner_" + mode, rendered_mode=mode, round_id="route-switch:round:2")
        assert fit.action == "send"
        assert fit.measurement.capacity_total_tokens == (window or None)
        assert fit.measurement.response_reserve_tokens == (min(NANO_MIN_HEADROOM_TOKENS, reserve) if mode == "nano" else reserve)


def test_actual_local_connection_error_is_not_a_synthetic_overflow(manager, monkeypatch):
    manager._context_length = 0
    monkeypatch.setattr(manager, "health_check", Mock(side_effect=ConnectionError("health unavailable")))
    client = LLMClient(api_key="unused")
    error = ConnectionError("local transport unavailable")
    create = Mock(side_effect=error)
    monkeypatch.setattr(client, "_get_local_client", lambda: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    monkeypatch.setattr(client, "_resolve_remote_target", lambda *a, **k: pytest.fail("cloud fallback"))
    with pytest.raises(ConnectionError) as caught:
        client.chat(messages=[{"role": "user", "content": "source " * 5000}],
                    model="local-fixture", max_tokens=65536, use_local=True)
    assert caught.value is error
    create.assert_called_once()


_STAND_IN_SERVER = '''
"""Loopback stand-in for ouroboros.local_model_server; it loads no model."""
import json, os, socket, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from socketserver import TCPServer

from ouroboros.local_model_server import input_fingerprint


def no_reverse_dns(*args):
    raise AssertionError("loopback fixture must not resolve hostnames")


socket.getfqdn = no_reverse_dns

PORT, N_CTX = (int(sys.argv[sys.argv.index(flag) + 1]) for flag in ("--port", "--n_ctx"))


class Handler(BaseHTTPRequestHandler):
    def reply(self, body):
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # /v1/models reports training metadata only.
        self.reply({"data": [{"id": "local-fixture", "meta": {"n_ctx_train": 131072}}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with open(os.environ["LOCAL_STAND_IN_LOG"], "a", encoding="utf-8") as log:
            log.write(json.dumps({"path": self.path, "pid": os.getpid(), "ppid": os.getppid(),
                                  "body": body}) + "\\n")
        if self.path == "/extras/measure_chat":
            self.reply({"supported": True, "input_is_exact": True, "input_tokens": 100,
                        "context_window": N_CTX, "process_id": os.getpid(),
                        "native_input_sha256": input_fingerprint(body), "output_limit_enforced": True,
                        "reasoning_included_in_limit": True, "reason": None})
        else:
            self.reply({"id": "fixture", "object": "chat.completion", "created": 0, "model": "local-model",
                        "choices": [{"index": 0, "finish_reason": "stop",
                                     "message": {"role": "assistant", "content": "local answer"}}],
                        "usage": {"prompt_tokens": 100, "completion_tokens": 2, "total_tokens": 102}})

    def log_message(self, *args):
        pass


class NumericLoopbackServer(ThreadingHTTPServer):
    def server_bind(self):
        # This numeric fixture needs no reverse DNS before it can listen.
        TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address


NumericLoopbackServer(("127.0.0.1", PORT), Handler).serve_forever()
'''


def _consumer_worker_entry(*args):
    """The actual pooled worker_main; only agent construction and extension loading differ.

    worker_main marks this spawned process a worker, and the process derives its data
    root and local port from the inherited environment like every pooled worker.
    """
    from ouroboros import agent, extension_loader
    from supervisor.worker_process import worker_main

    class ConsumerAgent:
        def handle_task(self, task):
            import os

            from ouroboros.context_fit import resolve_context_fit_route
            from ouroboros.reviewer_window import resolve_reviewer_window
            from ouroboros.utils import in_worker_process

            try:
                manager = local_model.get_manager()
                _route, fit = resolve_context_fit_route(
                    {"model": "local-fixture", "use_local_model": True}, allow_fetch=False)
                reviewer = resolve_reviewer_window("local-fixture", use_local=True)
                reply = None
                if task.get("chat"):
                    message, usage = LLMClient(api_key="unused").chat(
                        messages=[{"role": "user", "content": "hello"}], model="local-fixture",
                        max_tokens=65536, use_local=True)
                    reply = (message.get("content"), usage.get("provider"))
                seen = {"pid": os.getpid(), "in_worker": in_worker_process(), "data_dir": str(config.DATA_DIR),
                        "owns_server": manager._proc is not None,
                        "evidence": manager.serving_context_evidence(),
                        "limits": local_context_limits(65536),
                        "fit": (fit.window_tokens, fit.status, fit.source),
                        "reviewer": (reviewer.window_tokens, reviewer.status), "reply": reply}
            except Exception as error:  # Report consumer failures instead of hanging the parent.
                seen = {"error": repr(error)}
            return [{"type": "fixture_consumers", "task_id": task["id"], **seen}]

    agent.make_agent = lambda **_: ConsumerAgent()
    extension_loader.reload_all = lambda *_a, **_k: None
    worker_main(*args)


@pytest.fixture
def offline_manager(monkeypatch, tmp_path):
    manager = local_model.LocalModelManager()
    monkeypatch.setattr(local_model, "_manager", manager)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(manager, "check_runtime", lambda: True)
    yield manager
    manager.stop_server()


@pytest.mark.parametrize("source", ["", "   ", "fixture/source"])
def test_autostart_guard_admits_only_a_configured_source(offline_manager, monkeypatch, caplog, source):
    from ouroboros.local_model_autostart import auto_start_local_model

    download = Mock(return_value="/models/fixture.gguf")
    start = Mock()
    monkeypatch.setattr(offline_manager, "download_model", download)
    monkeypatch.setattr(offline_manager, "start_server", start)
    auto_start_local_model({"LOCAL_MODEL_SOURCE": source, "LOCAL_MODEL_FILENAME": "fixture.gguf",
                            "LOCAL_MODEL_PORT": 9123, "LOCAL_MODEL_N_GPU_LAYERS": 7,
                            "LOCAL_MODEL_CONTEXT_LENGTH": 8192, "LOCAL_MODEL_CHAT_FORMAT": "chatml"})
    if source.strip():
        download.assert_called_once_with("fixture/source", "fixture.gguf")
        start.assert_called_once_with("/models/fixture.gguf", port=9123, n_gpu_layers=7, n_ctx=8192,
                                      chat_format="chatml", source="fixture/source", filename="fixture.gguf")
        assert "LOCAL_MODEL_SOURCE is empty" not in caplog.text
    else:
        download.assert_not_called()
        start.assert_not_called()
        assert "LOCAL_MODEL_SOURCE is empty" in caplog.text




@pytest.mark.serial
def test_actual_pooled_worker_reads_the_serving_instance_its_server_process_owns(
    offline_manager, monkeypatch, tmp_path,
):
    """Pooled workers are spawned processes; the owned server lives in the server process."""
    import itertools
    import json
    import multiprocessing
    import os
    import pathlib
    import queue
    import socket
    import subprocess
    import sys
    import time

    from ouroboros.config import apply_settings_to_env
    from ouroboros.local_model_autostart import auto_start_local_model
    from ouroboros.server_process import read_service_bindings, record_service_binding
    from supervisor import worker_process
    from tests._shared import stop_socket_sharer

    stand_in = tmp_path / "stand_in_server.py"
    stand_in.write_text(_STAND_IN_SERVER, encoding="utf-8")
    real_popen, real_run = subprocess.Popen, subprocess.run

    def popen(cmd, **kwargs):  # Only the vendor server is replaced; the launched argv is kept.
        assert cmd[1:3] == ["-m", "ouroboros.local_model_server"]
        return real_popen([cmd[0], str(stand_in), *cmd[3:]], **kwargs)

    def run(cmd, **kwargs):  # llama-cpp-python is not installed here.
        if cmd[1:] == ["-c", "import llama_cpp"]:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return real_run(cmd, **kwargs)

    monkeypatch.setattr(local_model, "subprocess", SimpleNamespace(**{
        **vars(subprocess), "Popen": popen, "run": run}))
    log_path = tmp_path / "stand_in.jsonl"
    monkeypatch.setenv("LOCAL_STAND_IN_LOG", str(log_path))
    model = tmp_path / "fixture.gguf"  # A local source resolves without a download; nothing loads it.
    model.write_bytes(b"stand-in")

    def free_port():
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]

    health_events = []
    original_health = offline_manager.health_check

    def traced_health():
        event = {"started": time.monotonic(), "port": offline_manager._port}
        health_events.append(event)
        try:
            event["result"] = original_health()
            return event["result"]
        except Exception as error:
            event["error_type"] = type(error).__name__
            raise
        finally:
            event["finished"] = time.monotonic()

    monkeypatch.setattr(offline_manager, "health_check", traced_health)

    def readiness_failure():
        import threading
        import traceback

        process = offline_manager._proc
        facts = {**offline_manager.status_dict(), "pid": getattr(process, "pid", None),
                 "returncode": process.poll() if process else None, "health_events": list(health_events),
                 "stderr_tail": offline_manager._stderr_buf.decode("utf-8", errors="replace"), "threads": {}}
        frames = sys._current_frames()
        for thread in threading.enumerate():
            frame = frames.get(thread.ident)
            if frame is None or thread.name not in {"local-model-health", "local-model-stderr"}:
                continue
            facts["threads"][thread.name] = "".join(traceback.format_stack(frame))
            while frame:
                if frame.f_code.co_name == "_drain_stderr" and frame.f_locals.get("self") is offline_manager:
                    facts["stderr_live_tail"] = frame.f_locals.get("buf", b"").decode("utf-8", errors="replace")
                frame = frame.f_back
        # Printed only on failure so short tracebacks retain the diagnostic, without environment values.
        print("LOCAL_MODEL_STARTUP_DIAGNOSTIC " + json.dumps(facts, ensure_ascii=False), flush=True)
        return facts

    def autostart(port, n_ctx):
        auto_start_local_model({"LOCAL_MODEL_SOURCE": str(model), "LOCAL_MODEL_PORT": port,
                                "LOCAL_MODEL_CONTEXT_LENGTH": n_ctx})
        deadline = time.monotonic() + 30
        while not offline_manager.is_running and time.monotonic() < deadline:
            time.sleep(0.05)
        assert offline_manager.is_running, readiness_failure()
        return offline_manager.serving_context_evidence()

    # The server process exports settings into the environment its pooled workers inherit.
    port, exported = free_port(), {}
    apply_settings_to_env({"LOCAL_MODEL_PORT": port}, environ=exported)
    for key, value in {"LOCAL_MODEL_PORT": exported["LOCAL_MODEL_PORT"], "OUROBOROS_DATA_DIR": str(tmp_path),
                       "OUROBOROS_SETTINGS_PATH": str(tmp_path / "settings.json"),
                       "OUROBOROS_MODEL_CONTEXT_WINDOWS": "{}"}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("OUROBOROS_IN_WORKER", raising=False)  # worker_main itself must mark the worker.
    monkeypatch.setattr(worker_process, "worker_main", _consumer_worker_entry)
    ctx = multiprocessing.get_context("spawn")
    repo = pathlib.Path(__file__).resolve().parents[1]
    workers, steps = [], itertools.count()

    def spawn():
        incoming, outgoing = ctx.Queue(), ctx.Queue()
        proc = worker_process.spawn_worker_process(ctx, len(workers), incoming, outgoing, repo, tmp_path)
        workers.append((proc, incoming, outgoing))
        return workers[-1]

    def ask(worker, chat=False):
        proc, incoming, outgoing = worker
        task_id = f"step-{next(steps)}"
        incoming.put({"id": task_id, "type": "task", "chat": chat})
        crashes, deadline = tmp_path / "logs" / "supervisor.jsonl", time.monotonic() + 60
        while True:
            assert proc.is_alive() and time.monotonic() < deadline, (
                task_id, proc.exitcode, crashes.read_text(encoding="utf-8") if crashes.exists() else "")
            try:
                seen = outgoing.get(timeout=0.5)
            except queue.Empty:
                continue
            if seen.get("type") == "fixture_consumers" and seen.get("task_id") == task_id:
                break
        assert "error" not in seen, seen
        assert seen["pid"] == proc.pid != os.getpid() and seen["in_worker"] and not seen["owns_server"]
        assert seen["data_dir"] == str(tmp_path)
        return seen

    def assert_unknown(seen):  # Neither training metadata, a cached row nor an invented window.
        assert seen["evidence"] == {"context_window": None, "confirmed": False,
                                    "source": "serving_window_unobserved", "process_id": None}
        assert seen["limits"] == (0, 2048)
        assert seen["fit"] == (0, ce.STATUS_UNPROBEABLE, ce.SOURCE_NONE)
        assert seen["reviewer"] == (0, ce.STATUS_UNPROBEABLE)

    def assert_serving(seen, window, pid):
        assert seen["evidence"] == {"context_window": window, "confirmed": True, "port": port,
                                    "source": "published_server_arguments", "process_id": pid}
        assert seen["limits"] == (window, window // 4)
        assert seen["fit"] == (window, ce.STATUS_CONFIRMED, ce.SOURCE_LOCAL_HEALTH)
        assert seen["reviewer"] == (window, ce.STATUS_CONFIRMED)

    bindings_path, foreign = tmp_path / "state" / "server_port.bindings.json", None
    try:
        worker = spawn()
        auto_start_local_model({"LOCAL_MODEL_SOURCE": " ", "LOCAL_MODEL_PORT": port})
        assert not offline_manager.is_running and not bindings_path.exists()
        assert_unknown(ask(worker))

        owned = autostart(port, 16384)
        seen = ask(worker, chat=True)  # The same worker adopts an instance published after its spawn.
        assert_serving(seen, 16384, owned["process_id"])
        assert seen["reply"] == ("local answer", "local")
        sent = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
        assert [row["path"] for row in sent] == ["/extras/measure_chat", "/v1/chat/completions"]
        # The launched server itself, or the interpreter a Windows venv redirector runs as its child.
        assert all(owned["process_id"] in ((row["pid"], row["ppid"]) if os.name == "nt" else (row["pid"],))
                   for row in sent)
        assert sent[-1]["body"]["max_tokens"] == 4096

        monkeypatch.setenv("LOCAL_MODEL_PORT", str(free_port()))
        elsewhere = spawn()  # This worker would dispatch to another endpoint.
        monkeypatch.setenv("LOCAL_MODEL_PORT", str(port))
        assert_unknown(ask(elsewhere))

        published = bindings_path.read_text(encoding="utf-8")
        # A live, identity-checked process whose argv does not END with the launch pair proves no window.
        foreign = real_popen([sys.executable, "-c", "import time; time.sleep(60)",
                              "--n_ctx", "99999", "--port", str(port)])
        record_service_binding(tmp_path, "local_model", "127.0.0.1", port, pid=foreign.pid)
        assert_unknown(ask(worker))
        reused = json.loads(published)  # The pid now names a process born after the published one.
        fingerprint = reused["local_model"]["fingerprint"]
        fingerprint.update({key: "0" for key in ("start_time", "creation_time") if key in fingerprint})
        bindings_path.write_text(json.dumps(reused), encoding="utf-8")
        assert_unknown(ask(worker))
        bindings_path.write_text("{unreadable", encoding="utf-8")
        assert_unknown(ask(worker))
        bindings_path.write_text(published, encoding="utf-8")
        assert_serving(ask(worker), 16384, owned["process_id"])

        offline_manager.stop_server()
        assert "local_model" not in read_service_bindings(tmp_path)
        assert_unknown(ask(worker))

        restarted = autostart(port, 8192)  # The same worker holds no stale window.
        assert restarted["process_id"] != owned["process_id"]
        assert_serving(ask(worker), 8192, restarted["process_id"])

        offline_manager._proc.kill()  # A crash leaves the published binding behind.
        offline_manager._proc.wait(timeout=10)
        assert read_service_bindings(tmp_path)["local_model"]["pid"] == restarted["process_id"]
        assert_unknown(ask(worker))
    finally:
        if foreign is not None:
            foreign.kill()
            foreign.wait(timeout=10)
        for proc, incoming, outgoing in workers:
            incoming.put(None)
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
            proc._ouroboros_stop_socket.close()
            for channel in (incoming, outgoing):
                channel.close()
                channel.cancel_join_thread()
        stop_socket_sharer()


@pytest.mark.serial
@pytest.mark.parametrize("worker", [False, True], ids=["server-owner", "pooled-worker"])
def test_owned_local_health_never_consults_ambient_proxy_configuration(monkeypatch, worker):
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import requests
    from ouroboros import utils

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            assert self.path == "/v1/models"
            body = json.dumps({"data": [{"id": "local-fixture", "meta": {"n_ctx_train": 8192}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    def unavailable_proxy_lookup(*_args, **_kwargs):
        raise RuntimeError("ambient proxy lookup unavailable")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.setattr(utils, "in_worker_process", lambda: worker)
        monkeypatch.setattr(requests.sessions, "get_environ_proxies", unavailable_proxy_lookup)
        manager = local_model.LocalModelManager()
        manager._port = server.server_port
        assert manager.health_check() == {"ok": True, "model_name": "local-fixture", "context_length": 8192}
        # Only this owned-loopback session bypasses discovery; global Requests policy is intact.
        with requests.Session() as ordinary:
            with pytest.raises(RuntimeError, match="ambient proxy lookup unavailable"):
                ordinary.get(f"http://127.0.0.1:{server.server_port}/v1/models", timeout=5)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()
