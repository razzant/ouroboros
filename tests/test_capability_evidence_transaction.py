"""#1520: every whole-file writer of ``capability_evidence.json`` is one cross-process transaction.

Real separate processes contend for one store: a holder pauses between its fresh load and its
atomic save while writers from all three modules (capability_evidence, response_limits,
context_window) wait for it, so no stale snapshot erases a record or revives an older one.
A busy store fails honestly: owner actions report it, best-effort observations skip their
write and never fail the read or send they observed.
"""
from __future__ import annotations

import asyncio
import json
import os
import pathlib
import subprocess
import sys
import textwrap
import threading
from types import SimpleNamespace

import httpx
import pytest

from ouroboros import capability_evidence as ce, context_window, response_limits, vision_routing as vision
from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock

REPO = pathlib.Path(__file__).resolve().parents[1]
OPENROUTER = "https://openrouter.ai/api/v1"
IMAGE_TARGET = {"provider": "openai-compatible", "base_url": "https://compat.invalid/v1", "api_key": "fixture-key"}
WRITES = {  # one disjoint record per writer module, run in its own process
    # Signature-tolerant so the same choreography runs, and shows the race, on the pre-#1520 tree too.
    "image": "vision.record_catalog_image_input(TARGET['provider'], TARGET['base_url'], [{'id': 'm', "
             "'input_modalities': image}], source='a catalog', **({'api_key': TARGET['api_key']} if 'api_key' in "
             "inspect.signature(vision.record_catalog_image_input).parameters else {}))",
    "limits": "response_limits.record_catalog_limits(root, [('vendor/limited', 4096)], provider='openrouter', "
              f"base_url='{OPENROUTER}', field='max_completion_tokens', source='a catalog')",
    "window": "context_window.record_observation(root, 'observed-route', accepted=50000)",
    "density": "ce.record_token_density(root, 'vendor/dense', prompt_chars=40000, prompt_tokens=11000)",
    "ack": "ce.record_owner_ack(root, provider='openrouter', model='vendor/acked', window_tokens=200000)",
}
CHILD = textwrap.dedent("""
    import inspect, json, pathlib, sys, time
    root, role, sync, write = pathlib.Path(sys.argv[1]), sys.argv[2], pathlib.Path(sys.argv[3]), sys.argv[4]
    contenders, image = int(sys.argv[5]), json.loads(sys.argv[6])
    from ouroboros import capability_evidence as ce, context_window, response_limits, vision_routing as vision
    ce.canonical_evidence_root = lambda: root
    if hasattr(ce, "_STORE_LOCK_WAIT_SEC"):
        ce._STORE_LOCK_WAIT_SEC = 60.0  # contention, not the bounded wait, is under test here
    TARGET = json.loads(sys.argv[7])
    if role == "holder":
        real = ce._save
        def paused(path_root, data):  # loaded, not yet saved: the window a stale snapshot exploits
            (sync / "loaded").write_text("1")
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline and len(list(sync.glob("done-*"))) < contenders:
                time.sleep(0.02)
            return real(path_root, data)
        ce._save = paused
    else:
        deadline = time.monotonic() + 60
        while not (sync / "loaded").exists():
            assert time.monotonic() < deadline, "the holder never loaded"
            time.sleep(0.01)
    exec(write)
    (sync / ("done-" + role) if role != "holder" else sync / "holder-saved").write_text("1")
""")


def _run(root: pathlib.Path, holder: tuple, contenders: dict, tmp_path: pathlib.Path) -> None:
    sync = tmp_path / "sync"
    sync.mkdir()
    env = {**os.environ, "PYTHONPATH": str(REPO)}

    def start(role, write, image):
        return subprocess.Popen([sys.executable, "-c", CHILD, str(root), role, str(sync), write, str(len(contenders)),
                                 json.dumps(image), json.dumps(IMAGE_TARGET)], cwd=REPO, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    processes = [("holder", start("holder", *holder))]
    processes += [(role, start(role, write, image)) for role, (write, image) in contenders.items()]
    try:
        for role, process in processes:
            output, _ = process.communicate(timeout=120)
            assert process.returncode == 0, f"{role}: {output}"
    finally:
        for _role, process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()


def _image_verdict(root):
    record = (ce._read(root).get(vision.EVIDENCE_NAMESPACE) or {}).get(vision.image_route_scope(IMAGE_TARGET)) or {}
    return (record.get("models") or {}).get("m")


@pytest.mark.serial
def test_disjoint_records_of_all_three_writer_modules_survive_a_paused_holder(tmp_path):
    root = tmp_path / "store"
    contenders = {role: (WRITES[role], None) for role in ("limits", "window", "density", "ack")}
    _run(root, (WRITES["image"], ["text"]), contenders, tmp_path)
    data = json.loads(ce._store_path(root).read_text(encoding="utf-8"))
    assert _image_verdict(root) is False
    assert response_limits._response_fingerprint(provider="openrouter", model="vendor/limited",
                                                 base_url=OPENROUTER) in data["response_limits"]
    assert "observed-route" in data["main_windows"]
    assert "vendor/dense" in data["token_density"]
    assert ce.route_fingerprint(provider="openrouter", model="vendor/acked") in data["owner_acks"]
    assert "fixture-key" not in ce._store_path(root).read_text(encoding="utf-8")


@pytest.mark.serial
def test_a_stale_unrelated_snapshot_never_revives_an_older_image_no(tmp_path):
    root = tmp_path / "store"
    old = vision.image_route_scope(IMAGE_TARGET)
    ce._store_evidence(root, vision.EVIDENCE_NAMESPACE, old, {
        "source": "an older catalog", "observed_at": ce.utc_now_iso(), "models": {"m": False}})
    assert _image_verdict(root) is False
    _run(root, (WRITES["window"], None), {"image": (WRITES["image"], ["text", "image"])}, tmp_path)
    assert _image_verdict(root) is True  # the newer catalog's yes, not the holder's stale no
    assert "observed-route" in ce._read(root)["main_windows"]


@pytest.fixture
def held(tmp_path, monkeypatch):
    """Another process holds the store's sidecar; the bounded wait is shortened for the test."""
    root = tmp_path / "store"
    monkeypatch.setattr(ce, "_STORE_LOCK_WAIT_SEC", 0.2)
    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: root)
    ack = ce.record_owner_ack(root, provider="openrouter", model="vendor/acked", window_tokens=100000)
    before = ce._store_path(root).read_bytes()
    sidecar = ce._store_path(root).with_name("capability_evidence.json.lock")
    fd = acquire_exclusive_file_lock(sidecar, timeout_sec=1)
    assert fd is not None
    yield SimpleNamespace(root=root, before=before, ack=ack, sidecar=sidecar)
    release_exclusive_file_lock(sidecar, fd)


def test_owner_actions_report_a_busy_store_instead_of_a_save(held):
    with pytest.raises(OSError):
        ce.record_owner_ack(held.root, provider="openrouter", model="vendor/other", window_tokens=50000)
    with pytest.raises(TimeoutError):
        ce.revoke_owner_ack(held.root, held.ack["route_fp"])
    with pytest.raises(TimeoutError):
        response_limits.record_response_ack(held.root, provider="openrouter", model="vendor/acked",
                                            max_output_tokens=2048)
    assert ce._store_path(held.root).read_bytes() == held.before


def test_best_effort_observations_skip_a_busy_store_without_failing(held):
    ce._DENSITY_MEMO.clear()
    assert ce._store_evidence(held.root, "probes", "fp", {"status": "confirmed"}) is False
    vision.record_catalog_image_input("openai-compatible", IMAGE_TARGET["base_url"], [{"id": "m"}],
                                      source="a catalog", api_key="fixture-key")
    response_limits.record_catalog_limits(held.root, [("vendor/limited", 4096)], provider="openrouter",
                                          base_url=OPENROUTER, field="max_completion_tokens", source="a catalog")
    context_window.record_observation(held.root, "observed-route", accepted=50000)
    ce.record_token_density(held.root, "vendor/dense", prompt_chars=40000, prompt_tokens=11000)
    assert not ce._DENSITY_MEMO  # no false "already recorded" throttle after a skipped write
    assert ce._store_path(held.root).read_bytes() == held.before
    assert _thread_lock_free()  # each refused acquire gave the process lock back


def test_catalog_reads_survive_a_busy_store(held, monkeypatch):
    import requests

    from ouroboros.gateway import models
    from ouroboros.llm import LLMClient
    from tests._shared import _PROVIDER_CATALOG_PRISTINE, _restore_provider_catalog

    rows = [{"id": "vendor/m", "context_length": 200000, "supported_parameters": ["tools"],
             "top_provider": {"max_completion_tokens": 4096}, "architecture": {"input_modalities": ["text"]}}]
    answer = SimpleNamespace(status_code=200, json=lambda: {"data": rows})
    monkeypatch.setattr(requests, "get", lambda url, **_kwargs: answer)
    _restore_provider_catalog(LLMClient, _PROVIDER_CATALOG_PRISTINE)
    try:
        LLMClient._read_openrouter_catalogue()
        assert LLMClient._CAPABILITIES_FETCH_OK is True and LLMClient._CONTEXT_LENGTH_CACHE == {"vendor/m": 200000}
    finally:
        _restore_provider_catalog(LLMClient, _PROVIDER_CATALOG_PRISTINE)

    original_client = httpx.AsyncClient
    monkeypatch.setattr(models, "load_settings", lambda: dict(OPENAI_COMPATIBLE_BASE_URL=IMAGE_TARGET["base_url"],
                                                              OPENAI_COMPATIBLE_API_KEY="fixture-key"))
    monkeypatch.setattr(models, "_subscription_model_catalog", lambda *args: {"items": [], "errors": []})
    monkeypatch.setattr(models.httpx, "AsyncClient", lambda **kwargs: original_client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"data": [{**rows[0], "id": "m"}]})),
        **kwargs))
    payload = json.loads(asyncio.run(models.api_model_catalog(None)).body)
    assert payload["errors"] == [] and [item["id"] for item in payload["items"]] == ["m"]
    assert ce._store_path(held.root).read_bytes() == held.before


def _thread_lock_free() -> bool:
    """Whether another thread can take ``_STORE_LOCK`` now; the RLock always readmits its own holder."""
    taken = []

    def probe():
        if ce._STORE_LOCK.acquire(timeout=1):
            ce._STORE_LOCK.release()
            taken.append(True)

    worker = threading.Thread(target=probe)
    worker.start()
    worker.join()
    return taken == [True]


def test_transactions_share_the_update_sidecar_and_release_exactly_what_they_took(tmp_path, monkeypatch):
    from ouroboros.utils import update_json_locked

    monkeypatch.setattr(ce, "_STORE_LOCK_WAIT_SEC", 0.2)
    first, second = tmp_path / "first", tmp_path / "second"
    sidecar = lambda root: ce._store_path(root).with_name("capability_evidence.json.lock")  # noqa: E731
    with ce._store_transaction(first):
        assert sidecar(first).exists() and not _thread_lock_free()
        with pytest.raises(TimeoutError):  # the generic locked writer of this file waits for the same sidecar
            update_json_locked(ce._store_path(first), lambda data: data, timeout_sec=0.1)
        with pytest.raises(TimeoutError):  # not re-entrant per store: a nested one waits out the bound
            with ce._store_transaction(first):
                pass
        with ce._store_transaction(second):  # another store is its own lock
            assert sidecar(first).exists() and sidecar(second).exists()
        assert sidecar(first).exists() and not sidecar(second).exists()
    assert not sidecar(first).exists() and _thread_lock_free()
    with pytest.raises(RuntimeError):
        with ce._store_transaction(first):
            raise RuntimeError("a failed mutation")
    assert not sidecar(first).exists() and _thread_lock_free()
    assert ce._store_evidence(first, "probes", "after", {"status": "confirmed"}) is True
    assert ce._read(first)["probes"]["after"] == {"status": "confirmed"}


def test_revoke_failure_reaches_the_http_consumer_without_claiming_removal(tmp_path, monkeypatch):
    from ouroboros.gateway import settings

    ack = ce.record_owner_ack(tmp_path, provider="openrouter", model="vendor/acked", window_tokens=100000)
    before = ce._store_path(tmp_path).read_bytes()
    audit = []

    async def body(_request):
        return {"provider": "openrouter", "model": "vendor/acked", "window_tokens": 0}

    monkeypatch.setattr(settings, "_json_body_or_empty", body)
    monkeypatch.setattr(settings, "request_drive_root", lambda _request: tmp_path)
    monkeypatch.setattr(settings, "_owner_audit", lambda *args: audit.append(args))
    with monkeypatch.context() as failed:
        failed.setattr(ce, "_save", lambda *_args: False)
        response = asyncio.run(settings.api_acknowledge_capability(SimpleNamespace()))
        assert response.status_code == 500
        assert ce._store_path(tmp_path).read_bytes() == before
        assert not audit
    response = asyncio.run(settings.api_acknowledge_capability(SimpleNamespace()))
    assert response.status_code == 200 and json.loads(response.body)["ack"]["removed"] is True
    assert ack["route_fp"] not in ce._read(tmp_path)["owner_acks"]
    assert len(audit) == 1


def _save_response_maximum(root, monkeypatch):
    """Settings' Maximum response save over real Starlette HTTP; an escaping error is the server's plain 500."""
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros.gateway import settings

    audit = []
    monkeypatch.setattr(settings, "request_drive_root", lambda _request: root)
    monkeypatch.setattr(settings, "_owner_audit", lambda *args: audit.append(args))
    app = Starlette(routes=[Route("/api/owner/capability-ack", settings.api_acknowledge_capability, methods=["POST"])])
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/api/owner/capability-ack",
                               json={"provider": "openrouter", "model": "vendor/acked", "max_output_tokens": 2048})
    return response, audit


def _unsaved(response):
    """What the page's fetchJson shows: the JSON ``error``, never "non-json response (HTTP 500)"."""
    assert response.status_code == 500, response.text
    assert response.headers["content-type"].startswith("application/json"), response.text
    return response.json()["error"]


def test_a_busy_store_answers_a_response_maximum_save_as_json(held, monkeypatch):
    response, audit = _save_response_maximum(held.root, monkeypatch)
    assert "capability_evidence.json.lock" in _unsaved(response)
    assert not audit and ce._store_path(held.root).read_bytes() == held.before


def test_an_unwritable_store_answers_a_response_maximum_save_as_json(tmp_path, monkeypatch):
    ce.record_owner_ack(tmp_path, provider="openrouter", model="vendor/acked", window_tokens=100000)
    before = ce._store_path(tmp_path).read_bytes()
    with monkeypatch.context() as failed:
        failed.setattr(ce, "_save", lambda *_args: False)
        response, audit = _save_response_maximum(tmp_path, monkeypatch)
        assert _unsaved(response) == "Could not persist response acknowledgement"
        assert not audit and ce._store_path(tmp_path).read_bytes() == before
    response, audit = _save_response_maximum(tmp_path, monkeypatch)
    assert response.status_code == 200 and response.json()["ack"]["max_output_tokens"] == 2048
    assert len(audit) == 1 and response_limits.resolve_response_limit(
        tmp_path, provider="openrouter", model="vendor/acked").max_output_tokens == 2048
