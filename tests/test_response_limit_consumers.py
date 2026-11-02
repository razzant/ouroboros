"""Response maxima reach planning and native/API consumers."""
import asyncio
from types import SimpleNamespace

import pytest

from ouroboros import capability_evidence as ce, usage_accounting as ua
from ouroboros.response_limits import record_response_ack, resolve_response_limit
from tests import test_llm_claudexor as claudexor_fixtures, test_processing_transport as transport_fixtures
from tests.test_llm_claudexor import MODEL

transport = transport_fixtures.transport
native_setup = claudexor_fixtures.setup


def test_native_catalog_maximum_remains_planning_evidence_not_an_engine_option(native_setup, monkeypatch):
    from ouroboros.response_limits import response_allowance
    from ouroboros.llm import LLMClient
    root, gateway, client = native_setup
    monkeypatch.setattr(LLMClient, "claudexor_model_catalog", staticmethod(lambda source, profile=None, **kw: {
        "source": source, "credentialProfileId": profile or "account-a", "accountFingerprint": "fingerprint-a",
        "observedAt": ce.utc_now_iso(), "provenance": "fixture",
        "models": [{"id": "exact-model", "maxOutputTokens": 4096}]}))
    allowance = response_allowance(MODEL, 65536, credential_profile_id="account-a", allow_fetch=True)
    assert allowance == 4096
    _, usage = client.chat([{"role": "user", "content": "reply"}], MODEL, max_tokens=allowance, model_account_override="account-a")
    assert usage["claudexor"]["output_reserve_tokens"] == 4096
    assert usage["claudexor"]["output_cap_applied"] is False
    assert "maxOutputTokens" not in gateway.uploads[-1][0]["options"]


def test_a_real_reviewer_send_and_the_forced_candidate_share_the_physical_ceiling(transport):
    from ouroboros.review_substrate import ReviewRequest, ReviewSlot, run_review_request
    from ouroboros.task_pacing import prospective_wrapup_attempt_request
    from tests._usage_store_testing import ledger_rows
    root, client, sent = transport
    model = "openai::test-model"
    route = dict(provider="openai", model=model, base_url="https://api.openai.com/v1")
    record_response_ack(root, **route, max_output_tokens=4096)
    ce.record_owner_ack(root, **route, window_tokens=100000)
    # The review_change wave's surface (a task's result review dispatches under it too).
    result = run_review_request(ReviewRequest(surface="multi_model_review", goal="assess", task_id="processing",
        no_proxy=False, max_tokens=65536, policy={"min_successful_slots": 1}),
        slots=[ReviewSlot(slot_id="one", model=model)], drive_root=root, llm=client)
    # One answered send: the format-repair resend left with the task-acceptance machine.
    assert len(sent) == 1, result.actors
    assert sent[0]["max_completion_tokens"] == 4096
    assert len(ledger_rows(root)) == 1
    prospective = prospective_wrapup_attempt_request(llm=client, messages=[{"role": "user", "content": "finish"}],
        model=model, reasoning_effort="high")
    assert prospective.max_completion_tokens == 4096
    assert len(ledger_rows(root)) == 1  # Preparing a forced send reserves nothing.


@pytest.mark.parametrize("role", ["main", "light", "reviewer:triad_1", "consciousness"])
def test_async_initial_continuation_and_rebinding_use_route_cap_before_capture(transport, monkeypatch, role):
    root, client, sent = transport
    model = "openai-compatible::new-model"
    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL", "https://one.invalid/v1")
    target = client._resolve_remote_target(model)
    record_response_ack(root, provider=target["provider"], model=model, base_url=target["base_url"], max_output_tokens=4096)
    messages = [{"role": "user", "content": "initial"}]
    async def send():
        message, _ = await client.chat_async(messages, model, max_tokens=65536, model_role=role)
        assert ua.last_physical_attempt_capture().max_completion_tokens == 4096
        return message
    for index in range(2):
        message = asyncio.run(send())
        assert sent[-1]["max_tokens"] == 4096
        messages.extend([message, {"role": "user", "content": "continue"}])
    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL", "https://two.invalid/v1")
    asyncio.run(client.chat_async(messages, model, max_tokens=65536, model_role=role))
    assert sent[-1]["max_tokens"] == 65536


def test_legacy_compatible_endpoint_and_fit_reserve_use_the_same_cap(transport, monkeypatch):
    from ouroboros.context_fit import main_output_reserve_tokens
    from ouroboros.gateway.settings import _active_main_route
    from ouroboros.response_limits import response_limit_preview
    root, client, sent = transport
    monkeypatch.delenv("OPENAI_COMPATIBLE_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://legacy.invalid/v1")
    model = "openai-compatible::new-model"
    settings = {"OUROBOROS_MODEL": model, "OPENAI_BASE_URL": "https://legacy.invalid/v1", "OPENAI_API_KEY": "synthetic"}
    route = _active_main_route(settings)
    assert route["base_url"] == ""  # The context-window identity keeps its configured endpoint.
    # The Settings maximum binds the endpoint the send actually reaches.
    owner_route = response_limit_preview(root, route)["route"]
    assert owner_route["base_url"] == client._resolve_remote_target(model)["base_url"] == settings["OPENAI_BASE_URL"]
    ack = {key: owner_route[key] for key in ("provider", "model", "base_url")}
    record_response_ack(root, **ack, max_output_tokens=4096)
    evidence = ce.probe(root, **route, allow_fetch=False)
    assert main_output_reserve_tokens(use_local=False, evidence=evidence) == 4096
    client.chat([{"role": "user", "content": "reply"}], model, max_tokens=65536)
    assert sent[-1]["max_tokens"] == 4096
    # Clearing the assertion changes only this output record; stale metadata supplies no ceiling.
    record_response_ack(root, **ack, max_output_tokens=0)
    assert resolve_response_limit(root, **ack).ceiling(65536) == 65536


def test_old_output_metadata_cannot_borrow_the_fresh_context_timestamp(tmp_path, monkeypatch):
    from ouroboros.response_limits import record_metadata_limit
    route = {"provider": "openai-compatible", "model": "openai-compatible::new", "base_url": "https://old.invalid/v1"}
    ce.record_owner_ack(tmp_path, **route, window_tokens=100000)
    record_metadata_limit(tmp_path, **route, maximum=4096, observed_at="2000-01-01T00:00:00Z")
    monkeypatch.setattr("httpx.get", lambda *a, **k: (_ for _ in ()).throw(OSError("metadata offline")))
    evidence = ce.probe(tmp_path, **route)
    assert not evidence.stale and evidence.response_limit["stale"]
    assert resolve_response_limit(tmp_path, **route).ceiling(65536) == 65536


@pytest.mark.parametrize("fails", [False, True])
def test_a_route_its_catalog_omits_reads_that_catalog_once_per_failed_evidence_ttl(tmp_path, monkeypatch, fails):
    from ouroboros.llm import LLMClient
    from ouroboros.response_limits import record_catalog_limits
    reads = []
    def read(_cls):
        reads.append("catalog")
        if fails:
            raise OSError("catalog offline")
        record_catalog_limits(tmp_path, [("vendor/listed", 4096)], provider="openrouter",
                              base_url="https://openrouter.ai/api/v1", field="max_completion_tokens", source="fixture")
    monkeypatch.setattr(LLMClient, "_fetch_openrouter_capabilities", classmethod(read))
    route = dict(provider="openrouter", model="vendor/unlisted")
    for _ in range(3):  # every window probe asks again; the observed absence answers
        assert resolve_response_limit(tmp_path, **route, allow_fetch=True).ceiling(65536) == 65536
    assert len(reads) == 1
    listed = resolve_response_limit(tmp_path, provider="openrouter", model="vendor/listed", allow_fetch=True)
    assert listed.ceiling(65536) == (65536 if fails else 4096) and len(reads) == (2 if fails else 1)
    monkeypatch.setattr(ce, "_FAILED_TTL_SEC", -1)  # an expired absence reads the catalog again ...
    resolve_response_limit(tmp_path, **route, allow_fetch=True)
    monkeypatch.setattr(ce, "_FAILED_TTL_SEC", 600)  # ... and is observed afresh, not re-read per call
    resolve_response_limit(tmp_path, **route, allow_fetch=True)
    assert len(reads) == (3 if fails else 2)


def test_an_expired_openrouter_maximum_is_refreshed_by_one_real_catalog_read(monkeypatch):
    import requests
    from ouroboros.llm import LLMClient
    from ouroboros.response_limits import record_metadata_limit
    root, gets = ce.canonical_evidence_root(), []
    catalog = {"data": [{"id": "vendor/listed", "context_length": 100_000,
                         "top_provider": {"max_completion_tokens": 4096}}]}
    monkeypatch.setattr(requests, "get", lambda url, **_kw: gets.append(url) or SimpleNamespace(
        status_code=200, json=lambda: catalog))
    # This process already holds the window catalog, so its window cache alone never reads again.
    LLMClient._SUPPORTED_PARAMS_FETCHED = LLMClient._CAPABILITIES_FETCH_OK = True
    LLMClient._CONTEXT_LENGTH_CACHE["vendor/listed"] = 100_000
    route = dict(provider="openrouter", model="vendor/listed")
    record_metadata_limit(root, **route, maximum=4096, source="OpenRouter top_provider.max_completion_tokens",
                          observed_at="2000-01-01T00:00:00Z")
    assert resolve_response_limit(root, **route).ceiling(65536) == 65536  # expired evidence never caps
    refreshed = resolve_response_limit(root, **route, allow_fetch=True)
    assert refreshed.ceiling(65536) == 4096 and refreshed.observed_at > "2000-01-02" and not refreshed.stale
    assert gets == ["https://openrouter.ai/api/v1/models"]
    assert resolve_response_limit(root, **route, allow_fetch=True).ceiling(65536) == 4096 and len(gets) == 1


@pytest.mark.parametrize("pin", ["", "account-a"])
def test_subscription_reviewer_sizing_reads_the_maximum_of_the_account_its_window_observed(
        native_setup, monkeypatch, pin):
    from ouroboros import config
    from ouroboros.llm import LLMClient
    from ouroboros.review_records import ReviewSlot
    from ouroboros.tools.review_helpers import calibrated_input_token_limit
    from ouroboros.tools.review_synthesis import per_slot_input_token_limits
    root, _gateway, _client = native_setup
    monkeypatch.setattr(config, "DATA_DIR", root)  # the window store and the response store are one root
    reads = []
    def catalog(source, profile=None, **_kw):
        reads.append(profile)
        return {"source": source, "credentialProfileId": profile or "account-a", "accountFingerprint": "fingerprint-a",
                "observedAt": ce.utc_now_iso(), "provenance": "fixture",
                "models": [{"id": "exact-model", "contextWindow": 200_000, "maxOutputTokens": 4096}]}
    monkeypatch.setattr(LLMClient, "claudexor_model_catalog", staticmethod(catalog))
    slot = ReviewSlot(slot_id="one", model=MODEL, session_profile=pin)
    def limit(reserve):
        return calibrated_input_token_limit(MODEL, context_window=200_000, output_reserve=reserve, tokenizer_margin=25_000)
    first = per_slot_input_token_limits([MODEL], output_reserve=65536, tokenizer_margin=50_000, slots=[slot])
    assert first == {"one": limit(4096)}  # without the observed account: the 50,000 reserve of a 200K window
    reads_per_sizing = len(reads)
    record_response_ack(root, provider="claudexor", model=MODEL, max_output_tokens=2048, options={
        "source_id": "codex", "credential_profile_id": "account-a", "account_fingerprint": "fingerprint-a"})
    assert per_slot_input_token_limits([MODEL], output_reserve=65536, tokenizer_margin=50_000,
                                       slots=[slot]) == {"one": limit(2048)}
    assert len(reads) == 2 * reads_per_sizing + 1  # the ack's own exact-account read; sizing adds none


def test_gigachat_numeric_field_is_capped_before_its_real_accounted_send(transport, monkeypatch):
    root, client, sent = transport
    model = 'gigachat::GigaChat-2'
    target = client._resolve_remote_target(model)
    record_response_ack(root, provider='gigachat', model=model, base_url=target['base_url'], max_output_tokens=4096)
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='ok', function_call=None), finish_reason='stop')],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=1, precached_prompt_tokens=0))
    monkeypatch.setattr(client, '_get_gigachat_client', lambda *a, **k: SimpleNamespace(chat=lambda candidate: sent.append(candidate) or response))
    client.chat([{'role': 'user', 'content': 'reply'}], model, max_tokens=65536)
    assert sent[-1]['max_tokens'] == ua.last_physical_attempt_capture().max_completion_tokens == 4096


def test_local_cap_is_bound_to_the_served_artifact_and_local_endpoint(transport, monkeypatch):
    from ouroboros import local_model
    root, client, sent = transport
    monkeypatch.setenv('LOCAL_MODEL_SOURCE', 'fixture/repo')
    monkeypatch.setenv('LOCAL_MODEL_FILENAME', 'one.gguf')
    serving = {'model_path': '/models/one.gguf', 'port': 8766}
    manager = SimpleNamespace(serving_context_evidence=lambda: {'context_window': 32768, 'confirmed': True},
                              serving_artifact=lambda: dict(serving))
    monkeypatch.setattr(local_model, 'get_manager', lambda: manager)
    monkeypatch.setattr(client, '_get_local_client', lambda: client._get_remote_client({}))
    def reply():
        client.chat([{'role': 'user', 'content': 'reply'}], 'unused-remote-label', use_local=True, max_tokens=65536)
        return sent[-1]['max_tokens']
    record_response_ack(root, provider='local', model='unused-remote-label', max_output_tokens=4096)
    assert reply() == ua.last_physical_attempt_capture().max_completion_tokens == 4096
    # Saving file B leaves it pending until Stop/Start: A still serves and keeps its own maximum.
    monkeypatch.setenv('LOCAL_MODEL_FILENAME', 'two.gguf')
    assert reply() == 4096
    serving.update(model_path='/models/two.gguf')  # restarted on B
    assert reply() == 8192  # Existing local room rule, no inherited 4096 assertion.


@pytest.mark.parametrize("source,filename,served", [
    ("/models/model.gguf", "", "/models/model.gguf"),
    ("org/repo", "model-Q4.gguf", "/hf/hub/models--org--repo/snapshots/abc123/quant/model-Q4.gguf"),
    ("org/repo", "q4/model.gguf", "/hf/hub/models--org--repo/snapshots/abc123/q4/model.gguf"),
    ("org/repo", "q8/model.gguf", "/hf/hub/models--org--repo/snapshots/abc123/q8/model.gguf"),
])
def test_a_maximum_applied_while_stopped_holds_after_the_model_starts(transport, monkeypatch, source, filename, served):
    import sys
    from unittest.mock import Mock
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient
    from ouroboros import local_model
    from ouroboros.gateway import settings as gateway
    root, client, sent = transport
    monkeypatch.delenv("OUROBOROS_IN_WORKER", raising=False)
    for key, value in (("LOCAL_MODEL_SOURCE", source), ("LOCAL_MODEL_FILENAME", filename), ("LOCAL_MODEL_PORT", "8766")):
        monkeypatch.setenv(key, value)
    manager = local_model.LocalModelManager()  # the real manager; nothing serves yet
    monkeypatch.setattr(local_model, "get_manager", lambda *_a, **_k: manager)
    monkeypatch.setattr(gateway, "load_settings", lambda: {"OUROBOROS_MODEL": "local-label", "USE_LOCAL_MAIN": "true"})
    monkeypatch.setattr(gateway, "_owner_audit", lambda *a: None)
    app = Starlette(routes=[Route("/api/settings", gateway.api_settings_get),
                            Route("/api/owner/capability-ack", gateway.api_acknowledge_capability, methods=["POST"])])
    app.state.drive_root = root
    api = TestClient(app)
    def preview():
        return api.get("/api/settings", params={"response_limit_preview": "1", "model": "local-label", "local": "true"}).json()
    # Use the real resolver, downloader and launcher; only HF/process boundaries are fake.
    downloads = []
    def download(**kw):
        downloads.append(kw)
        assert kw["repo_id"] == "org/repo"
        assert "/".join(filter(None, (kw["subfolder"], kw["filename"]))) == served.split("/abc123/")[1]
        return served
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        list_repo_files=lambda repo: ["quant/model-Q4.gguf", "q4/model.gguf", "q8/model.gguf"],
        hf_hub_download=download))
    monkeypatch.setattr(local_model.os.path, "isfile", lambda path: str(path) == source and source.startswith("/"))
    monkeypatch.setattr(local_model, "subprocess", SimpleNamespace(**{
        **vars(local_model.subprocess), "run": lambda *a, **k: SimpleNamespace(returncode=0),
        "Popen": Mock(return_value=SimpleNamespace(poll=lambda: None, pid=7))}))
    monkeypatch.setattr(local_model, "threading", SimpleNamespace(**{
        **vars(local_model.threading), "Thread": Mock(return_value=SimpleNamespace(start=lambda: None))}))
    monkeypatch.setattr("ouroboros.process_custody.record_process", lambda *a, **k: None)
    stopped = preview()
    assert stopped["route"]["provider"] == "local"
    acked = api.post("/api/owner/capability-ack", json={**stopped["route"], "max_output_tokens": 4096,
                                                         "route_fp": stopped["response_limit"]["route_fp"]})
    assert acked.status_code == 200, acked.text
    assert downloads == []  # Preview/ack may resolve metadata, never download a model.
    path = manager.download_model(source, filename)
    manager.start_server(path, port=8766, n_ctx=32768, source=source, filename=filename)
    assert manager._model_path == served
    assert local_model.subprocess.Popen.call_args.args[0][3:7] == ["--model", served, "--port", "8766"]
    manager._status = "ready"  # Health is external to the identity path.
    assert len(downloads) == (0 if source.startswith("/") else 1)
    monkeypatch.setattr(manager, "measure_prepared_input", lambda _payload: None)
    monkeypatch.setattr(client, "_get_local_client", lambda: client._get_remote_client({}))
    client.chat([{"role": "user", "content": "reply"}], "local-label", use_local=True, max_tokens=65536)
    assert sent[-1]["max_tokens"] == ua.last_physical_attempt_capture().max_completion_tokens == 4096
    started = preview()
    assert started["response_limit"] == {**stopped["response_limit"], **acked.json()["ack"]}
    assert started["response_limit"]["route_fp"] == stopped["response_limit"]["route_fp"]
    if filename.startswith(("q4/", "q8/")):
        # Same repo and basename, different quantization; a pending save cannot change A.
        other = ("q8/" if filename.startswith("q4/") else "q4/") + "model.gguf"
        monkeypatch.setenv("LOCAL_MODEL_FILENAME", other)
        client.chat([{"role": "user", "content": "pending"}], "local-label", use_local=True, max_tokens=65536)
        assert sent[-1]["max_tokens"] == 4096
        assert preview()["response_limit"]["route_fp"] == stopped["response_limit"]["route_fp"]
        manager._proc, manager._status = None, "offline"
        other_preview = preview()
        assert other_preview["response_limit"]["route_fp"] != stopped["response_limit"]["route_fp"]
        served = served.rsplit("/", 2)[0] + "/" + other
        path = manager.download_model(source, other)
        manager.start_server(path, port=8766, n_ctx=32768, source=source, filename=other)
        manager._status = "ready"
        client.chat([{"role": "user", "content": "restarted"}], "local-label", use_local=True, max_tokens=65536)
        assert sent[-1]["max_tokens"] == ua.last_physical_attempt_capture().max_completion_tokens == 8192
        assert preview()["response_limit"]["route_fp"] == other_preview["response_limit"]["route_fp"]


def test_local_artifact_keys_match_between_settings_and_served_files(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(list_repo_files=lambda repo: []))
    from ouroboros.local_model import local_artifact_key
    assert local_artifact_key("/models/model.gguf", "") == local_artifact_key(model_path="/models/model.gguf")
    assert local_artifact_key("org/repo", "quant/m-Q4.gguf") == \
        local_artifact_key(model_path="/c/models--org--repo/snapshots/r1/quant/m-Q4.gguf") == "hf:org/repo:quant/m-Q4.gguf"
    # Different artifacts keep different keys.
    assert len({local_artifact_key("org/repo", "a.gguf"), local_artifact_key("org/repo", "b.gguf"),
                local_artifact_key("org/other", "a.gguf"), local_artifact_key("/models/a.gguf"),
                local_artifact_key("org/repo", "q4/model.gguf"), local_artifact_key("org/repo", "q8/model.gguf")}) == 6


def test_local_basename_identity_uses_resolution_and_never_guesses_a_subfolder(monkeypatch):
    import sys
    from unittest.mock import Mock
    from ouroboros.local_model import LocalModelManager, local_artifact_key

    listing = Mock(return_value=["q4/model.gguf", "q8/model.gguf"])
    download = Mock(side_effect=AssertionError("No artifact download during identity resolution"))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        list_repo_files=listing, hf_hub_download=download))
    # A full path remains exact even when the basename is ambiguous.
    q4 = local_artifact_key("org/repo", "q4/model.gguf")
    assert q4 != local_artifact_key("org/repo", "q8/model.gguf")
    assert q4 == local_artifact_key(model_path=r"C:\cache\models--org--repo\snapshots\r1\q4\model.gguf")
    listing.assert_not_called()
    for operation in (local_artifact_key, LocalModelManager().download_model):
        with pytest.raises(ValueError, match="Ambiguous filename"):
            operation("org/repo", "model.gguf")
    # A failed metadata read cannot authorize applying the root-file maximum to q4.
    listing.side_effect = OSError("offline")
    assert local_artifact_key("org/repo", "model.gguf") != q4
    download.assert_not_called()


@pytest.mark.parametrize('in_worker', [False, True])
def test_the_serving_artifact_is_the_live_launch_not_the_saved_settings(monkeypatch, in_worker):
    from ouroboros import local_model, platform_layer
    manager = local_model.LocalModelManager.__new__(local_model.LocalModelManager)
    manager._proc, manager._status, manager._model_path, manager._port = None, 'offline', '/models/a b.gguf', 8766
    monkeypatch.setenv('OUROBOROS_IN_WORKER', '1') if in_worker else monkeypatch.delenv('OUROBOROS_IN_WORKER', raising=False)
    monkeypatch.setattr(manager, '_published_serving_evidence', lambda: {'confirmed': False})
    assert manager.serving_artifact() == {}
    manager._proc, manager._status = SimpleNamespace(poll=lambda: None, pid=42), 'ready'
    monkeypatch.setattr(manager, '_published_serving_evidence', lambda: {'confirmed': True, 'process_id': 42, 'port': 8766})
    monkeypatch.setattr(platform_layer, 'process_command', lambda pid: (
        'python -m ouroboros.local_model_server --model /models/a b.gguf --port 8766 --n_gpu_layers 0 --n_ctx 4096'))
    assert manager.serving_artifact() == {'model_path': '/models/a b.gguf', 'port': 8766}


@pytest.mark.parametrize("expired", [False, True])
@pytest.mark.parametrize("account", ["", "account-b"])
def test_settings_preview_binds_the_exact_account_and_stores_nothing(tmp_path, monkeypatch, expired, account):
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient
    from ouroboros.gateway import settings as gateway
    from ouroboros.llm import LLMClient
    model, writes, reads = "claudexor::test-source=exact-model", [], []
    def catalog(source, profile=None, *, requested_model=None):
        reads.append((source, profile, requested_model))
        return {"source": source, "credentialProfileId": profile or "account-a", "accountFingerprint": "fingerprint-a",
                "observedAt": "2000-01-01T00:00:00Z" if expired else ce.utc_now_iso(), "provenance": "fixture",
                "models": [{"id": "exact-model", "maxOutputTokens": 4096}]}
    monkeypatch.setattr(LLMClient, "claudexor_model_catalog", staticmethod(catalog))
    monkeypatch.setattr(gateway, "load_settings", lambda: {"OUROBOROS_MODEL": model})
    monkeypatch.setattr(gateway, "_owner_audit", lambda *a: None)
    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: tmp_path)
    app = Starlette(routes=[Route("/api/settings", gateway.api_settings_get),
                            Route("/api/owner/capability-ack", gateway.api_acknowledge_capability, methods=["POST"])])
    app.state.drive_root = tmp_path
    client = TestClient(app)
    saved = ce._save
    monkeypatch.setattr(ce, "_save", lambda root, data: writes.append(root) or saved(root, data))
    params = {"response_limit_preview": "1", "model": model, "local": "false", "account": account}
    preview = client.get("/api/settings", params=params).json()
    assert writes == []  # A passive Settings read writes no evidence.
    assert reads == [("test-source", account or None, "exact-model")]
    assert not ce._store_path(tmp_path).exists()  # Even a cold preview reads the local catalog.
    assert preview["route"]["options"] == {"source_id": "test-source", "credential_profile_id": account or "account-a",
                                           "account_fingerprint": "fingerprint-a"}
    assert preview["response_limit"]["max_output_tokens"] == 4096
    # An expired catalog is shown as such (Settings reads it as unknown), never as a current maximum.
    assert preview["response_limit"]["stale"] is expired
    assert preview["window"]["reserve"] == (65536 if expired else 4096)
    acked = client.post("/api/owner/capability-ack", json={**preview["route"], "route_fp": preview["response_limit"]["route_fp"],
                                                           "max_output_tokens": 2048})
    if expired:  # nor can an expired account binding be acknowledged
        assert acked.status_code == 400 and "Refresh" in acked.text
        return
    assert acked.status_code == 200, acked.text
    writes.clear()
    again = client.get("/api/settings", params=params).json()
    assert again["response_limit"]["source"] == "owner_ack" and again["response_limit"]["max_output_tokens"] == 2048
    assert again["window"]["reserve"] == 2048 and writes == []


def test_a_malformed_subscription_identity_keeps_its_failed_window_evidence(tmp_path):
    evidence = ce.probe(tmp_path, provider="claudexor", model="claudexor::missing-model", base_url="")
    assert evidence.status == ce.STATUS_FAILED and evidence.response_limit["max_output_tokens"] == 0
    from ouroboros.reviewer_window import window_scaled_reserves
    assert window_scaled_reserves(0, output_reserve=65536, tokenizer_margin=1, model_id="claudexor::missing-model") == (65536, 1)


def test_commit_gate_seats_reserve_each_route_maximum(transport, tmp_path):
    """Each paid seat reserves its own route's maximum response. (A catalog row's price is no
    longer a reservation estimate: ``review_price`` measures it from the usage ledger.)"""
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.tools.review_admission import commit_gate_paid_seats
    from ouroboros.tools.review_multi_model import _review_output_budget
    root, _client, _sent = transport
    capped, other = "openai::test-model", "openai::other-model"
    record_response_ack(root, provider="openai", model=capped, max_output_tokens=4096)
    prepared = {"prompt": "PACKET", "stable_prefix_len": 0, "session_task": "", "target_repo": tmp_path,
                "models": [capped, other], "routes": [ReviewRouteKind.API_CHAT] * 2,
                "row_plan": {"slot_ids": ["one", "two"], "retrieves": [False, False],
                             "session_profiles": ["", ""], "use_local": [False, False]}}
    seats = commit_gate_paid_seats(prepared, False)
    assert [seat["max_completion_tokens"] for seat in seats] == [4096, _review_output_budget()]


def test_light_consolidation_and_memory_fit_plan_the_route_maximum(monkeypatch):
    from dataclasses import asdict
    from ouroboros import consolidator, context_fit, memory_fallback
    from ouroboros.response_limits import ResponseLimit
    from tests.test_consolidator_context_fit import _LLM, _tokens
    prompt = "Summarize this source.\n" + "source " * 400
    window = _tokens(prompt) + 4096  # holds the request only beside a 4096 reserve, never the shipped 16384

    def resolve(task, *, allow_fetch):
        evidence = ce.CapabilityEvidence(window, "confirmed", "test", "route-test", model=task["model"], provider="openrouter")
        evidence.response_limit = asdict(ResponseLimit(4096, "asserted", "owner_ack", ce.utc_now_iso(), route_fp="route-test"))
        return {"model": task["model"], "provider": "openrouter"}, evidence
    monkeypatch.setattr(consolidator, "_consolidation_route", lambda: ("test/model", False))
    monkeypatch.setattr(context_fit, "resolve_context_fit_route", resolve)
    monkeypatch.setattr(context_fit, "_route_calibration_ratio", lambda *_, **_kw: 1.0)
    llm = _LLM()
    consolidator._call_consolidation_llm(llm, prompt, "Probe")
    assert len(llm.calls) == 1 and llm.calls[0]["max_tokens"] == 4096
    assert memory_fallback._light_fit({"model": "test/model", "use_local": False})[0] == window - 4096
