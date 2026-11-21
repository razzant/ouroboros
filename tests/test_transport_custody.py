from __future__ import annotations

from types import SimpleNamespace

import pytest
import httpx

from ouroboros import usage_accounting as ua
from tests._usage_store_testing import ledger_rows


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(root / "settings.json"))
    monkeypatch.setenv("TOTAL_BUDGET", "100")
    (root / "state").mkdir(parents=True)
    return root


def _request(root):
    return ua.AttemptRequest(
        model="openai/gpt-5.2", provider="openai", reservation_usd=1.0,
        drive_root=root, task_id="transport", root_task_id="transport",
        source="test.transport_custody",
    )


def test_implicit_fallback_context_cannot_release_a_read_timeout(data_root):
    """A later fallback exception inherits the prior leg as implicit context."""
    from ouroboros.transport_custody import (
        is_pre_dispatch_transport_failure, release_pre_dispatch_attempt,
    )

    reservation = ua.reserve_attempt(_request(data_root))
    ua.mark_dispatched(reservation)
    try:
        raise httpx.ConnectError("first leg refused")
    except httpx.ConnectError:
        try:
            raise httpx.ReadTimeout("second leg timed out after dispatch")
        except httpx.ReadTimeout as exc:
            assert exc.__context__ is not None
            assert not is_pre_dispatch_transport_failure(exc)
            assert not release_pre_dispatch_attempt(reservation, exc)
            ua.mark_unresolved(reservation, "read timeout after dispatch")

    assert ua.usage_projection(data_root)["unresolved_upper_bound_usd"] == 1.0


def test_explicit_transport_cause_still_proves_pre_dispatch(data_root):
    from ouroboros.transport_custody import is_pre_dispatch_transport_failure

    try:
        raise httpx.ConnectError("socket refused")
    except httpx.ConnectError as cause:
        try:
            raise RuntimeError("provider connection wrapper") from cause
        except RuntimeError as wrapper:
            assert is_pre_dispatch_transport_failure(wrapper)


def test_requests_new_connection_error_proves_pre_dispatch(data_root):
    import requests
    import urllib3
    from ouroboros.transport_custody import is_pre_dispatch_transport_failure

    reason = urllib3.exceptions.NewConnectionError(None, "connection refused")
    wrapped = requests.exceptions.ConnectionError(
        urllib3.exceptions.MaxRetryError(None, "/messages", reason=reason)
    )
    assert is_pre_dispatch_transport_failure(wrapped)


def test_requests_read_timeout_connection_error_does_not_prove_pre_dispatch(data_root):
    import requests
    import urllib3
    from ouroboros.transport_custody import is_pre_dispatch_transport_failure

    reason = urllib3.exceptions.ReadTimeoutError(None, "/messages", "read timed out")
    wrapped = requests.exceptions.ConnectionError(
        urllib3.exceptions.MaxRetryError(None, "/messages", reason=reason)
    )
    assert not is_pre_dispatch_transport_failure(wrapped)


def test_requests_proxy_error_with_nested_connect_evidence_proves_pre_dispatch(data_root):
    """The standard unreachable-proxy chain (native Anthropic behind a dead
    proxy): requests.exceptions.ProxyError -> MaxRetryError -> urllib3
    ProxyError -> NewConnectionError is typed pre-dispatch evidence."""
    import requests
    import urllib3
    from ouroboros.transport_custody import is_pre_dispatch_transport_failure

    nested = urllib3.exceptions.NewConnectionError(
        None, "Failed to establish a new connection: [Errno 111] Connection refused"
    )
    proxy = urllib3.exceptions.ProxyError("Cannot connect to proxy.", nested)
    wrapped = requests.exceptions.ProxyError(
        urllib3.exceptions.MaxRetryError(None, "/messages", reason=proxy)
    )
    assert is_pre_dispatch_transport_failure(wrapped)


def test_requests_proxy_error_with_connect_timeout_evidence_proves_pre_dispatch(data_root):
    import requests
    import urllib3
    from ouroboros.transport_custody import is_pre_dispatch_transport_failure

    nested = urllib3.exceptions.ConnectTimeoutError("timed out connecting to proxy")
    proxy = urllib3.exceptions.ProxyError("Cannot connect to proxy.", nested)
    wrapped = requests.exceptions.ProxyError(
        urllib3.exceptions.MaxRetryError(None, "/messages", reason=proxy)
    )
    assert is_pre_dispatch_transport_failure(wrapped)


def test_requests_proxy_error_without_connect_evidence_stays_untyped(data_root):
    """A proxy failure that is NOT connect-time (a proxy HTTP response, a
    post-dispatch read failure) must never release custody."""
    import requests
    import urllib3
    from ouroboros.transport_custody import is_pre_dispatch_transport_failure

    proxy = urllib3.exceptions.ProxyError(
        "Your proxy appears to only use HTTP and not HTTPS",
        urllib3.exceptions.HTTPError("bad proxy response"),
    )
    wrapped = requests.exceptions.ProxyError(
        urllib3.exceptions.MaxRetryError(None, "/messages", reason=proxy)
    )
    assert not is_pre_dispatch_transport_failure(wrapped)


@pytest.mark.parametrize("url,expected", [
    ("http://localhost:11434/v1", True),
    ("http://127.0.0.2:11434/v1", True),  # the whole 127.0.0.0/8 range is this host
    ("http://10.0.0.5:8000/v1", False),
    ("http://example.com/v1", False),
    ("http://localhost.example/v1", False),  # a name is loopback only when it IS localhost
    ("https://openrouter.ai/api/v1", False),
    ("https://api.anthropic.com/v1", False),
    ("", False),
    ("not a url", False),
    # every inet_aton spelling the OS accepts for a local server, and a trailing-dot name
    ("http://10.1:8000/v1", False),  # an inet_aton shorthand of a remote address stays remote
    ("http://8.8.8.8/v1", False),
    ("http://example.com./v1", False),
])
def test_is_loopback_base_url(url, expected):
    from ouroboros.transport_custody import is_loopback_base_url

    assert is_loopback_base_url(url) is expected


def test_attempt_custody_event_fields_bind_ledger_and_cause():
    """Nanny-leaf S3: durable error events carry the attempt-ledger join key,
    the custody state, and the bounded transport cause TYPE (never raw text)."""
    import httpx

    from ouroboros import usage_accounting as ua
    from ouroboros.transport_custody import attempt_custody_event_fields

    capture = ua.PhysicalAttemptCapture(
        attempt_id="pa-s3", model="m", provider="openrouter", state="unresolved",
        candidate_measurement_kind="opaque",
    )
    cause = httpx.RemoteProtocolError("peer closed connection")
    try:
        raise RuntimeError("Connection error.") from cause
    except RuntimeError as exc:
        exc.physical_attempt_capture = capture
        fields = attempt_custody_event_fields(exc)
    assert fields["physical_attempt_id"] == "pa-s3"
    assert fields["attempt_custody_state"] == "unresolved"
    assert fields["transport_cause_type"] == "RemoteProtocolError"


def test_attempt_custody_event_fields_absent_safe():
    from ouroboros.transport_custody import attempt_custody_event_fields

    assert attempt_custody_event_fields(RuntimeError("plain")) == {}


def test_attempt_custody_capture_found_on_explicit_cause():
    """Sol lane B #2: wrappers (LocalContextTooLargeError, recovery RuntimeError)
    can carry the capture only on their explicit cause — the join key must
    survive the wrapping."""
    from ouroboros import usage_accounting as ua
    from ouroboros.transport_custody import attempt_custody_event_fields

    capture = ua.PhysicalAttemptCapture(
        attempt_id="pa-wrapped", model="m", provider="openrouter", state="unresolved",
        candidate_measurement_kind="opaque", provider_error_type="overflow",
    )
    inner = RuntimeError("provider said no")
    inner.physical_attempt_capture = capture
    try:
        raise ValueError("wrapper without capture") from inner
    except ValueError as exc:
        fields = attempt_custody_event_fields(exc)
    assert fields["physical_attempt_id"] == "pa-wrapped"
    assert fields["provider_error_type"] == "overflow"


def test_provider_error_type_is_the_providers_own_never_an_exception_class_name():
    """The durable error row names the provider's own type: a typed vendor code on the error or its cause
    (Claudexor's vendorCode, also when no capture rides the chain), never the class name a capture
    substitutes for the classifier; a capture's real provider type still passes (the overflow test above)."""
    from ouroboros.llm_claudexor import ClaudexorModelError
    from ouroboros.transport_custody import attempt_custody_event_fields

    busy = ClaudexorModelError({"code": "provider_failed", "message": "Model operation failed",
                                "context": {"vendorCode": "server_is_overloaded"}})
    assert attempt_custody_event_fields(busy) == {"provider_error_type": "server_is_overloaded"}
    try:
        raise RuntimeError("recovery wrapper") from busy
    except RuntimeError as exc:
        assert attempt_custody_event_fields(exc)["provider_error_type"] == "server_is_overloaded"

    class APIStatusError(Exception):
        pass

    sdk = APIStatusError("no structured body")
    sdk.physical_attempt_capture = ua.PhysicalAttemptCapture(
        attempt_id="pa-name", model="m", provider="openrouter", state="released",
        candidate_measurement_kind="opaque", provider_error_type="APIStatusError")
    fields = attempt_custody_event_fields(sdk)
    assert fields["physical_attempt_id"] == "pa-name" and "provider_error_type" not in fields


def test_attempt_custody_cause_walk_matches_bare_builtin_transport_errors():
    """Fable lane B F5: a bare builtins ConnectionResetError/TimeoutError cause
    (no httpx wrapper) still yields a transport cause type."""
    from ouroboros.transport_custody import attempt_custody_event_fields

    try:
        raise RuntimeError("wrapped") from ConnectionResetError("peer reset")
    except RuntimeError as exc:
        fields = attempt_custody_event_fields(exc)
    assert fields["transport_cause_type"] == "ConnectionResetError"


# ------------------------------------------- bounded paid repeat: the death class

def _unresolved_capture(provider: str = "openrouter", **extra) -> ua.PhysicalAttemptCapture:
    return ua.PhysicalAttemptCapture(
        attempt_id="pa-death", model="m", provider=provider, state="unresolved",
        candidate_measurement_kind="opaque", **extra,
    )


def _sdk_wrapped(cause: BaseException, capture=None):
    """The OpenAI SDK shape: ``raise APIConnectionError(request=request) from err``
    with the physical-attempt capture attached by execute_physical_attempt."""
    try:
        raise RuntimeError("Connection error.") from cause
    except RuntimeError as exc:
        if capture is not None:
            exc.physical_attempt_capture = capture
        return exc


def test_httpx_proxy_error_is_pre_dispatch():
    """A failed CONNECT tunnel supplies typed evidence that nothing reached the provider."""
    from ouroboros.transport_custody import is_pre_dispatch_transport_failure

    exc = _sdk_wrapped(httpx.ProxyError("CONNECT tunnel failed"), _unresolved_capture())
    assert is_pre_dispatch_transport_failure(exc) is True


def test_requests_lane_event_fields_name_the_innermost_typed_cause(data_root):
    """The durable row and the ledger's bounded cause text see the requests
    lane's typed death, which lives in ``args``/``reason``, not ``__cause__``:
    RemoteDisconnected over its ProtocolError wrapper; ProtocolError alone when
    urllib3 hands it over as MaxRetryError.reason; wrappers keep it."""
    import http.client

    import requests
    import urllib3
    from ouroboros.transport_custody import attempt_custody_event_fields

    disconnected = http.client.RemoteDisconnected("Remote end closed connection without response")
    bare = requests.exceptions.ConnectionError(urllib3.exceptions.ProtocolError("Connection aborted.", disconnected))
    bare.physical_attempt_capture = _unresolved_capture(provider="anthropic")
    fields = attempt_custody_event_fields(bare)
    assert fields["transport_cause_type"] == "RemoteDisconnected"
    assert fields["physical_attempt_id"] == "pa-death"
    assert attempt_custody_event_fields(_sdk_wrapped(bare))["transport_cause_type"] == "RemoteDisconnected"
    retried = requests.exceptions.ConnectionError(urllib3.exceptions.MaxRetryError(
        None, "/messages", reason=urllib3.exceptions.ProtocolError("Connection aborted."),
    ))
    assert attempt_custody_event_fields(retried)["transport_cause_type"] == "ProtocolError"
    # A pre-dispatch requests shape carries no typed death and stays unnamed, as before.
    refused = requests.exceptions.ConnectionError(urllib3.exceptions.MaxRetryError(
        None, "/messages", reason=urllib3.exceptions.NewConnectionError(None, "connection refused"),
    ))
    assert "transport_cause_type" not in attempt_custody_event_fields(refused)
    # The ledger terminalization path reads the same fact into its bounded reason.
    reservation = ua.reserve_attempt(_request(data_root))
    ua.mark_dispatched(reservation)
    assert ua._terminalize_failed_attempt(reservation, bare) == "unresolved"
    rows = ledger_rows(data_root)
    assert rows[-1]["state"] == "unresolved"
    assert rows[-1]["reason"].startswith("ConnectionError [cause: RemoteDisconnected]:")


def test_requests_chunked_body_disconnect_keeps_its_typed_cause():
    import http.client
    import requests
    import urllib3
    from ouroboros.transport_custody import attempt_custody_event_fields, is_pre_dispatch_transport_failure

    broken = urllib3.exceptions.ProtocolError("Connection broken", http.client.RemoteDisconnected("closed"))
    error = requests.exceptions.ChunkedEncodingError(broken)
    assert not is_pre_dispatch_transport_failure(error)
    assert attempt_custody_event_fields(error)["transport_cause_type"] == "RemoteDisconnected"
    untyped = requests.exceptions.ChunkedEncodingError("Connection broken")
    assert "transport_cause_type" not in attempt_custody_event_fields(untyped)


def test_requests_proxy_error_needs_connect_evidence_to_prove_pre_dispatch():
    import http.client

    import requests
    import urllib3
    from ouroboros.transport_custody import is_pre_dispatch_transport_failure

    tunnel_died = requests.exceptions.ProxyError(urllib3.exceptions.MaxRetryError(
        None, "/messages", reason=urllib3.exceptions.ProxyError(
            "Unable to connect to proxy", http.client.RemoteDisconnected("Remote end closed connection without response"),
        ),
    ))
    tunnel_died.physical_attempt_capture = _unresolved_capture(provider="anthropic")
    assert is_pre_dispatch_transport_failure(tunnel_died) is False
    refused = requests.exceptions.ProxyError(urllib3.exceptions.MaxRetryError(
        None, "/messages", reason=urllib3.exceptions.ProxyError(
            "Cannot connect to proxy.", urllib3.exceptions.NewConnectionError(None, "connection refused"),
        ),
    ))
    refused.physical_attempt_capture = _unresolved_capture(provider="anthropic")
    assert is_pre_dispatch_transport_failure(refused) is True
    bare = requests.exceptions.ProxyError("proxy refused the CONNECT")
    bare.physical_attempt_capture = _unresolved_capture(provider="anthropic")
    assert is_pre_dispatch_transport_failure(bare) is False


def _capture(state, status=None):
    return ua.PhysicalAttemptCapture(attempt_id="a", model="m", provider="p", state=state,
                                     candidate_measurement_kind="canonical_json_v1", provider_status_code=status)


def _failure(**facts):
    error = RuntimeError("synthetic attempt failure")
    for name, value in facts.items():
        setattr(error, name, value)
    return error


@pytest.mark.parametrize(("state", "status", "error", "known"), [
    ("released", None, _failure(), True),  # proven unsent, a typed no-start included
    ("released", None, None, True),
    ("settled", None, None, True),  # its response came back
    ("unresolved", None, None, True),  # unpriced is not unanswered
    ("dispatched", None, None, True),  # response read, accounting failed
    ("unresolved", 400, _failure(), True),  # the provider answered an HTTP error
    ("unresolved", 400, _failure(stream_incomplete=True, stream_receipt={"model_output_received": False}), True),
    ("settled", None, _failure(), True),  # a settled typed failure (an engine verdict without a status)
    ("unresolved", 400, _failure(stream_incomplete=True, stream_receipt={"model_output_received": True}), False),
    ("unresolved", None, _failure(), False),  # no answer
    ("unresolved", 504, _failure(explicit_timeout=True), False),
    ("dispatched", 400, _failure(receiver_abandoned=True), False),
    ("settled", None, _failure(code="model_outcome_unknown"), False),
    ("settled", None, _failure(stream_incomplete=True, stream_receipt={"model_output_received": True}), False),
    ("reserved", None, _failure(), False),  # its release could not be recorded
])
def test_attempt_end_known_reads_only_typed_attempt_facts(state, status, error, known):
    from ouroboros.transport_custody import attempt_end_known

    assert attempt_end_known(_capture(state, status), error) is known


def test_attempt_end_known_without_a_capture_proves_nothing():
    from ouroboros.transport_custody import attempt_end_known

    assert attempt_end_known(None, None) is False


def test_report_attempt_end_ignores_the_pre_dispatch_capture_and_an_unbound_tracker():
    from ouroboros.transport_custody import report_attempt_end, track_physical_attempts

    heard = []
    tracker = type("Tracker", (), {"attempt_ended": lambda self, capture, error: heard.append((capture.state, error))})()
    report_attempt_end(_capture("settled"), None)  # no tracker bound
    with track_physical_attempts(tracker):
        report_attempt_end(_capture("reserved"), None)
        report_attempt_end(_capture("reserved", 400), error := _failure())  # a failed release is a reported end
        report_attempt_end(_capture("settled"), None)
    assert heard == [("reserved", error), ("settled", None)]


def test_a_loopback_listener_never_proves_upstream_recovery_but_an_external_endpoint_does(monkeypatch):
    """A HEAD answer from a listener on this host says nothing of the old generation's end or of any upstream, so it
    cannot admit a new send after an unknown outcome; the same answer from an external endpoint still observes."""
    from ouroboros import llm_probe
    from ouroboros.llm import LLMClient

    seen, original = [], httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: original(
        **kw, transport=httpx.MockTransport(lambda request: seen.append(str(request.url)) or httpx.Response(200))))
    llm = LLMClient()
    for url, observed in (("http://127.0.0.1:1234/v1", False), ("http://localhost:11434/v1", False),
                          ("https://exact-provider.invalid/api", True)):
        monkeypatch.setattr(llm, "_resolve_remote_target", lambda _model, url=url: {"base_url": url})
        seen.clear()
        reached = llm_probe.upstream_transport_reachable(llm, "vendor/model", timeout=3)
        assert bool(reached) is observed and (seen == [url]) is observed  # a loopback URL is not even asked
        if observed:
            assert reached["kind"] == "upstream_http"


@pytest.mark.parametrize(("control", "status_sec"), [("deadline", 1.0), ("deadline", 5.0), ("finalize_requested", 5.0),
                                                     ("owner_pause", 5.0)],
                         ids=["inside_grace", "grace_arrives", "stop_arrives", "pause_arrives"])
def test_a_control_during_the_status_read_starts_no_result_read_and_cancels_nothing(monkeypatch, control, status_sec):
    """The status GET begins inside the grace and answers terminal-ready after it: no result GET follows, no
    cancel is posted for the already terminal operation, no create or retry is sent, and the interruption keeps
    the exact operation identity for the existing read-only recovery. Inside the grace the result is read once."""
    from ouroboros import llm_claudexor
    from ouroboros.gateways.claudexor_metadata import model_rejoin_binding

    now, calls = [0.0], []  # a controlled clock: the control (the grace at 4 s, or a Stop or Pause) is live from 4 s on
    invocation = llm_claudexor._ModelInvocation({"usage_model": "claudexor::codex=m"}, {}, {
        "timeout": 1, "model_poll_control": lambda: control if now[0] >= 4.0 else None})
    invocation.operation_id, invocation.invocation_id, invocation.create_attempted = "accepted-op", "attempt", True
    monkeypatch.setattr(llm_claudexor, "current_model_wait", lambda: SimpleNamespace(control_reason=lambda: None))

    class Gateway:
        store_identity = "fixture-store"

        def get_model_operation(self, operation, **_kwargs):
            calls.append("status")
            now[0] += status_sec  # this read began at 0 s, inside the grace
            return {"id": operation, "state": "succeeded", "dispatch": {"state": "response_received"},
                    "response": {"state": "ready", "ref": {"sha256": "exact"}}}

        def get_model_result(self, operation, **_kwargs):
            calls.append("result")
            return b'{"outcome":"completed"}'

        def cancel_model_operation(self, *_args, **_kwargs):
            pytest.fail("a cancel was posted for an operation already terminal")

        def create_model_operation(self, *_args, **_kwargs):
            pytest.fail("a second POST over an accepted operation")

        def close(self):
            pass

    gateway = invocation.gateway = Gateway()
    invocation.create_binding = model_rejoin_binding(gateway)
    if status_sec < 4.0:
        assert invocation._wait_for_result() == b'{"outcome":"completed"}'
        assert calls == ["status", "result"]
        return
    with pytest.raises(llm_claudexor.ClaudexorModelError) as ended:
        invocation._wait_for_result()
    assert ended.value.control_reason == control and ended.value.operation_id == "accepted-op"
    assert ended.value.problem["context"] == {"control_reason": control, "cancellation": "not_requested"}
    assert calls == ["status"]
