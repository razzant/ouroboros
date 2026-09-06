from __future__ import annotations

import pytest
import httpx

from ouroboros import usage_accounting as ua


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
    ("http://127.0.0.1:1234/v1", True),
    ("http://[::1]:8000/v1", True),
    ("http://127.0.0.2:11434/v1", True),  # the whole 127.0.0.0/8 range is this host
    ("http://127.255.255.254:1/v1", True),
    ("http://[::ffff:127.0.0.1]:8000/v1", True),  # IPv4-mapped IPv6 loopback
    ("http://10.0.0.5:8000/v1", False),
    ("http://example.com/v1", False),
    ("http://localhost.example/v1", False),  # a name is loopback only when it IS localhost
    ("https://openrouter.ai/api/v1", False),
    ("https://api.anthropic.com/v1", False),
    ("", False),
    ("not a url", False),
    # every inet_aton spelling the OS accepts for a local server, and a trailing-dot name
    ("http://127.1:8000/v1", True),
    ("http://127.0.1:8000/v1", True),
    ("http://2130706433:8000/v1", True),
    ("http://0x7f000001:8000/v1", True),
    ("http://0177.0.0.1:8000/v1", True),
    ("http://localhost.:11434/v1", True),
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


@pytest.mark.parametrize("cause_cls", [httpx.ReadError, httpx.WriteError, httpx.RemoteProtocolError])
def test_typed_transport_death_on_dispatched_remote_route_is_retryable(cause_cls):
    from ouroboros.transport_custody import is_retryable_transport_death

    exc = _sdk_wrapped(cause_cls("socket died after dispatch"), _unresolved_capture())
    assert is_retryable_transport_death(exc) is True


@pytest.mark.parametrize("cause_cls", [
    httpx.ReadTimeout, httpx.WriteTimeout, httpx.ConnectError, httpx.ConnectTimeout,
    httpx.PoolTimeout,
])
def test_timeouts_and_pre_dispatch_failures_are_not_transport_deaths(cause_cls):
    """A timeout is "we gave up" (the provider may still be working); a connect
    failure is the free released class — neither earns a paid repeat."""
    from ouroboros.transport_custody import is_retryable_transport_death

    exc = _sdk_wrapped(cause_cls("not a death"), _unresolved_capture())
    assert is_retryable_transport_death(exc) is False


def test_httpx_proxy_error_is_pre_dispatch_never_a_death():
    """A tunnel/CONNECT failure is the free released class even when it rides a
    dispatched-looking capture: the pre-dispatch predicate is evaluated first,
    so the two predicates can never both be true."""
    from ouroboros.transport_custody import is_pre_dispatch_transport_failure, is_retryable_transport_death

    exc = _sdk_wrapped(httpx.ProxyError("CONNECT tunnel failed"), _unresolved_capture())
    assert is_pre_dispatch_transport_failure(exc) is True
    assert is_retryable_transport_death(exc) is False


def test_provider_status_error_is_not_a_transport_death():
    from ouroboros.transport_custody import is_retryable_transport_death

    exc = RuntimeError("HTTP 503 upstream unavailable")
    exc.status_code = 503
    exc.physical_attempt_capture = _unresolved_capture(provider_status_code=503)
    assert is_retryable_transport_death(exc) is False


def test_implicit_context_chain_never_proves_a_transport_death():
    """Only an explicit ``raise ... from`` carries transport provenance."""
    from ouroboros.transport_custody import is_retryable_transport_death

    try:
        raise httpx.ReadError("earlier leg died")
    except httpx.ReadError:
        try:
            raise RuntimeError("later wrapper without a cause")
        except RuntimeError as exc:
            exc.physical_attempt_capture = _unresolved_capture()
            assert exc.__context__ is not None
            assert is_retryable_transport_death(exc) is False


@pytest.mark.parametrize("capture", [
    _unresolved_capture(provider="local"),
    _unresolved_capture(provider="openai-compatible", route_is_loopback=True),
    ua.PhysicalAttemptCapture(
        attempt_id="pa-rel", model="m", provider="openrouter", state="released",
        candidate_measurement_kind="opaque",
    ),
    None,
])
def test_local_loopback_released_or_captureless_death_is_not_retryable(capture):
    """The classifier's locality gate: a dead local/loopback server is not a
    network fault worth paying for again; released custody is the free
    pre-dispatch class, never a paid repeat; a missing capture fails closed."""
    from ouroboros.transport_custody import is_retryable_transport_death

    exc = _sdk_wrapped(httpx.ReadError("socket died"), capture)
    assert is_retryable_transport_death(exc) is False


def test_requests_protocol_error_with_remote_disconnected_is_a_transport_death():
    """The Anthropic-native requests/urllib3 shape of a mid-request socket death."""
    import http.client

    import requests
    import urllib3
    from ouroboros.transport_custody import is_pre_dispatch_transport_failure, is_retryable_transport_death

    disconnected = http.client.RemoteDisconnected("Remote end closed connection without response")
    exc = requests.exceptions.ConnectionError(
        urllib3.exceptions.ProtocolError("Connection aborted.", disconnected)
    )
    exc.physical_attempt_capture = _unresolved_capture(provider="anthropic")
    assert is_retryable_transport_death(exc) is True
    assert is_pre_dispatch_transport_failure(exc) is False  # the two predicates are exclusive
    # The same fact through an explicit wrapper (the recovery ladder re-raises with a cause).
    assert is_retryable_transport_death(_sdk_wrapped(exc, _unresolved_capture(provider="anthropic"))) is True
    # urllib3 may hand the same fact over as MaxRetryError(reason=ProtocolError).
    retried = requests.exceptions.ConnectionError(urllib3.exceptions.MaxRetryError(
        None, "/messages", reason=urllib3.exceptions.ProtocolError("Connection aborted.", disconnected),
    ))
    retried.physical_attempt_capture = _unresolved_capture(provider="anthropic")
    assert is_retryable_transport_death(retried) is True


def test_requests_chunked_body_disconnect_is_a_transport_death():
    """The lane's OTHER wrapper. The Anthropic-native POST is non-streaming, so
    the body is read inside ``requests.post``: a socket that dies mid-BODY
    surfaces as ``ChunkedEncodingError(ProtocolError(RemoteDisconnected))``,
    which does NOT subclass ``ConnectionError``. It is the same post-dispatch
    death and earns the same bounded repeat; without a typed transport cause the
    wrapper proves nothing and keeps the base no-resend terminal."""
    import http.client

    import requests
    import urllib3
    from ouroboros.transport_custody import (
        attempt_custody_event_fields,
        is_pre_dispatch_transport_failure,
        is_retryable_transport_death,
    )

    assert not issubclass(
        requests.exceptions.ChunkedEncodingError, requests.exceptions.ConnectionError,
    )  # why this class has to be named explicitly
    disconnected = http.client.RemoteDisconnected("Remote end closed connection without response")
    exc = requests.exceptions.ChunkedEncodingError(
        urllib3.exceptions.ProtocolError("Connection broken: IncompleteRead(0 bytes read)", disconnected)
    )
    exc.physical_attempt_capture = _unresolved_capture(provider="anthropic")
    assert is_retryable_transport_death(exc) is True
    assert is_pre_dispatch_transport_failure(exc) is False  # the two predicates stay exclusive
    assert attempt_custody_event_fields(exc)["transport_cause_type"] == "RemoteDisconnected"
    # The same fact through an explicit wrapper (the recovery ladder re-raises with a cause).
    assert is_retryable_transport_death(_sdk_wrapped(exc, _unresolved_capture(provider="anthropic"))) is True
    # A body failure with no typed transport cause is not a death.
    untyped = requests.exceptions.ChunkedEncodingError("Connection broken: IncompleteRead(0 bytes read)")
    untyped.physical_attempt_capture = _unresolved_capture(provider="anthropic")
    assert is_retryable_transport_death(untyped) is False
    assert "transport_cause_type" not in attempt_custody_event_fields(untyped)


def test_requests_lane_event_fields_name_the_innermost_typed_cause(data_root):
    """The durable row and the ledger's bounded cause text see the requests
    lane's typed death, which lives in ``args``/``reason``, not ``__cause__``:
    RemoteDisconnected over its ProtocolError wrapper; ProtocolError alone when
    urllib3 hands it over as MaxRetryError.reason; wrappers keep it."""
    import http.client
    import json

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
    rows = [json.loads(line) for line in (data_root / ua.LEDGER_REL).read_text().splitlines() if line.strip()]
    assert rows[-1]["state"] == "unresolved"
    assert rows[-1]["reason"].startswith("ConnectionError [cause: RemoteDisconnected]:")


def test_requests_read_timeout_and_connect_shapes_are_not_transport_deaths():
    import requests
    import urllib3
    from ouroboros.transport_custody import is_retryable_transport_death

    capture = _unresolved_capture(provider="anthropic")
    timeout = requests.exceptions.ReadTimeout("read timed out")
    timeout.physical_attempt_capture = capture
    assert is_retryable_transport_death(timeout) is False
    connect_timeout = requests.exceptions.ConnectTimeout("connect timed out")
    connect_timeout.physical_attempt_capture = capture
    assert is_retryable_transport_death(connect_timeout) is False
    read_timeout_wrapped = requests.exceptions.ConnectionError(
        urllib3.exceptions.MaxRetryError(
            None, "/messages", reason=urllib3.exceptions.ReadTimeoutError(None, "/messages", "read timed out"),
        )
    )
    read_timeout_wrapped.physical_attempt_capture = capture
    assert is_retryable_transport_death(read_timeout_wrapped) is False
    refused = requests.exceptions.ConnectionError(
        urllib3.exceptions.MaxRetryError(
            None, "/messages", reason=urllib3.exceptions.NewConnectionError(None, "connection refused"),
        )
    )
    refused.physical_attempt_capture = capture
    assert is_retryable_transport_death(refused) is False


def test_requests_proxy_tunnel_death_is_neither_pre_dispatch_nor_a_death():
    """The tunnel to the proxy died (RemoteDisconnected nested in a urllib3
    ProxyError): not the base pre-dispatch class (no connect-time evidence —
    a proxy that ANSWERS is not an outage to wait out), and never a paid
    repeat either — the round keeps the base unknown no-resend terminal. With
    connect-time evidence the base pre-dispatch contract still holds, and a
    proxy failure is still not a death."""
    import http.client

    import requests
    import urllib3
    from ouroboros.transport_custody import is_pre_dispatch_transport_failure, is_retryable_transport_death

    tunnel_died = requests.exceptions.ProxyError(urllib3.exceptions.MaxRetryError(
        None, "/messages", reason=urllib3.exceptions.ProxyError(
            "Unable to connect to proxy", http.client.RemoteDisconnected("Remote end closed connection without response"),
        ),
    ))
    tunnel_died.physical_attempt_capture = _unresolved_capture(provider="anthropic")
    assert is_pre_dispatch_transport_failure(tunnel_died) is False
    assert is_retryable_transport_death(tunnel_died) is False
    refused = requests.exceptions.ProxyError(urllib3.exceptions.MaxRetryError(
        None, "/messages", reason=urllib3.exceptions.ProxyError(
            "Cannot connect to proxy.", urllib3.exceptions.NewConnectionError(None, "connection refused"),
        ),
    ))
    refused.physical_attempt_capture = _unresolved_capture(provider="anthropic")
    assert is_pre_dispatch_transport_failure(refused) is True
    assert is_retryable_transport_death(refused) is False
    bare = requests.exceptions.ProxyError("proxy refused the CONNECT")
    bare.physical_attempt_capture = _unresolved_capture(provider="anthropic")
    assert is_pre_dispatch_transport_failure(bare) is False
    assert is_retryable_transport_death(bare) is False
