"""One private credential capture for fitting and opening a physical connection.

Only the account fingerprint enters plans or evidence. Values live in this
process-local object and are exposed through the same granted PluginAPI read.
"""
from contextlib import contextmanager
from contextvars import ContextVar

_active = ContextVar('realtime_credentials', default=None)


class SessionCredentials:
    def __init__(self, api, values, scope):
        self._api, self._values, self.scope = api, dict(values), scope

    @contextmanager
    def activate(self):
        token = _active.set(self)
        try:
            yield
        finally:
            _active.reset(token)


def session_settings(api, current):
    """Pin only the captured credential keys; PluginAPI still checks its grant."""
    captured = _active.get()
    return {**current, **captured._values} if captured is not None and captured._api is api else current


def capture_session_credentials(turn, provider=None):
    from ouroboros.contracts.realtime import canonical_json, sha256
    from ouroboros.extension_registry_state import _extensions, _lock
    from ouroboros.realtime_setup import loaded_provider

    provider = provider or loaded_provider(turn.provider)
    required = provider['catalog'].credential_settings
    api = provider.get('_credential_api')
    with _lock:
        apis = (api,) if api is not None else tuple(getattr(_extensions.get(provider['skill']), 'api_instances', ()))
    for api in apis:
        values = api.get_settings(required)
        values = {key: values.get(key) for key in required}
        scope = sha256(canonical_json({'provider': turn.route.accounting_provider, 'model': turn.model,
            'endpoint': turn.route.endpoint, 'credentials': values}))[:24] if all(values.values()) else ''
        if scope or api is provider.get('_credential_api'):
            return SessionCredentials(api, values, scope)
    return SessionCredentials(None, {}, '')
