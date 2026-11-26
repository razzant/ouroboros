"""The interface-language envelopes are a Python contract with one writer.

Wire shapes live in ``ouroboros/gateway/ui_i18n_contracts.py`` without a
browser JSDoc twin. The generic settings writer refuses the language key;
the dedicated endpoint is its only writer.
"""
from __future__ import annotations


def test_the_language_key_has_one_writer_and_still_reaches_every_process():
    """The generic settings save skips the language key (its endpoint is the one writer) — and the
    key is still projected into the environment, where every process reads it after a restart."""
    from ouroboros.config import ENDPOINT_WRITTEN_SETTINGS, ENDPOINT_WRITERS, apply_settings_to_env, settings_env_keys

    assert "OUROBOROS_UI_LANGUAGE" in ENDPOINT_WRITTEN_SETTINGS
    assert ENDPOINT_WRITERS["OUROBOROS_UI_LANGUAGE"] == "POST /api/ui/i18n/language"
    from ouroboros.gateway import settings as gateway_settings

    merged = gateway_settings._merge_settings_payload({"OUROBOROS_UI_LANGUAGE": "en"}, {"OUROBOROS_UI_LANGUAGE": "de"})  # noqa: SLF001
    assert merged["OUROBOROS_UI_LANGUAGE"] == "en", "a generic save cannot move the language"
    # The save answer names the writer for the key it skipped (the handler builds `ignored_keys`
    # from the same set), so a CLI `settings set` learns where the key is written. Pinned on the
    # source: a request-level test of the settings POST writes the process-wide settings path.
    import inspect

    handler_source = inspect.getsource(gateway_settings)
    assert 'resp["ignored_keys"] = {k: _ENDPOINT_WRITERS.get(k' in handler_source
    assert "sorted(k for k in body if k in _ENDPOINT_WRITTEN_SETTINGS)" in handler_source
    # Restart-shaped: the saved choice is projected into a fresh environment.
    assert "OUROBOROS_UI_LANGUAGE" in settings_env_keys()
    env: dict = {}
    apply_settings_to_env({"OUROBOROS_UI_LANGUAGE": "ru"}, environ=env)
    assert env.get("OUROBOROS_UI_LANGUAGE") == "ru"


def test_the_server_lifespan_is_an_async_context_manager():
    """The generator's boot hook once pulled the lifespan decorator onto a helper; Starlette
    then ran a deprecated async-generator lifespan. The decorator stays on `lifespan`."""
    import inspect

    import server

    assert not inspect.isasyncgenfunction(server.lifespan)
    wrapped = getattr(server.lifespan, "__wrapped__", None)
    assert wrapped is not None and inspect.isasyncgenfunction(wrapped), "lifespan must be @asynccontextmanager-wrapped"
