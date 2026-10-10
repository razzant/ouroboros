"""Trusted live full-registry provider-contract canaries.

The integration marker is excluded from ordinary pull-request pytest.  These
tests run only in the trusted target-push/manual/tag lane, where core provider
credentials are mandatory and optional direct-provider credentials remain loud
skips.  Every first turn sends the same complete shipped built-in catalog, fitted
to the route's tool-schema ceiling as Main does (direct OpenAI: 128), and
validates a normalized ``delegate_start`` call without executing it.
"""

from __future__ import annotations

import uuid

import pytest

from tests.provider_contract_ci import (
    _emit_canary_response_warnings,
    full_registry_canary_tools,
    provider_canary_matrix,
    require_provider_canary_credential,
    run_provider_contract_canary,
    skip_on_provider_environmental_error,
)

integration = pytest.mark.integration
_PROVIDER_CANARIES = provider_canary_matrix()


def _get_llm_client():
    """Lazy import keeps secretless collection independent of runtime clients."""
    from ouroboros.llm import LLMClient

    return LLMClient()


@pytest.fixture(scope="module")
def shipped_builtin_tools():
    return full_registry_canary_tools()


@integration
@pytest.mark.parametrize(
    "canary",
    _PROVIDER_CANARIES,
    ids=[canary.canary_id for canary in _PROVIDER_CANARIES],
)
def test_full_registry_provider_contract(canary, shipped_builtin_tools, request):
    """Exercise one bounded public-chat canary per physical provider surface."""
    from tests.provider_contract_diagnostics import provider_evidence

    with provider_evidence(request, canary) as observer:
        require_provider_canary_credential(canary)
        try:
            _message, usage, _final_message, _final_usage = run_provider_contract_canary(
                _get_llm_client(),
                canary=canary,
                tools=shipped_builtin_tools,
                nonce=uuid.uuid4().hex,
                observer=observer,
            )
            _emit_canary_response_warnings(usage)
        except Exception as exc:  # noqa: BLE001
            skip_on_provider_environmental_error(canary.canary_id, exc)
            raise
