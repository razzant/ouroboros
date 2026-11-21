"""Native Ollama contracts for llm_fallback."""
import asyncio
import pytest
from tests._ollama import (
    TARGET, MESSAGES, LlamaCppHTTP, install_http as install_http,
)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_failed_compatible_send_forgets_the_cached_verdict(install_http, asynchronous):
    from ouroboros import llm_ollama
    from ouroboros.llm import LLMClient

    target = {**TARGET, "base_url": "http://127.0.0.1:11463/v1"}
    install_http(LlamaCppHTTP())
    assert not llm_ollama.is_ollama(target)
    assert llm_ollama._rejected_now(llm_ollama._identity(target))
    request = {"model": "mock-model", "messages": MESSAGES}

    def refuse(**kwargs):
        raise RuntimeError("connection refused by the program now listening here")

    async def refuse_async(**kwargs):
        refuse()

    with pytest.raises(RuntimeError):
        if asynchronous:
            asyncio.run(LLMClient()._create_chat_completion_with_retries_async(refuse_async, request, target))
        else:
            LLMClient()._create_chat_completion_with_retries(refuse, request, target)
    assert not llm_ollama._rejected_now(llm_ollama._identity(target))
