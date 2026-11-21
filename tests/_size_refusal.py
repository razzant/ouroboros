"""Build a real SDK exception from a fixture body without network traffic."""
import httpx
import openai
import pytest


def _sdk_error(status: int, body_bytes: bytes) -> Exception:
    transport = httpx.MockTransport(lambda request: httpx.Response(
        status, content=body_bytes, headers={"content-type": "application/json"}))
    client = openai.OpenAI(api_key="x", base_url="http://127.0.0.1:9/v1", max_retries=0,
                           http_client=httpx.Client(transport=transport))
    with pytest.raises(openai.APIStatusError) as caught:
        client.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])
    return caught.value
