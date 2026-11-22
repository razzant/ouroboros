"""Shared transport-death fixtures: the scripted LLM double and the first-attempt retry driver.

Moved from ``tests/test_transport_death_retry.py`` so test modules do not import each other
(Development 02, "Documentation contract").
"""
from ouroboros.loop_llm_call import call_llm_with_retry

MESSAGES = [{"role": "user", "content": "hi"}]
OK_RESPONSE = ({"role": "assistant", "content": "done"}, {"prompt_tokens": 1, "completion_tokens": 1})


class _ScriptedLLM:
    """chat() follows a script: an exception factory raises, anything else is returned."""

    def __init__(self, *script):
        self.script = list(script)
        self.calls = 0

    def default_model(self):
        return "test-model"

    def chat(self, **_kwargs):
        self.calls += 1
        step = self.script.pop(0) if self.script else OK_RESPONSE
        if callable(step):
            raise step()
        return step


def _primary_call(llm, drive_logs, usage, *, round_idx=1, max_retries=3, **kwargs):
    return call_llm_with_retry(
        llm, MESSAGES, "test-model", None, "low", max_retries, drive_logs, "t-death",
        round_idx, None, usage, **kwargs,
    )
