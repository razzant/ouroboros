"""Regression test for PR-B: retry backoff (#15).

(#14 — not billing provider-glitch empties — was moved to CONSULT-BUGS.md: doing
it correctly requires deciding whether the durable usage SSOT in events.jsonl
should exclude finish_reason=null responses, a provider-billing semantics call
left to the maintainer.)
"""

from __future__ import annotations


def test_backoff_doubled_with_cap():
    """Backoff stays exponential (x4 base) and per-class capped: 30s for
    generic retryable errors, the transient cap for provider classes."""
    from ouroboros.loop_llm_call import _TRANSIENT_BACKOFF_CAP_SEC, _retry_backoff_sec

    assert _retry_backoff_sec(0, False) == 4.0
    assert _retry_backoff_sec(1, False) == 8.0
    assert _retry_backoff_sec(2, False) == 16.0
    assert _retry_backoff_sec(3, False) == 30.0
    assert _retry_backoff_sec(4, False) == 30.0
    assert _retry_backoff_sec(0, True) == min(4.0, _TRANSIENT_BACKOFF_CAP_SEC)
    assert _retry_backoff_sec(10, True) == float(_TRANSIENT_BACKOFF_CAP_SEC)
