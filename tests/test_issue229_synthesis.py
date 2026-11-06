"""Cross-stream regressions for the issue #229 synthesis driver."""

from __future__ import annotations

import copy


import ouroboros.llm_observability as llm_observability
from ouroboros.llm import add_usage
from ouroboros.openai_chat_dispatch import CUSTOM_RECEIPTS_USAGE_KEY


def test_ordered_request_wire_history_survives_main_and_nested_usage_totals(
    tmp_path, monkeypatch,
):
    first = {
        "attempt_id": "attempt-main",
        "candidate_sha256": "a" * 64,
        "requested_effort": "medium",
        "applied_effort": "medium",
    }
    second = {
        "attempt_id": "attempt-compaction",
        "candidate_sha256": "b" * 64,
        "requested_effort": "medium",
        "applied_effort": "medium",
    }
    usage_first = {
        "request_wire": first,
        CUSTOM_RECEIPTS_USAGE_KEY: (object(),),
    }
    usage_second = {"request_wire": second}

    persisted = []
    monkeypatch.setattr(
        llm_observability,
        "persist_call",
        lambda *args, **kwargs: persisted.append(copy.deepcopy(kwargs)),
    )

    class _FakeLLM:
        def chat(self, **kwargs):
            del kwargs
            return {"role": "assistant", "content": "ok"}, usage_first

    _message, returned = llm_observability.chat_observed(
        _FakeLLM(),
        drive_root=tmp_path,
        task_id="synthesis",
        model="openai::future-reasoning-model",
    )
    response_payload = next(
        item["payload"] for item in persisted
        if item["call_type"].endswith("_response")
    )
    assert CUSTOM_RECEIPTS_USAGE_KEY not in response_payload["usage"]
    assert CUSTOM_RECEIPTS_USAGE_KEY in returned

    main_total = {}
    add_usage(main_total, usage_first)
    nested_total = {}  # a second request's own accumulator (once the helper summarizer's), merged twice
    add_usage(nested_total, usage_second)
    add_usage(main_total, nested_total)
    add_usage(main_total, nested_total)

    assert main_total["request_wire"] == second
    assert main_total["request_wire_history"] == [first, second]
