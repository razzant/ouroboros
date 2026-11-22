"""Sealed model sends of invented dialogues, and Beautiful Context reports the real reader writes over them.

``tests/test_model_send_seal.py`` and the Widgets browser case (``tests/test_widgets_ui_browser.py``) import these
scenario builders from here, so a change to one reaches both by import. ``_dispatch`` looks up ``_scope`` in this
module when it runs: another module binds its own scope by patching it here (``tests/test_boot_dependency_sync.py``).
"""
from __future__ import annotations

import copy
import dataclasses
import pathlib
from typing import Any, Dict

from ouroboros import usage_accounting as ua
from ouroboros.llm import LLMClient
from tests._usage_store_testing import ledger_rows


class _Response:
    _payload = {
        "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        "usage": {"prompt_tokens": 50, "completion_tokens": 2, "cost": 0.0},
    }

    def model_dump(self):
        return copy.deepcopy(self._payload)


def _scope(root: pathlib.Path, task_id: str) -> ua.UsageScope:
    return ua.UsageScope(
        drive_root=root, task_id=task_id, root_task_id=task_id,
        category="task", source="test.model_send",
    )


def _target() -> Dict[str, Any]:
    return {"provider": "openai", "usage_model": "openai/gpt-x", "resolved_model": "gpt-x"}


def _payload() -> Dict[str, Any]:
    return {
        "model": "gpt-x",
        "messages": [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": "OPENAI_API_KEY=sk-secret-model-send-seal summarize this",
            }],
        }],
        "tools": [{
            "type": "function",
            "function": {"name": "inspect", "parameters": {"type": "object"}},
        }],
        "max_tokens": 32,
    }


def _rows(root: pathlib.Path):
    return ledger_rows(root)


def _dispatch(root: pathlib.Path, task_id: str, payload=None, target=None, *, category=None):
    client = LLMClient(api_key="unused")
    scope = _scope(root, task_id)  # other modules bind their own scope here (tests/test_boot_dependency_sync.py)
    with ua.usage_scope(scope if category is None else dataclasses.replace(scope, category=category)):
        client._create_chat_completion_with_retries(
            lambda **_candidate: _Response(),
            payload if payload is not None else _payload(),
            target if target is not None else _target(),
        )
    return _rows(root)[-1]


def _rejected_then_answered(root: pathlib.Path, task_id: str, monkeypatch, payload=None):
    """One call whose first send the provider rejects and whose retry is answered: the two attempt ids, failed first."""
    from ouroboros.context_attempt_reader import list_attempts

    client = LLMClient(api_key="unused")

    class _Rejected(RuntimeError):
        status_code = 400
        body = {"error": {"code": "unsupported_parameter", "type": "invalid_request_error"}}

    replies = [_Rejected("temperature unsupported: private words"), _Response()]

    def create(**_candidate):
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(client, "_openrouter_signature_retry_kwargs", lambda *args: None)
    with ua.usage_scope(_scope(root, task_id)):
        client._create_chat_completion_with_retries(
            create, {**(payload if payload is not None else _payload()), "temperature": 0.7}, _target())
    answered, failed = (row["attempt_id"] for row in list_attempts(root, task_id)["attempts"][:2])
    return failed, answered


def _settle_abandoned(root: pathlib.Path, attempt_id: str) -> Dict[str, Any]:
    """Close an open attempt as its task's end does: settled with no receipt, its price unknown."""
    row = next(row for row in _rows(root) if row["attempt_id"] == attempt_id)
    reservation = ua.AttemptReservation(attempt_id, root, row["model"], row["provider"],
                                        row.get("reservation_upper_bound_usd"))
    assert ua.terminalize_abandoned_attempt(reservation, reason="owner_task_terminal") == "settled"
    return next(row for row in _rows(root) if row["attempt_id"] == attempt_id)


def context_reports(root: pathlib.Path, monkeypatch, capsys) -> dict:
    """Two attempts of one body, a report of a gap, and a send the provider refused whose retry was answered.

    The refused send is then settled as its task's end settles it: without a receipt, its price unknown.
    Each value is the exact bytes ``--report`` wrote; ``failed`` and ``retried`` pair an attempt id with its report.
    """
    from ouroboros import context_attempt_reader as reader

    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(root / "settings.json"))
    monkeypatch.setenv("TOTAL_BUDGET", "100")
    monkeypatch.setattr(ua, "estimate_cost_optional", lambda *args, **kwargs: 0.01)
    (root / "state").mkdir(parents=True)
    payload = _payload()
    payload["messages"] = [
        {"role": "system", "content": [{"type": "text", "text": "You answer from the tide table.\n\n## Rules\n\n- Heights in metres."}]},
        {"role": "user", "content": "Can the Marigold berth at 14?\nShe arrives at 06:30.\n"
                                    "OPENAI_API_KEY=sk-secret-model-send-seal <b>now</b>"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "inspect", "arguments": '{"berth": 14}'}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "open\nfenders replaced"},
    ]
    attempts = [_dispatch(root, "report-task", payload=copy.deepcopy(payload))["attempt_id"] for _ in range(2)]
    failed, retried = _rejected_then_answered(root, "retry-task", monkeypatch, payload=copy.deepcopy(payload))
    _settle_abandoned(root, failed)

    def report(task: str, attempt: str) -> bytes:
        capsys.readouterr()
        assert reader.main(["--data-root", str(root), task, "--attempt", attempt, "--report"]) == 0
        return capsys.readouterr().out.encode("utf-8")

    with monkeypatch.context() as bound:  # --max-bytes only raises the bound: a gap needs a smaller default
        bound.setattr(reader, "_BODY_LIMIT", 1)
        over_limit = report("report-task", attempts[0])
    return {"attempts": attempts, "first": report("report-task", attempts[0]),
            "second": report("report-task", attempts[1]), "over_limit": over_limit,
            "failed": (failed, report("retry-task", failed)), "retried": (retried, report("retry-task", retried))}
