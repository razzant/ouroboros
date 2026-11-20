"""Beautiful Context reports of one invented dialogue, written by the real reader over real sealed attempts.

The Widgets browser case (``tests/test_widgets_ui_browser.py``) opens them through the picker a person uses.
"""
from __future__ import annotations

import copy
import pathlib


def context_reports(root: pathlib.Path, monkeypatch, capsys) -> dict:
    """Two attempts of one body, a report of a gap, and a send the provider refused whose retry was answered.

    The refused send is then settled as its task's end settles it: without a receipt, its price unknown.
    Each value is the exact bytes ``--report`` wrote; ``failed`` and ``retried`` pair an attempt id with its report.
    """
    from ouroboros import usage_accounting as ua
    from ouroboros.context_attempt_reader import main
    from tests.test_model_send_seal import _dispatch, _payload, _rejected_then_answered, _settle_abandoned

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

    def report(task: str, attempt: str, *extra: str) -> bytes:
        capsys.readouterr()
        assert main(["--data-root", str(root), task, "--attempt", attempt, "--report", *extra]) == 0
        return capsys.readouterr().out.encode("utf-8")

    return {"attempts": attempts, "first": report("report-task", attempts[0]),
            "second": report("report-task", attempts[1]),
            "over_limit": report("report-task", attempts[0], "--max-bytes", "1"),
            "failed": (failed, report("retry-task", failed)), "retried": (retried, report("retry-task", retried))}
