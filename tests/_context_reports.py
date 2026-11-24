"""Sealed model sends of invented dialogues, and Beautiful Context reports the real reader writes over them.

``tests/test_model_send_seal.py`` and the Widgets browser case (``tests/test_widgets_ui_browser.py``) import these
scenario builders from here, so a change to one reaches both by import. ``_dispatch`` looks up ``_scope`` in this
module when it runs: another module binds its own scope by patching it here (``tests/test_boot_dependency_sync.py``).
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import pathlib
from typing import Any, Dict

from ouroboros import usage_accounting as ua
from ouroboros.llm import LLMClient
from ouroboros.physical_attempt_context import PhysicalAttemptContext
from tests._usage_store_testing import ledger_rows


class _Response:
    _payload = {
        "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        "usage": {"prompt_tokens": 50, "completion_tokens": 2, "prompt_tokens_details": {"cached_tokens": 0}, "cost": 0.0},
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


def _dispatch(root: pathlib.Path, task_id: str, payload=None, target=None, *, category=None, response_factory=_Response):
    client = LLMClient(api_key="unused")
    scope = _scope(root, task_id)  # other modules bind their own scope here (tests/test_boot_dependency_sync.py)
    with ua.usage_scope(scope if category is None else dataclasses.replace(scope, category=category)):
        client._create_chat_completion_with_retries(
            lambda **_candidate: response_factory(),
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
    system = 'You answer from the tide table.'
    bible = '- Heights in metres.'

    def witness(path: str, content: str) -> tuple:
        digest = hashlib.sha256(content.encode('utf-8')).hexdigest()
        return path, digest, len(content), digest

    payload["messages"] = [
        {"role": "system", "content": [{"type": "text", "text": f'{system}\n\n## BIBLE.md\n\n{bible}'}]},
        {"role": "user", "content": "Can the Marigold berth at 14?\nShe arrives at 06:30.\n"
                                    "OPENAI_API_KEY=sk-secret-model-send-seal <b>now</b>"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "inspect", "arguments": '{"berth": 14}'}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "open\nfenders replaced"},
    ]
    fit = PhysicalAttemptContext(
        profile="task_local_low", rendered_mode="low", measurement_basis="fresh_route_usage",
        route_fp="private-fixture-route", round_id="1", capacity_total_tokens=100_000,
        target_total_tokens=None, context_target_miss=False, automatic_pass_used=True,
        preferred_mode="max", mode_cause="window", response_reserve_tokens=4096,
        prefix={"departs": "system", "block": "shared understanding", "index": 2, "of": 5},
        base_prompt_source=witness('prompts/SYSTEM.md', system), bible_source=witness('BIBLE.md', bible),
    )
    from ouroboros.context_assembly_trace import bind_contribution_input, note_contribution
    from types import SimpleNamespace

    typed = copy.deepcopy(payload)
    typed['messages'][1]['_host_context_kind'] = 'owner_words'
    slot = SimpleNamespace()
    note_contribution(slot, typed['messages'][1], 'owner_input', form='direct_input')
    fit = bind_contribution_input(fit, typed, typed['messages'], slot)
    with ua.bind_physical_attempt_context(fit):
        first = _dispatch(root, "report-task", payload=copy.deepcopy(payload))["attempt_id"]
    attempts = [first, _dispatch(root, "report-task", payload=copy.deepcopy(payload))["attempt_id"]]
    noisy = copy.deepcopy(payload)
    noisy['messages'][-1]['content'] += '\n' + 'Tide sensor background output.\n' * 5_000
    noisy_id = _dispatch(root, "report-task", payload=noisy)["attempt_id"]
    media_id = _dispatch_staged_media(root, monkeypatch, fit)
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
            "noisy": report("report-task", noisy_id), "media": report("media-task", media_id),
            "failed": (failed, report("retry-task", failed)), "retried": (retried, report("retry-task", retried))}


def _dispatch_staged_media(root, monkeypatch, fit):
    from types import SimpleNamespace

    from PIL import Image
    from ouroboros.artifacts import task_artifact_dir_path
    from ouroboros.context import build_incoming_user_content
    from ouroboros.context_assembly_trace import bind_contribution_input, note_contribution
    from ouroboros.llm_messages import project_wire_messages
    from ouroboros.vision_routing import VisionRoutingContext, prepare_messages_for_send

    task = 'media-task'
    folder = task_artifact_dir_path(root, task, create=True) / 'attachments'
    folder.mkdir(exist_ok=True)
    manifest = []
    for index in range(6):
        name = f'harbor-{index}.png'
        Image.new('RGB', (24, 24), (80 + index, 120, 180)).save(folder / name)
        manifest.append({'relpath': f'attachments/{name}', 'is_image': True})
    observed = {}
    entry = {'text': 'Where did the attachment go?', 'attachment_manifest': manifest}
    row = {'role': 'user', 'content': build_incoming_user_content(entry, root, task, attachment_facts=observed)}
    assert observed['manifest_pixels'] == 5 and observed['manifest_path_only'] == 0
    assert observed['attachment_items'][-1] == {'number': 6, 'decision': 'image_count_limit'}
    with monkeypatch.context() as m:
        m.setenv('OUROBOROS_IMAGE_INPUT_MODE', 'off')
        projected = prepare_messages_for_send([row], routing=VisionRoutingContext('not/vision', object(), {}))
    slot = SimpleNamespace()
    note_contribution(slot, row, 'attachment_manifest', observation=observed)
    logical = {**_payload(), 'messages': projected}

    class MixedUsage(_Response):
        # The ledger's totals include non-text traffic; the cache subtraction is not a text subtotal.
        _payload = {**_Response._payload, 'usage': {
            'prompt_tokens': 1000, 'completion_tokens': 100, 'cost': 0.0,
            'prompt_tokens_details': {'audio_tokens': 5, 'cached_tokens': 100},
            'completion_tokens_details': {'audio_tokens': 8},
        }}

    with ua.bind_physical_attempt_context(bind_contribution_input(fit, logical, [row], slot)):
        return _dispatch(root, task, payload={**logical, 'messages': project_wire_messages(projected)},
                         response_factory=MixedUsage)['attempt_id']
