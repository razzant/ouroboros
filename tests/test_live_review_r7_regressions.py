"""Regressions from review round R7 (astra, 22.10): synthetic transports and a disposable filesystem only."""
import errno
from pathlib import Path

import pytest

from tests._live_world import host_world as host_world
from tests.test_live_conversation_config import config_api as config_api, request


@pytest.mark.parametrize("file_state", ["readable", "malformed", "permission"])
def test_owner_can_disable_when_old_settings_cannot_be_read(config_api, tmp_path, monkeypatch, file_state):
    from skills.live_conversation import plugin

    api, _, _, _ = config_api
    path = tmp_path / "feature/conversation.json"
    fields = {"provider": "fixture-engine:voice", "model": "fixture-model"}
    assert request(api, {**fields, "enabled": True})[0] == 200
    ended, refused = [], []
    monkeypatch.setattr(plugin, "end_feature_calls", ended.append)
    if file_state == "malformed":
        path.write_text("{incomplete", encoding="utf-8")
    elif file_state == "permission":
        # A chmod cannot make a file unreadable on Windows (nor for root): refuse exactly this file's read.
        read_text = Path.read_text
        def unreadable(self, *args, **kwargs):
            if self == path:
                refused.append(self)
                raise PermissionError(errno.EACCES, "Permission denied", str(self))
            return read_text(self, *args, **kwargs)
        monkeypatch.setattr(Path, "read_text", unreadable)
    status, result = request(api, {**fields, "enabled": False})
    assert status == 200 and result["enabled"] is False
    assert ended == ["live_conversation:call"]
    assert bool(refused) == (file_state == "permission"), "the endpoint itself met the unreadable file"


def test_openai_measured_window_reaches_the_ordinary_presend_check(tmp_path):
    from dataclasses import replace
    from ouroboros.capability_evidence import record_route_facts
    from ouroboros.llm_attempt import main_context_fit_refusal
    from ouroboros.usage_accounting import AttemptRequest, PhysicalAttemptContext
    from skills.openai_realtime.plugin import CATALOG, ROUTE

    model = CATALOG.model("")
    evidence = record_route_facts(tmp_path, provider=ROUTE.accounting_provider,
        model=ROUTE.model_prefix + model.id, base_url=ROUTE.endpoint, facts=model.facts)
    physical = PhysicalAttemptContext("owner_nano", "nano", "cold_estimate", evidence.route_fp, "round",
        None, evidence.window_tokens, False, False, 1.0)
    request = AttemptRequest(ROUTE.model_prefix + model.id, ROUTE.accounting_provider,
        max_completion_tokens=32000, prompt_tokens_bounded_estimate=120000,
        physical_context=physical, working_reserve_tokens=model.facts.working_reserve_tokens)
    assert main_context_fit_refusal(request) is None
    assert main_context_fit_refusal(replace(request,
        prompt_tokens_bounded_estimate=evidence.window_tokens - model.facts.working_reserve_tokens + 1)) is not None
