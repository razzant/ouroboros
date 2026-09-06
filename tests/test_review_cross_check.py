"""Preserve the contributor's disputed-finding examples without a host judge.

The original PR cited ouroborosproject_naming and Claudexor.unavailable.
Identifier absence alone cannot decide whether a missing-import finding is
true. Canonicalization keeps each judgment intact for the existing rebuttal
and review flow, on schema, strict, and extracted deliveries alike.
"""

import json
from types import SimpleNamespace

import pytest

from ouroboros.review_verdict_extraction import canonicalize_session_verdict


@pytest.mark.parametrize("reason", [
    "tests/test_attachment_staging.py imports `ouroborosproject_naming` which does not exist.",
    "claudexor_daemon.py:1234 uses `Claudexor.unavailable` — typo.",
    "module `ouroboros.config` is missing required setup.",
])
@pytest.mark.parametrize("method", ["schema", "strict", "light_model_extraction"])
def test_disputed_findings_keep_the_reviewers_verdict_and_severity(reason, method):
    findings = [{"item": "missing import", "verdict": "FAIL",
                 "severity": "critical", "reason": reason}]
    calls = []

    def chat(**kwargs):
        calls.append(kwargs)
        return {"content": json.dumps(findings)}, {"cost": 0.001}

    raw = (json.dumps({"findings": findings}) if method == "schema" else
           json.dumps(findings) if method == "strict" else "Review completed. " + reason)
    text, actual_method, usage = canonicalize_session_verdict(
        raw, conformance_passed=method == "schema", llm=SimpleNamespace(chat=chat),
    )
    assert actual_method == method
    assert json.loads(text) == findings
    assert "cross_check" not in usage
    assert len(calls) == (1 if method == "light_model_extraction" else 0)
    if method == "strict":
        assert text == raw
