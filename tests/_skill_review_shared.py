"""Reviewer-array builders, skill builders and the context factory shared by the skill-review suites.

Split out of ``tests/test_skill_review.py`` when that module was divided by theme; every
definition is verbatim, so each sibling suite keeps the exact checklist, actor shape and
payload it was written against.
"""

from __future__ import annotations

import json
import pathlib
from unittest.mock import patch

from ouroboros.tools.registry import ToolContext


_NEW_SKILL_REVIEW_PASS_ITEMS = [
    {"item": "inject_chat_minimization", "verdict": "PASS", "severity": "critical", "reason": "Not applicable"},
    {"item": "event_subscription_minimization", "verdict": "PASS", "severity": "critical", "reason": "Not applicable"},
    {"item": "companion_process_safety", "verdict": "PASS", "severity": "critical", "reason": "Not applicable"},
    {"item": "host_token_handling", "verdict": "PASS", "severity": "critical", "reason": "Not applicable"},
    {"item": "error_handling", "verdict": "PASS", "severity": "advisory", "reason": "ok"},
    {"item": "integration_preflight", "verdict": "PASS", "severity": "advisory", "reason": "ok"},
    {"item": "bug_hunting", "verdict": "PASS", "severity": "critical", "reason": "ok"},
    {"item": "completion_notification", "verdict": "PASS", "severity": "advisory", "reason": "Not applicable"},
]


def _pass_array_for_script_skill() -> str:
    """Return a JSON array that PASSes every applicable skill checklist item."""
    return json.dumps(
        [
            {"item": "manifest_schema", "verdict": "PASS", "severity": "critical", "reason": "ok"},
            {"item": "permissions_honesty", "verdict": "PASS", "severity": "critical", "reason": "ok"},
            {"item": "no_repo_mutation", "verdict": "PASS", "severity": "critical", "reason": "ok"},
            {"item": "path_confinement", "verdict": "PASS", "severity": "critical", "reason": "ok"},
            {"item": "env_allowlist", "verdict": "PASS", "severity": "critical", "reason": "ok"},
            {"item": "timeout_and_output_discipline", "verdict": "PASS", "severity": "advisory", "reason": "ok"},
            {
                "item": "extension_namespace_discipline",
                "verdict": "PASS",
                "severity": "critical",
                "reason": "Not applicable — type != extension",
            },
            {
                "item": "widget_module_safety",
                "verdict": "PASS",
                "severity": "critical",
                "reason": "Not applicable — no module widget",
            },
            *_NEW_SKILL_REVIEW_PASS_ITEMS,
        ]
    )


def _make_actor(model: str, text: str) -> dict:
    """Mimic the flattened actor shape produced by _parse_model_response."""
    return {
        "model": model,
        "request_model": model,
        "provider": "openrouter",
        "verdict": "REVIEW",
        "text": text,
        "tokens_in": 100,
        "tokens_out": 50,
    }


def _build_skill(
    tmp_path: pathlib.Path,
    *,
    name: str = "weather",
    env_from_settings: list[str] | None = None,
) -> pathlib.Path:
    skills_root = tmp_path / "skills"
    skill_dir = skills_root / name
    (skill_dir / "scripts").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        (
            "---\n"
            f"name: {name}\n"
            "description: Check the weather.\n"
            "version: 0.1.0\n"
            "type: script\n"
            "runtime: python3\n"
            "timeout_sec: 30\n"
            + (
                "env_from_settings: [" + ", ".join(env_from_settings) + "]\n"
                if env_from_settings else ""
            )
            + "scripts:\n"
            "  - name: fetch.py\n"
            "    description: Fetch data.\n"
            "---\n"
            "body\n"
        ),
        encoding="utf-8",
    )
    (skill_dir / "scripts" / "fetch.py").write_text("print('hi')\n", encoding="utf-8")
    return skills_root


def _make_ctx(tmp_path: pathlib.Path) -> ToolContext:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    return ToolContext(repo_dir=repo_dir, drive_root=drive_root)


def _patch_review(return_value: str):
    """Patch ``_handle_multi_model_review`` to return a canned result.

    The returned shape mirrors what the real function produces:
    ``json.dumps({"results": [...]})``.
    """
    return patch(
        "ouroboros.tools.review._handle_multi_model_review",
        return_value=return_value,
    )
