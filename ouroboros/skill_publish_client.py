"""Token-authenticated GitHub HTTPS for publication and the owner's skills repository.

No CLI, retries or domain policy: callers retain transaction/settlement decisions.
Errors carry GitHub's status and message, bounded and redacted before leaving here.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from ouroboros.secret_masking import redact_known_values
from ouroboros.tools.github import github_token_from_env_or_settings
from ouroboros.utils import truncate_within_limit

MISSING_TOKEN_MESSAGE = "Add a GitHub token in Settings to publish"


class GitHubError(RuntimeError):
    """An HTTP, transport or response failure; never contains the credential."""

    def __init__(self, message: str, *, http_status: int | None = None,
                 failure: str = "http", token: str = "") -> None:
        self.http_status = http_status
        self.failure = failure
        self.message = truncate_within_limit(
            redact_known_values(message, [token or github_token_from_env_or_settings()]), 600,
        )
        super().__init__(f"GitHub HTTP {http_status}: {self.message}" if http_status else self.message)


def request(method: str, path: str, *, body: dict[str, Any] | None = None,
            timeout: int = 30) -> Any:
    """One JSON request to api.github.com; no subprocess or credential discovery via gh."""
    token = github_token_from_env_or_settings()
    if not token:
        raise GitHubError(MISSING_TOKEN_MESSAGE, failure="no_token")
    try:
        response = httpx.request(
            method, "https://api.github.com" + path, json=body,
            headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                     "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "Ouroboros"},
            timeout=timeout, follow_redirects=True,
        )
    except httpx.TimeoutException:
        raise GitHubError(f"GitHub request timed out after {timeout}s; the outcome may be unknown.",
                          failure="timeout", token=token) from None
    except httpx.HTTPError as exc:
        raise GitHubError(str(exc), failure="transport", token=token) from None
    try:
        data = response.json() if response.content else {}
    except ValueError:
        data = None
    if not response.is_success:
        message = str(data.get("message") or response.reason_phrase) if isinstance(data, dict) else response.text
        if isinstance(data, dict) and data.get("errors"):
            message += ": " + json.dumps(data["errors"], ensure_ascii=False)
        raise GitHubError(message, http_status=response.status_code, token=token)
    if data is None:
        raise GitHubError("GitHub returned invalid JSON: " + response.text,
                          http_status=response.status_code, failure="invalid_response", token=token)
    if path == "/graphql" and isinstance(data, dict) and data.get("errors"):
        raise GitHubError(json.dumps(data["errors"], ensure_ascii=False),
                          http_status=response.status_code, failure="graphql", token=token)
    return data
