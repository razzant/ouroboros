"""Owner-click creation/connection of the GitHub skills source, with its actual visibility.

GitHub work runs off the event loop through the publication client. The source
registry's existing writer persists the choice under the settings document lock.
"""

from __future__ import annotations

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from ouroboros.gateway._helpers import run_sync_to_completion
from ouroboros.gateway.owner_settings import owner_audit
from ouroboros.skill_publish_client import GitHubError, MISSING_TOKEN_MESSAGE, request as github_request
from ouroboros.skill_sources import load_sources, save_sources, with_added_source


def _connect_my_skills(request: Request) -> JSONResponse:
    created = False
    try:
        account = github_request("GET", "/user")
        login = account.get("login")
        if not isinstance(login, str) or not login:
            raise ValueError("GitHub did not return the account login")
        repository = f"{login}/ouroboros-skills"
        try:
            repo = github_request("GET", f"/repos/{repository}")
        except GitHubError as exc:
            if exc.http_status != 404:
                raise
            try:
                repo = github_request("POST", "/user/repos", timeout=60, body={
                    "name": "ouroboros-skills", "private": True, "auto_init": True,
                    "description": "My Ouroboros skills, shared between my installations.",
                })
            except GitHubError as exc:
                if exc.http_status in (403, 404):
                    return JSONResponse({"ok": False, "code": "cannot_create",
                                         "error": "The GitHub token needs permission to create repositories."},
                                        status_code=403)
                raise
            created = True
        url = f"https://github.com/{repository}"
        rows = load_sources()
        existing = next((row for row in rows if row["kind"] == "git" and row.get("url") == url), None)
        if existing is None:
            rows, source = with_added_source(rows, kind="git", url=url, name="My skills", trust="trusted")
        else:
            source = existing
        saved = save_sources(rows)
        source = next(row for row in saved if row["id"] == source["id"])
    except GitHubError as exc:
        no_token = exc.failure == "no_token"
        return JSONResponse({"ok": False, "code": "no_token" if no_token else "failed",
                             "error": MISSING_TOKEN_MESSAGE if no_token else f"Could not connect My skills: {exc.message}"},
                            status_code=400 if no_token else 502)
    except Exception:
        return JSONResponse({"ok": False, "code": "failed", "error": "Could not connect My skills; try again."},
                            status_code=502)
    owner_audit(request, "skill_sources", {"change": "my_skills", "repository": repository, "created": created})
    return JSONResponse({"ok": True, "created": created, "repository": repository,
                         "private": repo["private"], "source": source})


async def api_skill_source_my_skills(request: Request) -> JSONResponse:
    return await run_sync_to_completion(_connect_my_skills, request)


def skill_source_routes() -> list[Route]:
    """Put the named action before the existing source-id route."""
    from ouroboros.gateway.skill_sources import skill_source_routes as source_routes

    return [Route("/api/skill-sources/my-skills", endpoint=api_skill_source_my_skills, methods=["POST"]),
            *source_routes()]
