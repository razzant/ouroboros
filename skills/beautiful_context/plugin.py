"""Optional Beautiful Context widget serving the shared local report viewer."""

from __future__ import annotations

from starlette.responses import HTMLResponse, PlainTextResponse

from ouroboros.server_web import resolve_web_dir


def viewer(request):
    """Serve the shared single-file report viewer without caching or server state."""
    app_state = getattr(getattr(request, "app", None), "state", None)
    repo_dir = getattr(app_state, "repo_dir", None)
    if repo_dir is None:
        from ouroboros.config import REPO_DIR

        repo_dir = REPO_DIR
    report_path = resolve_web_dir(repo_dir) / "context-report.html"
    try:
        html = report_path.read_text(encoding="utf-8")
    except OSError:
        return PlainTextResponse("Beautiful Context viewer unavailable", status_code=404)
    return HTMLResponse(
        html,
        headers={"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"},
    )


def register(api):
    api.register_route("viewer", viewer, methods=("GET",))
    api.register_ui_tab(
        "viewer",
        "Beautiful Context",
        icon="gauge",
        render={
            "kind": "iframe",
            "route": "viewer",
            "start": "auto",
            "height": 680,
            "span": 2,
        },
    )
