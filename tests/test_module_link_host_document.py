"""The authenticated Telegram SPA retains its presentation SDK after bootstrap."""
import pathlib

from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros.server_web import UI_OWNER_MARKER, make_index_page


def test_telegram_marker_only_changes_the_host_document():
    web = pathlib.Path(__file__).resolve().parents[1] / "web"
    source = (web / "index.html").read_bytes().decode("utf-8")  # as served: checkout line endings kept
    with TestClient(Starlette(routes=[Route("/", make_index_page(web))])) as client:
        plain = client.get("/")
        # The only response-time change of the builtin page is its core-ownership marker.
        assert plain.text == source.replace("<head>", "<head>" + UI_OWNER_MARKER, 1)
        assert UI_OWNER_MARKER not in source, "the marker is response-only, never committed to web/index.html"
        assert plain.headers["cache-control"] == "no-store"
        marked = client.get("/", headers={"X-Ouroboros-Telegram-MiniApp": "1"})
        assert marked.status_code == 200
        assert '<html data-ouroboros-host="telegram"' in marked.text
        assert '<script async src="https://telegram.org/js/telegram-web-app.js"></script>' in marked.text
        assert marked.text.count(UI_OWNER_MARKER) == 1 and marked.headers["cache-control"] == "no-store"
        assert 'data-ouroboros-host' not in plain.text
        assert client.get("/", headers={"X-Ouroboros-Telegram-MiniApp": "0"}).content == plain.content
