"""The real task detail and artifact routes on a test-owned drive."""
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient


def task_file_client(data):
    from ouroboros.gateway.tasks import api_task_artifact, api_task_get

    app = Starlette(routes=[
        Route("/api/tasks/{task_id}", endpoint=api_task_get, methods=["GET"]),
        Route("/api/tasks/{task_id}/artifacts/{name}", endpoint=api_task_artifact, methods=["GET", "HEAD"]),
    ])
    app.state.drive_root = data
    return TestClient(app)


def use_cli_http(monkeypatch, http):
    """Keep CLI request creation and byte/header validation real, serving urllib in process."""
    import io
    from urllib.error import HTTPError
    from urllib.parse import urlsplit
    from urllib.response import addinfourl

    from ouroboros import cli

    requests = []

    def urlopen(request, *, timeout):
        url = request.full_url
        method = request.get_method()
        requests.append((method, urlsplit(url).path))
        response = http.request(method, url, headers=dict(request.header_items()), content=request.data)
        stream = io.BytesIO(response.content)
        if response.status_code >= 400:
            raise HTTPError(url, response.status_code, response.reason_phrase, response.headers, stream)
        return addinfourl(stream, response.headers, url, response.status_code)

    monkeypatch.setattr(cli.urllib.request, "urlopen", urlopen)
    return cli.OuroborosHTTPClient("http://testserver"), requests
