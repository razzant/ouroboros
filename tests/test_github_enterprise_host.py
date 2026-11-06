"""A ``repo=""`` GitHub call stays on its repository's own host, with that host's own credential (#2571).

The first half drives the registered tools over a fake ``gh``. The second drives the installed ``gh``: its
``http_unix_socket`` setting carries every HTTP request to a local two-host fixture, in the clear and with the
request's own Host and Authorization, so nothing leaves the machine and the destination and the credential gh chose
are observed, not inferred. Closed proxies stand behind the socket: a gh that ignored it fails instead of leaving.
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import socket
import socketserver
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest

from ouroboros.tools import github
from tests.test_github_project_target import _api_parts, _context

ENTERPRISE, PUBLIC_API = "ghe.example.test", "api.github.com"
URL = f"https://{ENTERPRISE}/corp/secret"
VIEW = json.dumps({"nameWithOwner": "corp/secret", "url": URL})
SHA = "a" * 40
VIEWER_QUERY = "query { viewer { login } }"


# ---- the fake CLI ----------------------------------------------------------------------------------
@pytest.fixture
def fake(monkeypatch, tmp_path):
    """Records the resolver's reads apart from the legs it pins; ``answer(argv, kwargs)`` is stdout or ``(code, stderr)``."""
    world = SimpleNamespace(views=[], calls=[], view=VIEW, answer=lambda argv, kwargs: "{}")
    monkeypatch.setattr(github, "github_token_from_env_or_settings", lambda: "ghp_fixture")
    for name in ("GH_REPO", "GH_HOST", "GH_ENTERPRISE_TOKEN", "GITHUB_ENTERPRISE_TOKEN"):
        monkeypatch.delenv(name, raising=False)

    def outcome(value):
        if isinstance(value, BaseException):
            raise value
        if isinstance(value, tuple):
            return SimpleNamespace(returncode=value[0], stdout="", stderr=value[1])
        return SimpleNamespace(returncode=0, stdout=value, stderr="")

    def run(argv, **kwargs):
        if argv[1:3] == ["repo", "view"]:
            world.views.append(argv)
            return outcome(world.view)
        world.calls.append((argv, kwargs))
        return outcome(world.answer(argv, kwargs))

    monkeypatch.setattr(subprocess, "run", run)
    world.ctx, world.cwd = _context(tmp_path, "queued")
    world.ctx._active_builtin_tool_result = None
    tools = {entry.name: entry.handler for entry in github.get_tools()}

    def call(name, **args):
        world.ctx._active_builtin_tool_result = None
        text = tools[name](world.ctx, **args)
        world.result = world.ctx._active_builtin_tool_result
        return text

    world.call = call
    world.routes = lambda: [_api_parts(argv)[:2] for argv, _kw in world.calls if argv[1] == "api"]
    return world


def _gql_query(kwargs):
    return json.loads(kwargs["input"])["query"]


def _leg_answers(argv, kwargs):
    """Enough of GitHub for one call of each multi-leg flow; a send to ``comments``/``resolve`` loses its answer."""
    route = _api_parts(argv)[0]
    if route == "user":
        return json.dumps({"login": "ent-bot"})
    if route == "graphql":
        query = _gql_query(kwargs)
        if "mutation" in query:
            return (1, "gh: Bad Gateway (HTTP 502)")
        if "nodes(ids:" in query:
            return json.dumps({"data": {"nodes": [{"id": "PR_7", "commits": {"totalCount": 1}, "reviewDecision": None}]}})
        if "reviewThreads" in query:
            return json.dumps({"data": {"repository": {"pullRequest": {"reviewThreads": {"totalCount": 0, "nodes": []}}}}})
        if "search(" in query:
            return json.dumps({"data": {"search": {"discussionCount": 0, "nodes": []}}})
        return json.dumps({"data": {"node": {"id": "PRRT_1", "isResolved": True, "resolvedBy": {"login": "ent-bot"}}}})
    method = argv[argv.index("--method") + 1] if "--method" in argv else "GET"
    if method != "GET":
        return (1, "gh: Bad Gateway (HTTP 502)")
    if route.endswith("/issues/7"):
        return json.dumps({"number": 7, "state": "open", "html_url": f"{URL}/issues/7"})
    if route.endswith("/pulls/7"):
        return json.dumps({"number": 7, "node_id": "PR_7", "state": "open", "title": "T", "html_url": f"{URL}/pull/7",
                           "head": {"ref": "f", "sha": SHA}, "base": {"ref": "main"}})
    return json.dumps([{"id": 3, "body": "x", "user": {"login": "ent-bot"}, "html_url": f"{URL}/issues/7#c3"}])


@pytest.mark.parametrize("name,args,routes", [
    ("github_write", {"method": "comment", "target": 7, "body": "x", "expect": {"state": "open"}},
     ["repos/corp/secret/issues/7", "repos/corp/secret/issues/7/comments", "repos/corp/secret/issues/7/comments?", "user"]),
    ("github_write", {"method": "resolve_thread", "target": "PRRT_1"}, ["graphql", "graphql", "user"]),
    ("github_read", {"method": "get", "kind": "pr", "target": 7, "include": ["threads"]},
     ["repos/corp/secret/pulls/7", "graphql", "graphql"]),
    ("github_read", {"method": "search", "kind": "discussion", "query": "bug"}, ["graphql"]),
], ids=["expectation-send-lookup-actor", "node-only-send-check-actor", "rest-and-graphql-facets", "search-graphql"])
def test_one_call_resolves_once_and_every_leg_shares_that_target(fake, name, args, routes):
    fake.answer = _leg_answers
    fake.call(name, **args)
    assert len(fake.views) == 1
    assert [host for _route, host in fake.routes()] == [ENTERPRISE] * len(routes)
    for (route, _host), expected in zip(fake.routes(), routes):
        assert route.startswith(expected) and "{" not in route


def test_the_pin_lasts_exactly_one_call(fake):
    moved = json.dumps({"nameWithOwner": "corp/secret", "url": "https://moved.example.test/corp/secret"})

    def answer(argv, kwargs):
        fake.view = moved  # A repository re-pointed during a call cannot split it.
        return _leg_answers(argv, kwargs)

    fake.answer = answer
    fake.call("github_write", method="comment", target=7, body="x")
    assert {host for _route, host in fake.routes()} == {ENTERPRISE}
    fake.calls.clear()
    fake.answer = lambda argv, kwargs: "[]"
    fake.call("github_read", method="list", kind="label")
    assert len(fake.views) == 2 and {host for _route, host in fake.routes()} == {"moved.example.test"}


def test_pagination_keeps_the_resolved_host_on_every_page(fake):
    next_url = f"https://{ENTERPRISE}/api/v3/repositories/1/issues?per_page=1&page=2"
    head = f'HTTP/2.0 200 OK\r\nLink: <{next_url}>; rel="next"\r\n\r\n'
    fake.answer = lambda argv, kwargs: head + json.dumps([{"number": 7, "html_url": f"{URL}/issues/7"}])
    first = json.loads(fake.call("github_read", method="list", kind="issue").partition("\n")[2])
    assert first["next_cursor"] == "2"
    fake.call("github_read", method="list", kind="issue", cursor=first["next_cursor"])
    following = json.loads(fake.call("github_read", method="api", route="repos/{owner}/{repo}/issues").partition("\n")[2])
    fake.call("github_read", method="api", route=following["next_cursor"])
    routes = fake.routes()
    assert [host for _route, host in routes] == [ENTERPRISE] * 4
    assert "&page=2" in routes[1][0] and routes[3][0] == next_url


@pytest.mark.parametrize("view", [
    (1, "no git remotes found"), (1, "gh: Server Error (HTTP 500)"), subprocess.TimeoutExpired(["gh"], 30), "not json", "[]",
    json.dumps({"nameWithOwner": "corp/secret"}), json.dumps({"nameWithOwner": "corp/other", "url": URL}),
    json.dumps({"nameWithOwner": "corp/secret", "url": f"http://{ENTERPRISE}/corp/secret"}),
], ids=["no-remote", "http-500", "timeout", "not-json", "not-object", "no-url", "other-name", "not-https"])
@pytest.mark.parametrize("name,args", [
    ("github_write", {"method": "comment", "target": 7, "body": "x", "expect": {"state": "open"}}),
    ("github_write", {"method": "comment", "target": 7, "body": "x"}),
    ("github_write", {"method": "create_issue", "title": "T"}),
    ("github_write", {"method": "resolve_thread", "target": "PRRT_1"}),
    ("github_write", {"method": "api", "http_method": "POST", "route": "user/repos"}),
    ("github_write", {"method": "api", "graphql": "mutation { x }"}),
    ("github_read", {"method": "api", "route": "rate_limit"}),
    ("github_read", {"method": "api", "graphql": VIEWER_QUERY}),
    ("github_read", {"method": "search", "kind": "issue", "query": "bug"}),
], ids=lambda value: value if isinstance(value, str) else value["method"])
def test_an_unresolved_target_reaches_no_host_and_attests_no_effect(fake, name, args, view):
    fake.view = view
    text = fake.call(name, **args)
    assert text.startswith("⚠️ GH_TARGET_UNAVAILABLE:") and "repo='[HOST/]OWNER/REPO'" in text
    assert fake.calls == [] and len(fake.views) == 1
    assert (fake.result.status, fake.result.meta["operation_outcome"]) == ("unavailable", "completed_no_effect")


@pytest.mark.parametrize("route", ["https://api.github.com/user", f"http://{ENTERPRISE}/api/v3/user",
                                   f"https://{ENTERPRISE}.evil.test/api/v3/user", "repos/x?u=https://elsewhere.test"])
@pytest.mark.parametrize("repo", ["", f"{ENTERPRISE}/corp/secret"])
def test_an_absolute_route_cannot_leave_the_host_and_naming_the_target_is_no_effect(fake, route, repo):
    text = fake.call("github_write", method="api", http_method="POST", route=route, repo=repo)
    assert text.startswith("⚠️ GH_TARGET_INVALID:") and fake.calls == []
    assert fake.result.meta["operation_outcome"] == "completed_no_effect"


@pytest.mark.parametrize("route", [f"https://{ENTERPRISE}/api/v3/user", f"https://api.{ENTERPRISE}/user",
                                   f"https://uploads.{ENTERPRISE}/x", f"https://{ENTERPRISE.upper()}/api/v3/user"])
def test_an_absolute_route_on_the_host_is_followed(fake, route):
    fake.answer = lambda argv, kwargs: "{}"
    assert not fake.call("github_read", method="api", route=route).startswith("⚠️")
    assert fake.routes() == [(route, ENTERPRISE)]


def test_a_host_less_explicit_repo_keeps_following_gh_configuration(fake):
    fake.answer = lambda argv, kwargs: "{}"
    fake.call("github_read", method="api", route="https://elsewhere.test/x", repo="corp/secret")
    fake.call("github_read", method="get", kind="issue", target=7, repo="corp/secret")
    assert fake.views == [] and fake.routes() == [("https://elsewhere.test/x", ""), ("repos/corp/secret/issues/7", "")]


@pytest.mark.parametrize("repo,slug", [("", "corp/secret"), ("other/thing", "other/thing")])
@pytest.mark.parametrize("route,expected", [("repos/{owner}/{repo}/x", "repos/%s/x"), ("repos/:owner/:repo/x", "repos/%s/x")])
def test_every_repository_placeholder_form_is_filled_from_the_target(fake, repo, slug, route, expected):
    fake.call("github_read", method="api", route=route, repo=repo)
    assert fake.routes()[0][0] == expected % slug


def test_a_single_slot_is_filled_too_and_branch_stays_with_gh(fake):
    fake.call("github_read", method="api", route="orgs/{owner}/x/{branch}")
    assert fake.routes()[0][0] == "orgs/corp/x/{branch}"


def test_a_later_step_cannot_leave_the_resolved_host(fake):
    """A URL-derived leg meets the same pin: the list is withheld and the other host hears nothing."""
    row = {"number": 7, "node_id": "PR_7", "html_url": "https://github.com/corp/secret/pull/7", "head": {}, "base": {}}
    fake.answer = lambda argv, kwargs: json.dumps(row if "/pulls/7" in argv[2] else {})
    text = fake.call("github_read", method="get", kind="pr", target=7)
    assert text.startswith("⚠️ GH_TARGET_UNAVAILABLE:") and "github.com" in text
    assert [host for _route, host in fake.routes()] == [ENTERPRISE]


def test_an_invalid_explicit_repo_is_refused_before_any_launch(fake):
    text = fake.call("github_read", method="search", kind="issue", query="bug", repo="not a repo")
    assert text.startswith("⚠️ GH_TARGET_INVALID:") and not fake.views and not fake.calls


def test_the_target_read_does_not_spend_the_no_effect_fact_but_a_real_leg_does(fake):
    token = github._SUBMITTED.set([False])
    try:
        assert github._selected_target(fake.ctx)[0] == github.GhTarget(ENTERPRISE, "corp", "secret")
        assert github._no_launch_yet()
        github._gh_run(["api", "user"], fake.ctx)
        assert not github._no_launch_yet()
        assert github._selected_target(fake.ctx)[1] is None and not github._no_launch_yet()
    finally:
        github._SUBMITTED.reset(token)


def test_gh_repo_selects_the_target_only_where_no_project_replaced_it(fake, tmp_path, monkeypatch):
    monkeypatch.setenv("GH_REPO", "ambient/repo")
    fake.call("github_read", method="api", route="user")
    assert fake.views[-1] == ["gh", "repo", "view", "--json", "nameWithOwner,url"]
    system, _cwd = _context(tmp_path, "system")
    tools = {entry.name: entry.handler for entry in github.get_tools()}
    tools["github_read"](system, method="api", route="user")
    assert fake.views[-1] == ["gh", "repo", "view", "ambient/repo", "--json", "nameWithOwner,url"]


def test_repo_less_generic_api_does_not_require_a_repository_resolution(fake, tmp_path):
    system, _cwd = _context(tmp_path, "system")
    tools = {entry.name: entry.handler for entry in github.get_tools()}
    tools["github_read"](system, method="api", route="user")
    tools["github_read"](system, method="api", graphql=VIEWER_QUERY)
    assert fake.views == []
    assert [argv[1:] for argv, _kwargs in fake.calls] == [
        ["api", "--include", "user"], ["api", "graphql", "--input", "-"]
    ]


@pytest.mark.parametrize("host,env,kind,source", [
    ("github.com", {}, "classic personal access token", "env GH_TOKEN"),
    ("octo.ghe.com", {}, "classic personal access token", "env GH_TOKEN"),
    (ENTERPRISE, {}, "gh CLI login", "gh CLI host configuration"),
    (ENTERPRISE, {"GITHUB_ENTERPRISE_TOKEN": "ghs_alt"}, "GitHub App installation token", "env GITHUB_ENTERPRISE_TOKEN"),
    (ENTERPRISE, {"GH_ENTERPRISE_TOKEN": "gho_ent", "GITHUB_ENTERPRISE_TOKEN": "ghs_alt"}, "OAuth token (gh login)", "env GH_ENTERPRISE_TOKEN"),
])
def test_whoami_attributes_the_credential_gh_selects_for_the_host(fake, monkeypatch, host, env, kind, source):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setenv("GH_TOKEN", "ghp_fixture")
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    fake.view = json.dumps({"nameWithOwner": "corp/secret", "url": f"https://{host}/corp/secret"})
    fake.answer = lambda argv, kwargs: "HTTP/2.0 200 OK\r\n\r\n" + json.dumps({"login": "bot", "html_url": f"https://{host}/bot"})
    item = json.loads(fake.call("github_read", method="whoami").partition("\n")[2])["item"]
    assert fake.calls[0][0] == ["gh", "api", "--include", "user", "--hostname", host]
    assert (item["host"], item["token_kind"], item["token_source"]) == (host, kind, source)


def test_whoami_with_no_repository_keeps_the_generic_contract_and_says_which_host_answered(fake):
    fake.view = (1, "no git remotes found")
    fake.answer = lambda argv, kwargs: "HTTP/2.0 200 OK\r\n\r\n" + json.dumps({"login": "bot", "html_url": f"https://{ENTERPRISE}/bot"})
    payload = json.loads(fake.call("github_read", method="whoami").partition("\n")[2])
    assert fake.calls[0][0] == ["gh", "api", "--include", "user"] and fake.calls[0][1]["cwd"] == str(fake.ctx.repo_dir)
    assert payload["item"]["host"] == ENTERPRISE  # gh's default host, read from GitHub's answer, not assumed
    assert "gh's default host" in payload["omitted"][0] and "no git remotes found" in payload["omitted"][0]


def test_the_public_token_is_never_copied_into_an_enterprise_variable(fake):
    env = github._gh_env(fake.ctx)
    assert env["GH_TOKEN"] == env["GITHUB_TOKEN"] == "ghp_fixture"
    assert "GH_ENTERPRISE_TOKEN" not in env and "GITHUB_ENTERPRISE_TOKEN" not in env


# ---- the installed gh ------------------------------------------------------------------------------
@dataclass(frozen=True)
class Seen:
    host: str
    method: str
    path: str
    token: str
    request: object


@dataclass
class _Forge:
    """One GitHub instance: its own login and its own ``corp/secret``, so a misdirected call has a same-named target."""
    web: str
    api: str
    login: str
    rest: str = ""
    graphql: str = "/graphql"
    state: str = "open"
    resolved: bool = False

    def __post_init__(self):
        self.comments, self.created = [], []

    def issue(self, slug, number=7):
        return {"number": number, "title": f"{self.login} issue", "state": self.state, "user": {"login": self.login},
                "labels": [], "assignees": [], "comments": 0, "body": "b", "html_url": f"https://{self.web}/{slug}/issues/{number}",
                "created_at": "2026-11-01T00:00:00Z", "updated_at": "2026-11-01T00:00:00Z"}

    def pull(self, slug):
        return {"number": 7, "node_id": "PR_7", "title": "T", "state": "open", "user": {"login": self.login}, "body": "",
                "html_url": f"https://{self.web}/{slug}/pull/7", "head": {"ref": "f", "label": "corp:f", "sha": SHA},
                "base": {"ref": "main"}, "labels": []}


class _World:
    def __init__(self):
        self.enterprise = _Forge(ENTERPRISE, ENTERPRISE, "ent-bot", rest="/api/v3", graphql="/api/graphql")
        self.public = _Forge("github.com", PUBLIC_API, "pub-bot")
        self.seen, self.lost, self.view_status, self.settings = [], set(), 200, {}

    def hosts(self):
        return {seen.host for seen in self.seen}

    def tokens(self):
        return {seen.token for seen in self.seen}

    def respond(self, host, method, target, authorization, body):
        url = urlsplit(target)
        forge = next((forge for forge in (self.enterprise, self.public) if forge.api == host), None)
        request = json.loads(body) if body.strip() else None
        self.seen.append(Seen(host, method, url.path, authorization.partition(" ")[2], request))
        if forge is None:
            return 404, {"message": "Not Found"}, {}
        if url.path == forge.graphql:
            return self.graphql(forge, request)
        return self.rest(forge, method, url.path.removeprefix(forge.rest), parse_qs(url.query), request)

    def graphql(self, forge, request):
        query, variables = request["query"], request.get("variables") or {}
        if "RepositoryInfo" in query:
            if self.view_status != 200:
                return self.view_status, {"message": "unavailable"}, {}
            slug = f"{variables['owner']}/{variables['name']}"
            return 200, {"data": {"repository": {"nameWithOwner": slug, "url": f"https://{forge.web}/{slug}"}}}, {}
        if "resolveReviewThread" in query:
            forge.resolved = True
            if "resolve" in self.lost:
                return 502, {"message": "Bad Gateway"}, {}
            data = {"resolveReviewThread": {"thread": {"id": variables["id"], "isResolved": True, "resolvedBy": {"login": forge.login}}}}
        elif "PullRequestReviewThread" in query:
            data = {"node": {"id": variables["id"], "isResolved": forge.resolved, "resolvedBy": {"login": forge.login}}}
        elif "nodes(ids:" in query:
            data = {"nodes": [{"id": node, "commits": {"totalCount": 1}, "reviewDecision": "APPROVED"} for node in variables["ids"]]}
        elif "search(" in query:
            data = {"search": {"discussionCount": 0, "pageInfo": {"hasNextPage": False, "endCursor": None}, "nodes": []}}
        elif "viewer" in query:
            data = {"viewer": {"login": forge.login}}
        elif "repository(" in query:
            data = {"repository": {"id": f"R_{forge.login}"}}
        else:
            data = {"addReaction": {"clientMutationId": None}}
        return 200, {"data": data}, {}

    def rest(self, forge, method, path, query, request):
        api_root = f"https://{forge.api}{forge.rest}"
        if path == "/user":
            return 200, {"login": forge.login, "id": 1, "type": "User", "name": None, "html_url": f"https://{forge.web}/{forge.login}"}, {"X-OAuth-Scopes": "repo"}
        if path == "/search/issues":
            return 200, {"total_count": 1, "incomplete_results": False, "items": [forge.issue("corp/secret")]}, {}
        if path.startswith("/repositories/"):
            return 200, [], {}
        slug = "/".join(path.split("/")[2:4])
        tail = "/" + "/".join(path.split("/")[4:])
        if tail == "/issues" and method == "POST":
            forge.created.append(request)
            return 201, {**forge.issue(slug, 8), "title": request["title"], "body": request["body"]}, {}
        if tail == "/issues":
            page = int((query.get("page") or ["1"])[0])
            link = f'<{api_root}/repositories/1/issues?per_page=30&page=2>; rel="next"' if page == 1 else ""
            return 200, [forge.issue(slug, 6 + page)], {"Link": link} if link else {}
        if tail == "/issues/7" and method == "PATCH":
            forge.state = request["state"]
            return 200, {**forge.issue(slug), "closed_by": {"login": forge.login}}, {}
        if tail == "/issues/7":
            return 200, forge.issue(slug), {}
        if tail == "/pulls":
            return 200, [forge.pull(slug)], {}
        if tail == "/pulls/7":
            return 200, forge.pull(slug), {}
        if tail == "/issues/7/comments" and method == "POST":
            comment = {"id": len(forge.comments) + 1, "body": request["body"], "user": {"login": forge.login}, "created_at": "2026-11-01T00:00:00Z",
                       "html_url": f"https://{forge.web}/{slug}/issues/7#c1", "issue_url": f"{api_root}/repos/{slug}/issues/7"}
            forge.comments.append(comment)
            return (502, {"message": "Bad Gateway"}, {}) if f"POST {path}" in self.lost else (201, comment, {})
        if tail == "/issues/7/comments":
            return 200, forge.comments, {}
        if tail == "/issues/7/timeline":
            return 200, [{"event": "labeled"}], {"Link": f'<{api_root}/repositories/1/issues/7/timeline?page=2>; rel="next"'}
        return (201 if method == "POST" else 200), {"id": 99, "url": f"{api_root}{path}"}, {}


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _body(self):
        if "chunked" not in (self.headers.get("transfer-encoding") or ""):
            return self.rfile.read(int(self.headers.get("content-length") or 0))
        chunks = []
        while True:  # `gh api --input -` streams stdin, whose length it does not know.
            size = int(self.rfile.readline().split(b";")[0], 16)
            if not size:
                self.rfile.readline()
                return b"".join(chunks)
            chunks.append(self.rfile.read(size))
            self.rfile.readline()

    def answer(self):
        status, payload, headers = self.server.world.respond(
            self.headers.get("host") or "", self.command, self.path, self.headers.get("authorization") or "", self._body())
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        for key, value in {"Content-Type": "application/json", "Content-Length": str(len(data)), "Connection": "close", **headers}.items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(data)
        self.close_connection = True

    do_GET = do_POST = do_PATCH = do_PUT = do_DELETE = answer

    def log_message(self, *_args):
        pass


def _serve_unix(address):
    # Built only past the fixture's skip: socketserver has no UnixStreamServer without AF_UNIX (Windows CPython).
    class _Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
        def get_request(self):
            connection, _address = super().get_request()
            return connection, ("fixture", 0)

    return _Server(address, _Handler)


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True, timeout=30)


def _bind(world, kind, remote=f"https://{ENTERPRISE}/corp/secret.git"):
    world.ctx, world.cwd = _context(world.tmp_path, kind)
    world.ctx._active_builtin_tool_result = None
    if not (world.cwd / ".git").exists():
        _git(world.cwd, "init", "-q")
        _git(world.cwd, "remote", "add", "origin", remote)
    return world


@pytest.fixture
def two_hosts(tmp_path, monkeypatch):
    if os.name == "nt" or not hasattr(socket, "AF_UNIX"):
        pytest.skip("the fixture listens on a Unix socket")
    if shutil.which("gh") is None or shutil.which("git") is None:
        pytest.skip("the installed GitHub CLI is the evidence")
    base = pathlib.Path(tempfile.mkdtemp(prefix="gh", dir="/tmp"))  # Short: an AF_UNIX path has at most 104 bytes on macOS.
    world, address = _World(), base / "s"
    server = _serve_unix(str(address))
    server.world, world.tmp_path = world, tmp_path
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    config = tmp_path / "gh-config"
    config.mkdir()
    (config / "config.yml").write_text(f"http_unix_socket: {address}\nprompt: disabled\n", encoding="utf-8")
    (config / "hosts.yml").write_text(
        f"github.com:\n    oauth_token: public-hosts-token\n    user: pub-bot\n"
        f"{ENTERPRISE}:\n    oauth_token: enterprise-hosts-token\n    user: ent-bot\n", encoding="utf-8")
    for name in ("GH_HOST", "GH_REPO", "GH_TOKEN", "GITHUB_TOKEN", "GH_ENTERPRISE_TOKEN", "GITHUB_ENTERPRISE_TOKEN", "GH_DEBUG", "NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("GH_CONFIG_DIR", str(config))
    for name in ("XDG_CONFIG_HOME", "XDG_STATE_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME"):
        monkeypatch.setenv(name, str(home / name.lower()))
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.setenv(name, "http://127.0.0.1:9")  # Closed: only the socket answers.
    for name, value in (("GH_NO_UPDATE_NOTIFIER", "1"), ("GH_PROMPT_DISABLED", "1"), ("NO_COLOR", "1"),
                        ("GIT_CONFIG_NOSYSTEM", "1"), ("GIT_CONFIG_GLOBAL", os.devnull)):
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("GH_TOKEN", "public-env-token")
    monkeypatch.setattr("ouroboros.config.load_settings", lambda: world.settings)
    tools = {entry.name: entry.handler for entry in github.get_tools()}

    def call(name, **args):
        world.ctx._active_builtin_tool_result = None
        return tools[name](world.ctx, **args)

    world.call = call
    try:
        subprocess.run(["gh", "api", "--hostname", "canary.example.test", "probe"], capture_output=True, timeout=30, check=False)
        assert world.hosts() == {"canary.example.test"}, "gh did not take the fixture's socket"
        world.seen.clear()
        yield _bind(world, "queued")
    finally:
        server.shutdown()
        server.server_close()
        serving.join(timeout=10)
        shutil.rmtree(base, ignore_errors=True)


def _on_the_enterprise_host_only(world, token):
    assert world.hosts() == {ENTERPRISE} and world.tokens() == {token}
    assert not world.public.comments and not world.public.created and world.public.state == "open" and not world.public.resolved
    for seen in world.seen:
        assert seen.path == "/api/graphql" or seen.path.startswith(("/api/v3/repos/corp/secret", "/api/v3/user", "/api/v3/search/")), seen


REAL_LEGS = [
    ("github_read", {"method": "get", "kind": "issue", "target": 7}),
    ("github_read", {"method": "get", "kind": "pr", "target": 7}),
    ("github_read", {"method": "list", "kind": "pr"}),
    ("github_read", {"method": "search", "kind": "issue", "query": "is:open"}),
    ("github_read", {"method": "search", "kind": "discussion", "query": "bug"}),
    ("github_read", {"method": "api", "graphql": "query($owner:String!,$name:String!){repository(owner:$owner,name:$name){id}}"}),
    ("github_read", {"method": "api", "graphql": VIEWER_QUERY}),
    ("github_read", {"method": "api", "route": "user"}),
    ("github_write", {"method": "comment", "target": 7, "body": "hello", "expect": {"state": "open"}}),
    ("github_write", {"method": "update", "target": 7, "state": "closed", "expect": {"state": "open"}}),
    ("github_write", {"method": "create_issue", "title": "T", "body": "B"}),
    ("github_write", {"method": "resolve_thread", "target": "PRRT_1"}),
    ("github_write", {"method": "api", "http_method": "POST", "route": "repos/{owner}/{repo}/issues/7/reactions", "input": {"content": "heart"}}),
    ("github_write", {"method": "api", "graphql": 'mutation { addReaction(input:{subjectId:"I_1",content:HEART}) { clientMutationId } }'}),
]


def test_the_fixture_sees_the_defect_a_bare_placeholder_call_has(two_hosts):
    """Without this repair's --hostname, gh fills the Enterprise repository's name and sends it to github.com."""
    subprocess.run(["gh", "api", "repos/{owner}/{repo}/issues/7"], cwd=str(two_hosts.cwd), capture_output=True, timeout=30, check=False)
    seen = two_hosts.seen[0]
    assert (seen.host, seen.path, seen.token) == (PUBLIC_API, "/repos/corp/secret/issues/7", "public-env-token")


@pytest.mark.parametrize("name,args", REAL_LEGS, ids=lambda value: value if isinstance(value, str) else "-".join(
    str(value.get(key)) for key in ("method", "kind", "http_method") if value.get(key)) + ("-graphql" if "graphql" in value else ""))
def test_a_two_host_project_keeps_every_leg_on_its_own_host_and_credential(two_hosts, name, args):
    text = two_hosts.call(name, **args)
    assert not text.startswith("⚠️"), text
    _on_the_enterprise_host_only(two_hosts, "enterprise-hosts-token")


@pytest.mark.parametrize("env,token", [({}, "enterprise-hosts-token"), ({"GITHUB_ENTERPRISE_TOKEN": "enterprise-alt-token"}, "enterprise-alt-token"),
                                       ({"GH_ENTERPRISE_TOKEN": "enterprise-env-token", "GITHUB_ENTERPRISE_TOKEN": "enterprise-alt-token"}, "enterprise-env-token")],
                         ids=["hosts-yml", "GITHUB_ENTERPRISE_TOKEN", "GH_ENTERPRISE_TOKEN"])
@pytest.mark.parametrize("public", ["env", "settings"])
def test_each_credential_family_reaches_only_its_own_host(two_hosts, monkeypatch, env, token, public):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    if public == "settings":
        monkeypatch.delenv("GH_TOKEN")
        two_hosts.settings["GITHUB_TOKEN"] = "public-settings-token"
    assert not two_hosts.call("github_write", method="comment", target=7, body="hello").startswith("⚠️")
    _on_the_enterprise_host_only(two_hosts, token)
    two_hosts.seen.clear()
    assert not two_hosts.call("github_read", method="get", kind="issue", target=7, repo="github.com/corp/secret").startswith("⚠️")
    assert two_hosts.hosts() == {PUBLIC_API}
    assert two_hosts.tokens() == {"public-settings-token" if public == "settings" else "public-env-token"}


@pytest.mark.parametrize("kind", ["room", "system"])
def test_room_and_system_repositories_take_the_same_path(two_hosts, kind):
    _bind(two_hosts, kind)
    assert not two_hosts.call("github_write", method="comment", target=7, body="hello").startswith("⚠️")
    _on_the_enterprise_host_only(two_hosts, "enterprise-hosts-token")


def test_list_pages_and_a_next_route_stay_on_the_project_host(two_hosts):
    first = json.loads(two_hosts.call("github_read", method="list", kind="issue").partition("\n")[2])
    assert first["next_cursor"] == "2" and first["items"][0]["number"] == 7
    second = json.loads(two_hosts.call("github_read", method="list", kind="issue", cursor="2").partition("\n")[2])
    assert second["complete"] is True and second["items"][0]["number"] == 8
    following = json.loads(two_hosts.call("github_read", method="api", route="repos/{owner}/{repo}/issues/7/timeline").partition("\n")[2])
    assert following["next_cursor"].startswith(f"https://{ENTERPRISE}/api/v3/")
    assert not two_hosts.call("github_read", method="api", route=following["next_cursor"]).startswith("⚠️")
    assert two_hosts.hosts() == {ENTERPRISE} and two_hosts.tokens() == {"enterprise-hosts-token"}
    assert two_hosts.seen[-1].path == "/api/v3/repositories/1/issues/7/timeline"


def test_an_unanswered_comment_is_reconciled_with_the_project_hosts_lookup_and_actor(two_hosts):
    two_hosts.lost = {"POST /repos/corp/secret/issues/7/comments"}
    text = two_hosts.call("github_write", method="comment", target=7, body="hello", expect={"state": "open"})
    receipt = json.loads(text.partition("\n")[2])
    assert (receipt["outcome"], receipt["readback"]["via"], receipt["actor"]) == ("confirmed", "lookup", "ent-bot")
    legs = [(seen.method, seen.path) for seen in two_hosts.seen if seen.path != "/api/graphql"]
    assert legs == [("GET", "/api/v3/repos/corp/secret/issues/7"), ("POST", "/api/v3/repos/corp/secret/issues/7/comments"),
                    ("GET", "/api/v3/repos/corp/secret/issues/7/comments"), ("GET", "/api/v3/user")]
    assert len(two_hosts.enterprise.comments) == 1
    _on_the_enterprise_host_only(two_hosts, "enterprise-hosts-token")


def test_an_unanswered_node_only_mutation_is_reconciled_on_the_project_host(two_hosts):
    two_hosts.lost = {"resolve"}
    receipt = json.loads(two_hosts.call("github_write", method="resolve_thread", target="PRRT_1").partition("\n")[2])
    assert (receipt["outcome"], receipt["readback"]["via"], receipt["actor"]) == ("confirmed", "lookup", "ent-bot")
    assert two_hosts.seen[-1].path == "/api/v3/user"
    _on_the_enterprise_host_only(two_hosts, "enterprise-hosts-token")


def test_explicit_targets_keep_their_contract_from_any_project(two_hosts, monkeypatch):
    two_hosts.call("github_read", method="get", kind="issue", target=7, repo=f"{ENTERPRISE}/corp/secret")
    assert [(seen.host, seen.path) for seen in two_hosts.seen] == [(ENTERPRISE, "/api/v3/repos/corp/secret/issues/7")]
    two_hosts.seen.clear()
    two_hosts.call("github_read", method="get", kind="issue", target=7, repo="github.com/corp/secret")
    assert [(seen.host, seen.token) for seen in two_hosts.seen] == [(PUBLIC_API, "public-env-token")]
    two_hosts.seen.clear()
    two_hosts.call("github_read", method="get", kind="issue", target=7, repo="corp/secret")  # gh's host configuration
    assert two_hosts.hosts() == {PUBLIC_API}
    two_hosts.seen.clear()
    monkeypatch.setenv("GH_HOST", ENTERPRISE)
    two_hosts.call("github_read", method="get", kind="issue", target=7, repo="corp/secret")
    assert two_hosts.hosts() == {ENTERPRISE}


def test_a_public_project_stays_public(two_hosts, monkeypatch):
    monkeypatch.setenv("GH_ENTERPRISE_TOKEN", "enterprise-env-token")
    two_hosts.tmp_path = two_hosts.tmp_path / "public"
    two_hosts.tmp_path.mkdir()
    _bind(two_hosts, "queued", "https://github.com/corp/secret.git")
    assert not two_hosts.call("github_write", method="comment", target=7, body="hello").startswith("⚠️")
    assert two_hosts.hosts() == {PUBLIC_API} and two_hosts.tokens() == {"public-env-token"}
    assert two_hosts.public.comments and not two_hosts.enterprise.comments


def test_every_leg_follows_the_base_gh_resolved_names(two_hosts):
    """`gh api` alone ignores `gh repo set-default`; a resolved call addresses the repository `gh pr` would."""
    _git(two_hosts.cwd, "remote", "add", "upstream", f"https://{ENTERPRISE}/corp/other.git")
    _git(two_hosts.cwd, "config", "remote.origin.gh-resolved", "base")
    assert not two_hosts.call("github_read", method="get", kind="issue", target=7).startswith("⚠️")
    assert two_hosts.seen[-1].path == "/api/v3/repos/corp/secret/issues/7"


@pytest.mark.parametrize("failure", ["no-remote", "no-repository", "GH_HOST-elsewhere", "host-refuses"])
def test_an_unresolved_project_sends_no_write_to_any_host(two_hosts, monkeypatch, failure):
    if failure == "no-remote":
        _git(two_hosts.cwd, "remote", "remove", "origin")
    elif failure == "no-repository":
        shutil.rmtree(two_hosts.cwd / ".git")
    elif failure == "GH_HOST-elsewhere":
        monkeypatch.setenv("GH_HOST", "other.example.test")
    else:
        two_hosts.view_status = 500
    text = two_hosts.call("github_write", method="comment", target=7, body="hello")
    assert text.startswith("⚠️ GH_TARGET_UNAVAILABLE:"), text
    assert two_hosts.ctx._active_builtin_tool_result.meta["operation_outcome"] == "completed_no_effect"
    assert all(seen.host == ENTERPRISE and "RepositoryInfo" in seen.request["query"] for seen in two_hosts.seen)
    assert not two_hosts.enterprise.comments and not two_hosts.public.comments


def test_a_foreign_absolute_route_reaches_no_host(two_hosts):
    text = two_hosts.call("github_read", method="api", route="https://api.github.com/user")
    assert text.startswith("⚠️ GH_TARGET_INVALID:") and PUBLIC_API not in two_hosts.hosts()


@pytest.mark.parametrize("env,source", [({}, "gh CLI host configuration"), ({"GH_ENTERPRISE_TOKEN": "enterprise-env-token"}, "env GH_ENTERPRISE_TOKEN")])
def test_whoami_names_the_project_hosts_actor_and_credential_source(two_hosts, monkeypatch, env, source):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    item = json.loads(two_hosts.call("github_read", method="whoami").partition("\n")[2])["item"]
    assert (item["login"], item["host"], item["token_source"]) == ("ent-bot", ENTERPRISE, source)
    assert two_hosts.hosts() == {ENTERPRISE}


def test_whoami_without_a_repository_stays_generic_and_names_the_host_that_answered(two_hosts):
    shutil.rmtree(two_hosts.cwd / ".git")
    payload = json.loads(two_hosts.call("github_read", method="whoami").partition("\n")[2])
    assert (payload["item"]["login"], payload["item"]["host"], payload["item"]["token_source"]) == ("pub-bot", "github.com", "env GH_TOKEN")
    assert two_hosts.hosts() == {PUBLIC_API} and "gh's default host" in payload["omitted"][0]
