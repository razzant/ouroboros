"""A separately hosted interface over a real core, behind an actual TLS reverse proxy.

``tests/test_separate_ui_browser.py`` drives these pieces, and ``python -m tests.separate_ui_support``
starts the same stack for a person to browse (see ``main``). Three processes, each with its own
lifetime and each bound to 127.0.0.1 only:

* the core: the real ``server.py`` of a candidate checkout of this tree, started with
  ``OUROBOROS_UI_DIR=none`` and a network password. It serves no main page; its API, sign-in,
  socket, setup and extension resources stay its own. Its provider is ``ModelStub``, a loopback
  OpenAI-compatible endpoint (no paid model, no key); a fresh install starts with no settings;
* the interface host: Caddy serving a COPY of the checkout's ``web/`` files through
  ``docs/examples/separate_ui/interface.Caddyfile`` as written;
* the front proxy: Caddy with a certificate made for this run and
  ``docs/examples/separate_ui/Caddyfile``. One line differs: the browser here is on loopback, which
  the core never gates, so the proxy forwards a fixed documentation address (TEST-NET-2) as the
  visitor instead of the real peer. ``verbatim=True`` runs the file unchanged.

Caddy comes from ``SEPARATE_UI_CADDY`` (an absolute path); without it the tests skip. Nothing here
installs a certificate: a browser trusts this run's certificate only inside the browser the test
launches (Chromium: the certificate's key pin; WebKit: its own context ignores certificate errors,
and the test checks which certificate it was shown).
"""
from __future__ import annotations

import base64
import datetime
import hashlib
import ipaddress
import json
import os
import shutil
import signal
import socket
import ssl
import subprocess
import sys
import time
import urllib.request
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from tests.ui_console_errors import PageErrors

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = Path("docs/examples/separate_ui")
CADDY_ENV = "SEPARATE_UI_CADDY"
PASSWORD = "separate-ui-password"  # synthetic: this run's own network password
SYNTHETIC_PEER = "198.51.100.23"  # TEST-NET-2: the visitor address the derived proxy forwards
CERT_SUBJECT = "ouroboros-separate-ui-test"
FORWARDED_LINE = "header_up X-Forwarded-For {client_ip}"

# Never the owner's: no admin endpoint, no autosaved config, no redirect listener on :80,
# no UDP (HTTP/3) socket, loopback only, certificates and state beneath this run's root.
_GLOBAL_OPTIONS = """{{
\tadmin off
\tpersist_config off
\tauto_https disable_redirects
\tdefault_bind 127.0.0.1
\tstorage file_system {storage}
\tservers {{
\t\tprotocols h1 h2
\t}}
}}
"""


def caddy_binary() -> Path | None:
    value = os.environ.get(CADDY_ENV, "")
    path = Path(value) if value else None
    return path if path and path.is_absolute() and path.is_file() and os.access(path, os.X_OK) else None


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_port(port: int, timeout: float, alive) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not alive():
            raise RuntimeError(f"the process meant to listen on {port} exited")
        with socket.socket() as sock:
            sock.settimeout(0.5)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    raise RuntimeError(f"nothing listened on 127.0.0.1:{port} within {timeout}s")


@dataclass(frozen=True)
class Certificate:
    cert: Path
    key: Path
    spki_pin: str  # base64(sha256(SubjectPublicKeyInfo)): Chromium's per-key trust
    sha256: str  # the certificate's fingerprint, for evidence

    def ssl_context(self) -> ssl.SSLContext:
        """A client context that trusts this certificate and nothing else."""
        context = ssl.create_default_context(cafile=str(self.cert))
        context.check_hostname = True
        return context


def make_certificate(directory: Path) -> Certificate:
    """A self-signed certificate for 127.0.0.1, valid two days, made for this run only."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    directory.mkdir(parents=True, exist_ok=True)
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, CERT_SUBJECT)])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(minutes=5)).not_valid_after(now + datetime.timedelta(days=2))
            .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]),
                           critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
            .sign(key, hashes.SHA256()))
    cert_path, key_path = directory / "cert.pem", directory / "key.pem"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                           serialization.NoEncryption()))
    key_path.chmod(0o600)
    spki = cert.public_key().public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
    return Certificate(cert_path, key_path, base64.b64encode(hashlib.sha256(spki).digest()).decode("ascii"),
                       cert.fingerprint(hashes.SHA256()).hex())


class OwnedProcess:
    """One process this run started, in its own container, its output appended to one file."""

    def __init__(self, name: str, argv: list[str], *, cwd: Path, env: dict, log_path: Path):
        self.name, self.argv, self.cwd, self.env, self.log_path = name, argv, cwd, env, log_path
        self.proc = self._container = self._log = None
        self.exits: list[int | None] = []
        self.pids: list[int] = []

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self) -> None:
        from ouroboros.process_containment import ProcessContainer

        if self._container is not None:
            raise RuntimeError(f"{self.name} is already started")
        self._container = ProcessContainer()
        self._log = self.log_path.open("ab")
        self.proc = self._container.spawn(self.argv, cwd=self.cwd, env=self.env,
                                          stdout=self._log, stderr=subprocess.STDOUT)
        self.pids.append(self.proc.pid)

    def wait_exit(self, timeout: float) -> int:
        """The exit status of a process that ends on its own (Panic), still in custody."""
        return self.proc.wait(timeout=timeout)

    def stop(self, timeout: float = 20) -> int | None:
        proc, container, stream = self.proc, self._container, self._log
        self.proc = self._container = self._log = None
        if container is None:
            return None
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    pass  # The reap below sweeps what is left and names it.
        finally:
            try:
                error = container.reap()
            finally:
                container.close()
                stream.close()
        if error:
            raise RuntimeError(f"{self.name}: process cleanup failed: {error}")
        code = proc.wait(timeout=5)
        self.exits.append(code)
        return code

    def log_text(self) -> str:
        return self.log_path.read_text(encoding="utf-8", errors="replace") if self.log_path.exists() else ""


class Caddy(OwnedProcess):
    """A Caddy process for one of the example's site files, under this run's global options."""

    def __init__(self, name: str, binary: Path, root: Path, *, site: Path, port: int, variables: dict):
        root.mkdir(parents=True, exist_ok=True)
        config = root / f"{name}.Caddyfile"
        config.write_text(_GLOBAL_OPTIONS.format(storage=root / "caddy-storage") + f"import {site}\n",
                          encoding="utf-8")
        home = root / "caddy-home"
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(home),
               "XDG_DATA_HOME": str(home / "data"), "XDG_CONFIG_HOME": str(home / "config"),
               **{key: str(value) for key, value in variables.items()}}
        super().__init__(name, [str(binary), "run", "--config", str(config), "--adapter", "caddyfile"],
                         cwd=root, env=env, log_path=root / f"{name}.log")
        self.port, self.config, self.site = port, config, site

    def start(self) -> None:
        super().start()
        try:
            _wait_port(self.port, 20, lambda: self.running)
        except RuntimeError as exc:
            raise RuntimeError(f"{self.name}: {exc}\n{self.log_text()[-3000:]}") from exc

    def access(self) -> list[dict]:
        """Every access-log record so far: one per request this Caddy answered."""
        records = []
        for line in self.log_text().splitlines():
            if line.startswith("{"):
                row = json.loads(line)
                if str(row.get("logger", "")).startswith("http.log.access"):
                    records.append(row)
        return records


def _owner_text(payload: dict) -> str:
    parts = []
    for message in payload.get("messages") or []:
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            parts.extend([content] if isinstance(content, str) else
                         [c.get("text", "") for c in content or [] if isinstance(c, dict)])
    return "\n".join(parts)


class ModelStub:
    """The cores' OpenAI-compatible provider on loopback. Every call gets "OK" (streamed when asked),
    ``GET /v1/models`` lists one model, and two levers act on main calls (those carrying tools) whose
    owner text contains a marker: ``hold`` keeps them waiting until ``release``; ``ask`` makes the
    first one call the ``escalate`` tool with the given arguments. No key is checked."""

    def __init__(self):
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        self.lock, self.released = threading.Lock(), threading.Event()
        self.calls: list[dict] = []
        self.holds: dict[str, object] = {}
        self.asks: dict[str, dict] = {}
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(handler):  # noqa: N802, N805 - stdlib callback
                if handler.path.rstrip("/") != "/v1/models":
                    handler.send_error(404)
                    return
                stub._send(handler, json.dumps({"object": "list", "data": [{"id": "mock-model", "object": "model"}]}),
                           "application/json")

            def do_POST(handler):  # noqa: N802, N805
                stub._handle(handler)

            def log_message(handler, *_args):  # noqa: N805
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, name="model-stub", daemon=True)
        self.thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def hold(self, marker: str):
        import threading

        entered = threading.Event()
        self.holds[marker] = entered
        return entered

    def ask(self, marker: str, arguments: dict) -> None:
        self.asks[marker] = dict(arguments)

    def release(self) -> None:
        self.released.set()

    def _handle(self, handler) -> None:
        payload = json.loads(handler.rfile.read(int(handler.headers.get("Content-Length", 0) or 0)) or b"{}")
        text, main = _owner_text(payload), bool(payload.get("tools"))
        message, finish = {"role": "assistant", "content": "OK"}, "stop"
        with self.lock:
            marker = next((m for m in [*self.asks, *self.holds] if m in text), "")
            self.calls.append({"marker": marker, "main": main, "stream": payload.get("stream") is True})
            arguments = self.asks.pop(marker, None) if main and marker in self.asks else None
        if arguments is not None:
            message, finish = {"role": "assistant", "content": "", "tool_calls": [{
                "id": f"call_{len(self.calls)}", "type": "function",
                "function": {"name": "escalate", "arguments": json.dumps(arguments)}}]}, "tool_calls"
        elif main and marker in self.holds:
            self.holds[marker].set()
            self.released.wait(180)
        streaming = payload.get("stream") is True
        if streaming and message.get("tool_calls"):
            message = {**message, "tool_calls": [{**c, "index": i} for i, c in enumerate(message["tool_calls"])]}
        answer = {"id": "separate-ui-stub", "object": "chat.completion", "model": "mock-model",
                  "choices": [{"index": 0, "finish_reason": finish, ("delta" if streaming else "message"): message}],
                  "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
        if streaming:
            self._send(handler, f"data: {json.dumps(answer)}\n\ndata: [DONE]\n\n", "text/event-stream")
        else:
            self._send(handler, json.dumps(answer), "application/json")

    @staticmethod
    def _send(handler, text: str, content_type: str) -> None:
        body = text.encode("utf-8")
        try:
            handler.send_response(200)
            handler.send_header("Content-Type", content_type)
            handler.send_header("Content-Length", str(len(body)))
            handler.end_headers()
            handler.wfile.write(body)
        except OSError:
            pass  # the calling worker was stopped while this call was held

    def close(self) -> None:
        self.released.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(10)


def listening_sockets(pid: int) -> list[str] | None:
    """``lsof``'s view of every internet socket the process holds; None when lsof is unavailable."""
    lsof = shutil.which("lsof") or ("/usr/sbin/lsof" if Path("/usr/sbin/lsof").exists() else None)
    if not lsof:
        return None
    result = subprocess.run([lsof, "-nP", "-a", "-p", str(pid), "-i"], capture_output=True, text=True, timeout=20)
    return [line for line in result.stdout.splitlines()[1:] if line.strip()]


def _fetch_json(url: str, timeout: float = 5) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - loopback core
        return json.loads(response.read().decode("utf-8"))


class Core:
    """The real ``server.py`` of a candidate checkout: started, stopped and restarted on request."""

    def __init__(self, root: Path, checkout: Path, candidate, env: dict, port: int, log_path: Path,
                 model: ModelStub):
        self.root, self.checkout, self.candidate, self.env, self.port = root, checkout, candidate, env, port
        self.data_dir = Path(env["OUROBOROS_DATA_DIR"])
        self.url = f"http://127.0.0.1:{port}"
        self.model = model
        self.process = OwnedProcess("core", [_python(), "server.py"], cwd=checkout, env=env, log_path=log_path)

    @property
    def pid(self) -> int | None:
        return self.process.proc.pid if self.process.proc is not None else None

    def start(self, *, configured: bool = True) -> None:
        """Start and prove the answering process is this checkout's; an unconfigured core has no workers."""
        from tests.candidate_checkout import verify_checkout
        from tests.test_ui_smoke_playwright import _wait_health, _wait_supervisor_ready

        verify_checkout(self.checkout, self.candidate)
        self.candidate.hold()  # pinned before the spawn; released only once the tree is proven gone
        self.process.start()
        _wait_health(self.url, timeout_sec=60)
        _wait_supervisor_ready(self.url, timeout_sec=90)
        if configured:
            self.prove_origin()
        else:  # the static and Python origin proofs, without the worker-pool requirement
            with urllib.request.urlopen(self.url + "/onboarding/static/candidate-sentinel.txt", timeout=5) as r:  # noqa: S310
                assert r.read() == self.candidate.sentinel_bytes
            assert _fetch_json(self.url + "/api/health")["runtime_version"] == self.candidate.version_text

    def prove_origin(self) -> None:
        from tests.candidate_checkout import assert_served_candidate

        assert_served_candidate(self.url, self.checkout, self.data_dir, self.process.proc.pid, self.candidate)

    def stop(self) -> int | None:
        if self.process._container is None:
            return None
        code = self.process.stop()
        self.candidate.release()
        return code

    def restart(self) -> None:
        self.stop()
        self.start()

    def wait_ready(self, timeout: float = 90) -> dict:
        from tests.test_ui_smoke_playwright import _wait_health, _wait_supervisor_ready

        _wait_health(self.url, timeout_sec=int(timeout))
        _wait_supervisor_ready(self.url, timeout_sec=int(timeout))
        return _fetch_json(self.url + "/api/state")

    def boots(self) -> int:
        """How many times a server generation started listening in this core's log."""
        return self.process.log_text().count("Starting Ouroboros server on")

    def jsonl(self, relative: str) -> list[dict]:
        """Core log rows: dialogue from SQL, other streams from their JSONL records."""
        path = self.data_dir / relative
        if relative in {"logs/chat.jsonl", "logs/progress.jsonl"}:
            from tests._dialogue import log_rows

            return log_rows(path)
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def task_for_message(self, client_message_id: str) -> dict | None:
        """The task the core received for one owner message (``task_received`` in events.jsonl)."""
        for row in self.jsonl("logs/events.jsonl"):
            task = row.get("task") if row.get("type") == "task_received" else None
            origin = ((task or {}).get("metadata") or {}).get("origin_message_ref") or {}
            if origin.get("client_message_id") == client_message_id:
                return task
        return None


def _python() -> str:
    return str(getattr(sys, "_base_executable", sys.executable)) if os.name == "nt" else sys.executable


@contextmanager
def real_core(root: Path, *, provider: bool = True, password: str = PASSWORD, settings: dict | None = None,
              ui_dir: Path | str = "none"):
    """A real core; main UI defaults off, and ``provider=False`` leaves settings absent.

    ``ui_dir`` lets a client-entry test exercise selected files through this same core instead
    of mistaking separately hosted files for the core's selected setup/sign-in pages.
    """
    from ouroboros.test_environment import isolated_environment
    from tests.candidate_checkout import candidate_checkout, require_candidate_interpreter

    # ``root`` itself holds the isolated home, data and tmp: macOS caps a socket path at 104
    # bytes, and the core's worker manager listens beneath TMPDIR.
    root.mkdir(parents=True, exist_ok=True)
    checkout = root / "repo"
    with ExitStack() as stack:
        candidate = stack.enter_context(candidate_checkout(REPO_ROOT, checkout, origin_proof=True))
        env = isolated_environment(root, checkout)
        data_dir = Path(env["OUROBOROS_DATA_DIR"])
        values = dict(settings or {})
        model = ModelStub()  # a fresh install points its setup at it
        stack.callback(model.close)
        if provider:
            values.update(dict.fromkeys(("OUROBOROS_MODEL", "OUROBOROS_MODEL_LIGHT", "OUROBOROS_MODEL_FALLBACKS"),
                                        "openai-compatible::mock-model"))
            values.update(OPENAI_COMPATIBLE_API_KEY="separate-ui-key", OPENAI_COMPATIBLE_BASE_URL=model.base_url,
                          OUROBOROS_MAX_WORKERS=1, OUROBOROS_RUNTIME_MODE="light",
                          # a fresh root with a ready provider would clone the skill hub at boot (#1995)
                          OUROBOROS_STARTER_SKILLS="[]")
        if values:
            (data_dir / "settings.json").write_text(json.dumps(values), encoding="utf-8")
        port = free_port()
        files_root = root / "files"
        files_root.mkdir(exist_ok=True)
        env.update({
            "OUROBOROS_SERVER_HOST": "127.0.0.1", "OUROBOROS_SERVER_PORT": str(port),
            "OUROBOROS_HOST_SERVICE_PORT": str(free_port()), "OUROBOROS_NETWORK_PASSWORD": password,
            "OUROBOROS_MAX_WORKERS": "1",  # fresh setup also keeps this fixture's pool bounded
            # The fixture's explicit choice, never silently persisted by a read.
            "OUROBOROS_UI_DIR": str(ui_dir),
            # A Restart must not reset this copy to a managed branch (no managed remote exists).
            "OUROBOROS_DISABLE_MANAGED_UPDATES": "1",
            # A visitor through the proxy is not loopback: Files needs the operator's root.
            "OUROBOROS_FILE_BROWSER_DEFAULT": str(files_root),
        })
        require_candidate_interpreter(_python(), env, checkout)
        core = Core(root, checkout, candidate, env, port, root / "core.log", model)
        try:
            core.start(configured=provider)
            yield core
        finally:
            core.stop()


def copy_interface(source: Path, target: Path) -> dict[str, str]:
    """Copy the static interface files only; return their sha256 manifest."""
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("tests", "node_modules", "package*.json", "*.py"))
    return {path.relative_to(target).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(target.rglob("*")) if path.is_file()}


@dataclass
class Stack:
    core: Core
    certificate: Certificate
    interface: Caddy
    proxy: Caddy
    interface_root: Path
    manifest: dict
    root: Path
    verbatim: bool
    extra: dict = field(default_factory=dict)

    @property
    def url(self) -> str:
        return f"https://127.0.0.1:{self.proxy.port}"

    def role(self, record: dict) -> str:
        upstream = record.get("upstream") or ""
        return {f"127.0.0.1:{self.core.port}": "core",
                f"127.0.0.1:{self.interface.port}": "interface"}.get(upstream, upstream or "proxy")

    def routed(self) -> list[dict]:
        """The proxy's access log as (method, uri, status, upstream role, protocol) rows."""
        return [{"method": r["request"]["method"], "uri": r["request"]["uri"], "status": r.get("status"),
                 "role": self.role(r), "proto": r["request"].get("proto"),
                 "websocket": (r["request"].get("headers", {}).get("Upgrade") or [""])[0].lower() == "websocket"}
                for r in self.proxy.access()]

    def interface_paths(self) -> list[str]:
        return [r["request"]["uri"] for r in self.interface.access()]

    def stop(self) -> None:
        errors = []
        for process in (self.proxy, self.interface):
            try:
                process.stop()
            except Exception as exc:  # keep stopping the rest; report every failure
                errors.append(str(exc))
        if errors:
            raise RuntimeError("; ".join(errors))


def derived_site(example: Path, target: Path, *, verbatim: bool) -> Path:
    """The example's front-proxy file, with the synthetic visitor unless ``verbatim``."""
    text = example.read_text(encoding="utf-8")
    if not verbatim:
        assert text.count(FORWARDED_LINE) == 1, "the example must overwrite X-Forwarded-For exactly once"
        text = text.replace(FORWARDED_LINE, f"header_up X-Forwarded-For {SYNTHETIC_PEER}")
    target.write_text(text, encoding="utf-8")
    return target


@contextmanager
def separate_stack(core: Core, root: Path, *, verbatim: bool = False):
    """An interface host and a TLS front proxy in front of ``core``, both stopped on exit."""
    binary = caddy_binary()
    if binary is None:
        raise RuntimeError(f"set {CADDY_ENV} to an absolute Caddy v2 binary")
    root.mkdir(parents=True, exist_ok=True)
    certificate = make_certificate(root / "tls")
    interface_root = root / "interface-web"
    manifest = copy_interface(core.checkout / "web", interface_root)
    example = core.checkout / EXAMPLE  # the example bytes of this same candidate
    interface_port, proxy_port = free_port(), free_port()
    interface = Caddy("interface", binary, root / "interface", site=example / "interface.Caddyfile", port=interface_port,
                      variables={"OUROBOROS_INTERFACE_PORT": interface_port, "OUROBOROS_INTERFACE_ROOT": interface_root})
    proxy = Caddy("proxy", binary, root / "proxy", port=proxy_port,
                  site=derived_site(example / "Caddyfile", root / "proxy-site.Caddyfile", verbatim=verbatim),
                  variables={"OUROBOROS_SITE": f"https://127.0.0.1:{proxy_port}",
                             "OUROBOROS_TLS_CERT": certificate.cert, "OUROBOROS_TLS_KEY": certificate.key,
                             "OUROBOROS_CORE_UPSTREAM": f"127.0.0.1:{core.port}",
                             "OUROBOROS_INTERFACE_UPSTREAM": f"127.0.0.1:{interface_port}"})
    stack = Stack(core, certificate, interface, proxy, interface_root, manifest, root, verbatim)
    try:
        interface.start()
        proxy.start()
        yield stack
    finally:
        stack.stop()


def launch_browser(playwright, name: str, certificate: Certificate):
    """A headless browser trusting this run's certificate in this browser only; and its context options."""
    if name == "chromium":
        return playwright.chromium.launch(
            headless=True, args=[f"--ignore-certificate-errors-spki-list={certificate.spki_pin}"]), {}
    return getattr(playwright, name).launch(headless=True), {"ignore_https_errors": True}


PROBE = "separate_ui_probe"
_PROBE_PLUGIN = """\
from pathlib import Path

from starlette.responses import JSONResponse, Response

HITS = {"data": 0, "disposed": 0}


def data(_request):
    HITS["data"] += 1
    return JSONResponse({"probe": "separate-ui"}, headers={"Cache-Control": "no-store"})


def logo(_request):
    return Response(Path(__file__).with_name("logo.png").read_bytes(), media_type="image/png")


def lib(_request):
    return Response("export const value = 'blob-module-ok';", media_type="text/javascript")


def disposed(_request):
    HITS["disposed"] += 1
    return JSONResponse({"ok": True})


def hits(_request):
    return JSONResponse(dict(HITS), headers={"Cache-Control": "no-store"})


def register(api):
    for name, handler in (("data", data), ("logo", logo), ("lib", lib), ("hits", hits)):
        api.register_route(name, handler, methods=("GET",))
    api.register_route("disposed", disposed, methods=("POST",))
    api.register_ui_tab("probe", "Separate host probe", render={"kind": "module", "entry": "widget.js", "start": "auto"})
    api.register_ui_tab("fault", "Deliberate fault", render={"kind": "module", "entry": "fault.js", "start": "auto"})
"""
_PROBE_WIDGET = """\
(async () => {
    const root = document.getElementById('root');
    const base = '/api/extensions/separate_ui_probe/';
    const mark = (key, value) => { root.dataset[key] = String(value); };
    root.innerHTML = '<h2>Separate host probe</h2><p data-line></p><img data-blob alt=""><img data-passive alt="">';
    window.__ouroWidgetOnDispose?.(async () => { await OuroborosWidget.fetch(base + 'disposed', {method: 'POST'}); });
    const line = (text) => { root.querySelector('[data-line]').textContent += text + ' '; };
    try {
        const body = await (await OuroborosWidget.fetch(base + 'data')).json();
        mark('data', body.probe); line('bridge fetch: ' + body.probe + ';');
        const source = await (await OuroborosWidget.fetch(base + 'lib')).text();
        const url = URL.createObjectURL(new Blob([source], {type: 'text/javascript'}));
        const loaded = await import(url);
        mark('module', loaded.value); line('blob module: ' + loaded.value + ';');
        const image = root.querySelector('[data-blob]');
        image.onload = () => { mark('blobImage', image.naturalWidth); line('blob image loaded;'); };
        image.onerror = () => mark('blobImage', 'error');
        image.src = URL.createObjectURL(await (await OuroborosWidget.fetch(base + 'logo')).blob());
    } catch (error) { mark('failure', error.message); }
    const passive = root.querySelector('[data-passive]');
    passive.onload = () => mark('passive', 'loaded');
    passive.onerror = () => { mark('passive', 'refused'); line('passive image refused (no session in the frame);'); };
    passive.src = base + 'logo?passive=1';
})();
"""
_PNG_1X1 = bytes.fromhex("89504e470d0a1a0a0000000d4948445200000001000000010806000000"
                         "1f15c4890000000d49444154789c63f8cfc0f01f0005000201a3e0f50500"
                         "00000049454e44ae426082")


def install_skill(core, name: str, files: dict) -> None:
    """A reviewed extension in the core's data root, the way an approved install leaves it (no paid review)."""
    from ouroboros.skill_loader import SkillReviewState, compute_content_hash, save_review_state

    skill = core.data_dir / "skills" / "external" / name
    skill.mkdir(parents=True, exist_ok=True)
    for relative, content in files.items():
        (skill / relative).write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))
    digest = compute_content_hash(skill, manifest_entry="plugin.py")
    save_review_state(core.data_dir, name, SkillReviewState(status="pass", content_hash=digest))


def install_widgets(core) -> None:
    """The published author-kit example and the probe, reviewed and in place (each still needs enabling)."""
    kit = core.checkout / "docs" / "examples" / "author_ui_kit"
    install_skill(core, "author_ui_kit", {f.name: f.read_bytes() for f in kit.iterdir() if f.is_file()})
    install_skill(core, PROBE, {
        "SKILL.md": f"---\nname: {PROBE}\ndescription: Separate host widget probe.\nversion: 0.1.0\n"
                    "type: extension\nentry: plugin.py\npermissions: [\"route\", \"widget\"]\n---\n# Probe\n",
        "plugin.py": _PROBE_PLUGIN, "widget.js": _PROBE_WIDGET, "logo.png": _PNG_1X1,
        "fault.js": "throw new Error('separate-ui deliberate fault');\n"})


# Every top-level document this tab loads, recorded before the page can rewrite its address.
_RECORD_LOADS = """(() => { if (window.top !== window) return;
    const loads = JSON.parse(sessionStorage.getItem('__loads') || '[]');
    sessionStorage.setItem('__loads', JSON.stringify([...loads, location.href])); })()"""
CONNECTED = "() => window.__ouroWs?.isConnected?.() === true"
DISCONNECTED = "() => window.__ouroWs?.isConnected?.() === false"


class BrowserSession:
    """One headless browser and tab on the proxy origin, with its page errors and document loads."""

    def __init__(self, playwright, stack, name: str, evidence: Path, *, viewport=(1400, 900)):
        self.stack, self.name, self.evidence = stack, name, evidence
        self.browser, options = launch_browser(playwright, name, stack.certificate)
        self.context = self.browser.new_context(viewport={"width": viewport[0], "height": viewport[1]}, **options)
        self.context.add_init_script(_RECORD_LOADS)
        self.page = self.context.new_page()
        self.errors = PageErrors(self.page, name)

    def loads(self) -> list[str]:
        return self.page.evaluate("() => JSON.parse(sessionStorage.getItem('__loads') || '[]')")

    def shot(self, label: str, **options) -> Path:
        path = self.evidence / f"{self.name}-{label}.png"
        self.page.screenshot(path=str(path), **options)
        return path

    def sign_in(self, password: str = PASSWORD) -> None:
        self.page.get_by_label("Password", exact=True).fill(password)
        self.page.get_by_role("button", name="Unlock", exact=True).click()

    def open_signed_in(self) -> None:
        """Empty cookie jar → the interface's page → the core's sign-in → back, connected."""
        self.page.goto(self.stack.url + "/")
        self.page.wait_for_url(lambda address: address.split("?")[0].endswith("/auth/login"), timeout=30_000)
        self.sign_in()
        self.page.wait_for_url(self.stack.url + "/", timeout=30_000)
        self.page.wait_for_function(CONNECTED, timeout=60_000)

    def api(self, method: str, path: str, body=None):
        """A same-origin request from the page itself: its cookie, through the proxy."""
        return self.page.evaluate("""async ([method, path, body]) => {
            const response = await fetch(path, {method, cache: 'no-store', headers: body === null ? {} :
                {'Content-Type': 'application/json'}, body: body === null ? undefined : JSON.stringify(body)});
            const text = await response.text();
            let json = null; try { json = JSON.parse(text); } catch (_) {}
            return {status: response.status, json, text: json === null ? text.slice(0, 2000) : ''};
        }""", [method, path, body])

    def fetch_bytes(self, path: str, headers: dict | None = None) -> dict:
        """Fetch from the page (its cookie and TLS trust): status, headers, sha256 and first bytes."""
        return self.page.evaluate("""async ([path, headers]) => {
            const response = await fetch(path, {cache: 'no-store', headers});
            const bytes = new Uint8Array(await response.arrayBuffer());
            const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
            return {status: response.status, size: bytes.length, head: [...bytes.slice(0, 16)],
                    sha256: [...digest].map(b => b.toString(16).padStart(2, '0')).join(''),
                    headers: Object.fromEntries(response.headers.entries())};
        }""", [path, headers or {}])

    def unexpected_errors(self) -> list[str]:
        return self.errors.unexpected()

    def close(self) -> None:
        self.browser.close()


def wait_until(predicate, timeout: float, message: str, interval: float = 0.25):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    raise AssertionError(message)


def evidence_dir(name: str, fallback: Path) -> Path:
    path = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR") or fallback) / "separate-ui" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def main() -> int:
    """Start the stack for a person to browse; stop every process on Ctrl-C or SIGTERM.

    Run it under the repository's environment boundary, e.g.
    ``SEPARATE_UI_CADDY=/abs/caddy PLAYWRIGHT_BROWSERS_PATH=... python -I -S scripts/safe_test.py
    --temp-parent /tmp/x -- <venv>/bin/python -m tests.separate_ui_support``.
    """
    import argparse
    import tempfile

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--root", type=Path, default=None, help="where the run's files go (kept)")
    parser.add_argument("--fresh", action="store_true", help="a core with no provider (setup needed)")
    parser.add_argument("--no-widgets", action="store_true", help="skip the author kit and the probe widget")
    args = parser.parse_args()
    # Short: the core's worker manager listens on a socket beneath root/tmp (macOS caps the path at 104 bytes).
    root = (args.root or Path(tempfile.mkdtemp(prefix="sui-"))).resolve()
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(True))
    with real_core(root, provider=not args.fresh) as core, separate_stack(core, root / "stack") as stack:
        if not args.no_widgets:
            install_widgets(core)
            for name in ("author_ui_kit", PROBE):  # enabled as the Skills toggle does, over loopback
                request = urllib.request.Request(f"{core.url}/api/skills/{name}/toggle", method="POST",
                                                 data=b'{"enabled": true}', headers={"Content-Type": "application/json"})
                urllib.request.urlopen(request, timeout=60).close()  # noqa: S310 - loopback core
        print(json.dumps({
            "url": stack.url + "/", "password": PASSWORD, "certificate": str(stack.certificate.cert),
            "chromium_trust_flag": f"--ignore-certificate-errors-spki-list={stack.certificate.spki_pin}",
            "core_direct": core.url, "core_log": str(core.process.log_path),
            "proxy_log": str(stack.proxy.log_path), "interface_log": str(stack.interface.log_path),
            "visitor_forwarded_as": SYNTHETIC_PEER, "root": str(root),
        }, indent=2), flush=True)
        try:
            while not stop:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
    print(f"stopped; files kept under {root}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
