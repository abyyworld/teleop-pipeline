"""A local desktop app for the pipeline: one binary, one window, no terminal.

Serves a single page over loopback and opens it in the default browser. The UI
is a browser tab rather than a native toolkit for three reasons: it adds no
dependency (the whole server is standard library), it looks the same on macOS,
Windows and Linux, and it survives being bundled by PyInstaller, which a native
toolkit does not reliably do.

Security, because this process can start a training run and anything a browser
can reach, a web page you happen to be visiting can also try to reach:

* The socket binds to 127.0.0.1, never to a routable address.
* Every request must carry the session token minted at startup. The token is in
  the URL the app opens and nowhere else, so a page that merely guesses the port
  cannot drive the pipeline.
* The Host header must name loopback, which blocks DNS rebinding: an attacker
  pointing their own hostname at 127.0.0.1 is rejected before routing.
* Only two paths serve files, and neither takes a path from the request, so
  there is no traversal surface.
"""

from __future__ import annotations

import http.server
import json
import secrets
import socket
import threading
import webbrowser
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from ..config import Config
from . import pipeline as pipe

HERE = Path(__file__).resolve().parent
INDEX = HERE / "index.html"

#: Bodies larger than this are refused rather than buffered. The API takes
#: small JSON objects; anything bigger is a mistake or an attack.
MAX_BODY = 64 * 1024

ALLOWED_HOSTS = {"localhost", "127.0.0.1", "[::1]", "::1"}


def _host_is_loopback(header: str | None) -> bool:
    if not header:
        return False
    host = header.rsplit(":", 1)[0] if header.count(":") == 1 else header
    if header.startswith("["):  # bracketed IPv6, possibly with a port
        host = header.split("]")[0] + "]"
    return host.lower() in ALLOWED_HOSTS


def _settings(cfg: Config) -> dict[str, Any]:
    """The values worth seeing next to a result, read straight from params.yaml."""
    return {
        "params_path": str(cfg.path),
        "robot": {
            "name": cfg.get("robot.name"),
            "n_joints": cfg.n_joints,
            "control_hz": cfg.control_hz,
            "action_limit": cfg.action_limit,
        },
        "train": {
            "epochs": cfg.get("train.epochs", 40),
            "action_horizon": cfg.get("train.action_horizon", 1),
            "batch_size": cfg.get("train.batch_size", 256),
            "lr": cfg.get("train.lr", 3e-4),
            "hidden_sizes": cfg.get("train.hidden_sizes", [512, 512]),
            "device": cfg.get("train.device", "auto"),
        },
    }


class Studio:
    """Shared state for the handler: one runner, one token."""

    def __init__(self, params: str | None = None, token: str | None = None):
        self.runner = pipe.Runner(params)
        self.token = token or secrets.token_urlsafe(24)

    def snapshot(self) -> dict[str, Any]:
        cfg = self.runner.config()
        outputs = pipe.stage_outputs(cfg)
        return {
            "project": {"root": str(cfg.root), "name": cfg.root.name},
            "stages": [
                {
                    "key": s.key,
                    "title": s.title,
                    "blurb": s.blurb,
                    "optional": s.optional,
                    "needs": list(s.needs),
                    "has_output": outputs[s.key].exists(),
                }
                for s in pipe.STAGES
            ],
            "default_run": list(pipe.DEFAULT_RUN),
            "runner": self.runner.state(),
            "results": pipe.results_on_disk(cfg),
            "settings": _settings(cfg),
            "torch": pipe.torch_available(),
        }


class Handler(http.server.BaseHTTPRequestHandler):
    studio: Studio  # injected by make_server
    protocol_version = "HTTP/1.1"
    server_version = "teleop-studio"
    sys_version = ""

    # -- plumbing ------------------------------------------------------------

    def log_message(self, fmt: str, *args: Any) -> None:
        """Silence the default stderr access log; the app has its own log pane."""

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # This page is not meant to be embedded, framed, or fetched by another
        # origin; the headers say so rather than relying on the token alone.
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: Any) -> None:
        self._send(code, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

    def _authorised(self, query: dict[str, list[str]]) -> bool:
        supplied = (query.get("t") or [self.headers.get("X-Studio-Token", "")])[0]
        return secrets.compare_digest(supplied, self.studio.token)

    def _guard(self, query: dict[str, list[str]]) -> bool:
        if not _host_is_loopback(self.headers.get("Host")):
            self._json(403, {"error": "this app only answers requests addressed to localhost"})
            return False
        if not self._authorised(query):
            self._json(403, {"error": "bad or missing session token"})
            return False
        return True

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            raise ValueError("request body too large")
        if length <= 0:
            return {}
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("expected a JSON object")
        return payload

    # -- routes --------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - name fixed by the base class
        url = urlparse(self.path)
        query = parse_qs(url.query)
        if not self._guard(query):
            return

        if url.path == "/":
            self._send(200, INDEX.read_bytes(), "text/html; charset=utf-8")
        elif url.path == "/api/state":
            self._json(200, self.studio.snapshot())
        elif url.path == "/api/log":
            try:
                start = int((query.get("from") or ["0"])[0])
            except ValueError:
                start = 0
            self.studio.runner.log.promote_partial()
            lines, total = self.studio.runner.log.since(start)
            self._json(200, {"lines": lines, "next": total})
        else:
            self._json(404, {"error": "no such endpoint"})

    def do_POST(self) -> None:  # noqa: N802 - name fixed by the base class
        url = urlparse(self.path)
        query = parse_qs(url.query)
        if not self._guard(query):
            return

        try:
            body = self._body()
        except ValueError as exc:
            self._json(400, {"error": str(exc)})
            return

        if url.path == "/api/run":
            # Absent means "the default run"; present but wrong is an error, and
            # an explicitly empty list means nothing was asked for. Collapsing
            # those with `or` would turn `{"stages": []}` into a full pipeline
            # run, which is the opposite of what the caller said.
            stages = body.get("stages")
            if stages is None:
                stages = list(pipe.DEFAULT_RUN)
            elif not isinstance(stages, list) or not all(isinstance(s, str) for s in stages):
                self._json(400, {"error": "stages must be a list of stage names"})
                return
            elif not stages:
                self._json(400, {"error": "no stages given"})
                return
            options = body.get("options")
            if options is None:
                options = {}
            elif not isinstance(options, dict):
                self._json(400, {"error": "options must be an object"})
                return
            try:
                self.studio.runner.start(stages, options)
            except ValueError as exc:
                self._json(400, {"error": str(exc)})
                return
            except RuntimeError as exc:
                self._json(409, {"error": str(exc)})
                return
            self._json(202, {"started": stages})
        elif url.path == "/api/cancel":
            self.studio.runner.cancel()
            self._json(202, {"cancelling": True})
        else:
            self._json(404, {"error": "no such endpoint"})


def make_server(
    port: int = 0, params: str | None = None, token: str | None = None
) -> tuple[http.server.ThreadingHTTPServer, Studio]:
    """Bind a loopback server. Port 0 asks the OS for a free one."""
    studio = Studio(params, token)
    handler = type("BoundHandler", (Handler,), {"studio": studio})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    server.daemon_threads = True
    return server, studio


def url_for(server: http.server.ThreadingHTTPServer, studio: Studio) -> str:
    host, port = server.socket.getsockname()[:2]
    return f"http://{host}:{port}/?t={studio.token}"


def serve(port: int = 8765, params: str | None = None, open_browser: bool = True) -> None:
    """Run the app until interrupted."""
    try:
        server, studio = make_server(port, params)
    except OSError as exc:
        # A second copy of the app, or something else already on the port.
        # Falling back silently to a random port would hide a running instance.
        raise SystemExit(
            f"cannot listen on 127.0.0.1:{port} ({exc}).\n"
            "Another copy of the app is probably already running. Close it, or "
            "start this one with --port 0 to pick any free port."
        ) from exc

    address = url_for(server, studio)
    print("teleop studio")
    print(f"  project  {studio.runner.config().root}")
    print(f"  open     {address}")
    print("  stop     Ctrl-C")
    if open_browser:
        threading.Thread(target=webbrowser.open, args=(address,), daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.shutdown()
        server.server_close()


def free_port() -> int:
    """An OS-assigned free loopback port, for tests."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


__all__ = ["Handler", "Studio", "free_port", "make_server", "serve", "url_for"]
