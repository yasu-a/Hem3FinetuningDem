"""Serve the THeatA detector-correlation workbench on this computer."""

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib.resources import files

from ._theata.workbench import defaults, simulate, simulate_uniform


HOST = "127.0.0.1"
ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
}
SIMULATIONS = {"/api/simulate": simulate, "/api/uniform-dem": simulate_uniform}


class WorkbenchHandler(BaseHTTPRequestHandler):
    """Serve the packaged page and its same-origin simulation API."""

    def _check_origin(self) -> bool:
        """Accept the workbench's loopback URL and same-origin browser requests."""
        port = self.server.server_port
        hosts = {f"{HOST}:{port}", f"localhost:{port}"}
        if port == 80:
            hosts.update((HOST, "localhost"))
        host = self.headers.get("Host")
        origin = self.headers.get("Origin")
        if host in hosts and (origin is None or origin == f"http://{host}"):
            return True
        self.send_error(403)
        return False

    def _reply(self, status: int, content_type: str, data: bytes) -> None:
        """Write a response without sending simulation data to a cache."""
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def _json_reply(self, status: int, data: dict) -> None:
        """Serialize a result or validation error as JSON."""
        self._reply(status, "application/json; charset=utf-8",
                    json.dumps(data, allow_nan=False).encode("utf-8"))

    def do_GET(self) -> None:
        """Serve the application files or its hardware defaults."""
        if not self._check_origin():
            return
        if self.path == "/api/defaults":
            self._json_reply(200, defaults())
        elif self.path in ASSETS:
            filename, content_type = ASSETS[self.path]
            asset = files("fdem") / "web_assets" / filename
            self._reply(200, content_type, asset.read_bytes())
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        """Run a QCCD hardware simulation or the uniform DEM baseline."""
        if not self._check_origin():
            return
        run = SIMULATIONS.get(self.path)
        if run is None:
            self.send_error(404)
            return
        if self.headers.get_content_type() != "application/json":
            self.send_error(415)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 1 <= size <= 524_288:
                raise ValueError("Request size must be between 1 and 524,288 bytes.")
            result = run(json.loads(self.rfile.read(size)))
        except KeyError as error:
            self._json_reply(400, {"error": f"Missing simulation parameter {error}."})
        except (ValueError, RuntimeError, TypeError, IndexError, OverflowError) as error:
            self._json_reply(400, {"error": str(error)[:2400] or "Check the simulation settings."})
        else:
            self._json_reply(200, result)


def main() -> None:
    """Start the workbench server bound to the IPv4 loopback interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765, help="Local port (default 8765).")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("Port must be between 0 and 65,535.")
    with HTTPServer((HOST, args.port), WorkbenchHandler) as server:
        print(f"Open http://{HOST}:{server.server_port}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
