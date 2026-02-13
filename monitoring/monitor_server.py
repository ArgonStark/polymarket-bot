"""
WebSocket monitor server — real-time data hub for trading bot dashboards.

Usage inside bot:
    from monitoring.monitor_server import MonitorServer
    monitor = MonitorServer(port=8765)
    monitor.start()
    monitor.push({"symbol": "BTC", "price": 67145.49, ...})

Usage standalone (serves web dashboard):
    python monitoring/monitor_server.py
"""

import asyncio
import json
import logging
import os
import threading
from typing import Optional

import websockets
from aiohttp import web

logger = logging.getLogger(__name__)

_HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web_dashboard.html")


class MonitorServer:
    """
    Lightweight WebSocket server that broadcasts JSON snapshots
    to all connected dashboard clients.

    Runs its own asyncio event loop in a daemon thread so it
    never blocks the caller.
    """

    def __init__(self, port: int = 8765, http_port: Optional[int] = None):
        self.ws_port = port
        self.http_port = http_port or port + 1
        self._clients: set[websockets.WebSocketServerProtocol] = set()
        self._last_snapshot: Optional[str] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ws_server = None
        self._http_runner = None

    # ── public API (thread-safe) ─────────────────────────────────

    def start(self):
        """Start the WebSocket + HTTP servers in a background daemon thread."""
        if self._thread and self._thread.is_alive():
            logger.warning("MonitorServer already running")
            return
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="monitor-server")
        self._thread.start()
        logger.info(
            "MonitorServer started — ws://0.0.0.0:%d  http://0.0.0.0:%d",
            self.ws_port, self.http_port,
        )

    def stop(self):
        """Shut down servers and join the background thread."""
        if self._loop and self._loop.is_running():
            # Schedule graceful cleanup before stopping the loop
            future = asyncio.run_coroutine_threadsafe(self._cleanup(), self._loop)
            try:
                future.result(timeout=3)
            except Exception:
                pass
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("MonitorServer stopped")

    async def _cleanup(self):
        """Gracefully close WS and HTTP servers before loop stops."""
        # Close all WebSocket client connections
        for ws in list(self._clients):
            try:
                await ws.close()
            except Exception:
                pass
        self._clients.clear()

        # Close WebSocket server
        if self._ws_server:
            self._ws_server.close()
            try:
                await self._ws_server.wait_closed()
            except Exception:
                pass

        # Clean up HTTP runner
        if self._http_runner:
            try:
                await self._http_runner.cleanup()
            except Exception:
                pass

    def push(self, data: dict):
        """
        Broadcast a data snapshot to all connected clients.

        Thread-safe — can be called from any thread.
        """
        payload = json.dumps(data)
        self._last_snapshot = payload
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._broadcast(payload), self._loop)

    # ── internals ────────────────────────────────────────────────

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())
        self._loop.run_forever()

    async def _serve(self):
        # Start WebSocket server
        self._ws_server = await websockets.serve(
            self._ws_handler,
            "0.0.0.0",
            self.ws_port,
            ping_interval=20,
            ping_timeout=10,
        )

        # Start HTTP server for web dashboard
        app = web.Application()
        app.router.add_get("/", self._http_dashboard)
        app.router.add_get("/health", self._http_health)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", self.http_port)
        await site.start()
        self._http_runner = runner

    async def _ws_handler(self, ws: websockets.WebSocketServerProtocol, path: str = "/"):
        """Handle a new WebSocket client connection."""
        self._clients.add(ws)
        remote = ws.remote_address
        logger.info("Dashboard client connected: %s (%d total)", remote, len(self._clients))

        # Send last known snapshot immediately so the client isn't blank
        if self._last_snapshot:
            try:
                await ws.send(self._last_snapshot)
            except Exception:
                pass

        try:
            async for _ in ws:
                pass  # We don't expect messages from clients
        except (websockets.ConnectionClosed, RuntimeError, GeneratorExit):
            pass
        except Exception:
            pass
        finally:
            self._clients.discard(ws)

    async def _broadcast(self, payload: str):
        """Send payload to all connected clients, dropping broken ones."""
        if not self._clients:
            return
        stale = set()
        for ws in list(self._clients):
            try:
                await ws.send(payload)
            except (websockets.ConnectionClosed, Exception):
                stale.add(ws)
        self._clients -= stale

    async def _http_dashboard(self, request: web.Request) -> web.Response:
        """Serve the web dashboard HTML file."""
        if os.path.exists(_HTML_PATH):
            with open(_HTML_PATH, "r") as f:
                html = f.read()
            # Inject the WebSocket port so the dashboard auto-connects
            html = html.replace("{{WS_PORT}}", str(self.ws_port))
            return web.Response(text=html, content_type="text/html")
        return web.Response(text="web_dashboard.html not found", status=404)

    async def _http_health(self, request: web.Request) -> web.Response:
        return web.Response(
            text=json.dumps({
                "status": "ok",
                "clients": len(self._clients),
                "has_data": self._last_snapshot is not None,
            }),
            content_type="application/json",
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    server = MonitorServer(port=8765)
    server.start()
    print(f"\n  WebSocket:  ws://localhost:{server.ws_port}")
    print(f"  Dashboard:  http://localhost:{server.http_port}")
    print(f"\n  Waiting for data pushes... (Ctrl+C to quit)\n")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        server.stop()
