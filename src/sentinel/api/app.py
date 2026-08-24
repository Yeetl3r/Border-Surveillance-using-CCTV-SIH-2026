"""Tier 6/8 -- FastAPI application: live stream, alert channel, query, evidence.

Two channels, deliberately separate:

* **MJPEG** at ``/api/stream/{camera_id}`` carries pixels only -- the raw or
  enhanced operator frame, with no annotations burned in.
* **WebSocket** at ``/ws`` carries state: track boxes, threat, confidence,
  health, queue counters, alerts.

Overlays are drawn by the client from the WebSocket state. That is not a
cosmetic choice. Burning boxes into the video would make the evidence keyframe
and the operator view the same artefact, and would mean a rendering change
requires re-encoding video. Keeping them apart is also what makes the
dual-axis rendering in F12 a property of the display rather than of the stream.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import cv2
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

from ..alerting import as_payload
from ..config import SystemCfg, load_config
from ..pipeline import SentinelSystem
from ..query import QueryRouter

UI_DIR = Path(__file__).resolve().parent.parent / "ui"


class Hub:
    """Fan-out of pipeline events to connected WebSocket clients."""

    def __init__(self) -> None:
        self.clients: set[WebSocket] = set()
        self.loop: asyncio.AbstractEventLoop | None = None

    def attach(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop

    async def register(self, ws: WebSocket) -> None:
        await ws.accept()
        self.clients.add(ws)

    def unregister(self, ws: WebSocket) -> None:
        self.clients.discard(ws)

    async def _send(self, message: dict) -> None:
        dead = []
        payload = json.dumps(message, default=str)
        for ws in list(self.clients):
            try:
                await ws.send_text(payload)
            except (WebSocketDisconnect, RuntimeError):
                dead.append(ws)
        for ws in dead:
            self.unregister(ws)

    def publish_threadsafe(self, message: dict) -> None:
        """Called from pipeline worker threads, which are not on the event loop."""
        if self.loop is None or not self.clients:
            return
        try:
            asyncio.run_coroutine_threadsafe(self._send(message), self.loop)
        except RuntimeError:
            pass


def create_app(cfg: SystemCfg | None = None) -> FastAPI:
    cfg = cfg or load_config(".")
    app = FastAPI(title="SENTINEL", version="1.0.0", docs_url="/api/docs")
    hub = Hub()

    def on_event(kind: str, payload: dict) -> None:
        hub.publish_threadsafe({"type": kind, **payload})

    system = SentinelSystem(cfg, on_event=on_event)
    router = QueryRouter(cfg, system.repo)
    app.state.system = system
    app.state.router = router
    app.state.cfg = cfg

    if (UI_DIR / "static").exists():
        app.mount("/static", StaticFiles(directory=UI_DIR / "static"), name="static")

    # -- lifecycle -----------------------------------------------------------
    @app.on_event("startup")
    async def _startup() -> None:
        hub.attach(asyncio.get_running_loop())
        system.start()
        asyncio.create_task(_broadcast_loop())

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        system.stop()

    async def _broadcast_loop() -> None:
        """Push per-frame overlay state at a fixed rate.

        Decoupled from the video frame rate on purpose: the browser cannot
        usefully redraw faster than this, and pushing every frame would spend
        bandwidth to no visible effect.
        """
        while True:
            await asyncio.sleep(0.1)                    # 10 Hz
            if not hub.clients:
                continue
            frames = {cid: p.latest_overlay
                      for cid, p in system.pipelines.items() if p.latest_overlay}
            if frames:
                await hub._send({"type": "tick", "cameras": frames,
                                 "status": _light_status(system)})

    def _light_status(sys_: SentinelSystem) -> dict[str, Any]:
        return {
            "threat": sys_.alerts.highest_live.value,
            "live_alerts": len(sys_.alerts.live),
            "queue_drops": sys_.registry.total_dropped(),
            "counters": sys_.alerts.counters,
            "alerts": sys_.alerts.snapshot(),
        }

    # -- pages ---------------------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return (UI_DIR / "templates" / "index.html").read_text(encoding="utf-8")

    # -- video ---------------------------------------------------------------
    @app.get("/api/stream/{camera_id}")
    async def stream(camera_id: str, raw: bool = Query(False)):
        pipe = system.pipelines.get(camera_id)
        if pipe is None:
            raise HTTPException(404, f"unknown camera {camera_id}")

        def gen():
            boundary = b"--frame\r\n"
            last = 0.0
            while True:
                frame = pipe.latest_frame
                if frame is None:
                    time.sleep(0.05)
                    continue
                now = time.time()
                if now - last < 1.0 / 25.0:
                    time.sleep(0.005)
                    continue
                last = now
                w = cfg.pipeline.stream_max_width
                if frame.shape[1] > w:
                    s = w / frame.shape[1]
                    frame = cv2.resize(frame, (w, int(frame.shape[0] * s)))
                ok, buf = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, cfg.pipeline.jpeg_quality])
                if not ok:
                    continue
                yield (boundary + b"Content-Type: image/jpeg\r\n"
                       + f"Content-Length: {len(buf)}\r\n\r\n".encode()
                       + buf.tobytes() + b"\r\n")

        return StreamingResponse(
            gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    # -- websocket -----------------------------------------------------------
    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await hub.register(ws)
        try:
            await ws.send_text(json.dumps({
                "type": "hello",
                "status": system.status(),
                "zones": {cid: p.zones.overlay()
                          for cid, p in system.pipelines.items()},
                "routes": router.describe(),
            }, default=str))
            while True:
                await ws.receive_text()                # keep-alive / client pings
        except WebSocketDisconnect:
            hub.unregister(ws)
        except RuntimeError:
            hub.unregister(ws)

    # -- state ---------------------------------------------------------------
    @app.get("/api/status")
    async def status() -> dict:
        return {**system.status(), "routes": router.describe()}

    @app.get("/api/zones/{camera_id}")
    async def zones(camera_id: str) -> dict:
        pipe = system.pipelines.get(camera_id)
        if pipe is None:
            raise HTTPException(404, f"unknown camera {camera_id}")
        return pipe.zones.overlay()

    @app.get("/api/alerts")
    async def alerts(limit: int = 50, camera_id: str | None = None) -> dict:
        return {"live": system.alerts.snapshot(),
                "recent": system.repo.recent_alerts(limit, camera_id)}

    @app.get("/api/alerts/{alert_id}")
    async def alert_detail(alert_id: str) -> dict:
        row = system.repo.alert_with_evidence(alert_id)
        if row is None:
            raise HTTPException(404, "unknown alert")
        return row

    @app.get("/api/stats")
    async def stats() -> dict:
        return system.repo.stats()

    # -- query ---------------------------------------------------------------
    @app.post("/api/query")
    async def query(request: Request) -> JSONResponse:
        body = await request.json()
        text = (body.get("q") or "").strip()
        if not text:
            raise HTTPException(400, "empty query")
        force = body.get("route")
        camera_id = body.get("camera_id") or next(iter(system.pipelines), None)

        frame = None
        tracks: dict[int, Any] = {}
        pipe = system.pipelines.get(camera_id) if camera_id else None
        if pipe is not None:
            tracks = pipe.tracks
            if body.get("alert_id"):
                row = system.repo.alert_with_evidence(body["alert_id"])
                if row and row.get("keyframe_path"):
                    kf = Path(cfg.root_dir) / row["keyframe_path"]
                    if kf.exists():
                        frame = cv2.imread(str(kf))
            if frame is None:
                # Grounding runs on a keyframe, not the live stream. Falling
                # back to the current frame keeps the route usable without an
                # alert selected, and it is still a single still image.
                frame = pipe.latest_frame

        # A 3B-parameter forward pass would block the event loop for seconds.
        result = await asyncio.to_thread(router.route, text, frame, tracks, force)
        return JSONResponse(result, status_code=result.get("status", 200))

    # -- evidence ------------------------------------------------------------
    @app.get("/evidence/{path:path}")
    async def evidence(path: str):
        root = Path(cfg.root_dir).resolve()
        target = (root / path).resolve()
        # Containment check: never serve a path that escapes the project root.
        if not str(target).startswith(str(root)) or not target.exists():
            raise HTTPException(404, "not found")
        allowed = {(root / cfg.storage.evidence_dir).resolve(),
                   (root / cfg.storage.clip_dir).resolve(),
                   (root / cfg.relay.out_dir).resolve()}
        if not any(str(target).startswith(str(a)) for a in allowed):
            raise HTTPException(403, "outside the evidence store")
        return FileResponse(target)

    return app
