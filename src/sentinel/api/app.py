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
from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Query,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
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

    @app.middleware("http")
    async def _no_store_ui(request: Request, call_next):
        """Serve the interface uncached.

        The operator console is edited in place during a build and a stale
        cached bundle looks exactly like a broken feature -- the browser runs
        yesterday's JavaScript against today's markup. The payload is a few
        tens of kilobytes on localhost, so revalidating every load costs
        nothing worth measuring.
        """
        response = await call_next(request)
        if request.url.path.startswith("/static") or request.url.path == "/":
            response.headers["Cache-Control"] = "no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
        return response

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
    async def index() -> HTMLResponse:
        html = (UI_DIR / "templates" / "index.html").read_text(encoding="utf-8")
        # Stamp the asset URLs with the files' own mtimes. A Cache-Control
        # header asks the browser to revalidate; a changed URL leaves it no
        # choice. During a build a stale bundle is indistinguishable from a
        # broken feature, so this is worth the two lines.
        for name in ("app.css", "app.js"):
            p = UI_DIR / "static" / name
            v = int(p.stat().st_mtime) if p.exists() else 0
            html = html.replace(f"/static/{name}", f"/static/{name}?v={v}")
        return HTMLResponse(html, headers={"Cache-Control": "no-store"})

    @app.get("/analysis", response_class=HTMLResponse)
    async def analysis() -> HTMLResponse:
        """The analysis screen, reached once a feed has been submitted.

        A separate page rather than a state of the landing page: the operator
        view and the pitch surface have different jobs, and mixing them means
        the thing you look at under pressure carries the marketing chrome.
        """
        html = (UI_DIR / "templates" / "analysis.html").read_text(encoding="utf-8")
        for name in ("app.css", "analysis.css", "analysis.js"):
            p = UI_DIR / "static" / name
            v = int(p.stat().st_mtime) if p.exists() else 0
            html = html.replace(f"/static/{name}", f"/static/{name}?v={v}")
        return HTMLResponse(html, headers={"Cache-Control": "no-store"})

    @app.get("/demos", response_class=HTMLResponse)
    async def demos_page() -> HTMLResponse:
        """The scripted demonstrations, one per screen.

        Separate from /analysis because the jobs differ: analysis is a free-run
        operator surface pointed at whatever you hand it, while this is a fixed
        sequence with a claim attached to each step.
        """
        html = (UI_DIR / "templates" / "demos.html").read_text(encoding="utf-8")
        for name in ("app.css", "analysis.css", "demos.css", "demos.js"):
            f = UI_DIR / "static" / name
            v = int(f.stat().st_mtime) if f.exists() else 0
            html = html.replace(f"/static/{name}", f"/static/{name}?v={v}")
        return HTMLResponse(html, headers={"Cache-Control": "no-store"})

    # -- video ---------------------------------------------------------------
    @app.get("/api/stream/{camera_id}")
    async def stream(camera_id: str, request: Request, raw: bool = Query(False)):
        """MJPEG for one camera.

        Async on purpose, and the reason is worth recording because the sync
        version deadlocked the whole server.

        A multipart/x-mixed-replace response never ends, so each open feed
        holds one connection for as long as it is on screen. Starlette runs a
        SYNC generator on a threadpool worker and can only notice the client
        has gone at a yield point. This generator has a path -- camera bound
        but not yet producing frames, which is exactly the state a demo camera
        is in for the first second after being retargeted -- where it loops on
        sleep and never yields at all. Every abandoned stream then pinned a
        worker permanently, and after a few demo switches the threadpool was
        gone and the server stopped answering anything, including the request
        that would have switched away.

        As an async generator the sleeps release the loop, the disconnect is
        checked explicitly, and a camera that never produces a frame ends the
        response instead of holding it open forever.
        """
        pipe = system.pipelines.get(camera_id)
        if pipe is None:
            raise HTTPException(404, f"unknown camera {camera_id}")

        def encode(frame):
            w = cfg.pipeline.stream_max_width
            if frame.shape[1] > w:
                sc = w / frame.shape[1]
                frame = cv2.resize(frame, (w, int(frame.shape[0] * sc)))
            ok, buf = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, cfg.pipeline.jpeg_quality])
            return buf.tobytes() if ok else None

        async def gen():
            boundary = b"--frame\r\n"
            last = 0.0
            starved = 0.0
            while True:
                if await request.is_disconnected():
                    return
                # ?raw=1 serves the detector's own input rather than the
                # enhanced operator view. They are deliberately separate
                # buffers, and showing them side by side is the honest way to
                # present what the model actually saw.
                frame = pipe.latest_raw if raw else pipe.latest_frame
                if frame is None:
                    starved += 0.05
                    # A camera that has produced nothing for this long is not
                    # about to. Ending the response frees the connection for
                    # the page that is trying to replace it.
                    if starved > 20.0:
                        return
                    await asyncio.sleep(0.05)
                    continue
                starved = 0.0

                now = time.time()
                if now - last < 1.0 / 25.0:
                    await asyncio.sleep(0.005)
                    continue
                last = now

                data = await asyncio.to_thread(encode, frame)
                if data is None:
                    continue
                yield (boundary + b"Content-Type: image/jpeg\r\n"
                       + f"Content-Length: {len(data)}\r\n\r\n".encode()
                       + data + b"\r\n")

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

    # -- source swap ---------------------------------------------------------
    @app.post("/api/source")
    async def swap_source(video: UploadFile = File(...)) -> dict:
        """Point a camera at an uploaded clip and restart its pipeline.

        The file is written to the local ingest directory and read from disk —
        it is never forwarded anywhere. F15 still holds: this is a local file
        handed to a local decoder.
        """
        suffix = Path(video.filename or "clip.mp4").suffix.lower()
        if suffix not in {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}:
            raise HTTPException(400, f"unsupported container: {suffix or 'none'}")

        ingest = Path(cfg.root_dir) / "data" / "ingest"
        ingest.mkdir(parents=True, exist_ok=True)
        dest = ingest / f"upload_{int(time.time())}{suffix}"

        size = 0
        with dest.open("wb") as fh:
            while chunk := await video.read(1 << 20):
                size += len(chunk)
                fh.write(chunk)
        if size == 0:
            dest.unlink(missing_ok=True)
            raise HTTPException(400, "empty file")

        camera_id = next(iter(system.pipelines), None)
        if camera_id is None:
            raise HTTPException(503, "no camera pipeline to retarget")

        pipe = system.pipelines[camera_id]
        pipe.stop()
        # Give the ingest and processing threads a moment to unwind before the
        # source is rebound; restarting into a live decoder loop double-reads.
        await asyncio.sleep(0.4)

        pipe.cam.source = str(dest)
        pipe.cam.loop = True
        pipe._stop.clear()
        pipe.q_decode._closed = False
        pipe.tracker.reset()
        pipe.spatial.reset()
        pipe.temporal.reset()
        pipe.frames_in = pipe.frames_processed = 0
        pipe.start()

        return {"ok": True, "camera_id": camera_id,
                "path": str(dest.relative_to(Path(cfg.root_dir))),
                "bytes": size}

    @app.post("/api/playback/{camera_id}")
    async def playback(camera_id: str, request: Request) -> dict:
        """Transport control for a recorded source.

        Refused on a live camera. There is no position to seek to and no past
        to replay, and a transport bar that silently does nothing is worse than
        no transport bar.
        """
        pipe = system.pipelines.get(camera_id)
        if pipe is None or pipe.source is None:
            raise HTTPException(404, f"unknown camera {camera_id}")
        if pipe.source.is_live:
            raise HTTPException(409, "live camera — playback control does not apply")

        body = await request.json()
        action = (body.get("action") or "").lower()
        if action == "pause":
            pipe.source.pause()
        elif action == "play":
            pipe.source.resume()
        elif action == "toggle":
            pipe.source.toggle()
        elif action == "rate":
            pipe.source.set_rate(float(body.get("value", 1.0)))
        elif action == "seek":
            pipe.source.seek_fraction(float(body.get("value", 0.0)))
        elif action == "restart":
            pipe.source.seek_fraction(0.0)
            pipe.source.resume()
        else:
            raise HTTPException(400, f"unknown action {action!r}")
        return pipe.source.transport()

    @app.post("/api/source/device")
    async def use_device_camera(request: Request) -> dict:
        """Retarget a camera at a locally attached capture device.

        The index is an OpenCV device index, so 0 is the built-in camera on
        most laptops. Nothing about the pipeline changes: the device becomes an
        ordinary source, and everything downstream — geometry, confirmation,
        identity, evidence — behaves exactly as it does for a file.
        """
        body = {}
        try:
            body = await request.json()
        except Exception:                                  # noqa: BLE001
            pass
        index = int(body.get("index", 0))

        probe = cv2.VideoCapture(index)
        ok = probe.isOpened()
        if ok:
            ok, _ = probe.read()
        probe.release()
        if not ok:
            raise HTTPException(
                400, f"no capture device at index {index}, or it is in use by "
                     f"another application")

        camera_id = next(iter(system.pipelines), None)
        if camera_id is None:
            raise HTTPException(503, "no camera pipeline to retarget")

        pipe = system.pipelines[camera_id]
        pipe.stop()
        await asyncio.sleep(0.4)

        pipe.cam.source = str(index)
        pipe.cam.loop = False
        # A live device has no ground calibration, so metric claims are not
        # available and the geometry falls back to pixel space -- flagged as
        # degraded rather than presented as metres.
        pipe._stop.clear()
        pipe.q_decode._closed = False
        pipe.tracker.reset()
        pipe.spatial.reset()
        pipe.temporal.reset()
        pipe.frames_in = pipe.frames_processed = 0
        pipe.start()

        return {"ok": True, "camera_id": camera_id, "device_index": index, "live": True}

    # -- scripted demonstrations --------------------------------------------
    async def _retarget(pipe, source: str) -> None:
        """Point a pipeline at a new source and restart it clean.

        Shared by the demo runner and the upload/device endpoints. The stop,
        the pause and the state resets all matter: restarting into a live
        decoder loop double-reads the source, and carrying a tracker or a
        confirmation window across a source change produces tracks that appear
        to teleport between two unrelated scenes.
        """
        if pipe.running:
            pipe.stop()
            await asyncio.sleep(0.4)

        if source.startswith("device:"):
            pipe.cam.source = source.split(":", 1)[1] or "0"
            pipe.cam.loop = False
        else:
            pipe.cam.source = source
            pipe.cam.loop = True

        pipe._stop.clear()
        pipe.q_decode._closed = False
        pipe.tracker.reset()
        pipe.spatial.reset()
        pipe.temporal.reset()
        pipe.frames_in = pipe.frames_processed = 0
        pipe.latest_overlay = {}
        pipe.start()

    @app.get("/api/demos")
    async def list_demos() -> dict:
        """The scripted demos, as configured. Read from configs/demos.yaml."""
        return {"demos": [
            {"id": d.id, "title": d.title, "claim": d.claim, "note": d.note.strip(),
             "extras": list(d.extras),
             "panels": [{"camera": p.camera, "label": p.label,
                         "source": p.source, "live": p.source.startswith("device:"),
                         "present": dict(p.present)}
                        for p in d.panels]}
            for d in cfg.demos]}

    @app.post("/api/demo/{demo_id}")
    async def run_demo(demo_id: int) -> dict:
        """Bind the demo's clips to its cameras and start only those.

        Every other pipeline is stopped, cam01 included. Three pipelines each
        running the detector would leave the one being watched at a third of
        its frame rate, and the demo is the only thing anyone is looking at.
        """
        demo = next((d for d in cfg.demos if d.id == demo_id), None)
        if demo is None:
            raise HTTPException(404, f"no demo {demo_id}")

        wanted = {p.camera for p in demo.panels}
        missing = wanted - set(system.pipelines)
        if missing:
            raise HTTPException(
                503, f"demo {demo_id} needs cameras that do not exist: "
                     f"{', '.join(sorted(missing))}")

        for cam_id, pipe in system.pipelines.items():
            if cam_id not in wanted and pipe.running:
                pipe.stop()

        panels = []
        for panel in demo.panels:
            pipe = system.pipelines[panel.camera]
            if panel.source.startswith("device:"):
                index = int(panel.source.split(":", 1)[1] or 0)
                probe = cv2.VideoCapture(index)
                ok = probe.isOpened()
                if ok:
                    ok, _ = probe.read()
                probe.release()
                if not ok:
                    raise HTTPException(
                        400, f"no capture device at index {index}, or it is in "
                             f"use by another application")
            else:
                if not (Path(cfg.root_dir) / panel.source).exists():
                    raise HTTPException(404, f"clip not found: {panel.source}")

            # Per-panel sensitivity, applied before the restart so the first
            # decoded frame already runs at the right threshold.
            pipe.cam.detector_overrides = dict(panel.detector)
            pipe._det_conf = panel.detector.get("conf")
            pipe.suppress = frozenset(panel.suppress)

            # The specialised detectors are shared across cameras -- one set of
            # weights, one config -- so a per-panel override is applied to that
            # shared config here. Within a demo that is unambiguous, because a
            # panel wanting different aerial settings from its sibling is a
            # panel whose sibling suppresses aerial entirely. Two panels both
            # running aerial with different ceilings would need per-camera
            # detector instances, and no demo asks for that.
            if system.threat is not None:
                if panel.aerial and system.threat.aerial is not None:
                    system.threat.aerial.cfg = (
                        system.threat.aerial.cfg.model_copy(update=panel.aerial))
                if panel.weapons and system.threat.weapons is not None:
                    system.threat.weapons.cfg = (
                        system.threat.weapons.cfg.model_copy(update=panel.weapons))

            await _retarget(pipe, panel.source)
            panels.append({"camera": panel.camera, "label": panel.label,
                           "source": panel.source,
                           "live": panel.source.startswith("device:"),
                           "present": dict(panel.present)})

        return {"ok": True, "demo": demo_id, "title": demo.title,
                "claim": demo.claim, "note": demo.note.strip(),
                "extras": list(demo.extras), "panels": panels}

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
