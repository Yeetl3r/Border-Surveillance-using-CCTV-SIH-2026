"""F14 -- sub-3 KB tactical relay payload.

Border posts have poor connectivity. The alert that matters is the one that
arrives, so the wire format is a compact JSON envelope plus a small base64
JPEG crop, held under a hard byte budget.

The budget is enforced by **degrading the image, never by truncating the JSON**.
A clipped JSON document is unparseable and therefore worthless; a grainier
thumbnail is still a thumbnail. If the crop cannot be squeezed under budget at
the minimum quality, it is dropped entirely and the payload goes without it --
the structured fields are what the receiving post acts on.

Keys are short because at 3 KB, key names are a measurable fraction of the
payload.
"""

from __future__ import annotations

import base64
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from ..config import RelayCfg, SystemCfg, resolve
from ..contracts import Alert, ThreatLevel

_THREAT_CODE = {
    ThreatLevel.INFO: 0,
    ThreatLevel.FRIENDLY: 1,
    ThreatLevel.WATCH: 2,
    ThreatLevel.WARNING: 3,
    ThreatLevel.CRITICAL: 4,
}

_KIND_CODE = {
    "TRIPWIRE_CROSS": "TW",
    "ZONE_PRESENCE": "ZP",
    "PROXIMITY": "PX",
    "BAG_UNATTENDED": "BG",
    "STANCE_ANOMALY": "ST",
    "IDENTITY_UNKNOWN": "ID",
    "HEALTH_FAULT": "HF",
    "SENSOR_CUE": "SC",
}


class RelayEncoder:
    """Encodes an alert into a size-bounded tactical payload."""

    def __init__(self, cfg: SystemCfg) -> None:
        self.cfg = cfg
        self.rcfg: RelayCfg = cfg.relay
        self.out_dir = resolve(cfg, self.rcfg.out_dir)
        if self.rcfg.enabled:
            self.out_dir.mkdir(parents=True, exist_ok=True)

    def _envelope(self, alert: Alert) -> dict:
        geo = alert.evidence.geometry or {}
        pos = geo.get("at") or geo.get("to")
        env = {
            "v": 1,
            "cam": alert.camera_id,
            "t": int(alert.opened_at.timestamp() * 1000),
            "k": _KIND_CODE.get(alert.kind.value, alert.kind.value[:2]),
            "th": _THREAT_CODE.get(alert.threat, 2),
            "c": round(alert.confidence, 2),
            "id": alert.alert_id[:8],
        }
        if pos and isinstance(pos, (list, tuple)) and len(pos) == 2:
            env["xy"] = [round(float(pos[0]), 1), round(float(pos[1]), 1)]
        if geo.get("direction"):
            env["d"] = geo["direction"][:3]          # INB / OUT
        if geo.get("distance_m") is not None:
            env["m"] = round(float(geo["distance_m"]), 1)
        if alert.track_ids:
            env["tr"] = list(alert.track_ids)[:3]
        return env

    def encode(self, alert: Alert, crop: np.ndarray | None = None) -> tuple[bytes, int]:
        """Return (payload_bytes, jpeg_quality_used). Quality -1 means no image."""
        env = self._envelope(alert)
        base_size = len(json.dumps(env, separators=(",", ":")).encode())
        budget = self.rcfg.max_bytes - base_size - 16      # 16 B for the "img" key

        if crop is None or crop.size == 0 or budget <= 128:
            return json.dumps(env, separators=(",", ":")).encode(), -1

        # Scale the crop down first: resolution buys more than JPEG quality at
        # this size, and a 160 px thumbnail is enough to tell a person from a
        # vehicle on a handheld.
        h, w = crop.shape[:2]
        m = self.rcfg.crop_max_px
        if max(h, w) > m:
            s = m / max(h, w)
            crop = cv2.resize(crop, (max(1, int(w * s)), max(1, int(h * s))),
                              interpolation=cv2.INTER_AREA)

        quality = self.rcfg.jpeg_quality_start
        while quality >= self.rcfg.jpeg_quality_min:
            ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if ok and (len(buf) * 4 // 3) <= budget:      # base64 inflates by 4/3
                env["img"] = base64.b64encode(buf.tobytes()).decode("ascii")
                payload = json.dumps(env, separators=(",", ":")).encode()
                if len(payload) <= self.rcfg.max_bytes:
                    return payload, quality
                env.pop("img", None)
            quality -= 10

        # Could not fit an image. Ship the structured fields -- they are what
        # the receiving post acts on.
        return json.dumps(env, separators=(",", ":")).encode(), -1

    def emit(self, alert: Alert, crop: np.ndarray | None) -> dict:
        """Encode, assert the budget, and persist the payload."""
        payload, quality = self.encode(alert, crop)
        size = len(payload)
        if size > self.rcfg.max_bytes:
            # Should be unreachable: the encoder degrades rather than overflows.
            raise AssertionError(
                f"relay payload {size} B exceeds the {self.rcfg.max_bytes} B budget"
            )

        path = ""
        if self.rcfg.enabled:
            p = self.out_dir / f"{alert.alert_id[:8]}.json"
            p.write_bytes(payload)
            path = str(p.relative_to(Path(self.cfg.root_dir)))

        return {"bytes": size, "jpeg_quality": quality, "path": path,
                "within_budget": size <= self.rcfg.max_bytes,
                "budget": self.rcfg.max_bytes}
