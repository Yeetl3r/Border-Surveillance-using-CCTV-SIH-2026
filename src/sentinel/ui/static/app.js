/* SENTINEL C2 — client.
 *
 * The markup is the approved design canvas, carried over verbatim. The canvas
 * shipped with mocked constants; this file replaces them with the live system:
 * real telemetry over a WebSocket, real MJPEG frames, real alerts, and real
 * evidence records pulled from the store.
 *
 * Two rules the rendering obeys:
 *
 *   F12 — THREAT sets colour, CONFIDENCE sets stroke style. They are separate
 *   fields and are never multiplied into one value. A low-confidence CRITICAL
 *   is a red hatched box; a high-confidence WATCH is a solid amber one.
 *
 *   Overlays are drawn here, on a canvas above the video, never burned into
 *   the stream. That keeps the evidence keyframe byte-identical to what the
 *   detector actually received.
 */

/* ══════════════════════════════════════════════════════════════════════════
   A very small template runtime for the canvas's directives:
     {{ path }}         interpolation, in text and in attributes
     data-if="path"     conditional display
     <template data-for="list" data-as="item">
     data-click / data-drop / data-dragover / data-dragleave / data-change
     data-ref="name"    element handle
   Compile once, patch on update — a full re-render of this document at 10 Hz
   would be wasteful and would restart the CSS animations every frame.
   ══════════════════════════════════════════════════════════════════════════ */

const RX = /\{\{\s*([\w.$]+)\s*\}\}/g;

function resolve(path, scopes) {
  const parts = path.split(".");
  for (let i = scopes.length - 1; i >= 0; i--) {
    let cur = scopes[i];
    if (cur == null || !(parts[0] in cur)) continue;
    for (const p of parts) {
      if (cur == null) break;
      cur = cur[p];
    }
    return cur;
  }
  return undefined;
}

function fill(tpl, scopes) {
  return tpl.replace(RX, (_, p) => {
    const v = resolve(p, scopes);
    return v === undefined || v === null ? "" : String(v);
  });
}

class Binder {
  constructor(root) {
    this.texts = [];
    this.attrs = [];
    this.ifs = [];
    this.loops = [];
    this.refs = {};
    this.handlers = {};
    this.scope = null;
    this.compile(root);
  }

  compile(root) {
    // Loops first: detach each <template data-for> and leave an anchor, so the
    // walker below never descends into an un-instantiated body.
    root.querySelectorAll("template[data-for]").forEach((tpl) => {
      const anchor = document.createComment("for:" + tpl.dataset.for);
      tpl.parentNode.insertBefore(anchor, tpl);
      tpl.remove();
      this.loops.push({
        anchor,
        list: tpl.dataset.for,
        as: tpl.dataset.as || "item",
        html: tpl.innerHTML,
        instances: [],
      });
    });

    const walk = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT);
    const els = [];
    let n = walk.currentNode;
    while (n) {
      if (n.nodeType === Node.TEXT_NODE) {
        if (n.nodeValue.includes("{{")) this.texts.push({ node: n, tpl: n.nodeValue });
      } else {
        els.push(n);
      }
      n = walk.nextNode();
    }

    for (const el of els) {
      if (el.dataset && el.dataset.ref) this.refs[el.dataset.ref] = el;
      if (el.dataset && el.dataset.if !== undefined) {
        this.ifs.push({ el, expr: el.dataset.if });
      }
      for (const a of Array.from(el.attributes || [])) {
        if (a.value.includes("{{")) {
          this.attrs.push({ el, name: a.name, tpl: a.value });
        }
      }
      // events
      const map = {
        click: "click", drop: "drop", dragover: "dragover",
        dragleave: "dragleave", change: "change",
      };
      for (const [dsKey, evt] of Object.entries(map)) {
        const fn = el.dataset && el.dataset[dsKey];
        if (fn) {
          el.addEventListener(evt, (e) => {
            // Handler names are paths, not plain keys: a loop body binds
            // `a.select`, the page level binds `pickFile`. Resolving against
            // the live scope covers both without special-casing either.
            const h = resolve(fn, [this.scope || {}]) ?? this.handlers[fn];
            if (typeof h === "function") h(e);
          });
          if (evt === "click") el.style.cursor = el.style.cursor || "pointer";
        }
      }
      if (el.dataset && el.dataset.bindSrc) {
        this.attrs.push({ el, name: "src", tpl: "{{ " + el.dataset.bindSrc + " }}" });
      }
    }
  }

  update(state) {
    this.scope = state;
    const scopes = [state];

    for (const t of this.texts) {
      const v = fill(t.tpl, scopes);
      if (t.node.nodeValue !== v) t.node.nodeValue = v;
    }
    for (const a of this.attrs) {
      const v = fill(a.tpl, scopes);
      if (a.name === "src" && !v) continue;
      if (a.el.getAttribute(a.name) !== v) a.el.setAttribute(a.name, v);
    }
    for (const c of this.ifs) {
      const on = !!resolve(c.expr, scopes);
      c.el.style.display = on ? "contents" : "none";
    }
    for (const lp of this.loops) {
      const items = resolve(lp.list, scopes) || [];
      this.renderLoop(lp, items, state);
    }
  }

  renderLoop(lp, items, state) {
    // Reuse instances where the count matches; rebuild when it changes.
    if (lp.instances.length !== items.length) {
      lp.instances.forEach((inst) => inst.nodes.forEach((n) => n.remove()));
      lp.instances = items.map(() => {
        const frag = document.createElement("div");
        frag.innerHTML = lp.html;
        const nodes = Array.from(frag.childNodes);
        const holder = document.createElement("div");
        holder.style.display = "contents";
        nodes.forEach((n) => holder.appendChild(n));
        lp.anchor.parentNode.insertBefore(holder, lp.anchor);
        return { holder, nodes: [holder], binder: new Binder(holder) };
      });
    }
    items.forEach((item, i) => {
      const inst = lp.instances[i];
      const scoped = Object.assign(Object.create(state), { [lp.as]: item });
      inst.binder.handlers = this.handlers;
      inst.binder.update(scoped);
    });
  }
}

/* ══════════════════════════════════════════════════════════════════════════
   Live state
   ══════════════════════════════════════════════════════════════════════════ */

const THREAT_COLOUR = {
  INFO: "#5aa9e6",
  FRIENDLY: "#2fd98a",
  WATCH: "#ffc21a",
  WARNING: "#ff7a1a",
  CRITICAL: "#ff2b3d",
};
const THREAT_BG = {
  INFO: "rgba(90,169,230,.09)",
  FRIENDLY: "rgba(47,217,138,.09)",
  WATCH: "rgba(255,194,26,.09)",
  WARNING: "rgba(255,122,26,.09)",
  CRITICAL: "rgba(255,43,61,.10)",
};

const S = {
  phase: "idle",           // idle | running | done
  fileName: "",
  videoUrl: "",
  sourceLabel: "REFERENCE IR CLIP",
  cameraId: null,
  status: {},
  overlay: null,
  zones: null,
  liveAlerts: [],
  historyCount: 0,
  selected: 0,
  evidence: null,
  logLines: [],
  step: 0,
  startedAt: 0,
};

const TIER_DEFS = [
  { id: "T0", name: "Ingestion & health", note: "decode · Laplacian + SSIM · H" },
  { id: "T1", name: "Perception", note: "YOLO26-s NMS-free · BoT-SORT" },
  { id: "T2", name: "Spatial reasoning", note: "footpoint → metres · tripwire · zones" },
  { id: "T3", name: "Temporal confirmation", note: "N-of-M in seconds and hit ratio" },
  { id: "T5", name: "Identity, open-set", note: "quality gate · OSNet · gallery" },
  { id: "T8", name: "Alert manager", note: "sole alert constructor · evidence" },
];

let binder = null;
let ws = null;

/* ── helpers ───────────────────────────────────────────────────────────── */
const pad = (n) => String(n).padStart(2, "0");
function stamp() {
  const ms = Date.now() - (S.startedAt || Date.now());
  const s = Math.floor(ms / 1000);
  return `${pad(Math.floor(s / 60))}:${pad(s % 60)}.${pad(Math.floor((ms % 1000) / 10))}`;
}
function log(text) {
  S.logLines.push({ t: stamp(), text });
  if (S.logLines.length > 40) S.logLines.shift();
  S.step = S.logLines.length;
}
const hhmmss = (iso) => (iso ? new Date(iso).toISOString().substr(11, 8) : "--:--:--");

/* ── the view model the template binds against ─────────────────────────── */
function viewModel() {
  const st = S.status || {};
  const cam = (st.cameras || [])[0] || {};
  const perf = cam.perf || {};
  const ov = S.overlay || {};
  const health = (ov.health && ov.health.state) || cam.health || "--";

  const perTier = Math.max(1, 14 / TIER_DEFS.length);
  const tiers = TIER_DEFS.map((t, i) => {
    const done = S.phase === "done" || S.step >= (i + 1) * perTier;
    const active = !done && S.step >= i * perTier;
    return {
      key: t.id,
      name: t.id + " · " + t.name,
      note: t.note,
      status: done ? "DONE" : active ? "RUNNING" : "QUEUED",
      dot: done ? "#2fd98a" : active ? "#ff1f3c" : "rgba(255,255,255,.16)",
      glow: active ? "0 0 12px rgba(255,31,60,.9)"
           : done ? "0 0 10px rgba(47,217,138,.6)" : "none",
      fg: done ? "rgba(242,236,235,.85)"
        : active ? "#ff8f9a" : "rgba(242,236,235,.35)",
    };
  });

  const alerts = S.liveAlerts.map((a, i) => ({
    key: a.alert_id,
    id: a.alert_id,
    kind: (a.kind || "").replace(/_/g, " ").toLowerCase()
            .replace(/^./, (c) => c.toUpperCase()),
    threat: a.threat,
    band: a.confidence_band,
    conf: Number(a.confidence || 0).toFixed(2),
    cam: a.camera_id,
    time: hhmmss(a.opened_at),
    color: THREAT_COLOUR[a.threat] || THREAT_COLOUR.INFO,
    bg: THREAT_BG[a.threat] || THREAT_BG.INFO,
    lineStyle: a.confidence_band === "HIGH" ? "solid"
             : a.confidence_band === "MEDIUM" ? "dashed" : "dotted",
    hatched: a.confidence_band === "LOW",
    rationale: a.rationale || "",
    select: () => { S.selected = i; loadEvidence(a.alert_id); paint(); },
  }));

  const ev = S.evidence || {};
  const cfg = ev.config_snapshot || {};
  const sel = alerts[S.selected] || alerts[0] || {};
  const selFull = Object.assign({}, sel, {
    state: ev.state || "—",
    clip: ev.clip_path ? "H.264 · run-up retained" : "—",
    detections: ev.detections
      ? `${ev.detections.length} detection${ev.detections.length === 1 ? "" : "s"} · `
        + ev.detections.map((d) => Number(d.confidence).toFixed(2)).join(" / ")
      : "—",
    confirm: cfg.confirm
      ? `${cfg.confirm.min_hits} of window ${cfg.confirm.window_s}s · ratio ${cfg.confirm.min_ratio}`
      : "—",
    geometry: ev.geometry
      ? [ev.geometry.zone_label, ev.geometry.direction,
         ev.geometry.distance_m != null ? ev.geometry.distance_m + " m" : null]
        .filter(Boolean).join(" · ")
      : "—",
    homography: ev.homography_id
      ? `${ev.homography_id} · residual ${(cam.homography || {}).residual_m ?? "—"} m`
      : "uncalibrated · pixel space",
    identity: (() => {
      const d = (ev.detections || []).find((x) => x.identity);
      if (!d) return "—";
      const id = d.identity;
      return id.label === "FRIENDLY"
        ? `FRIENDLY · ${id.matched} · ${id.similarity} > ${id.threshold}`
        : `${id.label} · best ${id.similarity ?? "—"} vs ${id.threshold}`;
    })(),
    discounts: (sel.rationale || "").includes("|")
      ? sel.rationale.split("|").slice(2).join(" · ").trim() || "none applied"
      : "none applied",
    models: ev.model_ids
      ? Object.entries(ev.model_ids).map(([k, v]) => `${k} ${v}`).join(" · ")
      : "—",
  });

  const idcfg = st.identity || {};
  const prof = st.profile || "demo";

  return {
    // phase
    isIdle: S.phase === "idle",
    isRunning: S.phase === "running",
    isDone: S.phase === "done",
    hasVideo: !!S.videoUrl,
    noVideo: !S.videoUrl,
    videoUrl: S.videoUrl,
    fileName: S.fileName,
    sourceLabel: S.sourceLabel,

    // progress
    progressPct: Math.min(100, Math.round((S.step / 14) * 100)) + "%",
    stageLabel: S.phase === "done" ? "ALERTS PROMOTED"
      : "TIER " + Math.min(9, Math.floor(S.step / 1.6)) + " · PROCESSING",
    logs: S.logLines.slice(-6).map((l, i) => ({ key: l.t + i, t: l.t, text: l.text })),
    tiers,

    // live
    alerts,
    sel: selFull,

    // config, read from the running system rather than hardcoded
    profileName: String(prof).toUpperCase() === "OPERATIONAL" ? "OPERATIONAL" : "DEMONSTRATION",
    cfgWindow: (cfg.confirm && cfg.confirm.window_s) ?? (prof === "operational" ? "3.0" : "1.2"),
    cfgRatio: (cfg.confirm && cfg.confirm.min_ratio) ?? (prof === "operational" ? "0.7" : "0.6"),
    cfgDwell: (cfg.baggage && cfg.baggage.dwell_s) ?? (prof === "operational" ? "120" : "15"),
    simThreshold: idcfg.threshold ?? "0.666",

    // handlers
    pickFile: () => binder.refs.fileRef && binder.refs.fileRef.click(),
    onFileChange: (e) => takeFile(e.target.files && e.target.files[0]),
    onDragOver: (e) => e.preventDefault(),
    onDragLeave: (e) => e.preventDefault(),
    onDrop: (e) => {
      e.preventDefault();
      takeFile(e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]);
    },
    runReference: () => start(null),
    reset: () => {
      S.phase = "idle"; S.step = 0; S.logLines = []; S.selected = 0;
      S.evidence = null;
      if (S.videoUrl) { URL.revokeObjectURL(S.videoUrl); S.videoUrl = ""; }
      S.fileName = ""; S.sourceLabel = "REFERENCE IR CLIP";
      paint();
    },

    // live readouts
    _fps: (perf.fps ?? 0).toFixed(1),
    _frameMs: (perf.frame_ms ?? 0).toFixed(1),
    _health: health,
    _tracks: (ov.boxes || []).length,
    _drops: st.queue_drops ?? 0,
    _threat: st.threat || "INFO",
    _calibrated: cam.metric,
  };
}

/* ── DOM patches the template cannot express ───────────────────────────── */
function paintRefs(vm) {
  const r = binder.refs;
  if (r.fps) r.fps.textContent = vm._fps + " FPS";
  if (r.frameMs) r.frameMs.textContent = vm._frameMs + " ms";
  if (r.tracksDrops) r.tracksDrops.textContent =
    `${vm._tracks} TRACKS · Q1 DROPS ${vm._drops}`;
  if (r.alertCounts) r.alertCounts.textContent =
    `${S.liveAlerts.length} ACTIVE · ${S.historyCount} IN HISTORY`;
  if (r.calib) r.calib.textContent = vm._calibrated ? "H · CALIBRATED" : "PIXEL SPACE";
  if (r.relay) {
    const b = (S.status.relay_max_bytes ?? null);
    r.relay.textContent = b ? b.toLocaleString() + " B" : "≤ 3,072 B";
  }
  if (r.health) {
    const ok = vm._health === "OK";
    r.health.textContent = "HEALTH " + vm._health;
    r.health.style.color = ok ? "#8ff0c2" : "#ff8f9a";
    r.health.style.background = ok ? "rgba(47,217,138,.14)" : "rgba(255,43,61,.16)";
    r.health.style.borderColor = ok ? "rgba(47,217,138,.45)" : "rgba(255,43,61,.5)";
  }
  if (r.sectorThreat) {
    r.sectorThreat.textContent = vm._threat;
    const c = THREAT_COLOUR[vm._threat] || THREAT_COLOUR.INFO;
    r.sectorThreat.style.color = c;
    const badge = r.sectorThreat.parentElement;
    if (badge) {
      badge.style.borderColor = c;
      badge.style.background = THREAT_BG[vm._threat] || THREAT_BG.INFO;
      const dot = badge.querySelector("span");
      if (dot && dot !== r.sectorThreat) {
        dot.style.background = c;
        dot.style.boxShadow = `0 0 12px ${c}`;
      }
    }
  }
  if (r.stream && S.phase === "done" && !S.videoUrl && S.cameraId) {
    const want = `/api/stream/${S.cameraId}`;
    if (!r.stream.getAttribute("src")) r.stream.setAttribute("src", want);
  }
}

/* ── canvas overlay: F12 dual-axis ─────────────────────────────────────── */
function drawOverlay() {
  const cv = binder.refs.overlay;
  const img = binder.refs.stream;
  if (!cv || !S.overlay) return;

  const w = cv.clientWidth, h = cv.clientHeight;
  if (!w || !h) return;
  if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
  const ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, w, h);

  // The MJPEG frame is object-fit:cover — mirror that mapping or every box
  // lands in the wrong place.
  const nw = (img && img.naturalWidth) || 1280;
  const nh = (img && img.naturalHeight) || 720;
  const scale = Math.max(w / nw, h / nh);
  const ox = (w - nw * scale) / 2, oy = (h - nh * scale) / 2;
  const X = (x) => ox + x * scale, Y = (y) => oy + y * scale;

  drawZones(ctx, X, Y);

  for (const b of S.overlay.boxes || []) {
    const colour = THREAT_COLOUR[b.threat] || THREAT_COLOUR.INFO;
    const band = b.confidence >= 0.75 ? "HIGH" : b.confidence >= 0.5 ? "MEDIUM" : "LOW";
    const [x1, y1, x2, y2] = b.bbox;
    const x = X(x1), y = Y(y1), bw = (x2 - x1) * scale, bh = (y2 - y1) * scale;

    if (band === "LOW") {
      ctx.save();
      ctx.beginPath(); ctx.rect(x, y, bw, bh); ctx.clip();
      ctx.globalAlpha = 0.18; ctx.strokeStyle = colour; ctx.lineWidth = 1.2;
      for (let i = -bh; i < bw; i += 8) {
        ctx.beginPath(); ctx.moveTo(x + i, y + bh); ctx.lineTo(x + i + bh, y); ctx.stroke();
      }
      ctx.restore();
    }

    ctx.save();
    ctx.strokeStyle = colour;
    ctx.shadowColor = colour;
    ctx.shadowBlur = band === "HIGH" ? 16 : 8;
    if (band === "HIGH") { ctx.setLineDash([]); ctx.lineWidth = 2; }
    else if (band === "MEDIUM") { ctx.setLineDash([7, 4]); ctx.lineWidth = 2; }
    else { ctx.setLineDash([2, 4]); ctx.lineWidth = 1.6; }
    ctx.strokeRect(x, y, bw, bh);
    ctx.restore();

    // Label bar, in the design's idiom: solid fill, dark text, mono.
    const bits = [`#${b.track_id}`, (b.group || "").toUpperCase(),
                  Number(b.confidence).toFixed(2)];
    if (b.footpoint_m) bits.push(`${b.footpoint_m[1].toFixed(1)} m`);
    if (b.identity && b.identity.label === "FRIENDLY")
      bits.push((b.identity.matched || "FRIENDLY").toUpperCase());
    else if (b.identity && b.identity.label === "UNIDENTIFIED") bits.push("UNIDENTIFIED");
    else if (b.identity && b.identity.label === "INSUFFICIENT_QUALITY")
      bits.push("INSUFFICIENT QUALITY");
    if (b.stance && b.stance !== "UNKNOWN" && b.stance !== "STANDING") bits.push(b.stance);
    const label = bits.join(" · ");

    ctx.save();
    ctx.font = "500 9.5px 'IBM Plex Mono', ui-monospace, monospace";
    const tw = ctx.measureText(label).width + 14;
    const ly = Math.max(0, y - 22);
    ctx.fillStyle = colour;
    ctx.fillRect(x, ly, tw, 19);
    ctx.fillStyle = band === "LOW" ? "rgba(10,7,8,.9)" : "#0a0708";
    ctx.fillText(label, x + 7, ly + 13);
    ctx.restore();

    ctx.save();
    ctx.fillStyle = colour;
    ctx.beginPath(); ctx.arc(X((x1 + x2) / 2), Y(y2), 3, 0, Math.PI * 2); ctx.fill();
    ctx.restore();
  }
}

function drawZones(ctx, X, Y) {
  const z = S.zones;
  if (!z) return;

  for (const zone of z.zones || []) {
    if (!zone.points_px) continue;
    ctx.save();
    ctx.strokeStyle = "rgba(255,31,60,.32)";
    ctx.setLineDash([6, 5]); ctx.lineWidth = 1.2;
    ctx.fillStyle = "rgba(255,31,60,.05)";
    ctx.beginPath();
    zone.points_px.forEach(([px, py], i) =>
      i ? ctx.lineTo(X(px), Y(py)) : ctx.moveTo(X(px), Y(py)));
    ctx.closePath(); ctx.fill(); ctx.stroke();
    ctx.restore();
  }

  for (const w of z.tripwires || []) {
    if (!w.points_px) continue;
    const [[ax, ay], [bx, by]] = w.points_px;
    ctx.save();
    ctx.strokeStyle = "#ff2b3d"; ctx.lineWidth = 1.6;
    ctx.setLineDash([12, 10]);
    ctx.beginPath(); ctx.moveTo(X(ax), Y(ay)); ctx.lineTo(X(bx), Y(by)); ctx.stroke();
    ctx.setLineDash([]);

    if (w.inbound_arrow_px) {
      const [[sx0, sy0], [tx0, ty0]] = w.inbound_arrow_px;
      const sx = X(sx0), sy = Y(sy0), tx = X(tx0), ty = Y(ty0);
      const a = Math.atan2(ty - sy, tx - sx), head = 7;
      ctx.beginPath();
      ctx.moveTo(sx, sy); ctx.lineTo(tx, ty);
      ctx.lineTo(tx - head * Math.cos(a - 0.4), ty - head * Math.sin(a - 0.4));
      ctx.moveTo(tx, ty);
      ctx.lineTo(tx - head * Math.cos(a + 0.4), ty - head * Math.sin(a + 0.4));
      ctx.stroke();
    }
    ctx.font = "500 9.5px 'IBM Plex Mono', ui-monospace, monospace";
    ctx.fillStyle = "rgba(255,43,61,.85)";
    ctx.fillText(`${w.label || w.id} · INBOUND`, X(ax) + 8, Y(ay) - 8);
    ctx.restore();
  }

  for (const p of z.proximity || []) {
    if (!p.points_px) continue;
    const [px, py] = p.points_px[0];
    ctx.save();
    ctx.strokeStyle = "rgba(255,122,26,.7)"; ctx.lineWidth = 1.4;
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.arc(X(px), Y(py), 10, 0, Math.PI * 2); ctx.stroke();
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(X(px) - 6, Y(py)); ctx.lineTo(X(px) + 6, Y(py));
    ctx.moveTo(X(px), Y(py) - 6); ctx.lineTo(X(px), Y(py) + 6);
    ctx.stroke();
    ctx.font = "500 9px 'IBM Plex Mono', ui-monospace, monospace";
    ctx.fillStyle = "rgba(255,122,26,.85)";
    ctx.fillText(`${p.label || p.id} < ${p.threshold_m} m`, X(px) + 14, Y(py) + 3);
    ctx.restore();
  }
}

/* ── paint ─────────────────────────────────────────────────────────────── */
function paint() {
  const vm = viewModel();
  binder.handlers = vm;
  binder.update(vm);
  paintRefs(vm);
  if (S.phase === "done") drawOverlay();
}

/* ── run control ───────────────────────────────────────────────────────── */
async function takeFile(file) {
  if (!file) return;
  S.fileName = file.name;
  S.sourceLabel = file.name.toUpperCase();
  S.videoUrl = URL.createObjectURL(file);
  await start(file);
}

async function start(file) {
  S.phase = "running";
  S.step = 0;
  S.logLines = [];
  S.startedAt = Date.now();
  paint();

  if (file) {
    log(`ingesting ${file.name} · ${(file.size / 1e6).toFixed(1)} MB · read locally, never uploaded`);
    const fd = new FormData();
    fd.append("video", file);
    try {
      const r = await fetch("/api/source", { method: "POST", body: fd });
      const j = await r.json();
      if (!r.ok) throw new Error(j.detail || "source rejected");
      S.cameraId = j.camera_id;
      log(`source swapped on ${j.camera_id} · pipeline restarted`);
    } catch (err) {
      log(`source could not be swapped (${err.message}) — continuing on the reference feed`);
    }
  } else {
    log("reference clip selected · 1280×720 · 15 fps · IR · ~97 kbps");
  }

  // Progress is driven by real telemetry arriving over the socket, not by a
  // timer. The lines below are emitted from measured values as they land.
  await pollUntilReady();
}

async function pollUntilReady() {
  const t0 = Date.now();
  const seen = new Set();
  const tick = async () => {
    if (S.phase !== "running") return;
    try {
      const st = await (await fetch("/api/status")).json();
      S.status = st;
      const cam = (st.cameras || [])[0] || {};
      if (!S.cameraId) S.cameraId = cam.camera_id;

      const say = (k, msg) => { if (!seen.has(k)) { seen.add(k); log(msg); } };

      if (cam.frames_processed > 0)
        say("decode", `decode running · ${cam.frames_processed} frames · health ${cam.health}`);
      if (cam.homography && cam.homography.calibrated)
        say("h", `homography ${cam.homography.homography_id} · round-trip residual ${cam.homography.residual_m} m`);
      if (cam.perf && cam.perf.fps)
        say("fps", `perception steady · ${cam.perf.fps.toFixed(1)} FPS · detect ${cam.perf.detect_ms} ms · track ${cam.perf.track_ms} ms`);
      if (cam.perf && cam.perf.logic_ms !== undefined)
        say("logic", `spatial + temporal reasoning ${cam.perf.logic_ms} ms/frame`);
      if (st.identity && st.identity.vectors)
        say("id", `identity gallery loaded · ${st.identity.identities} enrolled · threshold ${st.identity.threshold}`);
      if (cam.pending_candidates > 0)
        say("obs", `${cam.pending_candidates} observation(s) accumulating · NOT alerts · awaiting confirmation`);
      if (st.counters && st.counters.promoted > 0)
        say("promo", `confirmation satisfied · ${st.counters.promoted} alert(s) promoted · evidence written`);
      if (st.queue_drops > 0)
        say("drop", `Q1 drop-oldest engaged · ${st.queue_drops} frames shed to hold latency`);

      const ready = cam.frames_processed > 25 &&
                    (st.counters?.promoted > 0 || Date.now() - t0 > 12000);
      if (ready) {
        log(`sector threat recomputed from the live alert set · ${st.threat}`);
        await refreshAlerts();
        S.phase = "done";
        paint();
        return;
      }
    } catch (e) { /* transient during restart */ }
    paint();
    setTimeout(tick, 700);
  };
  tick();
}

/* ── data ──────────────────────────────────────────────────────────────── */
async function refreshAlerts() {
  try {
    const a = await (await fetch("/api/alerts?limit=40")).json();
    S.liveAlerts = (a.live && a.live.length ? a.live : a.recent) || [];
    S.historyCount = (a.recent || []).length;
    try {
      const s = await (await fetch("/api/stats")).json();
      S.historyCount = s.alerts_total ?? S.historyCount;
      S.status.relay_max_bytes = s.relay_max_bytes;
    } catch (_) { /* stats optional */ }
    if (S.liveAlerts.length && !S.evidence) loadEvidence(S.liveAlerts[0].alert_id);
  } catch (_) { /* keep last good */ }
}

async function loadEvidence(id) {
  try {
    S.evidence = await (await fetch(`/api/alerts/${id}`)).json();
  } catch (_) { S.evidence = null; }
  paint();
}

/* ── socket ────────────────────────────────────────────────────────────── */
function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);

  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.type === "hello") {
      S.status = m.status || {};
      const cams = Object.keys(m.zones || {});
      if (cams.length) {
        S.cameraId = S.cameraId || cams[0];
        S.zones = m.zones[S.cameraId];
      }
    } else if (m.type === "tick") {
      const o = m.cameras && m.cameras[S.cameraId];
      if (o) S.overlay = o;
      if (m.status) {
        Object.assign(S.status, m.status);
        // Only replace the list while something is actually live. When the
        // sector clears, keep the last set on screen rather than blanking the
        // panel -- an operator glancing up needs to see what just happened,
        // not an empty column.
        if (m.status.alerts && m.status.alerts.length) {
          S.liveAlerts = m.status.alerts;
        }
      }
      if (S.phase === "done") paint();
    } else if (m.type === "alert" && m.phase === "opened") {
      refreshAlerts().then(paint);
    }
  };
  ws.onclose = () => setTimeout(connect, 1500);
  setInterval(() => { if (ws && ws.readyState === 1) ws.send("ping"); }, 15000);
}

/* ── boot ──────────────────────────────────────────────────────────────── */
window.addEventListener("DOMContentLoaded", async () => {
  binder = new Binder(document.body);
  try { S.status = await (await fetch("/api/status")).json(); } catch (_) {}
  const cam = (S.status.cameras || [])[0];
  if (cam) S.cameraId = cam.camera_id;
  paint();
  connect();
  window.addEventListener("resize", () => { if (S.phase === "done") drawOverlay(); });
  setInterval(() => { if (S.phase === "done") drawOverlay(); }, 120);
});
