/* SENTINEL C2 client.
 *
 * Two inbound channels: an MJPEG <img> for pixels, and a WebSocket for state.
 * Overlays are drawn here, from state -- never burned into the video. That is
 * what lets the evidence keyframe stay the unannotated input the decision was
 * made on, and what makes F12 a property of the display.
 *
 * F12, the rule this file exists to implement:
 *     THREAT     -> stroke COLOUR
 *     CONFIDENCE -> stroke STYLE   (solid / dashed / hatched)
 * They are independent. A low-confidence CRITICAL is a red hatched box; a
 * high-confidence WATCH is a solid amber one. Collapsing them into one visual
 * variable would hide exactly the distinction the operator needs.
 */

const THREAT_COLOUR = {
  INFO: "#5aa9e6",
  FRIENDLY: "#3ddc84",
  WATCH: "#ffc23d",
  WARNING: "#ff8c1a",
  CRITICAL: "#ff3344",
};

const state = {
  cameraId: null,
  zones: {},
  overlay: null,
  alerts: [],
  selected: null,
  tab: "live",
  status: {},
  routes: {},
};

/* ---------------------------------------------------------------- helpers */
const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
};
const hhmmss = (iso) => (iso ? new Date(iso).toISOString().substr(11, 8) : "--:--:--");

function bandStyle(ctx, band) {
  // The style axis. Dash patterns are chosen to stay legible when a box is
  // only ~30 px tall, which is normal for a distant subject.
  if (band === "HIGH") { ctx.setLineDash([]); ctx.lineWidth = 2.2; ctx.globalAlpha = 1; }
  else if (band === "MEDIUM") { ctx.setLineDash([7, 4]); ctx.lineWidth = 2; ctx.globalAlpha = 0.92; }
  else { ctx.setLineDash([2, 4]); ctx.lineWidth = 1.6; ctx.globalAlpha = 0.78; }
}

function bandFor(conf) {
  return conf >= 0.75 ? "HIGH" : conf >= 0.5 ? "MEDIUM" : "LOW";
}

function hatch(ctx, x, y, w, h, colour) {
  // Low-confidence boxes get a hatched fill as well as a dotted stroke, so the
  // uncertainty reads at a glance rather than requiring a squint at the dashes.
  ctx.save();
  ctx.beginPath(); ctx.rect(x, y, w, h); ctx.clip();
  ctx.globalAlpha = 0.16; ctx.strokeStyle = colour; ctx.lineWidth = 1.2;
  ctx.setLineDash([]);
  for (let i = -h; i < w; i += 7) {
    ctx.beginPath(); ctx.moveTo(x + i, y + h); ctx.lineTo(x + i + h, y); ctx.stroke();
  }
  ctx.restore();
}

/* ------------------------------------------------------------- rendering */
function draw() {
  const cv = $("#overlay");
  const img = $("#video");
  const o = state.overlay;
  if (!cv || !img || !o) return;

  // The MJPEG frame is letterboxed by object-fit: contain. Map source pixels
  // into displayed pixels or every box lands in the wrong place.
  const cw = cv.clientWidth, ch = cv.clientHeight;
  if (cv.width !== cw || cv.height !== ch) { cv.width = cw; cv.height = ch; }
  const ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, cw, ch);

  const nw = img.naturalWidth || 1280, nh = img.naturalHeight || 720;
  const scale = Math.min(cw / nw, ch / nh);
  const ox = (cw - nw * scale) / 2, oy = (ch - nh * scale) / 2;
  const X = (x) => ox + x * scale, Y = (y) => oy + y * scale;

  drawZones(ctx, X, Y, scale);

  for (const b of o.boxes || []) {
    const colour = THREAT_COLOUR[b.threat] || THREAT_COLOUR.INFO;
    const band = bandFor(b.confidence);
    const [x1, y1, x2, y2] = b.bbox;
    const x = X(x1), y = Y(y1), w = (x2 - x1) * scale, h = (y2 - y1) * scale;

    if (band === "LOW") hatch(ctx, x, y, w, h, colour);

    ctx.save();
    bandStyle(ctx, band);
    ctx.strokeStyle = colour;
    ctx.strokeRect(x, y, w, h);

    // Corner brackets read as a targeting reticle and survive at small sizes.
    ctx.setLineDash([]); ctx.lineWidth = 2.6; ctx.globalAlpha = 1;
    const c = Math.min(13, w / 3, h / 3);
    [[x, y, 1, 1], [x + w, y, -1, 1], [x, y + h, 1, -1], [x + w, y + h, -1, -1]]
      .forEach(([cx, cy, sx, sy]) => {
        ctx.beginPath();
        ctx.moveTo(cx + sx * c, cy); ctx.lineTo(cx, cy); ctx.lineTo(cx, cy + sy * c);
        ctx.stroke();
      });
    ctx.restore();

    drawLabel(ctx, b, x, y, colour, band);

    // Footpoint marker: the point that is actually tested against geometry.
    ctx.save();
    ctx.fillStyle = colour; ctx.globalAlpha = 0.9;
    ctx.beginPath(); ctx.arc(X((x1 + x2) / 2), Y(y2), 3.2, 0, Math.PI * 2); ctx.fill();
    ctx.restore();
  }

  // VLM grounding results, when a grounding query has run.
  for (const g of state.grounding || []) {
    const [x1, y1, x2, y2] = g.bbox;
    ctx.save();
    ctx.strokeStyle = "#b7ff54"; ctx.lineWidth = 2; ctx.setLineDash([9, 5]);
    ctx.strokeRect(X(x1), Y(y1), (x2 - x1) * scale, (y2 - y1) * scale);
    ctx.fillStyle = "#b7ff54"; ctx.font = "600 11px ui-monospace, monospace";
    ctx.setLineDash([]);
    ctx.fillText("VLM", X(x1) + 4, Y(y1) - 5);
    ctx.restore();
  }
}

function drawLabel(ctx, b, x, y, colour, band) {
  // Operational group (person/vehicle/animal/bag) is what the operator
  // reasons in. The raw detector class is kept in the evidence record -- on
  // this camera's IR footage the parked car classifies as COCO `boat`, which
  // is worth recording but not worth putting on the operator's screen.
  const parts = [`#${b.track_id}`, b.group.toUpperCase(),
                 `${Math.round(b.confidence * 100)}%`];
  if (b.footpoint_m) parts.push(`${b.footpoint_m[1].toFixed(1)}m`);
  if (b.identity && b.identity.label === "FRIENDLY") parts.push(b.identity.matched || "FRIENDLY");
  else if (b.identity && b.identity.label === "UNIDENTIFIED") parts.push("UNIDENT");
  if (b.stance && b.stance !== "UNKNOWN" && b.stance !== "STANDING") parts.push(b.stance);
  const text = parts.join("  ");

  ctx.save();
  ctx.font = "600 11px ui-monospace, SFMono-Regular, monospace";
  const w = ctx.measureText(text).width + 12;
  const ly = Math.max(0, y - 19);
  ctx.fillStyle = colour; ctx.globalAlpha = band === "LOW" ? 0.72 : 0.95;
  ctx.fillRect(x, ly, w, 18);
  ctx.globalAlpha = 1; ctx.fillStyle = "#08140b";
  ctx.fillText(text, x + 6, ly + 13);
  ctx.restore();
}

function drawZones(ctx, X, Y, scale) {
  const z = state.zones[state.cameraId];
  if (!z) return;

  for (const zone of z.zones || []) {
    if (!zone.points_px) continue;
    ctx.save();
    ctx.strokeStyle = "rgba(183,255,84,.42)"; ctx.lineWidth = 1.4; ctx.setLineDash([6, 5]);
    ctx.fillStyle = "rgba(183,255,84,.05)";
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
    ctx.strokeStyle = "#ff8c1a"; ctx.lineWidth = 2.4; ctx.setLineDash([]);
    ctx.beginPath(); ctx.moveTo(X(ax), Y(ay)); ctx.lineTo(X(bx), Y(by)); ctx.stroke();

    // Inbound arrow. The direction is resolved server-side and delivered
    // already projected to pixels: metric Y and image y have opposite
    // handedness, so deriving the normal here would draw it backwards.
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

    ctx.font = "600 10px ui-monospace, monospace"; ctx.fillStyle = "#ff8c1a";
    ctx.fillText(w.label || w.id, X(ax) + 8, Y(ay) - 7);
    ctx.restore();
  }

  for (const p of z.proximity || []) {
    if (!p.points_px) continue;
    const [px, py] = p.points_px[0];
    ctx.save();
    ctx.strokeStyle = "rgba(255,51,68,.6)"; ctx.lineWidth = 1.6; ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.arc(X(px), Y(py), 9, 0, Math.PI * 2); ctx.stroke();
    ctx.setLineDash([]);
    ctx.beginPath(); ctx.moveTo(X(px) - 6, Y(py)); ctx.lineTo(X(px) + 6, Y(py));
    ctx.moveTo(X(px), Y(py) - 6); ctx.lineTo(X(px), Y(py) + 6); ctx.stroke();
    ctx.font = "600 9.5px ui-monospace, monospace"; ctx.fillStyle = "rgba(255,51,68,.85)";
    ctx.fillText(`${p.label} <${p.threshold_m}m`, X(px) + 12, Y(py) + 3);
    ctx.restore();
  }
}

/* ------------------------------------------------------------------- HUD */
function renderStatus(s) {
  if (!s) return;
  const bar = $("#threatbar");
  bar.dataset.threat = s.threat || "INFO";
  $("#threatlvl").textContent = s.threat || "INFO";
  $("#threatsub").textContent =
    s.live_alerts ? `${s.live_alerts} live alert${s.live_alerts === 1 ? "" : "s"}`
                  : "sector clear";
  if (typeof s.queue_drops === "number") {
    const q = $("#chip-drops");
    q.querySelector("b").textContent = s.queue_drops;
    q.className = "chip" + (s.queue_drops > 0 ? " warn" : "");
  }
  if (s.counters) {
    $("#chip-counters").querySelector("b").textContent =
      `${s.counters.promoted}/${s.counters.cleared}`;
  }
}

function renderOverlayHud(o) {
  if (!o) return;
  const p = o.perf || {};
  $("#hud-fps").textContent = (p.fps ?? 0).toFixed(1);
  $("#hud-detect").textContent = `${p.detect_ms ?? 0}ms`;
  $("#hud-track").textContent = `${p.track_ms ?? 0}ms`;
  $("#hud-logic").textContent = `${p.logic_ms ?? 0}ms`;
  $("#hud-tracks").textContent = (o.boxes || []).length;
  $("#hud-frame").textContent = o.frame_idx ?? 0;

  const h = o.health || {};
  const chip = $("#chip-health");
  chip.querySelector("b").textContent = h.state || "--";
  chip.className = "chip " + (h.state === "OK" ? "good"
    : h.state === "DEGRADED" ? "warn" : "bad");
  chip.title = h.detail || `blur ${h.blur} / ssim ${h.ssim}`;
  $("#hud-blur").textContent = (h.blur ?? 0).toFixed(0);
  $("#hud-ssim").textContent = (h.ssim ?? 0).toFixed(3);

  // Unattended-baggage dwell is shown while the timer runs, not only when it
  // expires -- an operator should see a countdown, not a surprise.
  const bags = (o.bags || []).filter((b) => b.state !== "ORPHAN");
  $("#hud-bags").textContent = bags.length
    ? bags.map((b) => `#${b.track_id}:${b.state}`).join(" ")
    : "none";
}

function renderAlerts() {
  const box = $("#alerts");
  const items = state.alerts;
  box.innerHTML = "";
  if (!items.length) {
    box.appendChild(el("div", "empty",
      "No active alerts.\nObservations are accumulating but none has met the confirmation threshold."));
    return;
  }
  for (const a of items) {
    const card = el("div", "alert" + (a.state === "CLEARED" ? " cleared" : "")
      + (state.selected === a.alert_id ? " sel" : ""));
    card.dataset.threat = a.threat;
    card.onclick = () => selectAlert(a.alert_id);

    const r1 = el("div", "row1");
    r1.appendChild(el("span", "kind", a.kind.replace(/_/g, " ")));
    r1.appendChild(el("span", "time", hhmmss(a.opened_at)));
    card.appendChild(r1);

    card.appendChild(el("div", "why", a.rationale || ""));

    const meta = el("div", "meta");
    const bandCls = a.confidence_band === "HIGH" ? "hi"
      : a.confidence_band === "MEDIUM" ? "med" : "lo";
    meta.appendChild(el("span", `pill ${bandCls}`,
      `CONF ${Math.round(a.confidence * 100)}% ${a.confidence_band}`));
    meta.appendChild(el("span", "pill", a.camera_id));
    if (a.track_ids && a.track_ids.length)
      meta.appendChild(el("span", "pill", `#${a.track_ids.join(",#")}`));
    if (a.relay_sent) meta.appendChild(el("span", "pill relay", "RELAY <3KB"));
    if (a.state === "CLEARED") meta.appendChild(el("span", "pill", "CLEARED"));
    card.appendChild(meta);
    box.appendChild(card);
  }
}

async function selectAlert(id) {
  state.selected = id;
  renderAlerts();
  setTab("evidence");
  const box = $("#evidence");
  box.innerHTML = "";
  box.appendChild(el("div", "empty", "Loading evidence..."));
  try {
    const r = await fetch(`/api/alerts/${id}`);
    if (!r.ok) throw new Error("not found");
    renderEvidence(await r.json());
  } catch (e) {
    box.innerHTML = "";
    box.appendChild(el("div", "empty", "Evidence could not be loaded."));
  }
}

function renderEvidence(d) {
  const box = $("#evidence");
  box.innerHTML = "";
  const wrap = el("div", "evidence");

  if (d.crop_path) {
    const i = el("img"); i.src = `/evidence/${d.crop_path}`; i.alt = "subject crop";
    wrap.appendChild(i);
  }
  if (d.keyframe_path) {
    const i = el("img"); i.src = `/evidence/${d.keyframe_path}`; i.alt = "keyframe";
    wrap.appendChild(i);
  }
  if (d.clip_path) {
    const v = el("video");
    v.src = `/evidence/${d.clip_path}`; v.controls = true; v.loop = true; v.muted = true;
    wrap.appendChild(v);
  }

  const t = el("table", "kvtable");
  const add = (k, v) => {
    if (v === null || v === undefined || v === "") return;
    const tr = el("tr");
    tr.appendChild(el("td", null, k));
    tr.appendChild(el("td", null, String(v)));
    t.appendChild(tr);
  };
  add("alert id", d.alert_id);
  add("kind", d.kind);
  add("threat", d.threat);
  add("confidence", `${(d.confidence * 100).toFixed(1)}% (${d.confidence_band})`);
  add("state", d.state);
  add("opened", d.opened_at);
  add("cleared", d.cleared_at);
  add("rationale", d.rationale);
  add("observations", (d.observations || []).length);
  if (d.geometry) {
    add("zone", d.geometry.zone_label || d.geometry.zone_id);
    add("direction", d.geometry.direction);
    if (d.geometry.distance_m != null) add("distance", `${d.geometry.distance_m} m`);
    add("metric", String(!!d.geometry.metric));
  }
  add("homography", d.homography_id);
  if (d.model_ids) for (const [k, v] of Object.entries(d.model_ids)) add(`model:${k}`, v);
  if (d.config_snapshot) {
    add("profile", d.config_snapshot.profile);
    if (d.config_snapshot.confirm)
      add("confirm rule", `${d.config_snapshot.confirm.window_s}s / ratio ${d.config_snapshot.confirm.min_ratio} / min ${d.config_snapshot.confirm.min_hits}`);
    if (d.config_snapshot.baggage)
      add("dwell threshold", `${d.config_snapshot.baggage.dwell_s}s`);
  }
  wrap.appendChild(t);

  for (const det of d.detections || []) {
    const dt = el("table", "kvtable");
    const add2 = (k, v) => {
      if (v === null || v === undefined) return;
      const tr = el("tr");
      tr.appendChild(el("td", null, k)); tr.appendChild(el("td", null, String(v)));
      dt.appendChild(tr);
    };
    add2("track", `#${det.track_id} ${det.class_name} (${det.group})`);
    add2("detector conf", det.confidence);
    add2("model", det.model_id);
    add2("footpoint", det.footpoint_m ? `${det.footpoint_m[0]}, ${det.footpoint_m[1]} m` : null);
    add2("edge clipped", String(det.edge_clipped));
    if (det.identity) {
      add2("identity", det.identity.label);
      add2("similarity", det.identity.similarity);
      add2("threshold", det.identity.threshold);
    }
    wrap.appendChild(dt);
  }
  box.appendChild(wrap);
}

/* ----------------------------------------------------------------- query */
async function runQuery(text) {
  const out = $("#qresult");
  out.innerHTML = "";
  out.appendChild(el("div", "qnote", "routing..."));
  $("#qbtn").disabled = true;
  state.grounding = null;

  try {
    const r = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        q: text, camera_id: state.cameraId,
        alert_id: state.selected || undefined,
      }),
    });
    renderQuery(await r.json());
  } catch (e) {
    out.innerHTML = "";
    out.appendChild(el("div", "qerr", `Query failed: ${e.message}`));
  } finally {
    $("#qbtn").disabled = false;
  }
}

function renderQuery(d) {
  const out = $("#qresult");
  out.innerHTML = "";

  const route = el("div", "qroute");
  route.appendChild(el("span", null, `ROUTE: ${(d.route || "?").toUpperCase()}`));
  if (d.backend) route.appendChild(el("span", "lat", d.backend));
  if (d.latency_ms != null) route.appendChild(el("span", "lat", `${d.latency_ms} ms`));
  out.appendChild(route);

  if (d.ok === false) {
    out.appendChild(el("div", "qerr", d.error || "unavailable"));
    if (d.detail) out.appendChild(el("div", "qnote", d.detail));
    if (d.fallback && d.fallback.answer) {
      out.appendChild(el("div", "qnote", "Answered by the structured route instead:"));
      out.appendChild(el("div", "qanswer", d.fallback.answer));
    }
    if (d.licence) out.appendChild(el("div", "qnote", `Licence: ${d.licence}`));
    return;
  }

  if (d.answer && d.route !== "grounding") out.appendChild(el("div", "qanswer", d.answer));
  if (d.sql) out.appendChild(el("div", "qsql", d.sql + "\n-- params: " + JSON.stringify(d.params)));

  if (d.route === "grounding") {
    state.grounding = d.boxes || [];
    out.appendChild(el("div", "qanswer",
      `${d.count} region${d.count === 1 ? "" : "s"} grounded for "${d.phrase}".`));
    if (d.answer) out.appendChild(el("div", "qsql", d.answer.slice(0, 400)));
    out.appendChild(el("div", "qnote", `${d.model_id} - ${d.licence}`));
  }

  if (d.rows && d.rows.length && d.route === "structured") {
    const t = el("table", "kvtable");
    for (const row of d.rows.slice(0, 12)) {
      const tr = el("tr");
      tr.appendChild(el("td", null, hhmmss(row.opened_at)));
      tr.appendChild(el("td", null,
        `${row.threat} ${row.kind} (${Math.round(row.confidence * 100)}%)`));
      tr.onclick = () => selectAlert(row.alert_id);
      tr.style.cursor = "pointer";
      t.appendChild(tr);
    }
    out.appendChild(t);
  }
  if (d.rows && d.route === "identity") {
    for (const row of d.rows) {
      out.appendChild(el("div", "qnote",
        `#${row.track_id} ${row.class_name}: ${row.label}` +
        (row.similarity != null ? ` (sim ${row.similarity}, thr ${row.threshold})` : "") +
        (row.note ? ` - ${row.note}` : "")));
    }
  }
}

/* ------------------------------------------------------------------ tabs */
function setTab(name) {
  state.tab = name;
  document.querySelectorAll(".tabs button").forEach((b) =>
    b.classList.toggle("on", b.dataset.tab === name));
  $("#pane-live").hidden = name !== "live";
  $("#pane-evidence").hidden = name !== "evidence";
  $("#pane-system").hidden = name !== "system";
  if (name === "system") renderSystem();
}

function renderSystem() {
  const box = $("#system");
  box.innerHTML = "";
  const s = state.status || {};
  const t = el("table", "kvtable");
  const add = (k, v) => {
    const tr = el("tr");
    tr.appendChild(el("td", null, k)); tr.appendChild(el("td", null, String(v)));
    t.appendChild(tr);
  };
  add("profile", s.profile);
  add("offline (F15)", s.offline);
  add("run id", (s.run_id || "").slice(0, 12));
  add("pose / F16", s.pose_enabled ? "enabled" : "disabled");
  for (const [k, v] of Object.entries(s.models || {})) add(`model:${k}`, v);
  if (s.identity) {
    add("re-id available", s.identity.embedder && s.identity.embedder.available);
    add("gallery identities", s.identity.identities);
    add("gallery vectors", s.identity.vectors);
    add("sim threshold", s.identity.threshold);
    if (s.identity.embedder && s.identity.embedder.reason)
      add("re-id note", s.identity.embedder.reason);
  }
  const g = (state.routes.routes || {}).grounding || {};
  add("F11 weights present", g.weights_present);
  add("F11 loaded", g.loaded);
  if (g.licence) add("F11 licence", g.licence);
  if (g.error) add("F11 error", g.error);
  box.appendChild(t);

  const qt = el("table", "kvtable");
  const hdr = el("tr");
  hdr.appendChild(el("td", null, "QUEUE"));
  hdr.appendChild(el("td", null, "pushed / dropped / peak / policy"));
  qt.appendChild(hdr);
  for (const q of s.queues || []) {
    const tr = el("tr");
    tr.appendChild(el("td", null, q.name));
    tr.appendChild(el("td", null,
      `${q.pushed} / ${q.dropped} / ${q.high_water} / ${q.policy}`));
    qt.appendChild(tr);
  }
  box.appendChild(qt);

  for (const c of s.cameras || []) {
    const ct = el("table", "kvtable");
    const add2 = (k, v) => {
      const tr = el("tr");
      tr.appendChild(el("td", null, k)); tr.appendChild(el("td", null, String(v)));
      ct.appendChild(tr);
    };
    add2("camera", `${c.camera_id} (${c.name})`);
    add2("health", c.health);
    add2("frames in / processed", `${c.frames_in} / ${c.frames_processed}`);
    add2("fps (measured)", c.perf.fps);
    add2("calibrated", c.metric ? `yes (${c.homography.homography_id})` : "no - pixel space");
    if (c.homography) add2("homography residual", `${c.homography.residual_m} m`);
    add2("pending candidates", c.pending_candidates);
    box.appendChild(ct);
  }
}

/* -------------------------------------------------------------- websocket */
function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);

  ws.onopen = () => {
    $("#chip-link").className = "chip good";
    $("#chip-link").querySelector("b").textContent = "LIVE";
  };
  ws.onclose = () => {
    $("#chip-link").className = "chip bad";
    $("#chip-link").querySelector("b").textContent = "LOST";
    setTimeout(connect, 1500);
  };
  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.type === "hello") {
      state.zones = m.zones || {};
      state.status = m.status || {};
      state.routes = m.routes || {};
      const cams = Object.keys(state.zones);
      if (cams.length && !state.cameraId) {
        state.cameraId = cams[0];
        $("#video").src = `/api/stream/${state.cameraId}`;
        $("#cam-name").textContent =
          (state.status.cameras || []).find((c) => c.camera_id === state.cameraId)?.name
          || state.cameraId;
        const z = state.zones[state.cameraId];
        $("#cal").innerHTML = z.metric
          ? `HOMOGRAPHY <b>${z.homography.homography_id}</b><br><span class="est">SCALE ESTIMATED, NOT SURVEYED</span>`
          : `<span class="est">UNCALIBRATED - PIXEL SPACE</span>`;
      }
      renderSystem();
    } else if (m.type === "tick") {
      const o = m.cameras[state.cameraId];
      if (o) { state.overlay = o; renderOverlayHud(o); draw(); }
      if (m.status) {
        state.status = { ...state.status, ...m.status };
        renderStatus(m.status);
        state.alerts = m.status.alerts || [];
        renderAlerts();
      }
    } else if (m.type === "alert") {
      // Fires on open and close; the tick already carries the live set, so
      // this only needs to handle the closed-alert flash.
      if (m.phase === "opened" && m.alert.threat === "CRITICAL") {
        document.body.animate(
          [{ boxShadow: "inset 0 0 0 0 rgba(255,51,68,0)" },
           { boxShadow: "inset 0 0 90px 0 rgba(255,51,68,.34)" },
           { boxShadow: "inset 0 0 0 0 rgba(255,51,68,0)" }],
          { duration: 850 });
      }
    }
  };
  setInterval(() => { if (ws.readyState === 1) ws.send("ping"); }, 15000);
}

/* ------------------------------------------------------------------ init */
window.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".tabs button").forEach((b) =>
    (b.onclick = () => setTab(b.dataset.tab)));

  $("#qform").onsubmit = (e) => {
    e.preventDefault();
    const v = $("#q").value.trim();
    if (v) runQuery(v);
  };
  document.querySelectorAll(".suggests button").forEach((b) =>
    (b.onclick = () => { $("#q").value = b.textContent; runQuery(b.textContent); }));

  window.addEventListener("resize", draw);
  setInterval(draw, 120);   // redraw between ticks so resize stays smooth
  connect();
});
