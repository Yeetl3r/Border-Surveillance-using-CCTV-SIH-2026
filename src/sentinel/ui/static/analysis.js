/* SENTINEL — analysis page.
 *
 * Reached after a feed is submitted on the landing page. Shows a loading state
 * driven by real pipeline telemetry, then the analysis itself: the footage as
 * given on the left, the same footage with detection on the right, and the
 * alerts beneath with their evidence.
 *
 * F12 still governs the overlay: threat sets colour, confidence sets stroke
 * style. They are separate fields and are never combined.
 */

const THREAT = {
  INFO:     { c: "#5aa9e6", bg: "rgba(90,169,230,.12)" },
  FRIENDLY: { c: "#2fd98a", bg: "rgba(47,217,138,.12)" },
  WATCH:    { c: "#ffc21a", bg: "rgba(255,194,26,.12)" },
  WARNING:  { c: "#ff7a1a", bg: "rgba(255,122,26,.12)" },
  CRITICAL: { c: "#ff2b3d", bg: "rgba(255,43,61,.14)" },
};
const col = (t) => (THREAT[t] || THREAT.INFO).c;

const $ = (s) => document.querySelector(s);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
};
const hhmmss = (iso) => (iso ? new Date(iso).toISOString().substr(11, 8) : "--:--:--");
const bandOf = (c) => (c >= 0.75 ? "HIGH" : c >= 0.5 ? "MEDIUM" : "LOW");

const S = {
  cameraId: null,
  sourceLabel: sessionStorage.getItem("sentinel.source") || "reference clip",
  status: {},
  zones: null,
  overlay: null,
  alerts: [],
  selected: null,
  evidence: null,
  ready: false,
};

/* ══════════════ loading ══════════════ */
const STEPS = 9;
let done = 0;
const seen = new Set();
const t0 = Date.now();

function stamp() {
  const ms = Date.now() - t0, s = Math.floor(ms / 1000);
  return String(Math.floor(s / 60)).padStart(2, "0") + ":" +
         String(s % 60).padStart(2, "0") + "." +
         String(Math.floor((ms % 1000) / 10)).padStart(2, "0");
}

function step(key, text, stage) {
  if (seen.has(key)) return;
  seen.add(key);
  done++;
  const li = el("li");
  li.appendChild(el("span", null, stamp()));
  li.appendChild(el("em", null, text));
  li.lastChild.style.fontStyle = "normal";
  const log = $("#loader-log");
  log.appendChild(li);
  while (log.children.length > 6) log.firstChild.remove();

  const pct = Math.min(100, Math.round((done / STEPS) * 100));
  $("#loader-fill").style.width = pct + "%";
  $("#loader-pct").textContent = pct + "%";
  if (stage) $("#loader-stage").textContent = stage;
}

// A warm pipeline is ready almost immediately, which made the loading screen
// flash past unread. Hold it briefly so the stages it reports can actually be
// followed -- the wait is real work being reported, not a fake spinner.
const MIN_LOADER_MS = 2400;

function finishLoading() {
  $("#loader-fill").style.width = "100%";
  $("#loader-pct").textContent = "100%";
  $("#loader-stage").textContent = "ready";
  const held = Math.max(0, MIN_LOADER_MS - (Date.now() - t0));
  setTimeout(() => {
    $("#loader").classList.add("gone");
    $("#app").hidden = false;
    setTimeout(() => { $("#loader").remove(); draw(); }, 520);
  }, held + 260);
}

async function warmUp() {
  $("#loader-source").textContent = S.sourceLabel;
  const started = Date.now();

  const poll = async () => {
    if (S.ready) return;
    try {
      const st = await (await fetch("/api/status")).json();
      S.status = st;
      const cam = (st.cameras || [])[0] || {};
      S.cameraId = S.cameraId || cam.camera_id;

      if (cam.frames_processed > 0)
        step("decode", `decode running · ${cam.frames_processed} frames · health ${cam.health}`,
             "tier 0 · ingest");
      if (cam.homography && cam.homography.calibrated)
        step("h", `homography ${cam.homography.homography_id} · residual ${cam.homography.residual_m} m`,
             "tier 0 · calibration");
      if (cam.perf && cam.perf.fps)
        step("fps", `perception ${cam.perf.fps.toFixed(1)} FPS · detect ${cam.perf.detect_ms} ms · track ${cam.perf.track_ms} ms`,
             "tier 1 · perception");
      if (cam.perf && cam.perf.logic_ms !== undefined)
        step("logic", `spatial + temporal reasoning ${cam.perf.logic_ms} ms/frame`,
             "tier 2–3 · reasoning");
      if (st.identity && st.identity.vectors)
        step("id", `identity gallery · ${st.identity.identities} enrolled · threshold ${st.identity.threshold}`,
             "tier 5 · identity");
      if (cam.pending_candidates > 0)
        step("obs", `${cam.pending_candidates} observation(s) accumulating — not yet alerts`,
             "tier 3 · confirmation");
      if (st.counters && st.counters.promoted > 0)
        step("promo", `${st.counters.promoted} alert(s) promoted · evidence written`,
             "tier 8 · alerting");
      if (st.queue_drops > 0)
        step("drop", `Q1 drop-oldest engaged · ${st.queue_drops} frames shed to hold latency`,
             "tier 0 · backpressure");

      const enough = cam.frames_processed > 20 &&
                     (st.counters?.promoted > 0 || Date.now() - started > 9000);
      if (enough) {
        step("threat", `sector threat computed from the live alert set · ${st.threat}`, "ready");
        await loadAlerts();
        S.ready = true;
        bindStreams();
        paint();
        finishLoading();
        return;
      }
    } catch (_) { /* server may still be rebinding the source */ }
    setTimeout(poll, 600);
  };
  poll();
}

/* ══════════════ streams ══════════════ */
function bindStreams() {
  if (!S.cameraId) return;
  // Left is the raw sensor buffer -- what the detector actually receives.
  // Right is the operator view, which is where enhancement is allowed to run.
  $("#raw").src = `/api/stream/${S.cameraId}?raw=1`;
  $("#det").src = `/api/stream/${S.cameraId}`;
  $("#src-label").textContent = S.sourceLabel;
}

/* ══════════════ overlay ══════════════ */
function draw() {
  const cv = $("#ovl"), img = $("#det");
  if (!cv || !S.overlay) return;
  const w = cv.clientWidth, h = cv.clientHeight;
  if (!w || !h) return;
  if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
  const ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, w, h);

  // The frame is object-fit:contain — mirror that or the boxes drift.
  const nw = img.naturalWidth || 1280, nh = img.naturalHeight || 720;
  const scale = Math.min(w / nw, h / nh);
  const ox = (w - nw * scale) / 2, oy = (h - nh * scale) / 2;
  const X = (x) => ox + x * scale, Y = (y) => oy + y * scale;

  drawZones(ctx, X, Y);

  for (const b of S.overlay.boxes || []) {
    const c = col(b.threat);
    const band = bandOf(b.confidence);
    const [x1, y1, x2, y2] = b.bbox;
    const x = X(x1), y = Y(y1), bw = (x2 - x1) * scale, bh = (y2 - y1) * scale;

    if (band === "LOW") {
      ctx.save();
      ctx.beginPath(); ctx.rect(x, y, bw, bh); ctx.clip();
      ctx.globalAlpha = .16; ctx.strokeStyle = c; ctx.lineWidth = 1.1;
      for (let i = -bh; i < bw; i += 8) {
        ctx.beginPath(); ctx.moveTo(x + i, y + bh); ctx.lineTo(x + i + bh, y); ctx.stroke();
      }
      ctx.restore();
    }

    ctx.save();
    ctx.strokeStyle = c;
    if (band === "HIGH") { ctx.setLineDash([]); ctx.lineWidth = 2; }
    else if (band === "MEDIUM") { ctx.setLineDash([7, 4]); ctx.lineWidth = 1.8; }
    else { ctx.setLineDash([2, 4]); ctx.lineWidth = 1.5; }
    ctx.strokeRect(x, y, bw, bh);
    ctx.restore();

    const bits = [`#${b.track_id}`, (b.group || "").toUpperCase(),
                  Number(b.confidence).toFixed(2)];
    if (b.footpoint_m) bits.push(`${b.footpoint_m[1].toFixed(1)} m`);
    const id = b.identity;
    if (id && id.label === "FRIENDLY") bits.push((id.matched || "FRIENDLY").toUpperCase());
    else if (id && id.label === "UNIDENTIFIED") bits.push("UNIDENTIFIED");
    if (b.stance && !["UNKNOWN", "STANDING"].includes(b.stance)) bits.push(b.stance);
    if (b.weapon) bits.push(b.weapon.toUpperCase());
    if (b.reclassified) bits.push("HUMAN");
    const label = bits.join(" · ");

    ctx.save();
    ctx.font = "500 10px 'IBM Plex Mono', ui-monospace, monospace";
    const tw = ctx.measureText(label).width + 12;
    const ly = Math.max(0, y - 20);
    ctx.fillStyle = c; ctx.fillRect(x, ly, tw, 17);
    ctx.fillStyle = "#080506"; ctx.fillText(label, x + 6, ly + 12);
    ctx.restore();
  }

  drawAerial(ctx, X, Y, scale);
}

// Drones are drawn separately because they are not tracked: they have no
// ground footpoint, and the tracker's motion model assumes a ground plane.
function drawAerial(ctx, X, Y, scale) {
  const a = S.overlay.aerial;
  if (!a) return;

  for (const d of a.drones || []) {
    const [x1, y1, x2, y2] = d.bbox_px;
    const x = X(x1), y = Y(y1), bw = (x2 - x1) * scale, bh = (y2 - y1) * scale;
    ctx.save();
    ctx.strokeStyle = col("CRITICAL"); ctx.lineWidth = 2; ctx.setLineDash([]);
    ctx.strokeRect(x, y, bw, bh);
    // A reticle: an aerial contact reads differently from a ground box.
    ctx.beginPath();
    ctx.moveTo(x + bw / 2, y - 7); ctx.lineTo(x + bw / 2, y - 1);
    ctx.moveTo(x + bw / 2, y + bh + 1); ctx.lineTo(x + bw / 2, y + bh + 7);
    ctx.stroke();
    const label = `AERIAL · ${Number(d.conf).toFixed(2)}`;
    ctx.font = "500 10px 'IBM Plex Mono', ui-monospace, monospace";
    const tw = ctx.measureText(label).width + 12;
    ctx.fillStyle = col("CRITICAL"); ctx.fillRect(x, Math.max(0, y - 20), tw, 17);
    ctx.fillStyle = "#080506"; ctx.fillText(label, x + 6, Math.max(0, y - 20) + 12);
    ctx.restore();
  }

  // Withheld candidates. Showing the rejection is the point: an operator who
  // can see the system decline a bird trusts it more than one that is simply
  // silent, and it is the only visible evidence that the check ran at all.
  for (const wv of a.withheld || []) {
    const [x1, y1, x2, y2] = wv.bbox_px;
    const x = X(x1), y = Y(y1), bw = (x2 - x1) * scale, bh = (y2 - y1) * scale;
    ctx.save();
    ctx.strokeStyle = "rgba(160,160,170,.75)";
    ctx.lineWidth = 1.2; ctx.setLineDash([3, 4]);
    ctx.strokeRect(x, y, bw, bh);
    ctx.setLineDash([]);
    const label = `BIRD · drone ${Number(wv.drone_conf).toFixed(2)} vs bird ${Number(wv.bird_conf).toFixed(2)}`;
    ctx.font = "500 9px 'IBM Plex Mono', ui-monospace, monospace";
    ctx.fillStyle = "rgba(160,160,170,.9)";
    ctx.fillText(label, x, Math.max(9, y - 5));
    ctx.restore();
  }
}

function drawZones(ctx, X, Y) {
  const z = S.zones;
  if (!z) return;
  for (const zone of z.zones || []) {
    if (!zone.points_px) continue;
    ctx.save();
    ctx.strokeStyle = "rgba(255,31,60,.3)"; ctx.setLineDash([6, 5]); ctx.lineWidth = 1.1;
    ctx.beginPath();
    zone.points_px.forEach(([px, py], i) =>
      i ? ctx.lineTo(X(px), Y(py)) : ctx.moveTo(X(px), Y(py)));
    ctx.closePath(); ctx.stroke(); ctx.restore();
  }
  for (const w of z.tripwires || []) {
    if (!w.points_px) continue;
    const [[ax, ay], [bx, by]] = w.points_px;
    ctx.save();
    ctx.strokeStyle = "#ff2b3d"; ctx.lineWidth = 1.6; ctx.setLineDash([11, 9]);
    ctx.beginPath(); ctx.moveTo(X(ax), Y(ay)); ctx.lineTo(X(bx), Y(by)); ctx.stroke();
    ctx.setLineDash([]);
    if (w.inbound_arrow_px) {
      const [[sx0, sy0], [tx0, ty0]] = w.inbound_arrow_px;
      const sx = X(sx0), sy = Y(sy0), tx = X(tx0), ty = Y(ty0);
      const a = Math.atan2(ty - sy, tx - sx), hd = 7;
      ctx.beginPath();
      ctx.moveTo(sx, sy); ctx.lineTo(tx, ty);
      ctx.lineTo(tx - hd * Math.cos(a - .4), ty - hd * Math.sin(a - .4));
      ctx.moveTo(tx, ty);
      ctx.lineTo(tx - hd * Math.cos(a + .4), ty - hd * Math.sin(a + .4));
      ctx.stroke();
    }
    ctx.font = "500 9.5px 'IBM Plex Mono', monospace";
    ctx.fillStyle = "rgba(255,43,61,.85)";
    ctx.fillText(`${w.label || w.id} · inbound`, X(ax) + 8, Y(ay) - 8);
    ctx.restore();
  }
  for (const p of z.proximity || []) {
    if (!p.points_px) continue;
    const [px, py] = p.points_px[0];
    ctx.save();
    ctx.strokeStyle = "rgba(255,122,26,.7)"; ctx.setLineDash([4, 4]); ctx.lineWidth = 1.3;
    ctx.beginPath(); ctx.arc(X(px), Y(py), 9, 0, Math.PI * 2); ctx.stroke();
    ctx.setLineDash([]);
    ctx.font = "500 9px 'IBM Plex Mono', monospace";
    ctx.fillStyle = "rgba(255,122,26,.85)";
    ctx.fillText(`${p.label || p.id} < ${p.threshold_m} m`, X(px) + 13, Y(py) + 3);
    ctx.restore();
  }
}

/* ══════════════ paint ══════════════ */
function paint() {
  const st = S.status, cam = (st.cameras || [])[0] || {}, perf = cam.perf || {};
  const threat = st.threat || "INFO";
  const t = THREAT[threat] || THREAT.INFO;

  $("#threat-val").textContent = threat;
  $("#threat-val").style.color = t.c;
  const pill = $("#threat-pill");
  pill.style.borderColor = t.c;
  pill.style.background = t.bg;
  pill.querySelector("i").style.background = t.c;
  pill.querySelector("i").style.boxShadow = `0 0 10px ${t.c}`;

  const h = (S.overlay && S.overlay.health) || {};
  const strip = [
    ["FRAME TIME", (perf.frame_ms ?? 0).toFixed(1) + " ms"],
    ["SPATIAL + TEMPORAL", (perf.logic_ms ?? 0).toFixed(2) + " ms"],
    ["THROUGHPUT", (perf.fps ?? 0).toFixed(1) + " FPS"],
    ["TRACKS", String((S.overlay?.boxes || []).length)],
    ["CAMERA", h.state || cam.health || "—"],
    ["CALIBRATION", cam.metric ? "metric" : "pixel space"],
  ];
  const box = $("#strip");
  box.innerHTML = "";
  for (const [k, v] of strip) {
    const d = el("div");
    d.appendChild(el("div", "k", k));
    const val = el("div", "v", v);
    if (k === "CAMERA" && v !== "OK" && v !== "—") val.style.color = "#ff8f9a";
    d.appendChild(val);
    box.appendChild(d);
  }

  renderAlerts();
}

function renderAlerts() {
  const list = $("#alert-list");
  $("#tab-count").textContent = S.alerts.length;
  list.innerHTML = "";
  if (!S.alerts.length) {
    list.appendChild(el("p", "muted",
      "No alerts yet. Observations are accumulating but none has met the confirmation threshold."));
    return;
  }
  for (const a of S.alerts) {
    const c = col(a.threat);
    const card = el("div", "alert" + (a.alert_id === S.selected ? " on" : ""));
    card.style.borderLeftColor = c;
    card.onclick = () => selectAlert(a.alert_id);

    const r1 = el("div", "r1");
    const kind = el("div", "kind", (a.kind || "").replace(/_/g, " ").toLowerCase()
      .replace(/^./, (m) => m.toUpperCase()));
    kind.style.color = c;
    r1.appendChild(kind);
    r1.appendChild(el("div", "time", hhmmss(a.opened_at)));
    card.appendChild(r1);
    card.appendChild(el("div", "why", a.rationale || ""));

    const r2 = el("div", "r2");
    const th = el("span", "pill solid", a.threat);
    th.style.background = c;
    r2.appendChild(th);
    r2.appendChild(el("span", "pill",
      `CONF ${a.confidence_band} · ${Number(a.confidence).toFixed(2)}`));
    if (a.track_ids && a.track_ids.length)
      r2.appendChild(el("span", "pill", "#" + a.track_ids.join(", #")));
    card.appendChild(r2);
    list.appendChild(card);
  }
}

async function selectAlert(id) {
  S.selected = id;
  renderAlerts();
  const box = $("#alert-detail");
  box.innerHTML = "";
  box.appendChild(el("p", "muted", "Loading evidence…"));
  try {
    S.evidence = await (await fetch(`/api/alerts/${id}`)).json();
  } catch (_) {
    box.innerHTML = "";
    box.appendChild(el("p", "muted", "Evidence could not be loaded."));
    return;
  }
  renderEvidence();
}

function renderEvidence() {
  const d = S.evidence, box = $("#alert-detail");
  box.innerHTML = "";
  if (!d) return;

  const h3 = el("h3", null, (d.kind || "").replace(/_/g, " ").toLowerCase()
    .replace(/^./, (m) => m.toUpperCase()));
  h3.style.color = col(d.threat);
  box.appendChild(h3);
  box.appendChild(el("p", "why", d.rationale || ""));

  // The two images the alert was actually raised on. These are real files from
  // the evidence store, served from disk — the keyframe is the unannotated
  // frame the detector saw, and the crop is the subject cut from it.
  const shots = el("div", "shots");
  const add = (cap, path, alt) => {
    const s = el("div", "shot");
    s.appendChild(el("div", "cap", cap));
    const b = el("div", "box");
    if (path) {
      const im = el("img");
      im.src = "/evidence/" + String(path).replace(/\\/g, "/");
      im.alt = alt;
      im.onerror = () => { b.classList.add("empty"); im.remove(); };
      b.appendChild(im);
    } else {
      b.classList.add("empty");
    }
    s.appendChild(b);
    shots.appendChild(s);
  };
  add("KEYFRAME · UNANNOTATED, AS THE DETECTOR SAW IT", d.keyframe_path, "alert keyframe");
  add("SUBJECT CROP", d.crop_path, "subject crop");
  // Only meaningful for a person, and only when a face was actually visible.
  // A subject facing away has no face to show, and the tile says so rather
  // than being filled with a guess.
  add("IDENTIFIED PERSON · FACE", d.face_path, "face of the subject");
  box.appendChild(shots);

  if (d.clip_path) {
    const v = el("video");
    v.src = "/evidence/" + String(d.clip_path).replace(/\\/g, "/");
    v.controls = true; v.loop = true; v.muted = true;
    v.style.cssText = "width:100%;margin-top:12px;border:1px solid rgba(255,255,255,.1);border-radius:12px";
    box.appendChild(v);
  }

  const t = el("table", "kv");
  const row = (k, v) => {
    if (v === null || v === undefined || v === "") return;
    const tr = el("tr");
    tr.appendChild(el("td", null, k));
    tr.appendChild(el("td", null, String(v)));
    t.appendChild(tr);
  };
  row("threat", d.threat);
  row("confidence", `${(d.confidence * 100).toFixed(0)}% · ${d.confidence_band}`);
  row("state", d.state);
  row("opened", (d.opened_at || "").replace("T", " ").slice(0, 19));
  const g = d.geometry || {};
  row("zone", g.zone_label || g.zone_id);
  row("direction", g.direction);
  if (g.distance_m != null) row("distance", g.distance_m + " m");
  row("homography", d.homography_id);
  const det = (d.detections || [])[0];
  if (det) {
    row("detector conf", Number(det.confidence).toFixed(3));
    row("model", det.model_id);
    if (det.identity) {
      row("identity", det.identity.label);
      if (det.identity.similarity != null)
        row("similarity", `${det.identity.similarity} vs threshold ${det.identity.threshold}`);
    }
  }
  const cs = d.config_snapshot || {};
  if (cs.confirm)
    row("confirmation", `${cs.confirm.min_hits} hits · window ${cs.confirm.window_s}s · ratio ${cs.confirm.min_ratio}`);
  row("observations", (d.observations || []).length);
  box.appendChild(t);
}

/* ══════════════ data ══════════════ */
async function loadAlerts() {
  try {
    const a = await (await fetch("/api/alerts?limit=40")).json();
    const next = (a.live && a.live.length ? a.live : a.recent) || [];
    if (next.length) S.alerts = next;
    if (!S.selected && S.alerts.length) selectAlert(S.alerts[0].alert_id);
  } catch (_) { /* keep last good */ }
}

/* ══════════════ ask ══════════════ */
async function ask(text) {
  const out = $("#ask-out");
  out.innerHTML = "";
  out.appendChild(el("div", "route", "routing…"));
  $("#ask-btn").disabled = true;
  try {
    const r = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ q: text, camera_id: S.cameraId,
                             alert_id: S.selected || undefined }),
    });
    renderAnswer(await r.json());
  } catch (e) {
    out.innerHTML = "";
    out.appendChild(el("div", "errbox", "Query failed: " + e.message));
  } finally {
    $("#ask-btn").disabled = false;
  }
}

function renderAnswer(d) {
  const out = $("#ask-out");
  out.innerHTML = "";
  const route = el("div", "route");
  route.appendChild(el("span", null, "ROUTE · " + (d.route || "?").toUpperCase()));
  if (d.backend) route.appendChild(el("span", "lat", d.backend));
  if (d.latency_ms != null) route.appendChild(el("span", "lat", d.latency_ms + " ms"));
  out.appendChild(route);

  if (d.ok === false) {
    out.appendChild(el("div", "errbox", d.error || "unavailable"));
    if (d.detail) out.appendChild(el("div", "sqlbox", d.detail));
    if (d.fallback && d.fallback.answer)
      out.appendChild(el("div", "answer", d.fallback.answer));
    if (d.licence) out.appendChild(el("div", "sqlbox", "Licence: " + d.licence));
    return;
  }
  if (d.route === "grounding") {
    out.appendChild(el("div", "answer",
      `${d.count} region${d.count === 1 ? "" : "s"} grounded for “${d.phrase}”.`));
    if (d.answer) out.appendChild(el("div", "sqlbox", d.answer.slice(0, 400)));
    if (d.licence) out.appendChild(el("div", "sqlbox", d.model_id + " — " + d.licence));
    return;
  }
  if (d.answer) out.appendChild(el("div", "answer", d.answer));
  if (d.sql) out.appendChild(el("div", "sqlbox", d.sql + "\n-- " + JSON.stringify(d.params)));
  if (d.rows && d.rows.length && d.route === "structured") {
    const t = el("table", "kv");
    for (const r of d.rows.slice(0, 10)) {
      const tr = el("tr");
      tr.appendChild(el("td", null, hhmmss(r.opened_at)));
      tr.appendChild(el("td", null,
        `${r.threat} · ${r.kind} · ${Math.round(r.confidence * 100)}%`));
      tr.style.cursor = "pointer";
      tr.onclick = () => { switchTab("alerts"); selectAlert(r.alert_id); };
      t.appendChild(tr);
    }
    out.appendChild(t);
  }
  if (d.rows && d.route === "identity") {
    for (const r of d.rows) {
      out.appendChild(el("div", "sqlbox",
        `#${r.track_id} ${r.class_name}: ${r.label}` +
        (r.similarity != null ? ` (sim ${r.similarity}, thr ${r.threshold})` : "") +
        (r.note ? ` — ${r.note}` : "")));
    }
  }
}

function switchTab(name) {
  document.querySelectorAll(".tabs button").forEach((b) =>
    b.classList.toggle("on", b.dataset.tab === name));
  $("#pane-alerts").hidden = name !== "alerts";
  $("#pane-ask").hidden = name !== "ask";
}

/* ══════════════ transport ══════════════ */
const fmt = (sec) => {
  if (!isFinite(sec) || sec < 0) sec = 0;
  const m = Math.floor(sec / 60), r = Math.floor(sec % 60);
  return `${m}:${String(r).padStart(2, "0")}`;
};

let seekHeld = false;

async function transport(action, value) {
  if (!S.cameraId) return;
  try {
    const r = await fetch(`/api/playback/${S.cameraId}`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action, value }),
    });
    if (r.ok) applyTransport(await r.json());
  } catch (_) { /* transient */ }
}

function applyTransport(t) {
  if (!t) return;
  S.transport = t;
  const bar = $("#transport"), live = $("#livebadge");

  // A live camera has no position to seek to and no past to replay, so the
  // controls are withheld entirely rather than shown doing nothing.
  bar.hidden = !!t.live;
  live.hidden = !t.live;
  if (t.live) return;

  $("#tp-play").textContent = t.paused ? "▶" : "❚❚";
  const fps = t.fps || 25;
  const pos = (t.position || 0) / fps;
  const dur = t.duration_s || (t.total_frames || 0) / fps;
  $("#tp-time").textContent = `${fmt(pos)} / ${fmt(dur)}`;
  if (!seekHeld && t.total_frames > 0) {
    $("#tp-seek").value = Math.round(((t.position || 0) / t.total_frames) * 1000);
  }
  document.querySelectorAll("#tp-rates button").forEach((b) =>
    b.classList.toggle("on", Math.abs(parseFloat(b.dataset.rate) - (t.rate || 1)) < 1e-6));
}

function initTransport() {
  $("#tp-play").onclick = () => transport("toggle");
  $("#tp-restart").onclick = () => transport("restart");

  const seek = $("#tp-seek");
  seek.addEventListener("input", () => { seekHeld = true; });
  seek.addEventListener("change", () => {
    seekHeld = false;
    transport("seek", Number(seek.value) / 1000);
  });

  document.querySelectorAll("#tp-rates button").forEach((b) =>
    (b.onclick = () => transport("rate", parseFloat(b.dataset.rate))));

  $("#use-device").onclick = async () => {
    const btn = $("#use-device");
    btn.disabled = true;
    btn.textContent = "STARTING CAMERA…";
    try {
      const r = await fetch("/api/source/device", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ index: 0 }),
      });
      const j = await r.json();
      if (!r.ok) throw new Error(j.detail || "no capture device");
      S.sourceLabel = "live capture · device camera 0";
      sessionStorage.setItem("sentinel.source", S.sourceLabel);
      $("#src-label").textContent = S.sourceLabel;
      // The pipeline was restarted; re-bind the streams so the MJPEG
      // connections attach to the new source rather than the closed one.
      setTimeout(() => { bindStreams(); switchTab("alerts"); }, 900);
      btn.textContent = "LIVE CAMERA RUNNING";
    } catch (e) {
      btn.textContent = "CAMERA UNAVAILABLE";
      btn.title = String(e.message);
    } finally {
      setTimeout(() => { btn.disabled = false; }, 1500);
    }
  };
}

/* ══════════════ socket ══════════════ */
function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
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
        if (m.status.alerts && m.status.alerts.length) S.alerts = m.status.alerts;
      }
      if (S.ready) { paint(); draw(); }
    } else if (m.type === "alert" && m.phase === "opened") {
      loadAlerts().then(() => S.ready && paint());
    }
  };
  ws.onclose = () => setTimeout(connect, 1500);
  setInterval(() => { if (ws.readyState === 1) ws.send("ping"); }, 15000);
}

/* ══════════════ boot ══════════════ */
window.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".tabs button").forEach((b) =>
    (b.onclick = () => switchTab(b.dataset.tab)));
  $("#ask-form").onsubmit = (e) => {
    e.preventDefault();
    const v = $("#ask-input").value.trim();
    if (v) ask(v);
  };
  document.querySelectorAll(".chips button").forEach((b) =>
    (b.onclick = () => { $("#ask-input").value = b.textContent; ask(b.textContent); }));

  initTransport();
  connect();
  warmUp();
  // Transport state comes from the source itself, so it stays correct when the
  // clip loops or the source is swapped underneath.
  setInterval(async () => {
    if (!S.ready || !S.cameraId) return;
    try {
      const st = await (await fetch("/api/status")).json();
      const cam = (st.cameras || [])[0];
      if (cam && cam.transport) applyTransport(cam.transport);
    } catch (_) { /* transient */ }
  }, 900);
  window.addEventListener("resize", draw);
  setInterval(() => { if (S.ready) draw(); }, 120);
});
