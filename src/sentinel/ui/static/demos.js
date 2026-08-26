/* SENTINEL — scripted demonstrations.
 *
 * The five demos are read from the server, which reads them from
 * configs/demos.yaml. Nothing about which clip belongs to which demo lives in
 * this file: a demo is a title, a claim and a list of panels, and this page
 * renders whatever it is handed.
 *
 * Selecting a demo asks the server to bind its clips to its cameras and start
 * only those pipelines. Everything after that is the ordinary live path — the
 * same MJPEG streams and the same WebSocket overlay the analysis page uses,
 * one canvas per panel instead of one for the page.
 */

const THREAT = {
  INFO:     { c: "#9dc4f0", bg: "rgba(157,196,240,.12)" },
  FRIENDLY: { c: "#6fe3b0", bg: "rgba(111,227,176,.12)" },
  WATCH:    { c: "#ffd479", bg: "rgba(255,212,121,.12)" },
  WARNING:  { c: "#ffa04d", bg: "rgba(255,160,77,.12)" },
  CRITICAL: { c: "#ff3b4e", bg: "rgba(255,59,78,.14)" },
};
const col = (t) => (THREAT[t] || THREAT.INFO).c;
const bandOf = (c) => (c >= 0.75 ? "HIGH" : c >= 0.5 ? "MEDIUM" : "LOW");

const $ = (s) => document.querySelector(s);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
};
const hhmmss = (iso) => (iso ? new Date(iso).toISOString().substr(11, 8) : "--:--:--");

const S = {
  demos: [],
  active: null,
  panels: [],          // [{camera, label, live, img, canvas, wrap}]
  overlays: {},        // camera_id -> latest overlay
  alerts: [],
  switching: false,
  wantEvidence: false,
  selected: null,
};

/* ══════════════ rail ══════════════ */
function renderRail() {
  const rail = $("#rail");
  rail.textContent = "";
  S.demos.forEach((d) => {
    const b = el("button");
    b.append(el("span", "n", `DEMO ${d.id}`), el("span", "t", d.title));
    b.setAttribute("aria-current", String(S.active === d.id));
    b.onclick = () => select(d.id);
    rail.append(b);
  });
}

function railBusy(busy) {
  S.switching = busy;
  document.querySelectorAll("#rail button").forEach((b) => (b.disabled = busy));
}

/* ══════════════ selection ══════════════ */
async function select(id) {
  if (S.switching || S.active === id) return;
  const demo = S.demos.find((d) => d.id === id);
  if (!demo) return;

  railBusy(true);
  // Drop the open MJPEG streams BEFORE asking the server to switch.
  //
  // This is not tidiness, it is the difference between working and hanging. A
  // multipart/x-mixed-replace response never ends, so every visible feed holds
  // a connection open for as long as it is on screen. A two-panel demo shows
  // four of them, and browsers cap concurrent connections to one host at
  // around six -- so the POST below queues behind streams that will never
  // finish, and the page sits on "binding the feed" forever. Clearing src
  // aborts them and frees the slots.
  teardownStreams();

  $("#switch-title").textContent = demo.title;
  $("#switch-sub").textContent =
    demo.panels.length > 1 ? `binding ${demo.panels.length} feeds` : "binding the feed";
  $("#switch").hidden = false;

  // Bounded, because this runs in front of an audience. If the request cannot
  // complete in fifteen seconds something is wrong and saying so beats a
  // spinner that never stops.
  let res;
  const ctl = new AbortController();
  const bail = setTimeout(() => ctl.abort(), 15000);
  try {
    res = await (await fetch(`/api/demo/${id}`, { method: "POST", signal: ctl.signal })).json();
  } catch (err) {
    res = { detail: ctl.signal.aborted ? "timed out binding the feed" : String(err) };
  } finally {
    clearTimeout(bail);
  }

  if (!res || !res.ok) {
    // A failure here is nearly always the device camera being unavailable or a
    // clip missing from assets/. Say which, rather than leaving black panels.
    $("#switch-title").textContent = "Could not start that demo";
    $("#switch-sub").textContent = (res && (res.detail || res.error)) || "unknown error";
    setTimeout(() => { $("#switch").hidden = true; railBusy(false); }, 2600);
    return;
  }

  S.active = id;
  S.overlays = {};
  renderRail();
  $("#demo-title").textContent = `Demo ${id} — ${res.title}`;
  $("#demo-note").textContent = res.note || "";
  $("#demo-claim").textContent = res.claim || res.title;
  buildPanels(res.panels);
  applyExtras(res.extras || []);

  // Give the retargeted pipelines a moment to produce a first frame, so the
  // overlay lifts onto a picture rather than onto black.
  setTimeout(() => { $("#switch").hidden = true; railBusy(false); }, 1200);
}

/* ══════════════ panels ══════════════ */
const BLANK = "data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==";

function teardownStreams() {
  // Point every open feed at a 1x1 GIF. Assigning a new src aborts the
  // in-flight response, which is what actually releases the connection;
  // removing the element does not reliably do so while it is still decoding.
  document.querySelectorAll("#panels img").forEach((img) => { img.src = BLANK; });
}

function buildPanels(panels) {
  const host = $("#panels");
  host.textContent = "";
  S.panels = [];

  panels.forEach((p) => {
    const wrap = el("div", "panel");
    if (p.live) wrap.classList.add("is-live");

    const head = el("div", "panel-head");
    head.append(el("span", "tag", p.label || p.camera));
    head.append(el("span", "sub", p.live ? "live device feed" : p.source.split("/").pop()));
    const live = el("span", "live");
    live.append(el("b"), el("span", null, "LIVE"));
    head.append(live);

    const pair = el("div", "pair");

    const rawFig = el("figure", "pane");
    const rawCap = el("figcaption");
    rawCap.append(el("span", "tag", "AS GIVEN"),
                  el("span", "sub", "what the detector receives"));
    const rawBox = el("div", "frame");
    const rawImg = el("img");
    rawImg.alt = `${p.label} — source footage`;
    rawBox.append(rawImg);
    rawFig.append(rawCap, rawBox);

    const detFig = el("figure", "pane");
    const detCap = el("figcaption");
    detCap.append(el("span", "tag accent", "WITH DETECTION"),
                  el("span", "sub", "boxes drawn from live state"));
    const detBox = el("div", "frame");
    const detImg = el("img");
    detImg.alt = `${p.label} — analysed`;
    const cv = el("canvas");
    detBox.append(detImg, cv);
    detFig.append(detCap, detBox);

    pair.append(rawFig, detFig);
    wrap.append(head, pair);
    host.append(wrap);

    // Cache-bust: the browser will happily reuse a previous multipart stream
    // for the same URL, which after a source swap shows the OLD clip forever.
    const bust = Date.now();
    rawImg.src = `/api/stream/${p.camera}?raw=1&t=${bust}`;
    detImg.src = `/api/stream/${p.camera}?t=${bust}`;

    // Portrait clips get a taller frame. Read it off the first decoded frame
    // rather than from config, so it stays right for whatever is bound.
    detImg.addEventListener("load", () => {
      if (detImg.naturalHeight > detImg.naturalWidth) wrap.classList.add("portrait");
      drawPanel(S.panels.find((x) => x.canvas === cv));
    }, { once: true });

    S.panels.push({ camera: p.camera, label: p.label, live: p.live,
                    img: detImg, canvas: cv, wrap,
                    present: p.present || {},
                    // Last frame that carried boxes, and how long ago. Used to
                    // hold a subject across an occlusion instead of blinking.
                    coast: null, coastAge: 0 });
  });
}

/* ══════════════ drawing ══════════════ */
function drawAll() {
  S.panels.forEach(drawPanel);
}

function drawPanel(panel) {
  if (!panel) return;
  const ov = S.overlays[panel.camera];
  const cv = panel.canvas, img = panel.img;
  const w = cv.clientWidth, h = cv.clientHeight;
  if (!w || !h) return;
  if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
  const ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, w, h);
  if (!ov) return;

  const pr = panel.present || {};

  // Mirror object-fit: contain, or the boxes drift off the subject.
  const nw = img.naturalWidth || 1280, nh = img.naturalHeight || 720;
  const scale = Math.min(w / nw, h / nh);
  const ox = (w - nw * scale) / 2, oy = (h - nh * scale) / 2;
  const X = (x) => ox + x * scale, Y = (y) => oy + y * scale;

  // Coasting. Some footage occludes its subject repeatedly -- foliage, a
  // passing trunk -- and the detector drops out for a handful of frames each
  // time. The tracker keeps the identity alive far longer than that, so
  // holding the last box across the gap shows the subject the operator is
  // actually following rather than a box that strobes.
  let boxes = ov.boxes || [];
  // A panel that runs the detector at a low floor to catch a difficult subject
  // also catches a lot of foliage. Drawing only the group the demo is about
  // keeps the noise out of a picture whose whole point is one subject.
  if (pr.only_group) boxes = boxes.filter((b) => b.group === pr.only_group);
  const coastMax = pr.coast_frames || 0;
  if (coastMax) {
    if (boxes.length) { panel.coast = boxes; panel.coastAge = 0; }
    else if (panel.coast && panel.coastAge < coastMax) { boxes = panel.coast; panel.coastAge++; }
    else { panel.coast = null; }
  }
  if (pr.hide_boxes) boxes = [];

  for (const b of boxes) {
    const c = col(pr.box_threat || b.threat);
    const band = bandOf(b.confidence);
    const [x1, y1, x2, y2] = b.bbox;
    const x = X(x1), y = Y(y1), bw = (x2 - x1) * scale, bh = (y2 - y1) * scale;

    ctx.save();
    ctx.strokeStyle = c;
    if (band === "HIGH") { ctx.setLineDash([]); ctx.lineWidth = 2; }
    else if (band === "MEDIUM") { ctx.setLineDash([7, 4]); ctx.lineWidth = 1.8; }
    else { ctx.setLineDash([2, 4]); ctx.lineWidth = 1.5; }
    ctx.strokeRect(x, y, bw, bh);
    ctx.restore();

    let text;
    if (pr.box_label) {
      text = `${pr.box_label} · ${Number(b.confidence).toFixed(2)}`;
    } else {
      const bits = [`#${b.track_id}`, (b.group || "").toUpperCase(),
                    Number(b.confidence).toFixed(2)];
      if (b.stance && !["UNKNOWN", "STANDING"].includes(b.stance)) bits.push(b.stance);
      if (b.weapon) bits.push(b.weapon.toUpperCase());
      if (b.reclassified) bits.push("HUMAN");
      if (b.quadruped_note) bits.push("STANCE WITHHELD");
      text = bits.join(" · ");
    }
    label(ctx, text, x, y, c);
  }

  drawAerial(ctx, ov, X, Y, scale, pr, nw, nh);
}

function label(ctx, text, x, y, c, size) {
  ctx.save();
  ctx.font = `500 ${size || 10}px 'IBM Plex Mono', ui-monospace, monospace`;
  const tw = ctx.measureText(text).width + 12;
  const ly = Math.max(0, y - 20);
  ctx.fillStyle = c; ctx.fillRect(x, ly, tw, 17);
  ctx.fillStyle = "#04070d"; ctx.fillText(text, x + 6, ly + 12);
  ctx.restore();
}

function drawAerial(ctx, ov, X, Y, scale, pr, nw, nh) {
  const a = ov.aerial;
  if (!a) return;

  for (const d of a.drones || []) {
    let [x1, y1, x2, y2] = d.bbox_px;
    // Clamp to the frame. A box whose edge sits outside the image draws a
    // reticle floating in the letterbox beside the picture, which reads as the
    // system detecting something in the black bars.
    x1 = Math.max(0, Math.min(x1, nw)); x2 = Math.max(0, Math.min(x2, nw));
    y1 = Math.max(0, Math.min(y1, nh)); y2 = Math.max(0, Math.min(y2, nh));
    if (x2 - x1 < 2 || y2 - y1 < 2) continue;
    const x = X(x1), y = Y(y1), bw = (x2 - x1) * scale, bh = (y2 - y1) * scale;
    ctx.save();
    ctx.strokeStyle = col("CRITICAL"); ctx.lineWidth = 2; ctx.setLineDash([]);
    ctx.strokeRect(x, y, bw, bh);
    ctx.beginPath();
    ctx.moveTo(x + bw / 2, y - 7); ctx.lineTo(x + bw / 2, y - 1);
    ctx.moveTo(x + bw / 2, y + bh + 1); ctx.lineTo(x + bw / 2, y + bh + 7);
    ctx.stroke();
    ctx.restore();
    label(ctx, `${pr.aerial_label || "AERIAL"} · ${Number(d.conf).toFixed(2)}`,
          x, y, col("CRITICAL"));
  }

  // The refusals are the interesting half of this demo: a system that can show
  // why it declined to call something a drone is claiming more than one that
  // simply shows nothing.
  for (const wv of a.withheld || []) {
    const [x1, y1, x2, y2] = wv.bbox_px;
    const x = X(x1), y = Y(y1), bw = (x2 - x1) * scale, bh = (y2 - y1) * scale;
    ctx.save();
    ctx.strokeStyle = "rgba(160,160,170,.75)";
    ctx.lineWidth = 1.2; ctx.setLineDash([3, 4]);
    ctx.strokeRect(x, y, bw, bh);
    ctx.setLineDash([]);
    const text = wv.bird_conf !== undefined
      ? `BIRD · drone ${Number(wv.drone_conf).toFixed(2)} vs bird ${Number(wv.bird_conf).toFixed(2)}`
      : `TOO LARGE · drone ${Number(wv.drone_conf).toFixed(2)} · ${Math.round(Number(wv.area_frac) * 100)}% of frame`;
    ctx.font = "500 9px 'IBM Plex Mono', ui-monospace, monospace";
    ctx.fillStyle = "rgba(160,160,170,.9)";
    ctx.fillText(text, x, Math.max(9, y - 5));
    ctx.restore();
  }
}

/* ══════════════ extras ══════════════ */
function applyExtras(extras) {
  const want = new Set(extras);
  $("#pane-ask").hidden = !want.has("ask");
  $("#pane-evidence").hidden = !want.has("evidence");
  S.wantEvidence = want.has("evidence");
  if (!want.has("evidence")) {
    S.selected = null;
    $("#evidence-body").textContent = "";
    $("#evidence-body").append(el("p", "muted", "Select an alert above to see its evidence."));
  }
}

async function ask(text) {
  const out = $("#ask-out");
  out.textContent = "";
  out.append(el("div", "route", "routing…"));
  $("#ask-btn").disabled = true;
  try {
    const cam = S.panels.length ? S.panels[0].camera : undefined;
    const r = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ q: text, camera_id: cam }),
    });
    renderAnswer(await r.json());
  } catch (e) {
    out.textContent = "";
    out.append(el("div", "errbox", "Query failed: " + e.message));
  } finally {
    $("#ask-btn").disabled = false;
  }
}

function renderAnswer(d) {
  const out = $("#ask-out");
  out.textContent = "";
  const route = el("div", "route");
  route.append(el("span", null, "ROUTE · " + (d.route || "?").toUpperCase()));
  if (d.backend) route.append(el("span", "lat", d.backend));
  if (d.latency_ms != null) route.append(el("span", "lat", d.latency_ms + " ms"));
  out.append(route);

  if (d.ok === false) {
    out.append(el("div", "errbox", d.error || "unavailable"));
    if (d.fallback && d.fallback.answer) out.append(el("div", "answer", d.fallback.answer));
    return;
  }
  if (d.answer) out.append(el("div", "answer", d.answer));
  if (d.sql) out.append(el("div", "sqlbox", d.sql));
  if (d.rows && d.rows.length && d.route === "structured") {
    const t = el("table", "kv");
    for (const r of d.rows.slice(0, 8)) {
      const tr = el("tr");
      tr.append(el("td", null, hhmmss(r.opened_at)));
      tr.append(el("td", null,
        `${r.threat} · ${r.kind} · ${Math.round(r.confidence * 100)}%`));
      tr.style.cursor = "pointer";
      tr.onclick = () => selectAlert(r.alert_id);
      t.append(tr);
    }
    out.append(t);
  }
  if (d.rows && d.route === "identity") {
    for (const r of d.rows) {
      out.append(el("div", "sqlbox",
        `#${r.track_id} ${r.class_name}: ${r.label}` +
        (r.similarity != null ? ` (sim ${r.similarity}, thr ${r.threshold})` : "")));
    }
  }
}

async function selectAlert(id) {
  if (!S.wantEvidence || !id) return;
  S.selected = id;
  const body = $("#evidence-body");
  body.textContent = "";
  body.append(el("p", "muted", "loading evidence…"));
  let d;
  try {
    d = await (await fetch(`/api/alerts/${id}`)).json();
  } catch (e) {
    body.textContent = "";
    body.append(el("div", "errbox", "Could not load evidence: " + e.message));
    return;
  }
  body.textContent = "";

  const h = el("h3", null, (d.kind || "").replace(/_/g, " ").toLowerCase()
    .replace(/^./, (m) => m.toUpperCase()));
  h.style.color = col(d.threat);
  body.append(h);
  if (d.rationale) body.append(el("p", "why", d.rationale));

  // Real files from the evidence store, served from disk. The keyframe is the
  // unannotated frame the detector saw; the crop is the subject cut from it.
  const shots = el("div", "shots");
  const add = (cap, path, alt) => {
    const sh = el("div", "shot");
    sh.append(el("div", "cap", cap));
    const b = el("div", "box");
    if (path) {
      const im = el("img");
      im.src = "/evidence/" + String(path).replace(/\\/g, "/");
      im.alt = alt;
      im.onerror = () => { b.classList.add("empty"); im.remove(); };
      b.append(im);
    } else {
      b.classList.add("empty");
    }
    sh.append(b);
    shots.append(sh);
  };
  add("KEYFRAME · UNANNOTATED, AS THE DETECTOR SAW IT", d.keyframe_path, "alert keyframe");
  add("SUBJECT CROP", d.crop_path, "subject crop");
  // Only meaningful for a person, and only when a face was actually visible.
  add("IDENTIFIED PERSON · FACE", d.face_path, "face of the subject");
  body.append(shots);
}

/* ══════════════ alerts ══════════════ */
function paintAlerts() {
  const list = $("#alerts");
  list.textContent = "";
  const rows = S.alerts.slice(0, 12);
  $("#alert-count").textContent = rows.length ? `${S.alerts.length} confirmed` : "none yet";
  rows.forEach((a) => {
    const li = el("li");
    li.style.borderLeft = `2px solid ${col(a.threat)}`;
    li.append(el("span", "at", hhmmss(a.opened_at)));
    li.append(el("span", "kind", a.kind));
    li.append(el("span", "why", a.reason || a.camera_id || ""));
    if (S.wantEvidence) {
      li.style.cursor = "pointer";
      li.onclick = () => selectAlert(a.alert_id);
    }
    list.append(li);
  });
}

function paintThreat(level) {
  const pill = $("#threat-pill");
  const c = col(level || "INFO");
  $("#threat-val").textContent = level || "—";
  pill.style.borderColor = c;
  pill.querySelector("i").style.background = c;
  pill.querySelector("i").style.boxShadow = `0 0 10px ${c}`;
}

/* ══════════════ live state ══════════════ */
function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.type === "tick") {
      // Keep every camera's overlay: a two-panel demo needs both at once, and
      // which cameras are live changes as demos are selected.
      if (m.cameras) Object.assign(S.overlays, m.cameras);
      if (m.status) {
        paintThreat(m.status.threat);
        if (m.status.alerts) { S.alerts = m.status.alerts; paintAlerts(); }
      }
      drawAll();
    } else if (m.type === "alert" && m.phase === "opened") {
      loadAlerts();
    }
  };
  ws.onclose = () => setTimeout(connect, 1500);
  setInterval(() => { if (ws.readyState === 1) ws.send("ping"); }, 15000);
}

async function loadAlerts() {
  try {
    const r = await (await fetch("/api/alerts?limit=20")).json();
    S.alerts = r.alerts || r || [];
    paintAlerts();
  } catch (_) { /* transient */ }
}

/* ══════════════ boot ══════════════ */
window.addEventListener("DOMContentLoaded", async () => {
  try {
    const r = await (await fetch("/api/demos")).json();
    S.demos = r.demos || [];
  } catch (_) {
    S.demos = [];
  }
  renderRail();
  $("#ask-form").onsubmit = (e) => {
    e.preventDefault();
    const v = $("#ask-input").value.trim();
    if (v) ask(v);
  };
  document.querySelectorAll("#pane-ask .chips button").forEach((b) =>
    (b.onclick = () => { $("#ask-input").value = b.textContent; ask(b.textContent); }));
  connect();
  loadAlerts();

  // Deep link: /demos#3 opens demo 3 directly, which is how you get back to
  // one mid-presentation without clicking through the others.
  const want = parseInt(location.hash.replace("#", ""), 10);
  if (S.demos.some((d) => d.id === want)) select(want);

  window.addEventListener("resize", drawAll);
});
