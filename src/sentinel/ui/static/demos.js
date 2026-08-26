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
                    img: detImg, canvas: cv, wrap });
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

  // Mirror object-fit: contain, or the boxes drift off the subject.
  const nw = img.naturalWidth || 1280, nh = img.naturalHeight || 720;
  const scale = Math.min(w / nw, h / nh);
  const ox = (w - nw * scale) / 2, oy = (h - nh * scale) / 2;
  const X = (x) => ox + x * scale, Y = (y) => oy + y * scale;

  for (const b of ov.boxes || []) {
    const c = col(b.threat);
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

    const bits = [`#${b.track_id}`, (b.group || "").toUpperCase(),
                  Number(b.confidence).toFixed(2)];
    if (b.stance && !["UNKNOWN", "STANDING"].includes(b.stance)) bits.push(b.stance);
    if (b.weapon) bits.push(b.weapon.toUpperCase());
    if (b.reclassified) bits.push("HUMAN");
    if (b.quadruped_note) bits.push("STANCE WITHHELD");
    label(ctx, bits.join(" · "), x, y, c);
  }

  drawAerial(ctx, ov, X, Y, scale);
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

function drawAerial(ctx, ov, X, Y, scale) {
  const a = ov.aerial;
  if (!a) return;

  for (const d of a.drones || []) {
    const [x1, y1, x2, y2] = d.bbox_px;
    const x = X(x1), y = Y(y1), bw = (x2 - x1) * scale, bh = (y2 - y1) * scale;
    ctx.save();
    ctx.strokeStyle = col("CRITICAL"); ctx.lineWidth = 2; ctx.setLineDash([]);
    ctx.strokeRect(x, y, bw, bh);
    ctx.beginPath();
    ctx.moveTo(x + bw / 2, y - 7); ctx.lineTo(x + bw / 2, y - 1);
    ctx.moveTo(x + bw / 2, y + bh + 1); ctx.lineTo(x + bw / 2, y + bh + 7);
    ctx.stroke();
    ctx.restore();
    label(ctx, `AERIAL · ${Number(d.conf).toFixed(2)}`, x, y, col("CRITICAL"));
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
  connect();
  loadAlerts();

  // Deep link: /demos#3 opens demo 3 directly, which is how you get back to
  // one mid-presentation without clicking through the others.
  const want = parseInt(location.hash.replace("#", ""), 10);
  if (S.demos.some((d) => d.id === want)) select(want);

  window.addEventListener("resize", drawAll);
});
