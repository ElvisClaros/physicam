/** Physicam UI: upload -> POST /api/jobs -> SSE stream -> live tracks, fits,
 * answer and report. Also renders a finished job standalone (GET /api/jobs/{id},
 * #job=<id> deep link) and a fully offline mock mode (?mock=1). */

import { renderMarkdown } from "./markdown.js";
import { drawBarChart, drawLineChart, wireChart } from "./charts.js";

const $ = (id) => document.getElementById(id);
const FALLBACK_COLORS = ["#ff4d4f", "#22d3ee", "#3ddc97", "#f5b544", "#f472b6", "#a3e635"];
const params = new URLSearchParams(location.search);
const MOCK = params.get("mock") === "1";

const state = {
  jobId: null,
  result: null,          // accumulated result JSON (contract shape)
  tracks: new Map(),     // id -> track
  fits: [],
  videoFile: null,
  objectUrl: null,
  eventSource: null,
  runningRow: null,
  mockTimers: [],
  strobeUrl: null,
  stageMode: "video",   // "video" | "strobe"
  currentStage: null,
};

function emptyResult() {
  return { id: null, status: "running", question: "", video: null, plan: null,
    calibration: null, tracks: [], fits: [], answer: null, report_markdown: "",
    trace: [], cost: null };
}
state.result = emptyResult();

/* ---------------- helpers ---------------- */

function fmt(n, digits = 3) {
  if (n == null || !isFinite(n)) return "—";
  return Number(n).toPrecision(digits);
}
/** Value ± uncertainty: 2 sig digits on uncertainty, matching decimals on value. */
function fmtPair(value, unc) {
  if (value == null || !isFinite(value)) return "—";
  if (unc == null || !isFinite(unc) || unc === 0) return `${fmt(value, 4)} ± —`;
  const places = 1 - Math.floor(Math.log10(Math.abs(unc)));
  const factor = Math.pow(10, places);
  const ru = Math.round(unc * factor) / factor;
  const rv = Math.round(value * factor) / factor;
  if (places >= 0) return `${rv.toFixed(places)} ± ${ru.toFixed(places)}`;
  return `${rv} ± ${ru}`;
}
function fmtParam(p) {
  if (!p || p.value == null || !isFinite(p.value)) return "—";
  if (p.stderr == null || !isFinite(p.stderr) || p.stderr === 0) return `${fmt(p.value, 5)} ± —`;
  return fmtPair(p.value, p.stderr);
}
function esc(s) {
  return String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
function effectiveFps() {
  const override = parseFloat($("fact-fps").value);
  if (isFinite(override) && override > 0) return override;
  return state.result?.video?.fps || 30;
}
function calibration() {
  const c = state.result?.calibration;
  return c?.m_per_px && c.method !== "none" ? c : null;
}
function toMeters(px) { const c = calibration(); return c ? px * c.m_per_px : null; }

function trackColor(track, idx) {
  return track.color || FALLBACK_COLORS[idx % FALLBACK_COLORS.length];
}

/** Index of the track sample nearest to time t (track.t is sorted). */
function nearestIndex(track, t) {
  const ts = track.t || [];
  if (!ts.length) return -1;
  let lo = 0, hi = ts.length - 1;
  while (lo < hi) { const mid = (lo + hi) >> 1; (ts[mid] < t) ? lo = mid + 1 : hi = mid; }
  if (lo > 0 && Math.abs(ts[lo - 1] - t) < Math.abs(ts[lo] - t)) return lo - 1;
  return lo;
}

/** Smoothed speed (px/s) from finite differences of the centroid path. */
function speedSeries(track) {
  const { t, x, y } = track;
  const v = new Array(t.length).fill(NaN);
  for (let i = 1; i < t.length - 1; i++) {
    const dt = t[i + 1] - t[i - 1];
    if (dt > 0) v[i] = Math.hypot(x[i + 1] - x[i - 1], y[i + 1] - y[i - 1]) / dt;
  }
  if (t.length > 1) { v[0] = v[1]; v[t.length - 1] = v[t.length - 2]; }
  // light smoothing, window 5
  const out = v.slice();
  for (let i = 2; i < v.length - 2; i++) {
    const w = [v[i - 2], v[i - 1], v[i], v[i + 1], v[i + 2]].filter(isFinite);
    if (w.length) out[i] = w.reduce((a, b) => a + b, 0) / w.length;
  }
  return out;
}

function setStatus(msg) { $("job-status").textContent = msg || ""; }
function showError(msg) {
  setStatus("");
  const tl = $("timeline");
  tl.querySelector(".timeline-empty")?.remove();
  const div = document.createElement("div");
  div.className = "error-box";
  div.textContent = msg;
  tl.appendChild(div);
  scrollTimeline();
}

/* ---------------- timeline ---------------- */

const ICONS = {
  watch: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M2 12s3.5-6.5 10-6.5S22 12 22 12s-3.5 6.5-10 6.5S2 12 2 12Z"/><circle cx="12" cy="12" r="3"/></svg>',
  track: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 19 Q12 5 20 19"/><circle cx="7" cy="14" r="1.6" fill="currentColor"/><circle cx="12" cy="10" r="1.6" fill="currentColor"/><circle cx="17" cy="14" r="1.6" fill="currentColor"/></svg>',
  calibrate: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 17h18M6 17v-4m6 4V7m6 10v-7"/></svg>',
  fit: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 19 Q12 4 20 19" stroke-dasharray="4 3"/><circle cx="6" cy="15" r="1.4" fill="currentColor"/><circle cx="12" cy="9" r="1.4" fill="currentColor"/><circle cx="18" cy="15" r="1.4" fill="currentColor"/></svg>',
  python: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m8 8-4 4 4 4m8-8 4 4-4 4"/></svg>',
  answer: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 6 9 17l-5-5"/></svg>',
  report: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M6 3h9l4 4v14H6z"/><path d="M9 12h7M9 16h7"/></svg>',
  plan: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 3"/></svg>',
};

function traceMeta(trace) {
  const name = trace.name || "";
  const kind = trace.kind || "";
  if (name === "watch_video" || kind === "plan" || name.includes("plan")) return { icon: "watch", badge: "SPARK" };
  if (name.startsWith("track") || name.includes("segment")) return { icon: "track", badge: "SAM" };
  if (name.startsWith("calibrate_reference")) return { icon: "calibrate", badge: "SAM" };
  if (name.startsWith("calibrate")) return { icon: "calibrate", badge: "local" };
  if (name.startsWith("fit")) return { icon: "fit", badge: "local" };
  if (name === "run_python") return { icon: "python", badge: "local" };
  if (name === "write_report" || kind === "report") return { icon: "report", badge: "WONDER" };
  if (name === "finish" || kind === "answer") return { icon: "answer", badge: "SPARK" };
  return { icon: "plan", badge: kind === "report" ? "WONDER" : "SPARK" };
}

function scrollTimeline() {
  const tl = $("timeline");
  // Auto-scroll to the newest row (after layout, so scrollHeight is final).
  requestAnimationFrame(() => { tl.scrollTop = tl.scrollHeight; });
  tl.scrollTop = tl.scrollHeight;
}

function addStageHead(text) {
  const tl = $("timeline");
  tl.querySelector(".timeline-empty")?.remove();
  const div = document.createElement("div");
  div.className = "stage-head";
  div.textContent = text;
  // Keep the running spinner row pinned below the newest content.
  if (state.runningRow) tl.insertBefore(div, state.runningRow); else tl.appendChild(div);
  scrollTimeline();
}

function clearRunning() {
  if (state.runningRow) { state.runningRow.remove(); state.runningRow = null; }
}

function showRunning(label) {
  const tl = $("timeline");
  tl.querySelector(".timeline-empty")?.remove();
  if (state.runningRow) {
    // Keep the same spinner row until the next event; only refresh its label.
    const labelEl = state.runningRow.querySelector(".tsummary");
    if (labelEl && label) labelEl.textContent = label;
    tl.appendChild(state.runningRow);
    scrollTimeline();
    return;
  }
  const row = document.createElement("div");
  row.className = "trow running-row";
  row.innerHTML = `<div class="trow-top"><span class="spinner"></span><span class="tsummary">${esc(label)}</span></div>`;
  tl.appendChild(row);
  state.runningRow = row;
  scrollTimeline();
}

function addTraceRow(trace) {
  const tl = $("timeline");
  tl.querySelector(".timeline-empty")?.remove();
  const meta = traceMeta(trace);
  const row = document.createElement("div");
  row.className = "trow";
  let detail = JSON.stringify({ args: trace.args ?? {}, summary: trace.summary ?? "" }, null, 1);
  if (detail.length > 1200) detail = detail.slice(0, 1200) + "\n…";
  const ms = trace.ms != null ? `${Math.round(trace.ms)} ms` : "";
  row.innerHTML =
    `<div class="trow-top"><span class="ticon">${ICONS[meta.icon] || ICONS.plan}</span>` +
    `<span class="tbadge ${meta.badge}">${meta.badge}</span>` +
    `<span class="tsummary">${esc(trace.title || trace.summary || trace.name || "")}</span>` +
    `<span class="tms">${esc(ms)}</span></div>` +
    `<details><summary>${esc(trace.name || trace.kind || "step")} details</summary><pre>${esc(detail)}</pre></details>`;
  // Insert above the running spinner row so the spinner stays last/alive
  // until the next event (done/error/stage replaces it).
  if (state.runningRow) tl.insertBefore(row, state.runningRow); else tl.appendChild(row);
  scrollTimeline();
}

/* ---------------- stage progress + strobe ---------------- */

const STAGE_LABELS = {
  ingest: "Preparing video…",
  plan: "Muse Spark is watching the video…",
  agent: "Agent at work · SAM 3.1 tracking…",
  report: "Muse Wonder is writing the report…",
};

function stageKeyFrom(value) {
  const s = String(value || "").toLowerCase();
  if (s.includes("ingest") || s.includes("prepar")) return "ingest";
  if (s.includes("plan") || s.includes("watch")) return "plan";
  if (s.includes("agent") || s.includes("track") || s.includes("sam")) return "agent";
  if (s.includes("report") || s.includes("wonder")) return "report";
  return null;
}

let stageFadeTimer = null;
function setStageProgress(key, detail) {
  if (!key) return;
  if (stageFadeTimer) { clearTimeout(stageFadeTimer); stageFadeTimer = null; }
  state.currentStage = key;
  const pill = $("stage-progress");
  pill.classList.remove("fading");
  pill.hidden = false;
  $("stage-progress-text").textContent = detail || STAGE_LABELS[key];
}

function updateStageProgressFromTrace(trace) {
  if (state.currentStage !== "agent") return;
  const summary = trace?.summary || trace?.name;
  if (summary) setStageProgress("agent", `Agent at work · ${summary}`);
}

function hideStageProgress() {
  state.currentStage = null;
  const pill = $("stage-progress");
  if (pill.hidden) return;
  pill.classList.add("fading");
  stageFadeTimer = setTimeout(() => { pill.hidden = true; pill.classList.remove("fading"); stageFadeTimer = null; }, 450);
}

function setStageMode(mode) {
  state.stageMode = mode === "strobe" && state.strobeUrl ? "strobe" : "video";
  const isStrobe = state.stageMode === "strobe";
  $("strobe-img").hidden = !isStrobe;
  $("video").hidden = isStrobe;
  $("overlay").hidden = isStrobe;
  if (isStrobe) $("hud").hidden = true;
  for (const btn of $("stage-mode").querySelectorAll("button")) {
    btn.classList.toggle("active", btn.dataset.mode === state.stageMode);
  }
  if (!isStrobe) drawOverlay();
}

function setStrobe(url) {
  if (!url) return;
  state.strobeUrl = url;
  state.result.strobe_url = url;
  $("strobe-img").src = url;
  $("stage-mode").hidden = false;
  const fig = $("report-strobe");
  $("report-strobe-img").src = url;
  // The report figure only shows once a report section exists.
  fig.hidden = $("report-wrap").hidden;
  setStageMode(state.stageMode);
}

function renderReportStrobe() {
  const has = Boolean(state.strobeUrl);
  $("report-strobe").hidden = !has;
  if (has) $("report-strobe-img").src = state.strobeUrl;
}

/* ---------------- results rendering ---------------- */

function isDimensionless(unit) {
  const u = String(unit || "").trim();
  return u === "" || u === "1" || u === "-" || u.toLowerCase() === "dimensionless";
}

function eHeightCrossCheck() {
  for (const fit of state.fits) {
    const fromImpacts = (fit.impacts || []).find((im) => im.e_height != null && im.valid !== false);
    if (fromImpacts) return { value: fromImpacts.e_height, stderr: null };
    const raw = fit.e_height ?? fit.params?.e_height ?? fit.params?.eHeight;
    if (raw == null) continue;
    if (typeof raw === "number" && isFinite(raw)) return { value: raw, stderr: null };
    if (typeof raw === "object" && isFinite(raw.value)) return { value: raw.value, stderr: raw.stderr ?? null };
  }
  return null;
}

function renderAnswer(answer, animate = true) {
  if (!answer) return;
  const card = $("answer-card");
  card.hidden = false;
  const value = $("answer-value");
  const showValue = answer.value != null && isFinite(answer.value);
  const dimensionless = isDimensionless(answer.unit);
  const unitHtml = dimensionless ? "" : ` <span class="unit">${esc(answer.unit || "")}</span>`;
  value.innerHTML = showValue
    ? `${esc(fmtPair(answer.value, answer.uncertainty))}${unitHtml}`
    : esc(answer.headline || "");
  if (animate) {
    value.classList.remove("count-up");
    void value.offsetWidth;
    value.classList.add("count-up");
    if (showValue) countUp(value, answer);
  }
  $("answer-headline").textContent = showValue ? (answer.headline || "") : "";
  $("answer-summary").textContent = answer.summary || "";
  const delta = $("answer-delta");
  const unit = String(answer.unit || "").trim().replace("²", "^2");
  if (showValue && unit === "m/s^2") {
    // Gravity only: Δ vs 9.81 m/s².
    const pct = (answer.value - 9.81) / 9.81 * 100;
    const sigma = answer.uncertainty ? Math.abs(answer.value - 9.81) / answer.uncertainty : NaN;
    delta.textContent = `Δ vs 9.81 m/s²: ${pct >= 0 ? "+" : ""}${pct.toFixed(1)}%` +
      (isFinite(sigma) ? ` (${sigma.toFixed(1)}σ)` : "");
  } else if (showValue && dimensionless) {
    const cross = eHeightCrossCheck();
    delta.textContent = cross
      ? `cross-check (apex heights): e = ${fmt(cross.value, 3)}${cross.stderr != null ? ` ± ${fmt(cross.stderr, 3)}` : ""}`
      : "";
  } else delta.textContent = "";
}

function countUp(el, answer) {
  const target = answer.value, unc = answer.uncertainty || 0, unit = answer.unit || "";
  const unitHtml = isDimensionless(unit) ? "" : ` <span class="unit">${esc(unit)}</span>`;
  const t0 = performance.now(), dur = 700;
  const tick = (now) => {
    const k = Math.min(1, (now - t0) / dur);
    const eased = 1 - Math.pow(1 - k, 3);
    el.innerHTML = `${esc(fmtPair(target * eased, unc * eased))}${unitHtml}`;
    if (k < 1) requestAnimationFrame(tick);
    else if (state.result.answer) renderAnswer(state.result.answer, false);
  };
  requestAnimationFrame(tick);
}

function metricNote(p) {
  const cal = calibration();
  if (!cal || !p || !String(p.unit || "").startsWith("px")) return "";
  const unit = String(p.unit).replace(/^px/, "m");
  return ` (= ${fmtPair(p.value * cal.m_per_px, (p.stderr || 0) * cal.m_per_px)} ${esc(unit)})`;
}
function fitBlockHtml(fit) {
  try {
    const rows = Object.entries(fit.params || {}).map(([name, p]) =>
      `<tr><td>${esc(name)}</td><td class="num">${esc(fmtParam(p))}${metricNote(p)}</td>` +
      `<td>${esc(p.unit || "")}</td></tr>`).join("");
    const info = Object.entries(fit.info || {}).map(([k, v]) =>
      `<tr><td>${esc(k)}</td><td class="num">${esc(String(v))}</td><td></td></tr>`).join("");
    const warnings = (fit.warnings || []).map((w) => `<span class="warn-chip">${esc(w)}</span>`).join("");
    return `<div class="fit-title">${esc(fit.id || "")} · ${esc(fit.model || "")} · track ${esc(fit.track_id || "")}` +
      ` · R² = ${fmt(fit.r2, 4)} · RMSE = ${fmt(fit.rmse_px, 3)} px · n = ${fit.n_points ?? "—"}</div>` +
      ((rows || info) ? `<table class="params"><thead><tr><th>Parameter</th><th>Value ± stderr</th><th>Unit</th></tr></thead><tbody>${rows}${info}</tbody></table>` : "") +
      (warnings ? `<div>${warnings}</div>` : "");
  } catch { return `<div class="fit-title">${esc(fit.id || "fit")} (could not render)</div>`; }
}
function renderFits() {
  const wrap = $("fits-wrap");
  if (!state.fits.length) { wrap.hidden = true; return; }
  wrap.hidden = false;
  const box = $("fits");
  box.innerHTML = "";
  const primaryId = state.result?.answer?.fit_ids?.[0];
  const primary = state.fits.find((f) => f.id === primaryId) || state.fits[0];
  const others = state.fits.filter((f) => f !== primary);
  const block = document.createElement("div");
  block.className = "fit-block";
  block.innerHTML = fitBlockHtml(primary);
  box.appendChild(block);
  if (others.length) {
    const det = document.createElement("details");
    det.className = "fit-block";
    det.innerHTML = `<summary>Other fits (${others.length})</summary>` +
      others.map((f) => fitBlockHtml(f)).join("");
    box.appendChild(det);
  }
}

function renderReport(md) {
  if (!md) return;
  $("report-wrap").hidden = false;
  renderReportStrobe();
  $("report").innerHTML = renderMarkdown(md);
}

function renderCost(cost) {
  if (!cost) return;
  const el = $("cost-line");
  el.hidden = false;
  const frames = cost.sam_frames != null ? `${cost.sam_frames} SAM frames` : "";
  const tokens = cost.spark_tokens != null
    ? `${cost.spark_tokens >= 1000 ? (cost.spark_tokens / 1000).toFixed(1).replace(/\.0$/, "") + "k" : cost.spark_tokens} Spark tokens` : "";
  const usd = cost.usd_estimate != null ? `≈ $${Number(cost.usd_estimate).toFixed(2)}` : "";
  el.textContent = [frames, tokens, usd].filter(Boolean).join(" · ");
}

function renderLegend() {
  const box = $("track-legend");
  box.innerHTML = "";
  let i = 0;
  for (const track of state.tracks.values()) {
    const item = document.createElement("span");
    item.className = "legend-item";
    item.innerHTML = `<span class="legend-dot" style="background:${esc(trackColor(track, i))}"></span>${esc(track.label || track.id)}`;
    box.appendChild(item);
    i++;
  }
}

/** Interpolated value of a fit curve at time t (curve.t sorted). Missing axis -> NaN. */
function curveAt(curve, t, axis = "y") {
  if (!curve || !curve.t || !curve.t.length || !curve[axis] || !curve[axis].length) return NaN;
  const ts = curve.t, ys = curve[axis];
  if (t <= ts[0]) return ys[0];
  if (t >= ts[ts.length - 1]) return ys[ys.length - 1];
  let lo = 0, hi = ts.length - 1;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; (ts[mid] <= t) ? lo = mid : hi = mid; }
  const k = (t - ts[lo]) / (ts[hi] - ts[lo] || 1);
  return ys[lo] + k * (ys[hi] - ys[lo]);
}
function trackPosAt(track, t, axis) {
  const ts = track?.t || [], vs = track?.[axis] || [];
  if (!ts.length || !vs.length) return 0;
  if (t <= ts[0]) return vs[0];
  if (t >= ts[ts.length - 1]) return vs[vs.length - 1];
  let lo = 0, hi = ts.length - 1;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; (ts[mid] <= t) ? lo = mid : hi = mid; }
  const k = (t - ts[lo]) / (ts[hi] - ts[lo] || 1);
  return vs[lo] + k * (vs[hi] - vs[lo]);
}

function restitutionBars() {
  const bars = [];
  for (const fit of state.fits) {
    if (fit.impacts && fit.impacts.length) {
      for (const im of fit.impacts) {
        if (im.e != null && isFinite(im.e)) {
          bars.push({ label: `impact ${im.index ?? bars.length + 1}`, value: im.e,
            stderr: im.e_stderr || 0, t: im.t, color: "#f5b544" });
        }
      }
    }
    const e = fit.params && fit.params.e;
    if (!bars.length && e && isFinite(e.value)) {
      bars.push({ label: "impact 1", value: e.value, stderr: e.stderr || 0,
        t: (fit.t_range || [0])[0], color: "#f5b544" });
    }
  }
  return bars;
}

/** Per-frame calibrated tracks plot X_m/Y_m (Y up) directly. */
function isPerFrame(track) {
  return state.result?.calibration?.kind === "per_frame"
    && Array.isArray(track.Y_m) && Array.isArray(track.X_m);
}

/** Affine Y_m = a*y_px + b fitted from a per-frame track (a < 0: Y is up). */
function perFrameAffine(track) {
  const ys = track.y || [], ms = track.Y_m || [];
  const n = Math.min(ys.length, ms.length);
  if (n < 2) return null;
  let sx = 0, sy = 0;
  for (let i = 0; i < n; i++) { sx += ys[i]; sy += ms[i]; }
  const mx = sx / n, my = sy / n;
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) { num += (ys[i] - mx) * (ms[i] - my); den += (ys[i] - mx) ** 2; }
  if (!den) return null;
  const a = num / den;
  return { a, b: my - a * mx };
}

function renderCharts(cursorT = null) {
  const wrap = $("charts-wrap");
  if (!state.tracks.size) { wrap.hidden = true; return; }
  wrap.hidden = false;
  const t = cursorT != null ? cursorT : ($("video").currentTime || 0);
  const cal = calibration();
  const unit = cal ? "m" : "px";
  const conv = (v) => (cal ? v * cal.m_per_px : v);

  // Height chart: h = (y_ref - y) with y_ref the largest (lowest) tracked y,
  // so height is positive-up. Labels/units follow the calibration.
  const ySeries = [], xSeries = [], speedSeriesAll = [], residSeries = [];
  let heightUnit = unit, anyPerFrame = false;
  const heightOf = new Map(); // track.id -> {data, fromPx, label unit}
  let idx = 0;
  for (const track of state.tracks.values()) {
    const color = trackColor(track, idx++);
    const perFrame = isPerFrame(track);
    if (perFrame) anyPerFrame = true;
    const finiteY = (track.y || []).filter(isFinite);
    const yRef = finiteY.length ? Math.max(...finiteY) : 0;
    let heights, fromPx, hUnit;
    if (perFrame) {
      heights = track.Y_m.slice();
      const aff = perFrameAffine(track);
      fromPx = aff ? (yPx) => aff.a * yPx + aff.b : () => NaN;
      hUnit = "m";
    } else if (cal) {
      heights = track.y.map((y) => (yRef - y) * cal.m_per_px);
      fromPx = (yPx) => (yRef - yPx) * cal.m_per_px;
      hUnit = "m";
    } else {
      heights = track.y.map((y) => yRef - y);
      fromPx = (yPx) => yRef - yPx;
      hUnit = "px";
    }
    heightUnit = hUnit;
    heightOf.set(track.id, { fromPx, perFrame });
    const markers = [];
    for (const fit of state.fits) {
      if (fit.track_id !== track.id) continue;
      for (const im of (fit.impacts || [])) {
        if (im.y != null) markers.push({ x: im.t, y: fromPx(im.y) });
      }
    }
    ySeries.push({ x: track.t, y: heights, color, points: true, markers });
    if (perFrame) {
      xSeries.push({ x: track.t, y: track.X_m.slice(), color, points: true });
      const mTrack = { t: track.t, x: track.X_m, y: track.Y_m };
      speedSeriesAll.push({ x: track.t, y: speedSeries(mTrack), color, width: 1.6 });
    } else {
      xSeries.push({ x: track.t, y: track.x.map(conv), color, points: true });
      speedSeriesAll.push({ x: track.t, y: speedSeries(track).map(conv), color, width: 1.6 });
    }
    // Residuals in height space: data height - fit height (sign flips vs y).
    const fit = state.fits.find((f) => f.track_id === track.id && f.curve);
    if (fit) {
      residSeries.push({
        x: track.t,
        y: track.t.map((tt, i) => heights[i] - curveHeightAt(fit, tt, track, fromPx)),
        color,
      });
    }
  }
  for (const fit of state.fits) {
    try {
      const track = state.tracks.get(fit.track_id);
      if (!track) continue;
      const color = trackColor(track, [...state.tracks.keys()].indexOf(fit.track_id));
      const { fromPx } = heightOf.get(fit.track_id);
      const curves = (fit.segments && fit.segments.length)
        ? fit.segments.map((s) => s.curve).filter(Boolean)
        : (fit.curve ? [fit.curve] : []);
      for (const c of curves) {
        if (!c || !c.t) continue;
        const cy = curveHeights(c, track, fromPx);
        if (cy) ySeries.push({ x: c.t, y: cy, color, dashed: true, width: 1.4 });
      }
    } catch { /* one bad fit must not blank the charts */ }
  }
  drawLineChart($("chart-y"), { series: ySeries, xLabel: "t (s)", yLabel: `height (${heightUnit})` }, t);
  drawLineChart($("chart-y-resid"), { series: residSeries, xLabel: "t (s)", yLabel: `resid (${heightUnit})` }, t);
  drawLineChart($("chart-x"), { series: xSeries, xLabel: "t (s)", yLabel: `x (${anyPerFrame ? "m" : unit})` }, t);
  drawLineChart($("chart-speed"), { series: speedSeriesAll, xLabel: "t (s)", yLabel: `speed (${anyPerFrame ? "m" : unit}/s)` }, t);

  const bars = restitutionBars();
  $("chart-e-box").hidden = !bars.length;
  if (bars.length) drawBarChart($("chart-e"), bars);
}

/** Fit-curve heights: Y_m/y_m if present, else px y transformed like data. */
function curveHeights(curve, track, fromPx) {
  const direct = curve.Y_m || curve.y_m;
  if (direct && direct.length) return direct.slice();
  if (curve.y && curve.y.length) return curve.y.map(fromPx);
  return null;
}

function curveHeightAt(fit, t, track, fromPx) {
  try {
    const curves = (fit.segments && fit.segments.length)
      ? fit.segments.map((s) => s.curve).filter(Boolean) : (fit.curve ? [fit.curve] : []);
    for (const c of curves) {
      if (!c || !c.t || !c.t.length) continue;
      if (t < c.t[0] || t > c.t[c.t.length - 1]) continue;
      const direct = c.Y_m || c.y_m;
      if (direct && direct.length) {
        const tmp = { t: c.t, y: direct };
        return curveAt(tmp, t, "y");
      }
      if (!c.y || !c.y.length) continue;
      return fromPx(curveAt(c, t, "y"));
    }
    const v = fit.curve ? curveAt(fit.curve, t, "y") : NaN;
    return isFinite(v) ? fromPx(v) : NaN;
  } catch { return NaN; }
}

function renderAll(animateAnswer = false) {
  renderLegend();
  renderFits();
  renderCharts();
  if (state.result.answer) renderAnswer(state.result.answer, animateAnswer);
  if (state.result.report_markdown) renderReport(state.result.report_markdown);
  renderCost(state.result.cost);
  drawOverlay();
}

/* ---------------- video + overlay ---------------- */

const video = $("video");
const overlay = $("overlay");

function frameGeometry() {
  const v = state.result.video || {};
  const fw = v.width || video.videoWidth || 1280;
  const fh = v.height || video.videoHeight || 720;
  const cw = overlay.clientWidth, ch = overlay.clientHeight;
  if (!cw || !ch) return null;
  // object-fit: contain letterbox
  const scale = Math.min(cw / fw, ch / fh);
  return { fw, fh, cw, ch, scale, ox: (cw - fw * scale) / 2, oy: (ch - fh * scale) / 2 };
}

function nearestContour(track, frame) {
  const contours = track.contours || {};
  if (contours[String(frame)] != null) return contours[String(frame)];
  let best = null, bestD = Infinity;
  for (const k of Object.keys(contours)) {
    const d = Math.abs(Number(k) - frame);
    if (d < bestD) { bestD = d; best = contours[k]; }
  }
  return best;
}

function drawOverlay() {
  if (state.stageMode === "strobe") return;
  const ctx = overlay.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const cw = overlay.clientWidth, ch = overlay.clientHeight;
  if (!cw || !ch) return;
  if (overlay.width !== Math.round(cw * dpr)) { overlay.width = Math.round(cw * dpr); overlay.height = Math.round(ch * dpr); }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cw, ch);
  const geo = frameGeometry();
  if (!geo || !state.tracks.size) { updateHud(-1, null, null); return; }
  const fps = effectiveFps();
  const tNow = video.currentTime || 0;
  const frame = Math.round(tNow * fps);
  const showMasks = $("tog-masks").checked, showTrail = $("tog-trail").checked, showFit = $("tog-fit").checked;
  let hudTrack = null, hudIndex = -1;

  let idx = 0;
  for (const track of state.tracks.values()) {
    const color = trackColor(track, idx++);
    const i = nearestIndex(track, tNow);
    if (i < 0) continue;
    if (hudTrack === null) { hudTrack = track; hudIndex = i; }
    const PX = (x) => geo.ox + x * geo.scale;
    const PY = (y) => geo.oy + y * geo.scale;

    if (showFit) {
      ctx.save();
      ctx.strokeStyle = color; ctx.globalAlpha = 0.85; ctx.lineWidth = 1.6;
      ctx.setLineDash([7, 6]);
      for (const fit of state.fits) {
        try {
        if (fit.track_id !== track.id) continue;
        const curves = (fit.segments && fit.segments.length)
          ? fit.segments.map((s) => s.curve).filter(Boolean)
          : (fit.curve ? [fit.curve] : []);
        for (const c of curves) {
          if (!c || !c.t) continue;
          ctx.beginPath();
          c.t.forEach((tt, k) => {
            const px = PX(c.x && c.x[k] != null ? c.x[k] : trackPosAt(track, tt, "x"));
            const py = PY(c.y && c.y[k] != null ? c.y[k] : trackPosAt(track, tt, "y"));
            k ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
          });
          ctx.stroke();
        }
        // impact markers
        for (const im of (fit.impacts || [])) {
          if (im.x == null || im.y == null) continue;
          const px = PX(im.x), py = PY(im.y);
          ctx.setLineDash([]);
          ctx.beginPath(); ctx.moveTo(px - 8, py - 8); ctx.lineTo(px + 8, py + 8);
          ctx.moveTo(px + 8, py - 8); ctx.lineTo(px - 8, py + 8); ctx.stroke();
          ctx.setLineDash([7, 6]);
          ctx.fillStyle = color;
          ctx.font = "11px ui-monospace, monospace";
          ctx.fillText(`impact ${im.index ?? ""}${im.e != null ? `  e=${Number(im.e).toFixed(2)}` : ""}`, px + 11, py - 8);
        }
        } catch { /* one bad fit must not blank the overlay */ }
      }
      ctx.restore();
    }

    if (showTrail && i > 0) {
      const lookback = Math.min(i, Math.round(fps * 1.2));
      for (let k = Math.max(0, i - lookback); k < i; k++) {
        const a = 0.08 + 0.6 * (k - (i - lookback)) / Math.max(1, lookback);
        ctx.strokeStyle = color; ctx.globalAlpha = a; ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(PX(track.x[k]), PY(track.y[k]));
        ctx.lineTo(PX(track.x[k + 1]), PY(track.y[k + 1]));
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }

    const d = nearestContour(track, track.frames ? track.frames[i] : frame);
    if (showMasks && d) {
      try {
        ctx.save();
        ctx.translate(geo.ox, geo.oy);
        ctx.scale(geo.scale, geo.scale);
        const path = new Path2D(d);
        ctx.fillStyle = color; ctx.globalAlpha = 0.28; ctx.fill(path);
        ctx.globalAlpha = 1; ctx.strokeStyle = color; ctx.lineWidth = 1.6 / geo.scale; ctx.stroke(path);
        ctx.restore();
      } catch { /* malformed path: skip this frame */ }
    }

    // centroid + label
    const cx = PX(track.x[i]), cy = PY(track.y[i]);
    ctx.fillStyle = color;
    ctx.beginPath(); ctx.arc(cx, cy, 4, 0, Math.PI * 2); ctx.fill();
    ctx.font = "12px sans-serif";
    const label = track.label || track.id;
    const wText = ctx.measureText(label).width;
    ctx.fillStyle = "rgba(4,10,20,.75)";
    ctx.fillRect(cx + 9, cy - 22, wText + 10, 17);
    ctx.fillStyle = "#eaf6fb";
    ctx.fillText(label, cx + 14, cy - 9);
  }
  updateHud(hudIndex, hudTrack, frame);
}

/** x position of a track at time t by interpolation (for fit ghost paths). */
function curveX(track, t) {
  const i = nearestIndex(track, t);
  return i >= 0 ? track.x[i] : 0;
}

function updateHud(index, track, frame) {
  const hud = $("hud");
  if (!track || index < 0) { hud.hidden = !state.tracks.size; if (!state.tracks.size) return; }
  if (!state.tracks.size) { hud.hidden = true; return; }
  hud.hidden = false;
  const fps = effectiveFps();
  const tNow = video.currentTime || 0;
  const fr = frame != null ? frame : Math.round(tNow * fps);
  let pos = "", vel = "";
  if (track && index >= 0) {
    const cal = calibration();
    const speeds = speedSeries(track);
    if (cal) {
      pos = `y = ${(track.y[index] * cal.m_per_px).toFixed(2)} m`;
      vel = `v = ${(speeds[index] * cal.m_per_px).toFixed(1)} m/s`;
    } else {
      pos = `y = ${track.y[index].toFixed(0)} px`;
      vel = `v = ${speeds[index].toFixed(0)} px/s`;
    }
  }
  hud.textContent = `t = ${tNow.toFixed(3)} s · frame ${fr}${pos ? ` · ${pos}` : ""}${vel ? ` · ${vel}` : ""}`;
  $("time-readout").textContent = `t = ${tNow.toFixed(3)} s`;
}

function syncLoop() {
  drawOverlay();
  renderChartsThrottled();
}
let chartTick = 0;
function renderChartsThrottled() {
  if (!state.tracks.size || $("charts-wrap").hidden) return;
  if (++chartTick % 3) return; // cursor redraw need not run every frame
  renderCharts(video.currentTime || 0);
}

function startSync() {
  if (typeof video.requestVideoFrameCallback === "function") {
    const cb = () => { syncLoop(); video.requestVideoFrameCallback(cb); };
    video.requestVideoFrameCallback(cb);
  } else {
    const cb = () => { if (!video.paused) syncLoop(); requestAnimationFrame(cb); };
    requestAnimationFrame(cb);
  }
  video.addEventListener("seeked", syncLoop);
  video.addEventListener("timeupdate", syncLoop);
}


/* ---------------- job lifecycle ---------------- */

function resetJob() {
  state.tracks.clear();
  state.fits = [];
  state.result = emptyResult();
  state.strobeUrl = null;
  state.stageMode = "video";
  $("timeline").innerHTML = "";
  $("answer-card").hidden = true;
  $("fits-wrap").hidden = true;
  $("charts-wrap").hidden = true;
  $("report-wrap").hidden = true;
  $("cost-line").hidden = true;
  $("track-legend").innerHTML = "";
  $("stage-mode").hidden = true;
  $("strobe-img").hidden = true;
  $("report-strobe").hidden = true;
  $("video").hidden = false;
  $("overlay").hidden = false;
  hideStageProgress();
  clearRunning();
  updateEmptyState();
}

/** Empty state shows only when no video source is available. */
function updateEmptyState() {
  const hasVideo = Boolean(state.result?.video?.url || video.getAttribute("src"));
  $("empty-state").hidden = hasVideo;
  if (hasVideo && state.stageMode !== "strobe") { video.hidden = false; overlay.hidden = false; }
}

function applyTrack(track) {
  if (!track || !track.id) return;
  state.tracks.set(track.id, track);
  state.result.tracks = [...state.tracks.values()];
  renderLegend();
  renderCharts();
  drawOverlay();
}

function applyFit(fit) {
  if (!fit) return;
  const i = state.fits.findIndex((f) => f.id === fit.id);
  if (i >= 0) state.fits[i] = fit; else state.fits.push(fit);
  state.result.fits = state.fits;
  renderFits();
  renderCharts();
  if (state.result.answer) renderAnswer(state.result.answer, false);
  drawOverlay();
}

function applyResult(result, { animateAnswer = false } = {}) {
  if (!result) return;
  state.result = { ...emptyResult(), ...result };
  state.jobId = result.id || state.jobId;
  state.tracks.clear();
  for (const tr of (result.tracks || [])) state.tracks.set(tr.id, tr);
  state.fits = result.fits || [];
  if (result.video && result.video.url) setVideoSource(result.video.url);
  if (result.strobe_url) setStrobe(result.strobe_url);
  if (result.question) $("question").value = result.question;
  clearRunning();
  $("timeline").innerHTML = "";
  for (const tr of (result.trace || [])) addTraceRow(tr);
  renderAll(animateAnswer);
  hideStageProgress();
  updateEmptyState();
}

function setVideoSource(url) {
  if (!url) return;
  video.preload = "auto";
  // Size the stage to the video's aspect ratio when the result declares it.
  const v = state.result?.video;
  if (v && v.width && v.height) $("stage").style.aspectRatio = `${v.width} / ${v.height}`;
  if (video.getAttribute("src") !== url) video.src = url;
  // Hide the empty state and show the video + overlay whenever a job or
  // example has a video (mock, example, live job and #job= reload all come
  // through here). The first frame is sought in the loadeddata handler.
  $("empty-state").hidden = true;
  if (state.stageMode !== "strobe") { video.hidden = false; overlay.hidden = false; }
}

async function loadJob(id) {
  const res = await fetch(`/api/jobs/${encodeURIComponent(id)}`);
  if (!res.ok) throw new Error(`job ${id} returned ${res.status}`);
  const result = await res.json();
  applyResult(result);
  return result;
}

function handleEvent(ev) {
  const type = ev.type || "";
  const trace = ev.trace || (type === "trace" ? ev : null);
  if (type === "stage") {
    const key = stageKeyFrom(ev.stage || ev.name || ev.title || "");
    addStageHead(ev.title || ev.stage || ev.name || "Stage");
    if (key) setStageProgress(key);
    showRunning(ev.summary || (key ? STAGE_LABELS[key] : "Working…"));
    return;
  }
  if (type === "strobe") { setStrobe(ev.url || ev.strobe_url || ev.strobe?.url); return; }
  if (type === "trace" && trace) {
    // tolerate flat events: {type:"trace", name, summary, ms, args}
    const row = { kind: ev.kind, name: ev.name, args: ev.args, summary: ev.summary, ms: ev.ms, step: ev.step, ...trace };
    addTraceRow(row);
    updateStageProgressFromTrace(row);
    state.result.trace.push(trace);
    return;
  }
  if (type === "track") { applyTrack(ev.track || ev); return; }
  if (type === "fit") { applyFit(ev.fit || ev); return; }
  if (type === "answer") {
    state.result.answer = ev.answer || ev;
    renderAnswer(state.result.answer, true);
    return;
  }
  if (type === "report") {
    state.result.report_markdown = ev.report_markdown || ev.markdown || ev.report || "";
    renderReport(state.result.report_markdown);
    return;
  }
  if (type === "plan" && ev.plan) { state.result.plan = ev.plan; return; }
  if (type === "calibration" && ev.calibration) { state.result.calibration = ev.calibration; renderCharts(); drawOverlay(); return; }
  if (ev.strobe_url && type !== "strobe") setStrobe(ev.strobe_url);
  if (type === "done") {
    clearRunning();
    hideStageProgress();
    setStatus("Done.");
    if (state.jobId && !MOCK) loadJob(state.jobId).catch(() => {});
    return;
  }
  if (type === "error") { clearRunning(); hideStageProgress(); showError(ev.message || ev.error || "Job failed."); return; }
}

function openEventStream(id) {
  if (state.eventSource) state.eventSource.close();
  const es = new EventSource(`/api/jobs/${encodeURIComponent(id)}/events`);
  state.eventSource = es;
  es.onmessage = (msg) => {
    try { handleEvent(JSON.parse(msg.data)); } catch { /* keep stream alive */ }
  };
  es.onerror = () => { /* browser retries; done/error close the job anyway */ };
  return es;
}

async function startJob() {
  if (!state.videoFile && !MOCK) { setStatus("Choose or drop a video first."); return; }
  const question = $("question").value.trim() || "What is g?";
  const facts = collectFacts();
  resetJob();
  state.result.question = question;
  $("measure").disabled = true;
  setStatus("Uploading…");
  addStageHead("Ingest");
  setStageProgress("ingest");
  showRunning("Uploading video…");
  try {
    let id;
    if (MOCK) {
      id = "mock-bounce-240";
      await runMockSequence();
      return;
    }
    const form = new FormData();
    form.append("video", state.videoFile);
    form.append("question", question);
    if (Object.keys(facts).length) form.append("facts", JSON.stringify(facts));
    const res = await fetch("/api/jobs", { method: "POST", body: form });
    if (!res.ok) throw new Error(`POST /api/jobs returned ${res.status}`);
    const data = await res.json();
    id = data.id;
    if (!id) throw new Error("POST /api/jobs did not return an id");
    state.jobId = id;
    history.replaceState(null, "", `#job=${encodeURIComponent(id)}`);
    if (state.objectUrl) setVideoSource(state.objectUrl);
    setStatus("Analysing…");
    openEventStream(id);
  } catch (err) {
    clearRunning();
    showError(err.message || String(err));
  } finally {
    $("measure").disabled = false;
  }
}

function collectFacts() {
  const facts = {};
  const ref = $("fact-ref").value.trim();
  const refSize = parseFloat($("fact-ref-size").value);
  const fps = parseFloat($("fact-fps").value);
  const diameter = parseFloat($("fact-diameter").value);
  if (ref) facts.reference_object = ref;
  if (isFinite(refSize) && refSize > 0) facts.reference_size_m = refSize;
  if (isFinite(fps) && fps > 0) facts.fps_override = fps;
  if (isFinite(diameter) && diameter > 0) facts.diameter_m = diameter;
  return facts;
}

/* ---------------- mock mode ---------------- */

async function loadMockResult() {
  const res = await fetch("mock/result.json");
  if (!res.ok) throw new Error("mock/result.json missing — run web/mock/make_mock.py");
  return res.json();
}

async function runMockSequence() {
  for (const t of state.mockTimers) clearTimeout(t);
  state.mockTimers = [];
  const result = await loadMockResult();
  state.result = emptyResult();
  state.result.question = result.question;
  state.result.video = result.video;
  state.jobId = result.id;
  setVideoSource("mock/clip.mp4");
  setStatus("Mock replay…");
  const at = (ms, fn) => state.mockTimers.push(setTimeout(fn, ms));
  const traces = result.trace || [];
  at(200, () => { addStageHead("Plan · Muse Spark"); setStageProgress("plan"); showRunning(STAGE_LABELS.plan); });
  traces.slice(0, 1).forEach((tr, i) => at(500 + i * 500, () => addTraceRow(tr)));
  at(1400, () => { addStageHead("Track · SAM 3.1"); setStageProgress("agent"); showRunning(STAGE_LABELS.agent); });
  traces.slice(1, 3).forEach((tr, i) => at(1900 + i * 600, () => { addTraceRow(tr); updateStageProgressFromTrace(tr); }));
  at(3300, () => { for (const tr of result.tracks || []) applyTrack(tr); if (result.strobe_url) setStrobe(result.strobe_url); });
  at(3800, () => addStageHead("Fit · local physics"));
  traces.slice(3, 4).forEach((tr, i) => at(4100 + i * 400, () => addTraceRow(tr)));
  at(4500, () => { state.result.calibration = result.calibration; for (const f of result.fits || []) applyFit(f); });
  at(5000, () => addStageHead("Answer · Muse Spark"));
  traces.slice(4, 5).forEach((tr) => at(5150, () => addTraceRow(tr)));
  at(5300, () => { state.result.answer = result.answer; renderAnswer(result.answer, true); });
  at(5600, () => { addStageHead("Report · Muse Wonder"); setStageProgress("report"); showRunning(STAGE_LABELS.report); });
  traces.slice(5).forEach((tr) => at(5750, () => addTraceRow(tr)));
  at(5950, () => {
    state.result.report_markdown = result.report_markdown;
    state.result.cost = result.cost;
    state.result.plan = result.plan;
    state.result.status = "done";
    if (result.strobe_url) setStrobe(result.strobe_url);
    renderReport(result.report_markdown);
    renderCost(result.cost);
    clearRunning();
    hideStageProgress();
    setStatus("Done (mock).");
  });
}

async function loadMockInstant() {
  const result = await loadMockResult();
  applyResult(result);
  setStatus("Mock example loaded.");
}

/* ---------------- examples ---------------- */

async function loadExamples() {
  if (MOCK) {
    renderExamples([{ id: "mock-bounce-240", title: "Bouncing ball · 240 fps",
      subtitle: "Real SAM fixture", thumb_url: "", question: "What is g?", job_id: "mock-bounce-240" }]);
    return;
  }
  try {
    const res = await fetch("/api/examples");
    if (!res.ok) return;
    const list = await res.json();
    if (Array.isArray(list) && list.length) renderExamples(list);
  } catch { /* backend offline: gallery stays hidden */ }
}

function renderExamples(list) {
  const wrap = $("examples-wrap"), box = $("examples");
  wrap.hidden = false;
  box.innerHTML = "";
  for (const ex of list) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "example";
    btn.innerHTML = `${ex.thumb_url ? `<img src="${esc(ex.thumb_url)}" alt="">` : ""}` +
      `<span class="ex-body"><span class="ex-title">${esc(ex.title || ex.id)}</span><br>` +
      `<span class="ex-sub">${esc(ex.subtitle || "")}</span></span>`;
    btn.addEventListener("click", async () => {
      if (ex.question) $("question").value = ex.question;
      try {
        if (MOCK) await loadMockInstant();
        else if (ex.job_id) { resetJob(); await loadJob(ex.job_id); setStatus("Example loaded."); }
      } catch (err) { showError(err.message || String(err)); }
    });
    box.appendChild(btn);
  }
}

/* ---------------- input: file, chips, mic ---------------- */

function setVideoFile(file) {
  if (!file) return;
  state.videoFile = file;
  if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
  state.objectUrl = URL.createObjectURL(file);
  setVideoSource(state.objectUrl);
  const dz = $("dz-file");
  dz.hidden = false;
  dz.textContent = `${file.name} · ${(file.size / 1e6).toFixed(1)} MB`;
  setStatus("Ready — ask a question and press Measure.");
}

function initInput() {
  const dz = $("dropzone"), input = $("file-input");
  dz.addEventListener("click", () => input.click());
  dz.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
  input.addEventListener("change", () => setVideoFile(input.files[0]));
  for (const ev of ["dragenter", "dragover"]) dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("dragover"); });
  for (const ev of ["dragleave", "drop"]) dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("dragover"); });
  dz.addEventListener("drop", (e) => setVideoFile(e.dataTransfer.files[0]));
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => e.preventDefault());

  $("chips").addEventListener("click", (e) => {
    const chip = e.target.closest(".chip");
    if (chip) $("question").value = chip.textContent.trim();
  });
  $("measure").addEventListener("click", startJob);
  initMic();
}

let mediaRecorder = null, audioChunks = [];
function initMic() {
  const mic = $("mic"), status = $("mic-status");
  const start = async (e) => {
    e.preventDefault();
    if (mediaRecorder && mediaRecorder.state === "recording") return;
    if (MOCK) {
      mic.classList.add("recording");
      status.textContent = "Listening… (mock)";
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioChunks = [];
      mediaRecorder = new MediaRecorder(stream);
      mediaRecorder.ondataavailable = (ev) => { if (ev.data.size) audioChunks.push(ev.data); };
      mediaRecorder.onstop = async () => {
        stream.getTracks().forEach((tr) => tr.stop());
        mic.classList.remove("recording");
        const blob = new Blob(audioChunks, { type: mediaRecorder.mimeType || "audio/webm" });
        status.textContent = "Transcribing…";
        try {
          const form = new FormData();
          form.append("audio", blob, "question.webm");
          const res = await fetch("/api/transcribe", { method: "POST", body: form });
          if (!res.ok) throw new Error(`transcribe returned ${res.status}`);
          const data = await res.json();
          if (data.transcript) $("question").value = data.transcript;
          status.textContent = "transcribed by Muse Voice";
        } catch (err) { status.textContent = err.message || String(err); }
      };
      mediaRecorder.start();
      mic.classList.add("recording");
      status.textContent = "Listening… release to transcribe";
    } catch { status.textContent = "Microphone unavailable."; }
  };
  const stop = () => {
    if (MOCK) {
      if (!mic.classList.contains("recording")) return;
      mic.classList.remove("recording");
      $("question").value = "What is g?";
      status.textContent = "transcribed by Muse Voice (mock)";
      return;
    }
    if (mediaRecorder && mediaRecorder.state === "recording") mediaRecorder.stop();
  };
  mic.addEventListener("pointerdown", start);
  mic.addEventListener("pointerup", stop);
  mic.addEventListener("pointerleave", stop);
}

/* ---------------- transport + keyboard ---------------- */

function initTransport() {
  const playBtn = $("btn-play");
  const syncPlayLabel = () => { playBtn.textContent = video.paused ? "▶" : "⏸"; };
  playBtn.addEventListener("click", () => { video.paused ? video.play() : video.pause(); });
  video.addEventListener("play", syncPlayLabel);
  video.addEventListener("pause", syncPlayLabel);
  const step = (dir) => {
    video.pause();
    const fps = effectiveFps();
    video.currentTime = Math.max(0, Math.min(video.duration || 0, video.currentTime + dir / fps));
  };
  $("btn-prev").addEventListener("click", () => step(-1));
  $("btn-next").addEventListener("click", () => step(1));
  $("speed").addEventListener("change", (e) => { video.playbackRate = parseFloat(e.target.value); });
  for (const id of ["tog-masks", "tog-trail", "tog-fit"]) $(id).addEventListener("change", drawOverlay);
  document.addEventListener("keydown", (e) => {
    const tag = (e.target.tagName || "").toLowerCase();
    if (tag === "textarea" || tag === "input" || tag === "select" || e.target.isContentEditable) return;
    if (e.code === "Space") { e.preventDefault(); video.paused ? video.play() : video.pause(); }
    if (e.key === "ArrowLeft") step(-1);
    if (e.key === "ArrowRight") step(1);
  });
  const seek = (t) => { if (isFinite(t)) { video.currentTime = Math.max(0, t); syncLoop(); } };
  for (const id of ["chart-y", "chart-y-resid", "chart-x", "chart-speed", "chart-e"]) {
    const canvas = $(id);
    const tip = canvas.parentElement.querySelector(".chart-tip");
    wireChart(canvas, tip, { onSeek: seek });
  }
  window.addEventListener("resize", () => { drawOverlay(); renderCharts(); });
  $("stage-mode").addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-mode]");
    if (btn) setStageMode(btn.dataset.mode);
  });
  video.addEventListener("loadeddata", () => {
    // Show the first frame before play.
    if (video.paused && video.currentTime === 0) {
      try { video.currentTime = 0.001; } catch { /* not seekable yet */ }
    }
    updateEmptyState();
    drawOverlay();
  });
  video.addEventListener("loadedmetadata", () => {
    const stage = $("stage");
    if (video.videoWidth && video.videoHeight) stage.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
    updateEmptyState();
    drawOverlay();
  });
}

/* ---------------- init ---------------- */

async function init() {
  try {
    if (localStorage.getItem("physicam-how-dismissed") === "1") $("how-it-works").hidden = true;
  } catch { /* storage unavailable */ }
  $("how-dismiss").addEventListener("click", () => {
    $("how-it-works").hidden = true;
    try { localStorage.setItem("physicam-how-dismissed", "1"); } catch { /* ignore */ }
  });
  initInput();
  initTransport();
  startSync();
  loadExamples();
  if (MOCK) {
    setStatus("Mock mode — press Measure to replay the analysis, or open the example.");
    $("dz-file").hidden = false;
    $("dz-file").textContent = "mock/clip.mp4 (real 240 fps bounce)";
    setVideoSource("mock/clip.mp4");
    return;
  }
  const hash = new URLSearchParams(location.hash.replace(/^#/, ""));
  const jobId = hash.get("job");
  if (jobId) {
    state.jobId = jobId;
    try { await loadJob(jobId); setStatus("Job loaded."); }
    catch (err) { showError(err.message || String(err)); }
  }
}

init().catch((err) => showError(err.message || String(err)));
