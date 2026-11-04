/** Hand-rolled canvas charts (no dependencies): line charts with optional
 * dashed fit series, a residuals strip, and a bar chart with error bars.
 * High-DPI aware, light grid, hover crosshair tooltip, click-to-seek and a
 * vertical cursor synced to the video time. */

function theme() {
  const cs = getComputedStyle(document.documentElement);
  return {
    grid: cs.getPropertyValue("--border-soft").trim() || "#dde6f2",
    text: cs.getPropertyValue("--muted").trim() || "#93a5c4",
    accent: cs.getPropertyValue("--accent").trim() || "#22d3ee",
  };
}

function setup(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (!w || !h) return null;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

const PAD = { l: 52, r: 12, t: 10, b: 26 };

function niceRange(values, includeZero = false) {
  let lo = Math.min(...values), hi = Math.max(...values);
  if (!isFinite(lo) || !isFinite(hi)) return [0, 1];
  if (includeZero) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }
  if (lo === hi) { lo -= 1; hi += 1; }
  const pad = (hi - lo) * 0.08;
  return [lo - pad, hi + pad];
}

function axes(ctx, w, h, xr, yr, th, xLabel, yLabel) {
  const iw = w - PAD.l - PAD.r, ih = h - PAD.t - PAD.b;
  const X = (v) => PAD.l + (v - xr[0]) / (xr[1] - xr[0]) * iw;
  const Y = (v) => PAD.t + ih - (v - yr[0]) / (yr[1] - yr[0]) * ih;
  ctx.strokeStyle = th.grid; ctx.fillStyle = th.text;
  ctx.font = "10.5px ui-monospace, monospace"; ctx.lineWidth = 1;
  for (let k = 0; k <= 4; k++) {
    const v = yr[0] + (yr[1] - yr[0]) * k / 4, y = Y(v);
    ctx.beginPath(); ctx.moveTo(PAD.l, y); ctx.lineTo(w - PAD.r, y); ctx.stroke();
    ctx.textAlign = "right"; ctx.fillText(v.toPrecision(3), PAD.l - 6, y + 3);
  }
  for (let k = 0; k <= 5; k++) {
    const v = xr[0] + (xr[1] - xr[0]) * k / 5, x = X(v);
    ctx.beginPath(); ctx.moveTo(x, PAD.t); ctx.lineTo(x, PAD.t + ih); ctx.stroke();
    ctx.textAlign = "center"; ctx.fillText(v.toFixed(2), x, h - 12);
  }
  ctx.textAlign = "center"; ctx.fillText(xLabel, PAD.l + iw / 2, h - 2);
  ctx.save(); ctx.translate(11, PAD.t + ih / 2); ctx.rotate(-Math.PI / 2);
  ctx.fillText(yLabel, 0, 0); ctx.restore();
  return { X, Y, iw, ih };
}

/** state: {x:[], y:[] per series…} — see drawLineChart config in app.js. */
export function drawLineChart(canvas, cfg, cursorT) {
  const s = setup(canvas);
  if (!s) return;
  const { ctx, w, h } = s;
  const th = theme();
  const allX = [], allY = [];
  for (const ser of cfg.series) {
    for (let i = 0; i < (ser.x || []).length; i++) {
      if (isFinite(ser.x[i]) && isFinite(ser.y[i])) { allX.push(ser.x[i]); allY.push(ser.y[i]); }
    }
  }
  if (!allX.length) {
    ctx.fillStyle = th.text; ctx.font = "12px sans-serif"; ctx.textAlign = "center";
    ctx.fillText("No data yet", w / 2, h / 2);
    canvas._geom = null;
    return;
  }
  const xr = cfg.xRange || niceRange(allX);
  const yr = cfg.yRange || niceRange(allY);
  const { X, Y } = axes(ctx, w, h, xr, yr, th, cfg.xLabel || "", cfg.yLabel || "");
  // zero line for residuals
  if (yr[0] < 0 && yr[1] > 0) {
    ctx.strokeStyle = th.text; ctx.globalAlpha = 0.5;
    ctx.beginPath(); ctx.moveTo(PAD.l, Y(0)); ctx.lineTo(w - PAD.r, Y(0)); ctx.stroke();
    ctx.globalAlpha = 1;
  }
  for (const ser of cfg.series) {
    ctx.strokeStyle = ser.color; ctx.fillStyle = ser.color; ctx.lineWidth = ser.width || 1.6;
    ctx.setLineDash(ser.dashed ? [6, 5] : []);
    ctx.beginPath();
    let pen = false;
    for (let i = 0; i < (ser.x || []).length; i++) {
      const xv = ser.x[i], yv = ser.y[i];
      if (!isFinite(xv) || !isFinite(yv)) { pen = false; continue; }
      const px = X(xv), py = Y(yv);
      if (!pen) { ctx.moveTo(px, py); pen = true; } else ctx.lineTo(px, py);
    }
    ctx.stroke();
    ctx.setLineDash([]);
    if (ser.points) {
      const step = Math.max(1, Math.floor(ser.x.length / 220));
      ctx.beginPath();
      for (let i = 0; i < ser.x.length; i += step) {
        if (!isFinite(ser.y[i])) continue;
        ctx.moveTo(X(ser.x[i]) + 2.2, Y(ser.y[i]));
        ctx.arc(X(ser.x[i]), Y(ser.y[i]), 2.2, 0, Math.PI * 2);
      }
      ctx.fill();
    }
    if (ser.markers) {
      for (const mk of ser.markers) {
        const px = X(mk.x), py = Y(mk.y);
        ctx.strokeStyle = ser.color; ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(px - 6, py - 6); ctx.lineTo(px + 6, py + 6);
        ctx.moveTo(px + 6, py - 6); ctx.lineTo(px - 6, py + 6);
        ctx.stroke();
      }
    }
  }
  if (cursorT != null && isFinite(cursorT) && cursorT >= xr[0] && cursorT <= xr[1]) {
    ctx.strokeStyle = th.accent; ctx.lineWidth = 1.4;
    ctx.beginPath(); ctx.moveTo(X(cursorT), PAD.t); ctx.lineTo(X(cursorT), h - PAD.b); ctx.stroke();
  }
  canvas._geom = { xr, yr, pad: PAD, cfg };
}

/** Bar chart with error bars. bars: [{label, value, stderr, color}]. */
export function drawBarChart(canvas, bars) {
  const s = setup(canvas);
  if (!s) return;
  const { ctx, w, h } = s;
  const th = theme();
  if (!bars.length) {
    ctx.fillStyle = th.text; ctx.font = "12px sans-serif"; ctx.textAlign = "center";
    ctx.fillText("No impacts", w / 2, h / 2);
    return;
  }
  const tops = bars.map((b) => b.value + (b.stderr || 0));
  const yr = [0, Math.max(1.15, ...tops) * 1.12];
  const xr = [0, bars.length];
  // Categorical x axis: no numeric x ticks (they overlap); labels are drawn
  // under each bar below. Only the y axis / gridlines are drawn here.
  const iw = w - PAD.l - PAD.r, ih = h - PAD.t - PAD.b;
  const X = (v) => PAD.l + (v - xr[0]) / (xr[1] - xr[0]) * iw;
  const Y = (v) => PAD.t + ih - (v - yr[0]) / (yr[1] - yr[0]) * ih;
  ctx.strokeStyle = th.grid; ctx.fillStyle = th.text;
  ctx.font = "10.5px ui-monospace, monospace"; ctx.lineWidth = 1;
  for (let k = 0; k <= 4; k++) {
    const v = yr[0] + (yr[1] - yr[0]) * k / 4, y = Y(v);
    ctx.beginPath(); ctx.moveTo(PAD.l, y); ctx.lineTo(w - PAD.r, y); ctx.stroke();
    ctx.textAlign = "right"; ctx.fillText(v.toPrecision(3), PAD.l - 6, y + 3);
  }
  ctx.textAlign = "center"; ctx.fillText("impact", PAD.l + iw / 2, h - 2);
  ctx.save(); ctx.translate(11, PAD.t + ih / 2); ctx.rotate(-Math.PI / 2);
  ctx.fillText("e", 0, 0); ctx.restore();
  const bw = (X(1) - X(0)) * 0.52;
  bars.forEach((b, i) => {
    const cx = X(i + 0.5), top = Y(Math.min(b.value, yr[1])), base = Y(0);
    ctx.fillStyle = b.color;
    ctx.globalAlpha = 0.85;
    ctx.fillRect(cx - bw / 2, top, bw, base - top);
    ctx.globalAlpha = 1;
    ctx.strokeStyle = b.color; ctx.lineWidth = 2;
    const eTop = Y(b.value + (b.stderr || 0)), eBot = Y(Math.max(0, b.value - (b.stderr || 0)));
    ctx.beginPath();
    ctx.moveTo(cx, eTop); ctx.lineTo(cx, eBot);
    ctx.moveTo(cx - 7, eTop); ctx.lineTo(cx + 7, eTop);
    ctx.moveTo(cx - 7, eBot); ctx.lineTo(cx + 7, eBot);
    ctx.stroke();
    // value above the error-bar cap, categorical label below the axis
    ctx.fillStyle = th.text; ctx.font = "11px ui-monospace, monospace"; ctx.textAlign = "center";
    ctx.fillText(b.value.toFixed(2), cx, Math.max(PAD.t + 10, eTop - 6));
    ctx.fillText(b.label, cx, h - 12);
  });
  // physical bound e = 1
  ctx.strokeStyle = th.accent; ctx.setLineDash([5, 5]); ctx.lineWidth = 1.2;
  ctx.beginPath(); ctx.moveTo(PAD.l, Y(1)); ctx.lineTo(w - PAD.r, Y(1)); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = th.accent; ctx.font = "10.5px ui-monospace, monospace"; ctx.textAlign = "right";
  ctx.fillText("e = 1", w - PAD.r - 2, Y(1) - 4);
  canvas._geom = { xr, yr, pad: PAD, cfg: { bar: true, bars } };
}

/** Wire hover tooltip + click-to-seek on a line chart. getCursor() returns t. */
export function wireChart(canvas, tip, { onSeek, seriesLabel }) {
  canvas.addEventListener("mousemove", (ev) => {
    const g = canvas._geom;
    if (!g || g.cfg.bar) { tip.hidden = true; return; }
    const rect = canvas.getBoundingClientRect();
    const mx = ev.clientX - rect.left, my = ev.clientY - rect.top;
    const iw = rect.width - g.pad.l - g.pad.r;
    if (mx < g.pad.l || mx > rect.width - g.pad.r) { tip.hidden = true; return; }
    const t = g.xr[0] + (mx - g.pad.l) / iw * (g.xr[1] - g.xr[0]);
    let best = null;
    for (const ser of g.cfg.series) {
      for (let i = 0; i < (ser.x || []).length; i++) {
        const d = Math.abs(ser.x[i] - t);
        if (isFinite(ser.y[i]) && (!best || d < best.d)) best = { d, ser, i };
      }
    }
    if (!best) { tip.hidden = true; return; }
    tip.hidden = false;
    tip.style.left = `${mx + 14}px`;
    tip.style.top = `${Math.max(4, my - 10)}px`;
    tip.textContent = `${seriesLabel || ""}t = ${best.ser.x[best.i].toFixed(3)} s · ${best.ser.y[best.i].toPrecision(4)}`;
  });
  canvas.addEventListener("mouseleave", () => { tip.hidden = true; });
  canvas.addEventListener("click", (ev) => {
    const g = canvas._geom;
    if (!g || !onSeek) return;
    const rect = canvas.getBoundingClientRect();
    const mx = ev.clientX - rect.left;
    if (g.cfg.bar) {
      const iw = rect.width - g.pad.l - g.pad.r;
      const idx = Math.floor((mx - g.pad.l) / iw * g.xr[1]);
      if (g.cfg.bars && g.cfg.bars[idx]) onSeek(g.cfg.bars[idx].t);
      return;
    }
    const iw = rect.width - g.pad.l - g.pad.r;
    const t = g.xr[0] + (mx - g.pad.l) / iw * (g.xr[1] - g.xr[0]);
    onSeek(t);
  });
}
