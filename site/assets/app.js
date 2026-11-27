/* LPAC range-transfer site. Every chart is recomputed here from site/data/*.json
   (built by site/build_data.py from the repository's result CSVs). */
(function () {
  "use strict";

  // ------------------------------------------------------------------ utils
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => Array.from(el.querySelectorAll(s));
  const NS = "http://www.w3.org/2000/svg";
  const RANGES = [128, 256, 512, 768, 1024];
  const fmt = (v, d = 2) => (v == null || Number.isNaN(v) ? "–" : (Math.round(v * 10 ** d) / 10 ** d).toFixed(d));
  const fmtS = (v, d = 3) => (v == null || Number.isNaN(v) ? "–" : (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(d));
  const mean = (a) => (a.length ? a.reduce((s, x) => s + x, 0) / a.length : NaN);
  const sd = (a) => { if (a.length < 2) return 0; const m = mean(a); return Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / (a.length - 1)); };
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  function el(tag, attrs = {}, ...kids) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") e.className = v; else if (k === "html") e.innerHTML = v;
      else if (k.startsWith("on")) e.addEventListener(k.slice(2), v); else if (v != null) e.setAttribute(k, v);
    }
    for (const k of kids) if (k != null) e.append(k);
    return e;
  }
  function svg(tag, attrs = {}) {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) if (v != null) e.setAttribute(k, v);
    return e;
  }
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  // ------------------------------------------------------------------ theme
  function applyTheme(t) {
    if (t === "system") document.documentElement.removeAttribute("data-theme");
    else document.documentElement.setAttribute("data-theme", t);
    $$("[data-theme-set]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.themeSet === t)));
    try { localStorage.setItem("lpac-theme", t); } catch (e) { /* storage unavailable */ }
    window.dispatchEvent(new Event("themechange"));
  }
  $$("[data-theme-set]").forEach((b) => b.addEventListener("click", () => applyTheme(b.dataset.themeSet)));
  try { const t = localStorage.getItem("lpac-theme"); if (t) applyTheme(t); } catch (e) { /* ignore */ }
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => window.dispatchEvent(new Event("themechange")));

  // toc highlight
  const tocLinks = $$(".toc a");
  const obs = new IntersectionObserver((entries) => {
    for (const en of entries) if (en.isIntersecting) {
      tocLinks.forEach((a) => a.classList.toggle("active", a.getAttribute("href") === "#" + en.target.id));
    }
  }, { rootMargin: "-30% 0px -60% 0px" });
  $$("section.chapter").forEach((s) => obs.observe(s));

  // ------------------------------------------------------------------ data
  async function getJSON(name) {
    try { const r = await fetch("data/" + name, { cache: "no-cache" }); if (!r.ok) return null; return await r.json(); }
    catch (e) { return null; }
  }
  function decode(t) {
    if (!t || !t.columns) return [];
    const runs = t.runs || {}, src = t.sources || [];
    return t.rows.map((r) => {
      const o = {};
      t.columns.forEach((c, i) => { o[c] = r[i]; });
      if (o.run_id != null && runs[o.run_id]) Object.assign(o, runs[o.run_id], { run_id: o.run_id });
      if (typeof o.source === "number") o.source = src[o.source];
      return o;
    });
  }

  // Color follows the model everywhere on the page.
  const COLORS = {
    published: "--s8", fix: "--s2", lpac128: "--s8", lpac256: "--s1", lpac512: "--s7", lpac1024: "--s6",
    v0mixed: "--s5", A_mixed: "--s4", A_128: "--s4", V1: "--s7", V2: "--s5", V3: "--s6", V3mixed: "--s6",
    V4_128: "--s1", V4mixed: "--s3", V5mixed: "--s4", V5_128: "--s4", V6mixed: "--s7", V0R: "--s5",
    V2res: "--s1", mixture: "--s7", clamp: "--s5", V2M: "--s2", V4half: "--s4", V4T: "--s6",
  };
  const colorOf = (k) => `var(${COLORS[k] || "--s1"})`;

  // Cost aggregation: per run -> mean over environments; per model -> mean and sd over runs.
  function pick(recs, f) {
    return recs.filter((r) => {
      if (f.model && r.model !== f.model) return false;
      if (f.env_set && r.env_set !== f.env_set) return false;
      if (f.range != null && r.range_m !== f.range) return false;
      if (f.map && !(Array.isArray(f.map) ? f.map.includes(r.map_level) : r.map_level === f.map)) return false;
      if (f.graph && (r.graph_level || "G0") !== f.graph) return false;
      if (f.primary !== false && r.primary === false) return false;
      if (f.run && r.run_id !== f.run) return false;
      return true;
    });
  }
  function byEnvSeedAvg(rows) { // env -> mean over runs
    const m = new Map();
    for (const r of rows) { if (!m.has(r.env)) m.set(r.env, []); m.get(r.env).push(r.cost_ratio); }
    const o = new Map(); for (const [k, v] of m) o.set(k, mean(v)); return o;
  }
  function stat(recs, f) {
    const rows = pick(recs, f);
    if (!rows.length) return null;
    const runs = new Map();
    for (const r of rows) { if (!runs.has(r.run_id)) runs.set(r.run_id, []); runs.get(r.run_id).push(r.cost_ratio); }
    const runMeans = Array.from(runs.values()).map(mean);
    const env = byEnvSeedAvg(rows);
    const envVals = Array.from(env.values());
    return { y: mean(runMeans), sdRuns: sd(runMeans), sdEnv: sd(envVals), nRuns: runMeans.length, nEnv: env.size, env, runMeans };
  }
  function paired(a, b) { // a,b: Map env->value. returns a-b
    const d = []; for (const [k, v] of a) if (b.has(k)) d.push(v - b.get(k));
    if (!d.length) return null;
    const s = sd(d);
    return { d: mean(d), se: s / Math.sqrt(d.length), n: d.length, better: d.filter((x) => x < 0).length };
  }

  // ------------------------------------------------------------------ chart
  let tip;
  function tooltip() { if (!tip) { tip = el("div", { class: "tooltip", hidden: "" }); document.body.append(tip); tip.style.position = "fixed"; } return tip; }
  function showTip(html, x, y) {
    const t = tooltip(); t.innerHTML = html; t.hidden = false;
    const r = t.getBoundingClientRect(); let left = x + 14, top = y + 14;
    if (left + r.width > innerWidth - 8) left = x - r.width - 14;
    if (top + r.height > innerHeight - 8) top = y - r.height - 14;
    t.style.left = Math.max(8, left) + "px"; t.style.top = Math.max(8, top) + "px";
  }
  const hideTip = () => { if (tip) tip.hidden = true; };

  /* spec: {xs:[...], xScale:'log2'|'lin', xLabel, yLabel, xFmt, series:[{key,label,color,dash,pts:[{x,y,lo,hi,note}]}],
            refs:[{y,label}], yMin, yMax, height} */
  function lineChart(host, spec) {
    host.innerHTML = "";
    const W = 680, H = spec.height || 360, m = { l: 52, r: spec.directLabels === false ? 16 : 150, t: 14, b: 46 };
    const hidden = host._hidden || new Set(); host._hidden = hidden;
    const vis = spec.series.filter((s) => !hidden.has(s.key) && s.pts.some((p) => p.y != null));
    const allY = [];
    for (const s of vis) for (const p of s.pts) if (p.y != null) { allY.push(p.y); if (p.lo != null) allY.push(p.lo); if (p.hi != null) allY.push(p.hi); }
    (spec.refs || []).forEach((r) => allY.push(r.y));
    if (!allY.length) { host.append(el("div", { class: "pending" }, spec.empty || "No data for this selection yet.")); return; }
    let y0 = spec.yMin != null ? spec.yMin : Math.min(...allY), y1 = spec.yMax != null ? spec.yMax : Math.max(...allY);
    const pad = (y1 - y0) * 0.08 || 0.1; if (spec.yMin == null) y0 -= pad; if (spec.yMax == null) y1 += pad;
    const step = niceStep((y1 - y0) / 5); y0 = Math.floor(y0 / step) * step; y1 = Math.ceil(y1 / step) * step;
    const xs = spec.xs, tx = spec.xScale === "log2" ? Math.log2 : (v) => v;
    const xa = tx(xs[0]), xb = tx(xs[xs.length - 1]);
    const X = (v) => m.l + ((tx(v) - xa) / (xb - xa || 1)) * (W - m.l - m.r);
    const Y = (v) => m.t + (1 - (v - y0) / (y1 - y0)) * (H - m.t - m.b);
    const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": spec.aria || spec.yLabel });
    const g = svg("g", { class: "grid" }); root.append(g);
    for (let v = y0; v <= y1 + 1e-9; v += step) {
      g.append(svg("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }));
      const t = svg("text", { x: m.l - 8, y: Y(v) + 4, "text-anchor": "end", class: "tick" }); t.textContent = fmt(v, step < 0.1 ? 2 : 1); root.append(t);
    }
    root.append(svg("line", { class: "baseline", x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b }));
    for (const v of xs) {
      const t = svg("text", { x: X(v), y: H - m.b + 18, "text-anchor": "middle", class: "tick" }); t.textContent = spec.xFmt ? spec.xFmt(v) : v; root.append(t);
    }
    const xl = svg("text", { x: (m.l + W - m.r) / 2, y: H - 8, "text-anchor": "middle", class: "axis-title" }); xl.textContent = spec.xLabel; root.append(xl);
    const yl = svg("text", { x: 14, y: (m.t + H - m.b) / 2, "text-anchor": "middle", class: "axis-title", transform: `rotate(-90 14 ${(m.t + H - m.b) / 2})` }); yl.textContent = spec.yLabel; root.append(yl);
    for (const r of spec.refs || []) {
      if (r.y < y0 || r.y > y1) continue;
      root.append(svg("line", { class: "ref", x1: m.l, x2: W - m.r, y1: Y(r.y), y2: Y(r.y) }));
      const t = svg("text", { x: m.l + 6, y: Y(r.y) - 5, class: "ref-label" }); t.textContent = r.label; root.append(t);
    }
    const labels = [];
    for (const s of vis) {
      const gs = svg("g", { class: "series" }); gs.dataset.key = s.key;
      const pts = s.pts.filter((p) => p.y != null);
      const band = pts.filter((p) => p.lo != null && p.hi != null);
      if (band.length > 1) {
        const d = band.map((p, i) => `${i ? "L" : "M"}${X(p.x)},${Y(p.hi)}`).join("") + band.slice().reverse().map((p) => `L${X(p.x)},${Y(p.lo)}`).join("") + "Z";
        gs.append(svg("path", { d, class: "band", style: `fill:${s.color}` }));
      }
      if (pts.length > 1) gs.append(svg("path", { class: "line", d: pts.map((p, i) => `${i ? "L" : "M"}${X(p.x)},${Y(p.y)}`).join(""), style: `stroke:${s.color}`, "stroke-dasharray": s.dash || null }));
      if (pts.length <= 12) for (const p of pts) gs.append(svg("circle", { cx: X(p.x), cy: Y(p.y), r: 4, style: `fill:${s.color}` }));
      root.append(gs);
      const last = pts[pts.length - 1];
      if (last && spec.directLabels !== false) labels.push({ y: Y(last.y), x: X(last.x), text: s.short || s.label });
    }
    // de-collide direct labels
    labels.sort((a, b) => a.y - b.y);
    for (let i = 1; i < labels.length; i++) if (labels[i].y - labels[i - 1].y < 13) labels[i].y = labels[i - 1].y + 13;
    for (const l of labels) { const t = svg("text", { x: l.x + 9, y: l.y + 4, class: "direct" }); t.textContent = l.text; root.append(t); }
    // hover
    const hl = svg("line", { class: "hover-line", y1: m.t, y2: H - m.b, visibility: "hidden" }); root.append(hl);
    const hit = svg("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent" }); root.append(hit);
    function move(ev) {
      const pt = root.createSVGPoint(); pt.x = ev.clientX; pt.y = ev.clientY;
      const p = pt.matrixTransform(root.getScreenCTM().inverse());
      let best = xs[0], bd = Infinity; for (const v of xs) { const d = Math.abs(X(v) - p.x); if (d < bd) { bd = d; best = v; } }
      hl.setAttribute("x1", X(best)); hl.setAttribute("x2", X(best)); hl.setAttribute("visibility", "visible");
      const rows = vis.map((s) => ({ s, p: s.pts.find((q) => q.x === best) })).filter((r) => r.p && r.p.y != null).sort((a, b) => a.p.y - b.p.y);
      if (!rows.length) return hideTip();
      showTip(`<div class="tt-head">${esc(spec.xFmt ? spec.xFmt(best) : best)} ${esc(spec.xUnit || "")}</div>` + rows.map((r) =>
        `<div class="tt-row"><i style="background:${r.s.color}"></i><span>${esc(r.s.label)}</span><b>${fmt(r.p.y)}${r.p.err != null ? " ± " + fmt(r.p.err) : ""}</b></div>${r.p.note ? `<div class="tt-note">${esc(r.p.note)}</div>` : ""}`).join(""), ev.clientX, ev.clientY);
    }
    hit.addEventListener("pointermove", move);
    hit.addEventListener("pointerleave", () => { hl.setAttribute("visibility", "hidden"); hideTip(); });
    host.append(el("div", { class: "chart" }, root));
    // legend
    if (spec.series.length > 1) {
      const lg = el("div", { class: "legend" });
      for (const s of spec.series) {
        const b = el("button", { type: "button", "aria-pressed": String(!hidden.has(s.key)) }, el("i", { style: `background:${s.color}` }), s.label);
        b.addEventListener("click", () => { hidden.has(s.key) ? hidden.delete(s.key) : hidden.add(s.key); lineChart(host, spec); });
        lg.append(b);
      }
      host.append(lg);
    }
    // table view
    const tbl = el("details", { class: "table-toggle" }, el("summary", { class: "small" }, "Show as a table"));
    const t = el("table", { class: "data" });
    t.append(el("thead", {}, el("tr", {}, el("th", {}, ""), ...xs.map((v) => el("th", {}, (spec.xFmt ? spec.xFmt(v) : v) + " " + (spec.xUnit || ""))))));
    const tb = el("tbody");
    for (const s of spec.series) tb.append(el("tr", {}, el("td", {}, s.label), ...xs.map((v) => { const p = s.pts.find((q) => q.x === v); return el("td", {}, p && p.y != null ? fmt(p.y) : "–"); })));
    t.append(tb); tbl.append(el("div", { class: "data-table-wrap" }, t)); host.append(tbl);
  }
  function niceStep(raw) { const p = 10 ** Math.floor(Math.log10(raw)); const f = raw / p; return (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * p; }

  function segmented(host, options, value, onChange, label) {
    const wrap = el("div", { class: "seg", role: "group", "aria-label": label || "" });
    for (const [v, text] of options) {
      const b = el("button", { type: "button", "aria-pressed": String(v === value) }, text);
      b.addEventListener("click", () => { $$("button", wrap).forEach((x) => x.setAttribute("aria-pressed", "false")); b.setAttribute("aria-pressed", "true"); onChange(v); });
      wrap.append(b);
    }
    host.append(label ? el("label", {}, label, wrap) : wrap);
    return wrap;
  }
  const rangeSeries = (recs, key, label, f, opts = {}) => ({
    key, label, short: opts.short, color: colorOf(opts.color || key), dash: opts.dash,
    pts: (opts.ranges || RANGES).map((R) => {
      const s = stat(recs, Object.assign({}, f, { range: R }));
      if (!s) return { x: R, y: null };
      const spread = opts.envBand ? s.sdEnv : s.nRuns > 1 ? s.sdRuns : null;
      return { x: R, y: s.y, lo: spread != null ? s.y - spread : null, hi: spread != null ? s.y + spread : null, err: spread,
        note: `${s.nRuns} run${s.nRuns > 1 ? "s" : ""} × ${s.nEnv} envs${spread != null ? (opts.envBand ? " · ± sd across envs" : " · ± sd across runs") : ""}` };
    }),
  });
  function cvtRef(recs, envSet) {
    const s = stat(recs, { model: "centralized_cvt", env_set: envSet });
    return s ? [{ y: s.y, label: `centralized CVT ${fmt(s.y)}` }] : [];
  }

  // ------------------------------------------------------------------ sections
  function renderAnomaly(C) {
    const host = $("#chart-anomaly"), ctl = $("#anomaly-controls");
    let set = "released11";
    const draw = () => lineChart(host, {
      xs: RANGES, xScale: "log2", xLabel: "test communication range (m)", xUnit: "m", yLabel: "cost / clairvoyant cost",
      series: [rangeSeries(C, "published", "LPAC, published checkpoint", { model: "published", env_set: set, map: "C0", graph: "G0" }, { envBand: true, short: "published" })],
      refs: cvtRef(C, set), yMin: 0,
    });
    segmented(ctl, [["released11", "11 released environments"], ["new40", "40 new environments"]], set, (v) => { set = v; draw(); }, "Environments");
    draw();
  }

  function renderFactorial(F, C) {
    const host = $("#factorial-table");
    const cells = decode(F && F.cells);
    const rows = cells.filter((r) => r.env_set === "released11" && r.model === "published" && r.range_m === 1024 && !r.noise_sigma);
    const maps = ["C0", "C1", "C2", "C3", "C4"], graphs = ["G0", "G1"];
    const names = { C0: "full map", C1: "4 nearest", C2: "128 m scale", C3: "256 m scale", C4: "empty map", G0: "full graph", G1: "4 nearest" };
    const val = (g, c) => { const v = rows.filter((r) => r.graph_level === g && r.map_level === c && (c !== "C1" || r.k == null || r.k === 4)); return v.length ? mean(Array.from(byEnvSeedAvg(v).values())) : null; };
    if (!rows.length) { host.append(el("div", { class: "pending" }, "Factorial data not in the bundle yet.")); return; }
    const t = el("table", { class: "data" });
    t.append(el("thead", {}, el("tr", {}, el("th", {}, "graph ↓ / map →"), ...maps.filter((c) => graphs.some((g) => val(g, c) != null)).map((c) => el("th", {}, `${c} ${names[c]}`)))));
    const tb = el("tbody");
    for (const g of graphs) {
      const vs = maps.filter((c) => graphs.some((gg) => val(gg, c) != null)).map((c) => val(g, c));
      const best = Math.min(...vs.filter((v) => v != null));
      tb.append(el("tr", {}, el("td", {}, `${g} ${names[g]}`), ...vs.map((v) => el("td", { class: v === best ? "best" : null }, fmt(v)))));
    }
    t.append(tb); host.append(t);
    host.append(el("p", { class: "small", style: "margin-top:8px" }, "Cells are means over the 11 released environments of the published model at 1024 m. Bold: best cell."));

    // dose response
    const D = decode(F && F.dose_response).filter((r) => r.range_m === 1024 && (r.graph_level || "G0") === "G0");
    const ks = Array.from(new Set(D.map((r) => r.k))).sort((a, b) => a - b);
    const full = stat(C, { model: "published", env_set: "released11", map: "C0", graph: "G0", range: 1024 });
    const xs = ks.concat(full ? [32] : []);
    const pts = ks.map((k) => { const v = D.filter((r) => r.k === k); const e = Array.from(byEnvSeedAvg(v).values()); return { x: k, y: mean(e), err: sd(e), note: `${v.length ? (mean(v.map((r) => r.map_neighbours_mean)) || 0).toFixed(1) : "–"} robots drawn on average` }; });
    if (full) pts.push({ x: 32, y: full.y, err: full.sdEnv, note: "every robot in range (the published map)" });
    lineChart($("#chart-dose"), {
      xs, xScale: "log2", xLabel: "robots drawn in the map (k nearest; 32 = all in range)", yLabel: "cost / clairvoyant cost",
      xFmt: (v) => (v === 32 ? "all" : v), series: [{ key: "dose", label: "published, 1024 m", color: colorOf("published"), pts }],
      refs: cvtRef(C, "released11"), directLabels: false, height: 300, empty: "Dose-response data not in the bundle yet.",
    });
  }

  function renderScale(C) {
    const scales = [128, 192, 256, 384, 512];
    const lvl = (s) => (s === 256 ? ["S256", "C3"] : ["S" + s]);
    const series = [["released11", "11 released environments", "published"], ["new40", "40 new environments", "fix"]].map(([set, label, col]) => ({
      key: set, label, short: set === "new40" ? "40 new" : "11 released", color: colorOf(col),
      pts: scales.map((s) => { const st = stat(C, { model: "published", env_set: set, range: 1024, map: lvl(s), graph: "G0" }); return { x: s, y: st ? st.y : null, err: st ? st.sdEnv : null, note: st ? `${st.nEnv} envs · ± sd across envs` : "" }; }),
    }));
    lineChart($("#chart-scale"), { xs: scales, xScale: "log2", xLabel: "scale of the neighbor map (m), test range 1024 m", xUnit: "m scale", yLabel: "cost / clairvoyant cost", series, refs: cvtRef(C, "new40"), height: 300 });
  }

  function renderClamp(C, M) {
    const rs = [512, 768, 1024];
    const mk = (key, label, lvl, col) => ({ key, label, color: colorOf(col), pts: rs.map((R) => { const s = stat(C, { model: "published", env_set: "new40", range: R, map: lvl, graph: "G0" }); return { x: R, y: s ? s.y : null, err: s ? s.sdEnv : null, note: s ? `${s.nEnv} envs` : "" }; }) });
    lineChart($("#chart-clamp"), {
      xs: rs, xScale: "log2", xLabel: "test range (m)", xUnit: "m", yLabel: "cost / clairvoyant cost", height: 300,
      series: [Object.assign(mk("drop", "drop far robots (C3)", "C3", "fix"), { short: "drop (C3)" }), Object.assign(mk("clamp", "clamp to border (K256)", "K256", "clamp"), { short: "clamp (K256)" }), Object.assign(mk("asis", "as published (C0)", "C0", "published"), { short: "published (C0)" })],
      refs: cvtRef(C, "new40"),
    });
    const host = $("#clamp-stats");
    const rows = decode(M).filter((r) => r.range_m === 1024);
    if (!rows.length) { host.append(el("div", { class: "pending" }, "Map statistics not in the bundle yet.")); return; }
    const lv = [["C3", "drop (C3)"], ["K256", "clamp (K256)"], ["C0", "as published (C0)"]];
    const metr = [["max_abs_cell", "largest cell value"], ["mass_per_robot", "total mass per map"], ["nonzero_cells_per_robot", "nonzero cells"], ["robots_drawn_per_map", "robots drawn"]];
    const t = el("table", { class: "data" });
    t.append(el("thead", {}, el("tr", {}, el("th", {}, ""), ...lv.map(([, n]) => el("th", {}, n)))));
    const tb = el("tbody");
    for (const [k, n] of metr) tb.append(el("tr", {}, el("td", {}, n), ...lv.map(([l]) => { const v = rows.filter((r) => r.level === l).map((r) => r[k]).filter((x) => x != null); return el("td", {}, v.length ? fmt(mean(v), k === "max_abs_cell" ? 2 : 1) : "–"); })));
    t.append(tb); host.append(t);
  }

  function renderFig20(C) {
    const host = $("#fig20"), ctl = $("#fig20-controls");
    let set = "released11";
    const models = [["lpac128", 128], ["lpac256", 256], ["lpac512", 512], ["lpac1024", 1024]];
    const draw = () => {
      host.innerHTML = "";
      const vals = models.map(([m]) => RANGES.map((R) => { const s = stat(C, { model: m, env_set: set, range: R, map: "C0", graph: "G0" }); return s ? s.y : null; }));
      const flat = vals.flat().filter((v) => v != null);
      if (!flat.length) { host.append(el("div", { class: "pending" }, "Per-range models not in the bundle yet.")); return; }
      const lo = Math.min(...flat), hi = Math.min(Math.max(...flat), 5);
      const t = el("table", { class: "data" });
      t.append(el("thead", {}, el("tr", {}, el("th", {}, "trained at ↓ / tested at →"), ...RANGES.map((R) => el("th", {}, R + " m")))));
      const tb = el("tbody");
      models.forEach(([m, tr], i) => tb.append(el("tr", {}, el("td", {}, `${tr} m`), ...vals[i].map((v, j) => {
        const a = v == null ? 0 : Math.max(0, Math.min(1, (v - lo) / (hi - lo)));
        const td = el("td", { class: RANGES[j] === tr ? "best" : null, style: v == null ? "" : `background: color-mix(in oklab, var(--s1) ${Math.round(8 + a * 62)}%, var(--surface)); color: ${a > 0.55 ? "#fff" : "var(--ink)"}` }, v == null ? "–" : fmt(v));
        return td;
      }))));
      t.append(tb); host.append(t);
      const cvt = stat(C, { model: "centralized_cvt", env_set: set });
      host.append(el("p", { class: "small", style: "margin-top:8px" }, `Bold: the diagonal (each model at its own range). 128 m row: mean of three seeds; other rows: one seed each. Centralized CVT on this set: ${cvt ? fmt(cvt.y) : "–"}.`));
    };
    segmented(ctl, [["released11", "11 released"], ["new40", "40 new"]], set, (v) => { set = v; draw(); }, "Environments");
    draw();
  }

  const PRESETS = {
    story: { label: "The story", keys: ["published", "fix", "A_mixed", "V4mixed"] },
    maps: { label: "Map-based", keys: ["published", "lpac256", "v0mixed", "A_mixed", "V5mixed", "V6mixed"] },
    edges: { label: "Edge geometry", keys: ["V1", "V2", "V3", "V4_128", "V4mixed", "V2M"] },
    final: { label: "Final runs", keys: ["published", "lpac256", "V4mixed", "V4half", "V2M"] },
    all: { label: "Everything", keys: null },
  };
  function variantSeries(C, set) {
    const S = [
      ["published", "LPAC published (128 m)", { model: "published", map: "C0" }],
      ["fix", "published + fixed 256 m map", { model: "published", map: "C3" }],
      ["lpac256", "LPAC retrained at 256 m", { model: "lpac256", map: "C0" }],
      ["v0mixed", "LPAC, mixed ranges", { model: "v0mixed", map: "C0" }],
      ["A_mixed", "A: fixed map, mixed", { model: "A_mixed", map: ["C3", "C0"] }],
      ["V1", "V1: no map", { model: "V1", map: ["none", "C0"] }],
      ["V2", "V2: GATv2", { model: "V2", map: ["none", "C0"] }],
      ["V3", "V3: PointNet edges, 128 m", { model: "V3", map: ["none", "C0"] }],
      ["V4_128", "V4: bounded edges, 128 m", { model: "V4_128", map: ["none", "C0"] }],
      ["V4mixed", "V4: bounded edges, mixed", { model: "V4mixed", map: ["none", "C0"] }],
      ["V4half", "V4: 128 m + 1024 m only", { model: "V4half", map: ["none", "C0"] }],
      ["V2M", "V2M: GATv2-style, max aggregation", { model: "V2M", map: ["none", "C0"] }],
      ["V5mixed", "V5: multi-scale maps, mixed", { model: "V5mixed", map: ["multiscale"] }],
      ["V6mixed", "V6: V5 + V4, mixed", { model: "V6mixed", map: ["multiscale"] }],
    ];
    return S.map(([k, label, f]) => rangeSeries(C, k, label, Object.assign({ env_set: set, graph: "G0" }, f), { short: k === "fix" ? "fixed map" : k === "published" ? "published" : label.split(":")[0] }));
  }
  function renderVariants(C) {
    const host = $("#chart-variants"), ctl = $("#variants-controls");
    let set = "new40", preset = "story";
    const draw = () => {
      const series = variantSeries(C, set);
      const keys = PRESETS[preset].keys;
      host._hidden = new Set(keys ? series.filter((s) => !keys.includes(s.key)).map((s) => s.key) : []);
      lineChart(host, { xs: RANGES, xScale: "log2", xLabel: "test communication range (m)", xUnit: "m", yLabel: "cost / clairvoyant cost", series, refs: cvtRef(C, set), height: 400 });
      renderPaired(C, set);
    };
    segmented(ctl, Object.entries(PRESETS).map(([k, v]) => [k, v.label]), preset, (v) => { preset = v; draw(); }, "Preset");
    segmented(ctl, [["new40", "40 new envs"], ["released11", "11 released"]], set, (v) => { set = v; draw(); }, "Environments");
    draw();
  }
  function renderPaired(C, set) {
    const host = $("#paired-box"); host.innerHTML = "";
    const f = (model, map, R) => { const s = stat(C, { model, map, env_set: set, range: R, graph: "G0" }); return s ? s.env : null; };
    const rows = [
      ["V4 mixed: cost at 1024 m − cost at 256 m", f("V4mixed", ["none", "C0"], 1024), f("V4mixed", ["none", "C0"], 256)],
      ["A (fixed map, mixed): 1024 m − 256 m", f("A_mixed", ["C3", "C0"], 1024), f("A_mixed", ["C3", "C0"], 256)],
      ["V5 mixed: 1024 m − 256 m", f("V5mixed", ["multiscale"], 1024), f("V5mixed", ["multiscale"], 256)],
      ["V4 mixed − LPAC trained at 128 m, both at 256 m", f("V4mixed", ["none", "C0"], 256), f("lpac128", "C0", 256)],
      ["V4 mixed − LPAC retrained at 256 m, both at 256 m", f("V4mixed", ["none", "C0"], 256), f("lpac256", "C0", 256)],
      ["V4 mixed − LPAC retrained at 256 m, both at 1024 m", f("V4mixed", ["none", "C0"], 1024), f("lpac256", "C0", 1024)],
      ["V4 at 128 m − V3, at 1024 m", f("V4_128", ["none", "C0"], 1024), f("V3", ["none", "C0"], 1024)],
    ];
    const t = el("table", { class: "data" });
    t.append(el("thead", {}, el("tr", {}, el("th", {}, `Paired by environment (${set === "new40" ? "40 new" : "11 released"}), seed-averaged`), el("th", {}, "mean difference"), el("th", {}, "SE"), el("th", {}, "envs where first is lower"))));
    const tb = el("tbody");
    for (const [name, a, b] of rows) {
      const p = a && b ? paired(a, b) : null;
      tb.append(el("tr", {}, el("td", {}, name), el("td", {}, p ? fmtS(p.d) : "–"), el("td", {}, p ? fmt(p.se, 3) : "–"), el("td", {}, p ? `${p.better} / ${p.n}` : "–")));
    }
    t.append(tb);
    host.append(el("figure", {}, el("div", { class: "panel" }, el("div", { class: "data-table-wrap" }, t)), el("figcaption", {}, "Computed live from the per-environment costs. A negative difference means the first quantity is lower (better).")));
  }

  function renderModels(Mo) {
    const host = $("#model-cards"); host.innerHTML = "";
    if (Mo && Mo.models) Mo = Mo.models;
    if (!Mo) { host.append(el("div", { class: "pending" }, "Model catalogue not in the bundle yet.")); return; }
    const order = ["V0", "V1", "V2", "V3", "V4", "V5", "V6", "V0F", "V0R", "V2R", "V2RL", "V2B", "V2M"];
    const keys = Object.keys(Mo).sort((a, b) => (order.indexOf(a) + 99) % 99 - (order.indexOf(b) + 99) % 99);
    const colorKey = { V0: "published", V1: "V1", V2: "V2", V3: "V3", V4: "V4mixed", V5: "V5mixed", V6: "V6mixed", V0F: "A_mixed", V0R: "V0R", V2R: "V2res", V2M: "V2M" };
    for (const k of keys) {
      const m = Mo[k];
      const dl = el("dl");
      const add = (dt, dd) => { if (dd != null && dd !== "") { dl.append(el("dt", {}, dt), el("dd", {}, String(dd))); } };
      add("parameters", m.parameters != null ? Number(m.parameters).toLocaleString("en-US") : null);
      add("map input", m.map_input); add("edges", m.edge_features); add("aggregation", m.aggregation);
      add("reach", m.receptive_field_hops != null ? m.receptive_field_hops + " hops" : null);
      add("trained on", (m.training_sets || []).join(", ")); add("epochs", m.epochs);
      host.append(el("div", { class: "model-card" },
        el("span", { class: "tag" }, el("i", { style: `background:${colorOf(colorKey[k] || "V1")}` }), k),
        el("h4", {}, m.label || k), el("p", {}, m.description || ""), dl));
    }
  }

  function renderHetero(Hraw) {
    const host = $("#chart-hetero"), ctl = $("#hetero-controls");
    const H = decode(Hraw);
    if (!H.length) { host.append(el("div", { class: "pending" }, "Heterogeneous-team data not in the bundle yet.")); return; }
    const FS = [0, 0.25, 0.5, 0.75, 1];
    let sem = "reception";
    const S = [
      ["published", "published, own map", { model: "published", map: "asis" }, "published"],
      ["fix", "published, fixed 256 m map", { model: "published", map: "fixed256" }, "fix"],
      ["V4mixed", "V4 mixed (6 seeds)", { model: "V4mixed" }, "V4mixed"],
      ["V4T", "V4 trained on transmission teams (2 seeds)", { model: "V4T" }, "V4T"],
      ["V4half", "V4 trained on 128 m + 1024 m", { model: "V4half" }, "V4half"],
      ["V0R", "V0R: own range as input", { model: "V0R" }, "V0R"],
      ["mixture", "mixture of specialists", { model: "mixture128+1024", mode: "mixture" }, "mixture"],
    ];
    const draw = () => {
      const series = S.map(([key, label, f, col]) => ({
        key, label, color: colorOf(col),
        pts: FS.map((fr) => {
          const rows = H.filter((r) => r.model === f.model && (!f.map || r.map === f.map) && (!f.mode || r.mode === f.mode) && (r.semantics || "reception") === sem && Math.abs(r.f - fr) < 1e-9);
          if (!rows.length) return { x: fr, y: null };
          const byRun = new Map(); for (const r of rows) { const k = r.run_id; if (!byRun.has(k)) byRun.set(k, []); byRun.get(k).push(r); }
          const runMeans = Array.from(byRun.values()).map((rs) => mean(Array.from(byEnvSeedAvg(rs).values())));
          const y = mean(runMeans), e = runMeans.length > 1 ? sd(runMeans) : null;
          return { x: fr, y, lo: e != null ? y - e : null, hi: e != null ? y + e : null, err: e, note: `${byRun.size} run(s) · ${rows.length} episodes` };
        }),
      }));
      lineChart(host, { xs: FS, xScale: "lin", xLabel: "fraction of robots with a 1024 m radio (others 128 m)", xFmt: (v) => ({ 0: "0", 0.25: "¼", 0.5: "½", 0.75: "¾", 1: "1" }[v]), yLabel: "cost / clairvoyant cost", series, height: 380, empty: "No runs with this link rule for these models." });
    };
    segmented(ctl, [["reception", "reception d < r_i"], ["transmission", "transmission d < r_j"], ["sym_max", "either (OR)"], ["sym_min", "both (AND)"]], sem, (v) => { sem = v; draw(); }, "Link rule");
    draw();

    // rules table: V4mixed paired against reception
    const host2 = $("#rules-table");
    const key = (r) => `${r.run_id}|${r.env}|${r.draw}|${r.f}`;
    const V = H.filter((r) => r.model === "V4mixed");
    const base = new Map(V.filter((r) => r.semantics === "reception").map((r) => [key(r), r.cost_ratio]));
    const t = el("table", { class: "data" });
    t.append(el("thead", {}, el("tr", {}, el("th", {}, "link rule"), ...[0.25, 0.5, 0.75].map((x) => el("th", {}, "f = " + ({ 0.25: "¼", 0.5: "½", 0.75: "¾" }[x]))))));
    const tb = el("tbody");
    for (const [s, name] of [["transmission", "transmission d < r_j"], ["sym_max", "either reaches (OR)"], ["sym_min", "both reach (AND)"]]) {
      tb.append(el("tr", {}, el("td", {}, name), ...[0.25, 0.5, 0.75].map((fr) => {
        const envD = new Map();
        for (const r of V.filter((q) => q.semantics === s && Math.abs(q.f - fr) < 1e-9)) {
          const b = base.get(key(Object.assign({}, r))); if (b == null) continue;
          if (!envD.has(r.env)) envD.set(r.env, []); envD.get(r.env).push(r.cost_ratio - b);
        }
        const d = Array.from(envD.values()).map(mean);
        if (!d.length) return el("td", {}, "–");
        return el("td", {}, `${fmtS(mean(d), 2)} (SE ${fmt(sd(d) / Math.sqrt(d.length), 2)}, ${d.filter((x) => x < 0).length}/${d.length})`);
      })));
    }
    t.append(tb); host2.append(t);
  }

  function renderCurves(Cu) {
    const host = $("#chart-curves"), ctl = $("#curves-controls");
    if (!Cu || !Cu.runs) { host.append(el("div", { class: "pending" }, "Training curves not in the bundle yet.")); return; }
    let which = "train";
    const ids = [["V2", "V2: GATv2 as configured", "V2"], ["V2lr", "V2, learning rate ×10", "fix"], ["V2B", "V2B: bounded edges only", "published"],["V2res", "V2res: residual + LayerNorm + bounded edges", "V3"], ["V2deep", "V2res, 15 blocks", "V4mixed"], ["V2logdeg", "V2res + node degree", "A_mixed"], ["V1", "V1: TAGConv, K = 3", "V1"], ["V1K1", "V1: TAGConv, K = 1", "lpac256"], ["V2M", "V2M: V2res with max aggregation", "V2M"]];
    const draw = () => {
      const series = ids.filter(([id]) => Cu.runs[id]).map(([id, label, col]) => {
        const r = Cu.runs[id];
        return { key: id, label, color: colorOf(col), dash: id.startsWith("V1") ? "5 4" : null, pts: r.epoch.map((e, i) => ({ x: e, y: r[which][i] })).filter((p) => p.x <= 30) };
      });
      const xs = Array.from({ length: 30 }, (_, i) => i + 1);
      lineChart(host, { xs, xScale: "lin", xLabel: "epoch", xUnit: "", xFmt: (v) => (v % 5 === 0 || v === 1 ? v : ""), yLabel: which === "train" ? "training MSE" : "validation MSE", series, height: 380, directLabels: false });
    };
    segmented(ctl, [["train", "training loss"], ["val", "validation loss"]], which, (v) => { which = v; draw(); }, "Show");
    draw();
  }

  function renderRuns(C) {
    const host = $("#chart-runs");
    const rows = C.filter((r) => r.model === "V5mixed" && r.range_m === 256 && (r.graph_level || "G0") === "G0");
    if (!rows.length) { host.append(el("div", { class: "pending" }, "Per-run data not in the bundle yet.")); return; }
    const runs = Array.from(new Set(rows.map((r) => r.run_id))).sort();
    const sets = [["released11", "11 released environments", "fix"], ["new40", "40 new environments", "V4mixed"]];
    const series = sets.map(([set, label, col]) => ({ key: set, label, color: colorOf(col), pts: runs.map((id, i) => { const v = rows.filter((r) => r.run_id === id && r.env_set === set); return { x: i + 1, y: v.length ? mean(v.map((r) => r.cost_ratio)) : null, note: id }; }) }));
    lineChart(host, { xs: runs.map((_, i) => i + 1), xScale: "lin", xLabel: "training run of V5 mixed (two launches of seeds 0, 1, 2)", xUnit: "", xFmt: (v) => "run " + v, yLabel: "cost at 256 m", series, height: 280, directLabels: false });
  }

  // ------------------------------------------------------------------ replays
  function idfImage(env) {
    const n = env.idf.size, data = env.idf.data;
    let mx = 0; for (const v of data) if (v > mx) mx = v;
    const cv = document.createElement("canvas"); cv.width = n; cv.height = n;
    const ctx = cv.getContext("2d"), img = ctx.createImageData(n, n);
    const dark = matchMedia("(prefers-color-scheme: dark)").matches && document.documentElement.dataset.theme !== "light" || document.documentElement.dataset.theme === "dark";
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const v = Math.sqrt(data[j * n + i] / (mx || 1)); // row j = y index
      const o = ((n - 1 - j) * n + i) * 4; // flip so y grows upward
      if (dark) { img.data[o] = 22 + v * 150; img.data[o + 1] = 27 + v * 110; img.data[o + 2] = 26 + v * 60; }
      else { img.data[o] = 252 - v * 120; img.data[o + 1] = 252 - v * 150; img.data[o + 2] = 251 - v * 200; }
      img.data[o + 3] = 255;
    }
    ctx.putImageData(img, 0, 0); return cv;
  }
  function drawSwarm(canvas, env, run, fi, bg) {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const size = canvas.clientWidth || 400;
    if (canvas.width !== Math.round(size * dpr)) { canvas.width = Math.round(size * dpr); canvas.height = Math.round(size * dpr); }
    const ctx = canvas.getContext("2d"), S = canvas.width, W = env.world_size_m;
    ctx.imageSmoothingEnabled = true; ctx.drawImage(bg, 0, 0, S, S);
    const P = run.positions[fi], n = P.length / 2;
    const px = (i) => (P[2 * i] / W) * S, py = (i) => S - (P[2 * i + 1] / W) * S;
    const ranges = run.robot_ranges || Array(n).fill(run.range_m);
    ctx.lineWidth = 1 * dpr; ctx.strokeStyle = css("--graph"); ctx.globalAlpha = 0.28;
    ctx.beginPath();
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
      const d = Math.hypot(P[2 * i] - P[2 * j], P[2 * i + 1] - P[2 * j + 1]);
      if (d < Math.max(ranges[i], ranges[j])) { ctx.moveTo(px(i), py(i)); ctx.lineTo(px(j), py(j)); }
    }
    ctx.stroke(); ctx.globalAlpha = 1;
    const foot = (64 / W) * S;
    for (let i = 0; i < n; i++) {
      ctx.strokeStyle = css("--ink"); ctx.globalAlpha = 0.35; ctx.lineWidth = 1 * dpr;
      ctx.strokeRect(px(i) - foot / 2, py(i) - foot / 2, foot, foot); ctx.globalAlpha = 1;
      ctx.fillStyle = ranges[i] > 512 ? css("--map") : css("--ink");
      ctx.beginPath(); ctx.arc(px(i), py(i), 3.6 * dpr, 0, 7); ctx.fill();
      ctx.strokeStyle = css("--surface"); ctx.lineWidth = 1.5 * dpr; ctx.stroke();
    }
  }
  function sparkline(canvas, run, fi, frames) {
    const dpr = Math.min(2, window.devicePixelRatio || 1), w = canvas.clientWidth || 300, h = 54;
    canvas.width = w * dpr; canvas.height = h * dpr;
    const ctx = canvas.getContext("2d"), c = run.cost_ratio, mx = Math.max(6, ...c), mn = 0;
    const X = (i) => (i / (frames - 1)) * canvas.width, Y = (v) => canvas.height - ((v - mn) / (mx - mn)) * (canvas.height - 6 * dpr) - 3 * dpr;
    ctx.strokeStyle = css("--rule"); ctx.lineWidth = 1 * dpr; ctx.beginPath(); ctx.moveTo(0, Y(1)); ctx.lineTo(canvas.width, Y(1)); ctx.stroke();
    ctx.strokeStyle = css("--ink-2"); ctx.lineWidth = 1.6 * dpr; ctx.beginPath(); c.forEach((v, i) => (i ? ctx.lineTo(X(i), Y(v)) : ctx.moveTo(X(i), Y(v)))); ctx.stroke();
    ctx.fillStyle = css("--graph"); ctx.beginPath(); ctx.arc(X(fi), Y(c[fi]), 3.5 * dpr, 0, 7); ctx.fill();
  }
  function renderReplays(R) {
    const host = $("#hero-replay");
    if (!R || !R.runs || !R.runs.length) { host.innerHTML = ""; host.append(el("div", { class: "pending", style: "grid-column:1/-1" }, "The episode replays are being generated; this panel fills in once they are in the data bundle.")); $(".replay-bar").hidden = true;
      const mv = $("#mapviz"); mv.innerHTML = ""; mv.append(el("div", { class: "pending", style: "grid-column:1/-1" }, "The map tensors come with the episode replays, which are being generated.")); return; }
    const find = (pred) => R.runs.find(pred);
    const left = find((r) => r.model === "published" && r.range_m === 1024 && (r.map_level === "C0" || !r.map_level) && !r.f);
    const right = find((r) => r.model === "published" && r.range_m === 1024 && r.map_level === "C3" && !r.f) || find((r) => r.model === "V4mixed" && r.range_m === 1024 && !r.f);
    const pair = [left, right].filter(Boolean);
    host.innerHTML = "";
    const frames = R.frame_steps.length;
    let bg = idfImage(R.env);
    const panels = pair.map((run) => {
      const cv = el("canvas", { role: "img", "aria-label": run.label });
      const sp = el("canvas", { class: "spark", "aria-hidden": "true" });
      const meta = el("div", { class: "meta" }, el("span", {}, run.label), el("span", {}, "mean ", el("b", {}, fmt(run.mean_cost_ratio))));
      host.append(el("div", {}, cv, meta, sp));
      return { run, cv, sp };
    });
    let fi = 0, playing = false, last = 0;
    const slider = $("#hero-step"), lab = $("#hero-step-label"), btn = $("#hero-play");
    slider.max = frames - 1;
    const paint = () => { panels.forEach((p) => { drawSwarm(p.cv, R.env, p.run, fi, bg); sparkline(p.sp, p.run, fi, frames); }); slider.value = fi; lab.textContent = `step ${R.frame_steps[fi]} / 600`; };
    const tick = (t) => { if (!playing) return; if (t - last > 70) { fi = (fi + 1) % frames; paint(); last = t; } requestAnimationFrame(tick); };
    btn.addEventListener("click", () => { playing = !playing; btn.textContent = playing ? "❚❚ Pause" : "▶ Play"; if (playing) requestAnimationFrame(tick); });
    slider.addEventListener("input", () => { fi = +slider.value; paint(); });
    window.addEventListener("resize", paint);
    window.addEventListener("themechange", () => { bg = idfImage(R.env); paint(); });
    fi = Math.min(frames - 1, 40); paint();
    if (!matchMedia("(prefers-reduced-motion: reduce)").matches) { playing = true; btn.textContent = "❚❚ Pause"; requestAnimationFrame(tick); }
    renderMapViz(R);
  }

  function renderMapViz(R) {
    const host = $("#mapviz"), ctl = $("#mapviz-controls");
    const run = R.runs.find((r) => r.maps && r.maps.C0 && r.maps.C0.length && r.range_m === 1024) || R.runs.find((r) => r.maps && r.maps.C0);
    if (!run) { host.innerHTML = ""; host.append(el("div", { class: "pending", style: "grid-column:1/-1" }, "Map tensors not in the bundle yet.")); return; }
    const mp = run.maps; let k = 0;
    const draw = () => {
      host.innerHTML = "";
      for (const [lvl, title] of [["C0", `C0, as published: scale ${run.range_m} m, ${(2 * run.range_m / 32).toFixed(0)} m per cell`], ["C3", "C3, fixed 256 m scale, far robots dropped"], ["K256", "K256, far robots clamped to the 256 m circle"]]) {
        const arr = mp[lvl] && mp[lvl][k]; if (!arr) continue;
        const cv = el("canvas", { width: 32, height: 32, role: "img", "aria-label": title });
        const ctx = cv.getContext("2d"), img = ctx.createImageData(32, 32);
        const mag = new Float32Array(1024); let mx = 0, nz = 0, maxv = 0;
        for (let c = 0; c < 1024; c++) { const v = Math.hypot(arr[c], arr[1024 + c]); mag[c] = v; if (v > 0) nz++; if (v > maxv) maxv = v; }
        mx = Math.max(maxv, 1.1);
        const hue = lvl === "C0" ? [194, 59, 59] : lvl === "C3" ? [168, 90, 6] : [180, 80, 160];
        for (let c = 0; c < 1024; c++) {
          const a = Math.min(1, mag[c] / mx); const yy = Math.floor(c / 32), xx = c % 32; const o = ((31 - yy) * 32 + xx) * 4;
          img.data[o] = 250 - a * (250 - hue[0]); img.data[o + 1] = 250 - a * (250 - hue[1]); img.data[o + 2] = 250 - a * (250 - hue[2]); img.data[o + 3] = 255;
        }
        img.data[((31 - 15) * 32 + 15) * 4 + 3] = 255;
        ctx.putImageData(img, 0, 0);
        host.append(el("div", {}, cv, el("div", { class: "cap" }, title, el("br"), "largest cell ", el("b", {}, fmt(maxv)), " · nonzero cells ", el("b", {}, String(nz)))));
      }
    };
    if (mp.steps && mp.steps.length > 1) segmented(ctl, mp.steps.map((s, i) => [i, "step " + s]), 0, (v) => { k = v; draw(); }, `Robot ${mp.robot}, run “${run.label}”`);
    draw();
  }

  // ------------------------------------------------------------------ boot
  (async function boot() {
    const [costs, hetero, factorial, curves, maps, models, replays] = await Promise.all(
      ["costs.json", "hetero.json", "factorial.json", "curves.json", "maps.json", "models.json", "replays.json"].map(getJSON));
    const C = decode(costs);
    if (C.length) {
      const n = C.length + (hetero ? hetero.rows.length : 0) + (factorial && factorial.cells ? factorial.cells.rows.length : 0);
      $("#fact-evals").textContent = n.toLocaleString("en-US");
    }
    const safe = (fn, ...a) => { try { fn(...a); } catch (e) { console.error(fn.name, e); } };
    safe(renderAnomaly, C); safe(renderFactorial, factorial, C); safe(renderScale, C); safe(renderClamp, C, maps);
    safe(renderFig20, C); safe(renderModels, models); safe(renderVariants, C); safe(renderHetero, hetero);
    safe(renderCurves, curves); safe(renderRuns, C); safe(renderReplays, replays);
  })();
})();
