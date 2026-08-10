/* omigamax web dashboard — vanilla JS client (no build step) */
"use strict";

const REFRESH_MS = 2500;
const VISIBLE_GAMES = 6;       // thumbnails shown in the grid
const COORD = "ABCDEFGHJKLMNOPQRST"; // standard go coordinates, I omitted

// ---------------------------------------------------------------------------
// state
// ---------------------------------------------------------------------------
const state = {
  train: null,
  games: [],                 // metadata list from /api/games
  gameCache: {},             // id -> {meta, replay} (full replay payload)
  open: null,                // id of currently-open game
  pos: 0,                    // replay position index
  playing: false,
  playTimer: null,
};

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
function $(sel) { return document.querySelector(sel); }

async function fetchJSON(url) {
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(url + " -> " + res.status);
  return res.json();
}

function fmt(x, digits = 4) {
  if (x === null || x === undefined || !isFinite(x)) return "—";
  return Number(x).toFixed(digits);
}

function lastUpdateStr(ts) {
  if (!ts) return "";
  const t = new Date(ts);
  return t.toLocaleTimeString("zh-CN", { hour12: false });
}

// ---------------------------------------------------------------------------
// canvas drawing (goban)
// ---------------------------------------------------------------------------
function stoneColorCode(grid, r, c) {
  return grid && grid[r] ? grid[r][c] : 0;
}

function drawGoban(canvas, grid, opts) {
  opts = opts || {};
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const cssW = canvas.clientWidth || opts.width || 120;
  const cssH = canvas.clientHeight || opts.height || 120;
  canvas.width = cssW * dpr;
  canvas.height = cssH * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const n = grid.length;
  const pad = Math.max(6, cssW * 0.055);
  const cell = (cssW - pad * 2) / (n - 1);

  // wood background
  const bg = ctx.createLinearGradient(0, 0, cssW, cssH);
  bg.addColorStop(0, "#8a5c30");
  bg.addColorStop(0.5, "#a06a38");
  bg.addColorStop(1, "#7a4f28");
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, cssW, cssH);

  // subtle wood grain
  ctx.strokeStyle = "rgba(60,35,10,0.14)";
  ctx.lineWidth = 1;
  for (let i = -2; i < cssH / 8; i++) {
    ctx.beginPath();
    const y = i * 8 + Math.sin(i) * 3;
    ctx.moveTo(0, y);
    ctx.bezierCurveTo(cssW * 0.3, y + 2, cssW * 0.7, y - 2, cssW, y + 1);
    ctx.stroke();
  }

  // grid lines
  ctx.strokeStyle = "rgba(30,18,8,0.85)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const p = pad + i * cell;
    ctx.moveTo(pad, p); ctx.lineTo(cssW - pad, p);
    ctx.moveTo(p, pad); ctx.lineTo(p, cssW - pad);
  }
  ctx.stroke();

  // star points
  const stars = starPoints(n);
  ctx.fillStyle = "rgba(25,14,6,0.9)";
  for (const [sr, sc] of stars) {
    ctx.beginPath();
    ctx.arc(pad + sc * cell, pad + sr * cell, Math.max(2, cell * 0.13), 0, Math.PI * 2);
    ctx.fill();
  }

  // coordinates (only when big enough)
  if (cell >= 24) {
    ctx.fillStyle = "rgba(30,18,8,0.7)";
    ctx.font = `${Math.max(9, cell * 0.34)}px sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    for (let i = 0; i < n; i++) {
      ctx.fillText(String(n - i), pad + i * cell, pad - cell * 0.45);
      ctx.fillText(COORD[i], pad - cell * 0.5, pad + i * cell);
    }
  }

  // stones
  const r = cell * 0.46;
  const last = opts.lastMove; // {r, c}
  for (let rIdx = 0; rIdx < n; rIdx++) {
    for (let cIdx = 0; cIdx < n; cIdx++) {
      const code = stoneColorCode(grid, rIdx, cIdx);
      if (code === 0) continue;
      const x = pad + cIdx * cell;
      const y = pad + rIdx * cell;
      drawStone(ctx, x, y, r, code === 1 ? "black" : "white");
    }
  }

  // last-move marker
  if (last) {
    const x = pad + last.c * cell;
    const y = pad + last.r * cell;
    ctx.beginPath();
    ctx.arc(x, y, r * 0.28, 0, Math.PI * 2);
    ctx.fillStyle = last.color === "black" ? "rgba(255,80,60,0.9)" : "rgba(220,60,40,0.9)";
    ctx.fill();
  }
}

function drawStone(ctx, x, y, r, color) {
  // soft shadow
  ctx.beginPath();
  ctx.arc(x + r * 0.08, y + r * 0.1, r, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(0,0,0,0.28)";
  ctx.fill();
  // body
  const g = ctx.createRadialGradient(x - r * 0.35, y - r * 0.35, r * 0.15, x, y, r);
  if (color === "black") {
    g.addColorStop(0, "#3d3d40");
    g.addColorStop(0.5, "#222224");
    g.addColorStop(1, "#0c0c0e");
  } else {
    g.addColorStop(0, "#ffffff");
    g.addColorStop(0.6, "#f2ece0");
    g.addColorStop(1, "#c9bfae");
  }
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fillStyle = g;
  ctx.fill();
  // rim
  ctx.strokeStyle = color === "black" ? "rgba(0,0,0,0.8)" : "rgba(140,120,90,0.5)";
  ctx.lineWidth = 1;
  ctx.stroke();
}

function starPoints(n) {
  if (n < 9) return [];
  if (n === 9) return [[2, 2], [2, 6], [6, 2], [6, 6], [4, 4]];
  if (n === 13) return [[3, 3], [3, 9], [9, 3], [9, 9], [6, 6]];
  const m = n - 4; // 19 -> 15
  const c = Math.floor(n / 2); // 19 -> 9
  return [[3, 3], [3, c], [3, m], [c, 3], [c, c], [c, m], [m, 3], [m, c], [m, m]];
}

// ---------------------------------------------------------------------------
// charts
// ---------------------------------------------------------------------------
function drawLineChart(canvas, points, opts) {
  opts = opts || {};
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const cssW = canvas.clientWidth || 380;
  const cssH = canvas.clientHeight || 110;
  canvas.width = cssW * dpr;
  canvas.height = cssH * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);

  const color = opts.color || "#d8a24c";
  const pad = { l: 6, r: 6, t: 8, b: 6 };
  const w = cssW - pad.l - pad.r;
  const h = cssH - pad.t - pad.b;

  if (!points || points.length < 2) {
    ctx.fillStyle = "rgba(168,152,127,0.5)";
    ctx.font = "11px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("暂无数据", cssW / 2, cssH / 2);
    return;
  }

  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  let minY = Math.min(...ys), maxY = Math.max(...ys);
  if (maxY === minY) { minY -= 1; maxY += 1; }
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const X = (x) => pad.l + (maxX === minX ? 0 : ((x - minX) / (maxX - minX)) * w);
  const Y = (y) => pad.t + (1 - (y - minY) / (maxY - minY)) * h;

  // grid
  ctx.strokeStyle = "rgba(255,255,255,0.06)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 3; i++) {
    const gy = pad.t + (h / 3) * i;
    ctx.beginPath(); ctx.moveTo(pad.l, gy); ctx.lineTo(cssW - pad.r, gy); ctx.stroke();
  }

  // line
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  points.forEach((p, i) => {
    if (i === 0) ctx.moveTo(X(p.x), Y(p.y));
    else ctx.lineTo(X(p.x), Y(p.y));
  });
  ctx.stroke();

  // area fill
  ctx.lineTo(X(xs[xs.length - 1]), pad.t + h);
  ctx.lineTo(X(xs[0]), pad.t + h);
  ctx.closePath();
  ctx.fillStyle = color + "1a";
  ctx.fill();

  // last value label
  ctx.fillStyle = color;
  ctx.font = "10px monospace";
  ctx.textAlign = "left";
  const lp = points[points.length - 1];
  ctx.fillText(fmt(lp.y, opts.digits === undefined ? 4 : opts.digits), X(lp.x) + 4, Y(lp.y) - 3);
}

// ---------------------------------------------------------------------------
// metrics + charts
// ---------------------------------------------------------------------------
function renderTrain(data) {
  state.train = data;
  const latest = data.latest || {};
  $("#m-step").textContent = latest.step ?? "—";
  $("#m-loss").textContent = fmt(latest.loss, 4);
  $("#m-lr").textContent = fmt(latest.lr, 6);
  $("#m-elo").textContent = latest.elo === null || latest.elo === undefined ? "—" : fmt(latest.elo, 1);
  $("#m-games").textContent = latest.games ?? "—";
  $("#m-cycle").textContent = latest.cycle ?? "—";

  // loss series (downsample for chart)
  const steps = data.steps || [];
  drawLineChart($("#chart-loss"), downsample(steps, 600, "loss", "step"), { color: "#e8a65c", digits: 4 });
  drawLineChart($("#chart-lr"), downsample(steps, 300, "lr", "step"), { color: "#7bc4c9", digits: 5 });

  const evals = (data.evals || []).map((e) => ({ x: e.step, y: e.elo }));
  drawLineChart($("#chart-elo"), evals, { color: "#8fd18a", digits: 1 });

  const ev = data.events || 0;
  const mtime = data.file_mtime ? lastUpdateStr(data.file_mtime) : "无日志";
  $("#log-meta").innerHTML =
    `train.jsonl: ${ev} 条事件<br>日志更新: ${mtime}<br>` +
    (latest.step !== undefined ? `评估: ${latest.winrate === null || latest.winrate === undefined ? "—" : fmt(latest.winrate, 3)} · 替换: ${latest.replaced ? "是" : "否"}` : "");

  // liveness
  const alive = !!data.alive;
  $("#live-dot").classList.toggle("on", alive);
  $("#live-text").textContent = alive ? "训练运行中" : "训练已停止";
  $("#last-update").textContent = data.now ? "更新 " + lastUpdateStr(data.now) : "";
}

function downsample(points, max, yKey, xKey) {
  if (!points || points.length <= max) return points.map((p) => ({ x: p[xKey], y: p[yKey] }));
  const out = [];
  const stride = Math.ceil(points.length / max);
  for (let i = 0; i < points.length; i += stride) out.push({ x: points[i][xKey], y: points[i][yKey] });
  const last = points[points.length - 1];
  out.push({ x: last[xKey], y: last[yKey] });
  return out;
}

// ---------------------------------------------------------------------------
// games grid
// ---------------------------------------------------------------------------
async function loadGames() {
  const data = await fetchJSON("/api/games");
  state.games = data.games || [];
  $("#game-count").textContent = `(${state.games.length})`;
  $("#games-empty").classList.toggle("hidden", state.games.length > 0);

  // find changed/new games among the visible window
  const window = state.games.slice(0, VISIBLE_GAMES);
  const needFetch = [];
  for (const g of window) {
    const cached = state.gameCache[g.id];
    if (!cached || cached.meta.mtime_raw !== g.mtime_raw) needFetch.push(g);
  }
  // also keep the open game fresh
  if (state.open) {
    const og = state.games.find((g) => g.id === state.open);
    const cached = state.gameCache[state.open];
    if (og && (!cached || cached.meta.mtime_raw !== og.mtime_raw)) needFetch.push(og);
  }

  await Promise.all(needFetch.map(async (g) => {
    try {
      const replay = await fetchJSON("/api/games/" + g.id);
      state.gameCache[g.id] = { meta: g, replay };
    } catch (e) { /* keep stale cache */ }
  }));

  renderGrid(window);
  if (state.open && state.gameCache[state.open]) renderReplay();
}

function renderGrid(windowGames) {
  const grid = $("#games-grid");
  grid.innerHTML = "";
  for (const g of windowGames) {
    const cached = state.gameCache[g.id];
    const card = document.createElement("div");
    card.className = "game-card";
    card.title = g.id + " · " + (g.move_count ?? "?") + " 手";

    const cv = document.createElement("canvas");
    cv.style.aspectRatio = "1/1";
    const finalGrid = cached ? cached.replay.positions[cached.replay.positions.length - 1] : null;
    if (finalGrid) drawGoban(cv, finalGrid, { width: 150, height: 150 });
    else drawGoban(cv, makeEmpty(19), { width: 150, height: 150 });

    const name = document.createElement("div");
    name.className = "gc-name";
    name.textContent = g.id;

    const meta = document.createElement("div");
    meta.className = "gc-meta";
    meta.textContent = `${g.winner || "?"} 胜 · ${g.move_count ?? "?"} 手 · ${g.result || ""}`;

    card.appendChild(cv);
    card.appendChild(name);
    card.appendChild(meta);
    card.addEventListener("click", () => openGame(g.id));
    grid.appendChild(card);
  }
}

function makeEmpty(n) {
  return Array.from({ length: n }, () => Array(n).fill(0));
}

// ---------------------------------------------------------------------------
// replay modal
// ---------------------------------------------------------------------------
async function openGame(id) {
  state.open = id;
  const cached = state.gameCache[id];
  if (!cached) {
    try {
      const replay = await fetchJSON("/api/games/" + id);
      const meta = state.games.find((g) => g.id === id) || { id };
      state.gameCache[id] = { meta, replay };
    } catch (e) {
      alert("加载棋局失败: " + id);
      return;
    }
  }
  renderReplay();
  $("#modal").classList.remove("hidden");
}

function renderReplay() {
  const c = state.gameCache[state.open];
  if (!c) return;
  const replay = c.replay;
  const meta = c.meta;

  $("#modal-title").textContent = "棋局 " + (replay.winner ? replay.winner + " 胜" : "");
  $("#modal-meta").textContent =
    `${meta.id} · ${replay.board_size}路 · komi ${replay.komi} · 结果 ${replay.result}` +
    (replay.forced_terminal ? " · 达到步数上限" : "");

  const positions = replay.positions || [];
  state.pos = Math.min(state.pos, positions.length - 1);
  $("#move-slider").max = Math.max(0, positions.length - 1);
  $("#move-slider").value = state.pos;
  drawBoardAt(state.pos);
}

function drawBoardAt(idx) {
  const c = state.gameCache[state.open];
  if (!c) return;
  const replay = c.replay;
  const positions = replay.positions || [];
  if (idx < 0 || idx >= positions.length) idx = 0;
  state.pos = idx;

  const grid = positions[idx];
  const n = grid.length;
  const last = idx > 0 && replay.moves && replay.moves[idx - 1] ? replay.moves[idx - 1] : null;
  const lastMove = last && !last.pass ? { r: last.r, c: last.c, color: last.color === "B" ? "black" : "white" } : null;

  drawGoban($("#replay-board"), grid, { lastMove });

  const total = positions.length - 1;
  $("#move-counter").textContent = `${idx} / ${total}`;
  $("#move-slider").value = idx;
  $("#btn-first").disabled = idx <= 0;
  $("#btn-prev").disabled = idx <= 0;
  $("#btn-next").disabled = idx >= total;
  $("#btn-last").disabled = idx >= total;

  let status = `第 ${idx} 手`;
  if (idx > 0 && last) {
    status += last.pass
      ? `  ${last.color} 停一手`
      : `  ${last.color} 落子 ${COORD[last.c]}${n - last.r}` + (last.captured ? ` · 提 ${last.captured} 子` : "");
  }
  if (idx === total) {
    status += ` · 终局 ${replay.result || ""}`;
  }
  $("#replay-status").textContent = status;
}

function playPause() {
  const c = state.gameCache[state.open];
  if (!c) return;
  const total = c.replay.positions.length - 1;
  if (state.playing) {
    stopPlay();
    return;
  }
  if (state.pos >= total) state.pos = 0;
  state.playing = true;
  $("#btn-play").textContent = "⏸";
  state.playTimer = setInterval(() => {
    if (state.pos >= total) { stopPlay(); return; }
    drawBoardAt(state.pos + 1);
  }, 600);
}

function stopPlay() {
  state.playing = false;
  clearInterval(state.playTimer);
  state.playTimer = null;
  const btn = $("#btn-play");
  if (btn) btn.textContent = "▶";
}

// ---------------------------------------------------------------------------
// wiring
// ---------------------------------------------------------------------------
function wireEvents() {
  $("#btn-first").addEventListener("click", () => { stopPlay(); drawBoardAt(0); });
  $("#btn-prev").addEventListener("click", () => { stopPlay(); drawBoardAt(state.pos - 1); });
  $("#btn-play").addEventListener("click", playPause);
  $("#btn-next").addEventListener("click", () => { stopPlay(); drawBoardAt(state.pos + 1); });
  $("#btn-last").addEventListener("click", () => {
    stopPlay();
    const c = state.gameCache[state.open];
    if (c) drawBoardAt(c.replay.positions.length - 1);
  });
  $("#move-slider").addEventListener("input", (e) => {
    stopPlay();
    drawBoardAt(Number(e.target.value));
  });
  $("#modal-close").addEventListener("click", () => {
    stopPlay();
    $("#modal").classList.add("hidden");
  });
  $("#modal").addEventListener("click", (e) => {
    if (e.target === $("#modal")) {
      stopPlay();
      $("#modal").classList.add("hidden");
    }
  });
  document.addEventListener("keydown", (e) => {
    if ($("#modal").classList.contains("hidden")) return;
    if (e.key === "ArrowLeft") { stopPlay(); drawBoardAt(state.pos - 1); }
    else if (e.key === "ArrowRight") { stopPlay(); drawBoardAt(state.pos + 1); }
    else if (e.key === " ") { e.preventDefault(); playPause(); }
    else if (e.key === "Escape") { stopPlay(); $("#modal").classList.add("hidden"); }
  });
}

async function poll() {
  try {
    const [train, games] = await Promise.all([fetchJSON("/api/train"), fetchJSON("/api/games")]);
    renderTrain(train);
    state.games = games.games || [];
    $("#game-count").textContent = `(${state.games.length})`;
    $("#games-empty").classList.toggle("hidden", state.games.length > 0);

    // fetch changed details, then re-render
    const changed = state.games
      .slice(0, VISIBLE_GAMES)
      .concat(state.open ? state.games.find((g) => g.id === state.open) : [])
      .filter(Boolean)
      .filter((g) => !state.gameCache[g.id] || state.gameCache[g.id].meta.mtime_raw !== g.mtime_raw);

    if (changed.length) {
      await Promise.all(changed.map(async (g) => {
        try {
          const replay = await fetchJSON("/api/games/" + g.id);
          state.gameCache[g.id] = { meta: g, replay };
        } catch (e) { /* keep stale */ }
      }));
      renderGrid(state.games.slice(0, VISIBLE_GAMES));
      if (state.open && state.gameCache[state.open]) renderReplay();
    }
  } catch (e) {
    console.warn("poll failed", e);
  }
}

// ---------------------------------------------------------------------------
// boot
// ---------------------------------------------------------------------------
wireEvents();
drawGoban($("#replay-board"), makeEmpty(19), { width: 560, height: 560 });
poll();
setInterval(poll, REFRESH_MS);
