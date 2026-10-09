/* Chapter 11 — The live training lab: "Watch a Pattern Form".

   A real transformer trains from scratch in the visitor's browser.

   1. The lab (scrollytelling stage). A Web Worker runs the page's own trainer
      (AMTrainerMain from js/model/api.js, the same code createTrainerWorker
      ships) on the reverse or sort task. The stage draws, for the task's fixed
      probe input, silk threads from every answer slot to the tokens it attends
      to (brightness = live attention weight), a woven heatmap for every head,
      and the live loss / exact-accuracy curves. Steps narrate the run and, until
      the visitor touches a control, start it and switch tasks for them.
   2. The test bench. The visitor types eight digits; the model they are
      training (or the shipped weights) writes its answer greedily, one token
      at a time, with per-digit confidence. A stress test scores 1,000 fresh
      random inputs.

   Why not AM.model.createTrainerWorker itself? The bench needs the weights the
   visitor is training, and that controller never sends them. So the worker
   here is built from exactly the same sources (AMTensorLib, AMTransformerLib,
   AMTaskLib, AMTrainerMain) plus a tiny hook that remembers the model the
   trainer creates and answers one extra message, {type:'weights'}. Everything
   else (the messages, the inline main-thread fallback) mirrors api.js. */
(() => {
  const ID = 'lab';
  const D = AM.draw;
  const MM = AM.math;
  const LEN = 8, TN = 16;
  const SLOW_MS = 140;         // slow motion: hold each step ~0.15 s
  const MAX_HIST = 3200;
  const LN10 = Math.log(10);
  const NICE = [60, 100, 150, 200, 300, 400, 600, 800, 1000, 1500, 2000, 3000, 4000];
  const HEAD_DYES = ['weld', 'woad', 'cochineal', 'lichen'];
  const headDye = (idx) => AM.dye[HEAD_DYES[idx % HEAD_DYES.length]];

  const fmtInt = (n) => Math.round(n).toLocaleString('en-US');
  const pct = (v, d = 0) => (Number.isFinite(v) ? (v * 100).toFixed(d) + '%' : '—');
  const fmtLoss = (v) => (!Number.isFinite(v) ? '—' : v >= 1 ? v.toFixed(2) : v >= 0.01 ? v.toFixed(3) : v.toFixed(4));
  const fmtSecs = (s) => (!Number.isFinite(s) ? '—' : (s < 10 ? s.toFixed(1) : String(Math.round(s))) + ' s');
  const isPhone = (w) => w < 560;

  // ------------------------------------------------------------------ colour + sprites
  const lutCache = new Map();
  /** 64-step ramp ink → dye → cream, as css strings (heatmap cells). */
  function heatLut(hex) {
    if (lutCache.has(hex)) return lutCache.get(hex);
    const a = AM.hexToRgb(AM.col.ink3), b = AM.hexToRgb(hex), c = AM.hexToRgb('#fff6dc');
    const out = [];
    for (let i = 0; i < 64; i++) {
      const t = i / 63;
      const [p, q, u] = t < 0.72 ? [a, b, t / 0.72] : [b, c, (t - 0.72) / 0.28];
      out.push(`rgb(${(p[0] + (q[0] - p[0]) * u) | 0},${(p[1] + (q[1] - p[1]) * u) | 0},${(p[2] + (q[2] - p[2]) * u) | 0})`);
    }
    lutCache.set(hex, out);
    return out;
  }
  const sprites = new Map();
  /** Pre-rendered glow bead per colour (drawImage is cheap). */
  function sprite(hex) {
    if (sprites.has(hex)) return sprites.get(hex);
    const S = 48, cv = document.createElement('canvas');
    cv.width = cv.height = S;
    const g = cv.getContext('2d'), r = S / 2;
    const grd = g.createRadialGradient(r, r, 0, r, r, r);
    grd.addColorStop(0, 'rgba(255,248,230,1)');
    grd.addColorStop(0.16, AM.rgba(hex, 1));
    grd.addColorStop(0.36, AM.rgba(hex, 0.4));
    grd.addColorStop(0.66, AM.rgba(hex, 0.08));
    grd.addColorStop(1, AM.rgba(hex, 0));
    g.fillStyle = grd;
    g.fillRect(0, 0, S, S);
    sprites.set(hex, cv);
    return cv;
  }
  const bez = (B, t) => {
    const u = 1 - t, a = u * u * u, b = 3 * u * u * t, c = 3 * u * t * t, d = t * t * t;
    return { x: a * B[0] + b * B[2] + c * B[4] + d * B[6], y: a * B[1] + b * B[3] + c * B[5] + d * B[7] };
  };
  const strokeBez = (g, B) => { g.beginPath(); g.moveTo(B[0], B[1]); g.bezierCurveTo(B[2], B[3], B[4], B[5], B[6], B[7]); g.stroke(); };

  // ------------------------------------------------------------------ model plumbing
  const LIB = (AM.model && AM.model.lib) || {};
  const TASKLIB = AM.model && AM.model.tasks;
  const HAVE_MODEL = !!(TASKLIB && LIB.AMTensorLib && LIB.AMTransformerLib && LIB.AMTrainerMain);
  let LT = null, TLm = null;
  /** A main-thread transformer library instance for the bench (separate tape from AM.model's). */
  const lib = () => { if (!TLm) { LT = LIB.AMTensorLib(); TLm = LIB.AMTransformerLib(LT); } return TLm; };

  // The hook: wrap AMTransformerLib so the trainer's freshly initialised model
  // (and its step count) are reachable, then answer {type:'weights'}.
  const HOOK = `
var __labModel = null, __labTL = null, __labStep = 0;
(function () {
  var orig = AMTransformerLib;
  AMTransformerLib = function (T) {
    var tl = orig(T), init = tl.init, mk = tl.createTrainer;
    tl.init = function (c, s) { __labModel = init(c, s); __labStep = 0; return __labModel; };
    tl.createTrainer = function (m, o) { var tr = mk(m, o), st = tr.step; tr.step = function (b) { var r = st(b); __labStep = r.step; return r; }; return tr; };
    __labTL = tl;
    return tl;
  };
})();
AMTrainerMain(self);
(function () {
  var inner = self.onmessage;
  self.onmessage = function (e) {
    var m = e.data || {};
    if (m.type === 'weights') {
      if (__labModel) self.postMessage({ type: 'weights', step: __labStep, config: __labModel.config, tensors: __labTL.exportWeights(__labModel, false) });
      return;
    }
    inner.call(self, e);
  };
})();`;
  let BODY = null;
  const body = () => BODY || (BODY = [LIB.AMTensorLib, LIB.AMTransformerLib, LIB.AMTaskLib, LIB.AMTrainerMain].map(String).join('\n;\n') + '\n;\n' + HOOK);

  /** Main-thread stand-in with the Worker interface (as in api.js), built without eval so it
      also runs under a Content-Security-Policy that blocks both blob workers and 'unsafe-eval'.
      AMTrainerMain runs here directly. It looks AMTransformerLib up by name once, when it starts,
      so for that one synchronous call the global is swapped for the same hook the worker gets
      (remember the model and its step), then put back. If the libraries are not globals the swap
      is skipped: training still works, and the bench keeps the weights it already has. */
  function inlineWorker() {
    const fake = { onmessage: null, terminated: false };
    const inner = { onmessage: null, postMessage(msg) { setTimeout(() => { if (!fake.terminated && fake.onmessage) fake.onmessage({ data: msg }); }, 0); } };
    const cap = { model: null, tl: null, step: 0 };
    const hooked = (T) => {
      const tl = LIB.AMTransformerLib(T), init = tl.init, mk = tl.createTrainer;
      tl.init = (c, sd) => { cap.model = init(c, sd); cap.step = 0; return cap.model; };
      tl.createTrainer = (mdl, o) => { const tr = mk(mdl, o), st = tr.step; tr.step = (b) => { const res = st(b); cap.step = res.step; return res; }; return tr; };
      cap.tl = tl;
      return tl;
    };
    const G = typeof globalThis !== 'undefined' ? globalThis : window;
    let swapped = false;
    try {
      if (G.AMTransformerLib === LIB.AMTransformerLib) { G.AMTransformerLib = hooked; swapped = G.AMTransformerLib === hooked; }
      LIB.AMTrainerMain(inner);
    } finally {
      if (swapped) G.AMTransformerLib = LIB.AMTransformerLib;
    }
    const handle = inner.onmessage;
    inner.onmessage = (e) => {
      const m = e.data || {};
      if (m.type === 'weights') {
        if (cap.model) inner.postMessage({ type: 'weights', step: cap.step, config: cap.model.config, tensors: cap.tl.exportWeights(cap.model, false) });
        return;
      }
      handle.call(inner, e);
    };
    fake.postMessage = (msg) => setTimeout(() => { if (!fake.terminated && inner.onmessage) inner.onmessage({ data: msg }); }, 0);
    fake.terminate = () => { fake.terminated = true; if (inner.onmessage) inner.onmessage({ data: { type: 'pause' } }); };
    return fake;
  }

  /** On the main thread: short work slices, and the (unsplittable) 200-input accuracy check at most every 0.5 s. */
  const inlineOpts = (o) => Object.assign({}, o, { chunkMs: 14, evalMs: Math.max(o.evalMs || 0, 500) });
  /** Trainer controller: Web Worker when possible, inline fallback if it dies before 'ready'. */
  function spawnTrainer(task, opts, onMsg) {
    const early = [];
    let w = null, ready = false, dead = false;
    const ctl = { inline: false };
    /** Switch to the main-thread trainer; if even that cannot start, say so through onMsg. */
    const fallBack = () => {
      try { attach(inlineWorker(), true); return true; } catch (err) {
        dead = true; w = null;
        const message = (err && err.message) || String(err);
        setTimeout(() => onMsg({ type: 'error', fatal: true, message }), 0);
        return false;
      }
    };
    const attach = (worker, isInline) => {
      w = worker;
      ctl.inline = isInline;
      w.onmessage = (e) => { if (dead) return; const m = e.data || {}; if (m.type === 'ready') ready = true; onMsg(m); };
      if (!isInline) {
        w.onerror = (e) => {
          if (e && e.preventDefault) e.preventDefault();
          if (dead) return;
          if (!ready) {
            try { w.terminate(); } catch (_) { /* already gone */ }
            if (fallBack()) early.forEach((m) => w.postMessage(m.type === 'init' ? Object.assign({}, m, { opts: inlineOpts(m.opts) }) : m));
          } else onMsg({ type: 'error', message: (e && e.message) || 'worker error' });
        };
      }
    };
    try {
      if (typeof Worker === 'undefined' || typeof Blob === 'undefined' || typeof URL === 'undefined') throw new Error('no workers');
      const url = URL.createObjectURL(new Blob(['(function (self) {\n' + body() + '\n})(self);'], { type: 'text/javascript' }));
      attach(new Worker(url), false);
      setTimeout(() => URL.revokeObjectURL(url), 10000);
    } catch (_) { w = null; }
    if (!w) fallBack();
    const send = (m) => { if (dead || !w) return; if (!ready) early.push(m); w.postMessage(m); };
    send({ type: 'init', task, opts: ctl.inline ? inlineOpts(opts) : opts });
    ctl.start = () => send({ type: 'start' });
    ctl.pause = () => send({ type: 'pause' });
    ctl.set = (o) => send(Object.assign({ type: 'set' }, o));
    ctl.weights = () => send({ type: 'weights' });
    ctl.terminate = () => { dead = true; try { if (w) w.terminate(); } catch (_) { /* ignore */ } };
    return ctl;
  }

  // ------------------------------------------------------------------ probe + analysis helpers
  const probeCache = {};
  /** The task's fixed probe as 17 ids: 8 digits, '>', 8 answer digits. */
  const probeSeq = (task) => probeCache[task] || (probeCache[task] = TASKLIB.example(task, null, TASKLIB.TASKS[task].probe).seq);
  const VOCAB = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '>'];
  const argmax = (p) => { let b = 0; for (let i = 1; i < p.length; i++) if (p[i] > p[b]) b = i; return b; };

  /** run().attn[l][h][q] (rows of length 16) → attn[l][h] = Float32Array(16·16), row q, column k. */
  const packAttn = (r) => r.attn.map((perHead) => perHead.map((rows) => {
    const a = new Float32Array(TN * TN);
    rows.forEach((row, q) => a.set(row.subarray ? row.subarray(0, TN) : row, q * TN));
    return a;
  }));

  /** What the worker's progress message holds, computed here for a given net (step 0 resting frame, shipped weights). */
  function snapshotNet(net, task) {
    const seq = probeSeq(task);
    const r = lib().run(net, seq.slice(0, TN), { capture: true });
    let pred = '';
    for (let j = 0; j < LEN; j++) pred += VOCAB[argmax(r.probs[LEN + j])];
    return { attn: packAttn(r), pred };
  }

  /** Per head: how much attention lands where the task's algorithm says it should, on the probe.
      reverse: answer slot j → input position 7 − j.
      sort: answer slot → input positions holding the smallest digit larger than the one just written. */
  function patternScores(task, attn) {
    const seq = probeSeq(task), xs = seq.slice(0, LEN);
    return attn.map((perHead) => perHead.map((A) => {
      let s = 0, n = 0;
      for (let j = 0; j < LEN; j++) {
        const q = LEN + j;
        if (task === 'reverse') { s += A[q * TN + (LEN - 1 - j)]; n++; continue; }
        const cur = j === 0 ? -1 : seq[q];
        let nl = 10;
        for (const d of xs) if (d > cur && d < nl) nl = d;
        if (nl === 10) continue;
        for (let k = 0; k < LEN; k++) if (xs[k] === nl) s += A[q * TN + k];
        n++;
      }
      return n ? s / n : 0;
    }));
  }
  /** Score an even spread would get (every query spreads 1/(q+1) over the keys it can see). */
  function uniformScore(task) {
    const A = new Float32Array(TN * TN);
    for (let q = 0; q < TN; q++) for (let k = 0; k <= q; k++) A[q * TN + k] = 1 / (q + 1);
    return patternScores(task, [[A]])[0][0];
  }

  const shippedNets = {};
  /** The shipped weights as a net of our own library instance (bench + "load trained weights"). */
  function shippedNet(task) {
    if (shippedNets[task]) return shippedNets[task];
    const m = AM.model.get(task);
    const tensors = m._net ? lib().exportWeights(m._net, false) : window.AM_WEIGHTS[task].tensors;
    return (shippedNets[task] = lib().fromWeights(m.config, tensors));
  }

  /** Greedy autoregressive answer for 8 input digits, with per-step attention rows. */
  function decode(net, digits) {
    const ids = digits.concat([10]);
    const out = [];
    for (let j = 0; j < LEN; j++) {
      const r = lib().run(net, ids, { capture: true });
      const q = ids.length - 1;
      const p = r.probs[q];
      const id = argmax(p);
      out.push({ id, p: p[id], probs: Array.from(p), attn: r.attn.map((ph) => ph.map((rows) => Float32Array.from(rows[q]))) });
      ids.push(id);
    }
    return out;
  }

  /** Exact-sequence and per-digit accuracy on n fresh random inputs (teacher-forced, one batched pass). */
  function stressTest(net, task, n) {
    const seqs = [];
    for (let i = 0; i < n; i++) seqs.push(TASKLIB.example(task, { int: (m) => Math.floor(Math.random() * m) }).seq);
    const bt = TASKLIB.batchFrom(task, seqs);
    const logits = LT.noGrad(() => lib().forward(net, bt.ids, bt.B, bt.T).logits);
    return TASKLIB.score(task, logits.data, bt);
  }

  // ------------------------------------------------------------------ chapter CSS
  AM.css(`
    #ch-${ID} .lab-stage .fig { gap: var(--space-2); }
    #ch-${ID} .lab-stats {
      display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 2px;
      padding: 7px 6px 6px; border: 1px solid var(--rule); border-radius: var(--radius-sm);
      background: color-mix(in srgb, var(--ink-2) 82%, transparent);
    }
    #ch-${ID} .lab-stat { display: grid; justify-items: center; gap: 1px; min-width: 0; cursor: help; }
    #ch-${ID} .lab-k { font-family: var(--font-mono); font-size: var(--fs-micro); letter-spacing: 0.06em; text-transform: uppercase; color: var(--mist); white-space: nowrap; }
    #ch-${ID} .lab-v { font-family: var(--font-mono); font-size: 13px; color: var(--linen); font-variant-numeric: tabular-nums; white-space: nowrap; transition: color 0.3s; }
    #ch-${ID} .lab-v.is-good { color: var(--verdigris); }
    #ch-${ID} .lab-v.is-weld { color: var(--weld); }
    #ch-${ID} .lab-meta { font-family: var(--font-mono); font-size: 10px; letter-spacing: 0.05em; color: var(--mist); line-height: 1.5; }
    #ch-${ID} .lab-meta b { color: var(--linen-dim); font-weight: 500; }
    #ch-${ID} .lab-meta .is-shipped { color: var(--weld); }
    #ch-${ID} .lab-controls { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 10px; }
    #ch-${ID} .lab-controls .btn { min-height: 34px; padding: 6px 14px; }
    #ch-${ID} .lab-controls .seg button { padding: 5px 11px; }
    #ch-${ID} .lab-controls .btn-primary { min-width: 104px; }
    #ch-${ID} .lab-head { display: inline-flex; align-items: center; gap: 8px; }
    #ch-${ID} .lab-head[hidden] { display: none; }
    #ch-${ID} .lab-head > span { font-family: var(--font-mono); font-size: var(--fs-micro); letter-spacing: 0.1em; text-transform: uppercase; color: var(--mist); }
    #ch-${ID} .lab-stage canvas:focus-visible, #ch-${ID} .lab-bench canvas:focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; border-radius: 6px; }
    #ch-${ID} .lab-kicker { font-family: var(--font-mono); font-size: var(--fs-micro); letter-spacing: 0.16em; text-transform: uppercase; color: var(--weld); }
    #ch-${ID} .lab-sec { display: grid; gap: var(--space-5); }
    #ch-${ID} .lab-bench-top { display: flex; flex-wrap: wrap; align-items: end; gap: 10px 14px; }
    #ch-${ID} .lab-in { display: grid; gap: 6px; }
    #ch-${ID} .lab-in > span { font-family: var(--font-mono); font-size: var(--fs-micro); letter-spacing: 0.1em; text-transform: uppercase; color: var(--mist); }
    #ch-${ID} .lab-digits {
      width: 10.5em; max-width: 100%; min-height: 44px; padding: 6px 10px 6px 16px;
      font-family: var(--font-mono); font-size: 1.15rem; letter-spacing: 0.32em; text-align: center;
      font-variant-numeric: tabular-nums;
    }
    #ch-${ID} .lab-digits.is-bad { border-color: var(--madder); }
    #ch-${ID} .lab-bench-grid { display: grid; grid-template-columns: minmax(0, 1.75fr) minmax(0, 1fr); gap: var(--space-5); align-items: center; }
    #ch-${ID} .lab-bench-side { display: grid; gap: var(--space-4); align-content: center; min-width: 0; }
    #ch-${ID} .lab-read { display: grid; gap: 4px; font-family: var(--font-mono); font-size: 12px; letter-spacing: 0.03em; color: var(--linen-dim); min-height: 6.5em; }
    #ch-${ID} .lab-ans { font-family: var(--font-mono); font-size: clamp(1.5rem, 1.1rem + 1.4vw, 2.1rem); letter-spacing: 0.18em; line-height: 1.2; color: var(--linen); }
    #ch-${ID} .lab-want { color: var(--mist); }
    @media (max-width: 900px) { #ch-${ID} .lab-bench-grid { grid-template-columns: minmax(0, 1fr); } #ch-${ID} .lab-read { min-height: 0; } }
    #ch-${ID} .lab-read b { font-weight: 500; color: var(--linen); }
    #ch-${ID} .lab-read .ok { color: var(--verdigris); }
    #ch-${ID} .lab-read .bad { color: var(--madder); }
    #ch-${ID} .lab-stress { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 14px; }
    #ch-${ID} .lab-stress-out { font-family: var(--font-mono); font-size: 12px; color: var(--linen-dim); }
    #ch-${ID} .lab-stress-out b { color: var(--verdigris); font-weight: 500; }
    #ch-${ID} .lab-limits { border-left: 2px solid var(--rule-strong); padding-left: var(--space-4); }
    @media (max-width: 900px) {
      #ch-${ID} .lab-stage .fig-top { min-height: 0; }
    }
    @media (max-width: 560px) {
      #ch-${ID} .lab-stats { grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 3px 10px; padding: 5px 8px; }
      #ch-${ID} .lab-stat { display: flex; justify-content: space-between; align-items: baseline; gap: 6px; }
      #ch-${ID} .lab-k { letter-spacing: 0.02em; }
      #ch-${ID} .lab-v { font-size: 12px; }
      #ch-${ID} .lab-controls { gap: 6px; }
      #ch-${ID} .lab-controls .btn { min-height: 32px; padding: 5px 11px; font-size: 10px; letter-spacing: 0.08em; }
      #ch-${ID} .lab-controls .btn-primary { min-width: 84px; }
      #ch-${ID} .lab-controls .seg button { padding: 4px 9px; font-size: 10px; letter-spacing: 0.04em; }
      #ch-${ID} .lab-meta { font-size: 9px; }
      #ch-${ID} .lab-head { display: none; }
      #ch-${ID} .lab-stage .fig { gap: 6px; }
    }
  `);

  // ------------------------------------------------------------------ stage geometry
  function stageLayout(w) {
    const phone = isPhone(w);
    const loomH = phone ? 160 : 250;
    const gap = phone ? 10 : 18;
    const lowH = phone ? 110 : 172;
    const mapsW = Math.round(w * (phone ? 0.5 : 0.47));
    return {
      phone, w, h: loomH + gap + lowH,
      loom: { x0: 2, x1: w - 2, y0: 0, y1: loomH },
      maps: { x0: 0, x1: mapsW, y0: loomH + gap, y1: loomH + gap + lowH },
      curve: { x0: mapsW + (phone ? 12 : 22), x1: w, y0: loomH + gap, y1: loomH + gap + lowH },
    };
  }
  function loomGeom(x0, x1, y0, y1, phone, opts = {}) {
    const colW = (x1 - x0) / 9;
    const tile = Math.min(colW * 0.74, phone ? 30 : 42);
    const yTop = y0 + (opts.topPad ?? (phone ? 25 : 34)) + tile / 2; // room above the tiles for per-key weights
    const yBot = y1 - (opts.botPad ?? (phone ? 26 : 34)) - tile / 2;
    return { x0, x1, colW, tile, yTop, yBot, cx: (i) => x0 + (i + 0.5) * colW };
  }
  /** Thread from answer slot j (query 8+j) to key k. Keys 0–8 sit on the top row; keys ≥ 9 are earlier answers (arcs). */
  function threadPts(G, j, k, arcsUp) {
    const sx = G.cx(j);
    if (k <= LEN) {
      const sy = G.yBot - G.tile / 2 - 3, ex = G.cx(k), ey = G.yTop + G.tile / 2 + 3, dy = sy - ey;
      return [sx, sy, sx, sy - dy * 0.56, ex, ey + dy * 0.56, ex, ey];
    }
    const i = k - LEN - 1, ex = G.cx(i);
    const depth = Math.min(arcsUp ? 30 : 24, 7 + 4 * (j - i));
    const y = arcsUp ? G.yBot - G.tile / 2 - 3 : G.yBot + G.tile / 2 + 3;
    const s = arcsUp ? -1 : 1;
    return [sx, y, sx, y + s * depth, ex, y + s * depth, ex, y];
  }

  /** Draw a digit tile. state: {fill, stroke, color, glow, glowColor, sub} */
  function tile(g, cx, cy, size, ch, o = {}) {
    const x = cx - size / 2, y = cy - size / 2;
    g.save();
    if (o.glow > 0.02) { g.shadowColor = AM.rgba(o.glowColor || AM.dye.weld, Math.min(0.9, o.glow)); g.shadowBlur = 6 + 16 * o.glow; }
    D.roundRect(g, x, y, size, size, Math.min(8, size * 0.22));
    g.fillStyle = o.fill || AM.col.ink2;
    g.fill();
    g.shadowBlur = 0;
    g.lineWidth = 1;
    g.strokeStyle = o.stroke || AM.col.ruleStrong;
    g.stroke();
    g.fillStyle = o.color || AM.col.linen;
    g.font = AM.font(Math.round(size * (ch === '>' ? 0.5 : 0.64)), ch === '>' ? 'mono' : 'display', ch === '>' ? 400 : 500);
    g.textAlign = 'center';
    g.textBaseline = 'middle';
    g.fillText(ch, cx, cy + size * 0.03);
    g.restore();
  }

  // ------------------------------------------------------------------ the chapter
  AM.chapter({
    id: ID,
    num: 11,
    kicker: 'Live training lab',
    title: 'Watch a Pattern <em>Form</em>',
    lede: 'A real transformer starts as random numbers and learns a small algorithm in your browser while you watch. Its attention begins as an even haze, then crystallises into a pattern you can read.',
    where: 'lab',
    mount(root, ctx) {
      ctx.header();
      const el = ctx.el;
      const ui = AM.ui;
      const body0 = el('div', { class: 'ch-body' });
      root.appendChild(body0);
      if (!HAVE_MODEL) {
        body0.appendChild(el('p', { class: 'caption' }, 'The live model scripts did not load, so the lab cannot run here.'));
        return;
      }

      // ================================================================ state
      const runs = { reverse: null, sort: null };
      const S = {
        task: 'reverse', autopilot: true, focus: -1,
        head: { reverse: 0, sort: 1 }, headManual: { reverse: false, sort: false },
        nowT: 0, lastWeightsReq: 0, statsDirty: true,
      };
      const defaultSpeed = (task) => (task === 'reverse' ? 'slow' : 'fast');
      const newSeed = () => 1 + Math.floor(Math.random() * 9999);
      const cur = () => runs[S.task];
      const view = (r) => (r.mode === 'shipped' && r.ship ? r.ship : r); // r.ship.shipped === true marks the shipped view
      const cfgOf = (task) => TASKLIB.TASKS[task].model;
      const UNIFORM = { reverse: uniformScore('reverse'), sort: uniformScore('sort') };

      function makeRun(task, seed, speed) {
        const net = lib().init(cfgOf(task), seed); // identical to the worker's init: same code, same seed
        const snap = snapshotNet(net, task);
        return {
          task, seed, speed, ctl: null, ready: false, running: false, started: false, done: false,
          step: 0, loss: NaN, acc: NaN, tokenAcc: NaN, elapsed: 0, sps: 0, hist: [], hit90: null, flareT: -99,
          attn: snap.attn, pred: snap.pred, disp: null, xMax: null,
          mode: 'live', ship: null, ownNet: net, ownStep: 0, params: lib().countParams(net), inline: false, resumeOnShow: false,
        };
      }
      runs.reverse = makeRun('reverse', newSeed(), 'slow');

      // ================================================================ DOM: stage
      const statDefs = [
        ['step', 'Step', 'Training steps so far'],
        ['seen', 'Seen', 'Training sequences so far, 32 per step'],
        ['loss', 'Loss', 'Cross-entropy on the eight answer digits, a running average over training batches (the chart shows it on a log scale)'],
        ['exact', 'Exact', 'Share of 200 held-out random inputs with all eight answer digits right'],
        ['digit', 'Per digit', 'Share of single answer digits right, on the same 200 held-out inputs'],
        ['time', 'Compute', 'Seconds spent training so far'],
      ];
      const statEls = {};
      const stats = el('div', { class: 'lab-stats', role: 'group', 'aria-label': 'Training statistics' },
        statDefs.map(([k, label, tip]) => el('div', { class: 'lab-stat', title: tip }, el('span', { class: 'lab-k' }, label), (statEls[k] = el('span', { class: 'lab-v' }, '—')))));
      const meta = el('div', { class: 'lab-meta' });
      const live = el('div', { class: 'sr-only', 'aria-live': 'polite' });
      const announce = (s) => { live.textContent = s; };

      const btnTrain = ui.button({ id: 'lab-train', label: 'Train', kind: 'primary', onClick: () => { touch(); toggleTrain(); } });
      const btnReset = ui.button({ id: 'lab-reset', label: 'Reset', title: 'Throw the weights away and start again from a new random seed', onClick: () => { touch(); resetRun(S.task); } });
      const btnLoad = ui.button({ id: 'lab-load', label: 'Load trained', title: 'Skip ahead: show the weights that ship with the page (trained offline with this same code)', onClick: () => { touch(); loadShipped(); } });
      const segTask = ui.segmented({ id: 'lab-task', label: null, options: [{ value: 'reverse', label: 'Reverse' }, { value: 'sort', label: 'Sort' }], value: 'reverse', onChange: (v) => { touch(); setTask(v); } });
      segTask.el.setAttribute('aria-label', 'Task');
      const segSpeed = ui.segmented({ id: 'lab-speed', options: [{ value: 'slow', label: 'Slow-mo' }, { value: 'fast', label: 'Full speed' }], value: 'slow', onChange: (v) => { touch(); setSpeed(v); } });
      segSpeed.el.setAttribute('aria-label', 'Training speed');
      const segHead = ui.segmented({ id: 'lab-head', options: ['L0H0', 'L0H1', 'L1H0', 'L1H1'].map((s, i) => ({ value: i, label: s })), value: 1, onChange: (v) => { touch(); S.head.sort = v; S.headManual.sort = true; } });
      segHead.el.setAttribute('aria-label', 'Head whose threads are drawn');
      const headBox = el('div', { class: 'lab-head' }, el('span', {}, 'Threads'), segHead.el);
      const controls = el('div', { class: 'lab-controls' }, btnTrain, btnReset, btnLoad, segSpeed.el, segTask.el, headBox);

      const stageHost = el('div');
      const capText = `Live model: a real transformer training in your browser, with the page's own code. Threads and maps show its attention on one fixed probe input. Each thread runs from an answer slot (bottom row) to a token it attends to (top row); map rows are the 8 answer slots, columns the 16 positions. The number above each map is the share of attention landing where the algorithm says it should (an even spread scores about ${UNIFORM.reverse.toFixed(2)} for reverse, ${UNIFORM.sort.toFixed(2)} for sort). Exact means all eight digits right on 200 held-out inputs.`;
      const figTitle = el('span', { class: 'fig-title' }, 'Training · reverse');
      const stage = el('div', { class: 'ch-stage lab-stage' },
        el('figure', { class: 'fig' },
          el('div', { class: 'fig-top' }, figTitle, ui.badge('live', 'Live training')),
          stats, stageHost, controls, meta, live));

      const step = (label, h3, ...html) => el('div', { class: 'step' },
        el('span', { class: 'step-label' }, label),
        el('h3', {}, h3),
        ...html.map((s) => (s.startsWith('<div') ? el('div', { html: s }) : el('p', { html: s }))));
      const revParams = fmtInt(runs.reverse.params);
      const sortParams = fmtInt(lib().countParams(lib().init(cfgOf('sort'), 1)));
      const steps = [
        step('1 · Random threads', 'A loom with no pattern',
          `The picture shows a real transformer: one layer, one attention head, <span class="math">d_model = 24</span>, and <strong>${revParams}</strong> parameters. A moment ago every one of them was drawn at random. The seed is shown under the picture.`,
          'Its task is tiny. Read eight digits and a <span class="math">&gt;</span>, then write the digits backwards: <span class="math">38152907&gt;70925183</span>. The bottom row is what it would write right now.',
          'Each thread runs from an answer slot to a token that slot attends to. With random weights every score <span class="math">q·k/√d_head</span> is close to zero, so the softmax spreads attention almost evenly. You see a haze.'),
        step('2 · Training', 'The same loop, on a toy task',
          'Training is the loop from the last chapter. Each step draws 32 fresh random sequences, predicts every next token, scores the eight answer positions with cross-entropy, backpropagates, and takes one AdamW step.',
          'It is running now, in a background thread of your browser (a Web Worker) if the browser allows one. It runs in slow motion, a few steps a second, so you can watch. The maths is the same at full speed. Use the buttons under the picture to pause, reset or skip ahead.'),
        step('3 · The plateau', 'Ten digits, no idea which',
          `For the first few dozen steps the loss barely moves. It starts near <strong>${Math.log(11).toFixed(2)}</strong>, which is <span class="math">ln 11</span>: an even guess over all eleven tokens. Then it creeps toward <strong>${LN10.toFixed(2)}</strong>, which is <span class="math">ln 10</span> (the dashed line): an even guess over just the ten digits. So far the model has learned that answers are digits, and little else.`,
          'Underneath, the attention is already shifting a little with every step. At first the change is too small to see.'),
        step('4 · The jump', 'Then it crystallises',
          'Somewhere around step 50 to 75 the haze snaps into a crisp X. Each slot locks onto one input digit, the loss falls off a cliff, and exact-sequence accuracy leaps from 0 to nearly 100% in about ten steps.',
          'Accuracy gets there first, because it only asks that the right digit be the most likely one. The loss keeps falling for hundreds of steps more, as the model grows confident.',
          'Sudden jumps like this are common when a network has to discover a mechanism. Until attention points at the right place, the MLP and the unembedding after it have nothing useful to read, so the loss barely moves. Once it does, everything improves at once.'),
        step('5 · The algorithm', 'Reading the pattern',
          'The finished pattern is an algorithm you can read off the map. The query at answer slot j puts nearly all its weight, typically around 0.98, on input position <span class="math">7\u00a0−\u00a0j</span>. That digit\'s value vector is copied into the slot, and the unembedding turns it into the prediction.',
          'Where a slot looks depends almost entirely on position, which is why the same X appears for every input. That information comes from the learned position embeddings: slot j\'s query is trained to match the key of position 7\u00a0−\u00a0j, whatever digit sits there.'),
        step('6 · Sorting', 'A harder pattern',
          `Sorting is harder: <span class="math">73519273&gt;12335779</span>. The stage switches to it here, unless you have taken the controls (then pick Sort). This model has two layers with two heads each and ${sortParams} parameters. It needs a few hundred steps to reach 90% exact, and the climb is gradual.`,
          'Watch the first-layer maps. One head picks up a rule: from the digit just written, look for the smallest input digit that is larger. On this probe that means: after <span class="math">&gt;</span> look at the 1, after the 1 look at the 2, after the 5 look at both 7s. Which head takes the job depends on the random start; its score is printed above each map.',
          'This head is a soft pointer. On random inputs, the shipped model\'s head puts 36% of its weight on that digit, about three times an even spread. The rest of the network finishes the job.'),
      ];
      const prose = el('div', { class: 'ch-prose' }, steps);
      body0.appendChild(el('div', { class: 'ch-split' }, stage, prose));
      body0.appendChild(el('p', { class: 'caption' }, capText));

      // ================================================================ stage canvas
      const cv = ctx.canvas(stageHost, {
        height: (w) => stageLayout(w).h,
        label: 'Live training view: silk threads from each answer slot to the input digits it attends to, a heatmap of attention for every head, and the loss and accuracy curves.',
      });
      cv.canvas.tabIndex = 0;
      let lay = stageLayout(cv.w || 600);
      let bg = null;
      let mapRects = [];
      const particles = [];

      function buildBg() {
        const { w, h, dpr } = cv;
        bg = document.createElement('canvas');
        bg.width = Math.round(w * dpr); bg.height = Math.round(h * dpr);
        const g = bg.getContext('2d');
        g.setTransform(dpr, 0, 0, dpr, 0, 0);
        const L = lay.loom;
        D.roundRect(g, L.x0, L.y0, L.x1 - L.x0, L.y1 - L.y0, 12);
        g.fillStyle = AM.rgba(AM.col.ink2, 0.55);
        g.fill();
        g.save();
        D.roundRect(g, L.x0, L.y0, L.x1 - L.x0, L.y1 - L.y0, 12);
        g.clip();
        D.weave(g, L.x0, L.y0, L.x1 - L.x0, L.y1 - L.y0, { step: 5, alpha: 0.035 });
        const grd = g.createRadialGradient(w / 2, L.y1 * 0.5, 10, w / 2, L.y1 * 0.5, w * 0.7);
        grd.addColorStop(0, AM.rgba(AM.dye.woad, 0.06));
        grd.addColorStop(1, AM.rgba(AM.col.ink, 0));
        g.fillStyle = grd;
        g.fillRect(L.x0, L.y0, L.x1 - L.x0, L.y1 - L.y0);
        g.restore();
        g.strokeStyle = AM.col.rule;
        g.lineWidth = 1;
        D.roundRect(g, L.x0 + 0.5, L.y0 + 0.5, L.x1 - L.x0 - 1, L.y1 - L.y0 - 1, 12);
        g.stroke();
      }
      cv.onResize((w) => { lay = stageLayout(w); buildBg(); particles.length = 0; S.statsDirty = true; drawStage(S.nowT, 0); });

      // smoothed display copy of the attention, so snapshots blend instead of jumping
      function smoothAttn(r, target, dt) {
        if (!r.disp || r.disp.length !== target.length) { r.disp = target.map((ph) => ph.map((A) => Float32Array.from(A))); return; }
        const k = AM.reducedMotion || dt === 0 ? 1 : 1 - Math.exp(-dt * 9);
        for (let l = 0; l < target.length; l++) for (let h = 0; h < target[l].length; h++) {
          const A = r.disp[l][h], T = target[l][h];
          for (let i = 0; i < A.length; i++) A[i] += (T[i] - A[i]) * k;
        }
      }

      function drawLoom(g, r, v, A, t, dt) {
        const L = lay.loom, phone = lay.phone;
        const G = loomGeom(L.x0 + 4, L.x1 - 4, L.y0, L.y1, phone);
        const seq = probeSeq(r.task);
        const nH = cfgOf(r.task).n_head;
        const hs = S.head[r.task], l = Math.floor(hs / nH), hh = hs % nH;
        const Ah = A[l][hh], dye = headDye(hs);
        const age = t - r.flareT;
        const flare = age >= 0 && age < 2.2 ? Math.pow(1 - age / 2.2, 2) : 0;

        // labels
        D.text(g, 'INPUT', G.x0 + 4, L.y0 + (phone ? 11 : 14), { size: phone ? 8 : 9, role: 'mono', color: AM.col.mist, letterSpacing: '0.14em' });
        D.text(g, `L${l} · H${hh}`, G.x1 - 4, L.y0 + (phone ? 11 : 14), { size: phone ? 8 : 9, role: 'mono', color: dye, align: 'right', letterSpacing: '0.1em' });
        const ax = G.cx(8);
        D.text(g, 'ANSWER', ax, G.yBot - 3, { size: phone ? 8 : 9, role: 'mono', color: AM.col.mist, align: 'center', letterSpacing: phone ? '0.02em' : '0.12em', maxWidth: G.colW - 2 });
        D.text(g, v.shipped ? 'shipped' : phone ? 'now' : 'right now', ax, G.yBot + (phone ? 9 : 11), { size: phone ? 8 : 9, role: 'mono', color: v.shipped ? AM.dye.weld : AM.col.mist, align: 'center' });

        // threads, additive so overlaps glow like silk
        g.save();
        g.globalCompositeOperation = 'lighter';
        g.lineCap = 'round';
        const focus = S.focus;
        for (let j = 0; j < LEN; j++) {
          const q = LEN + j;
          const fm = focus < 0 || focus === j ? 1 : 0.1;
          for (let k = 0; k <= q; k++) {
            const a = Ah[q * TN + k];
            if (a < 0.004) continue;
            const P = threadPts(G, j, k, false);
            const alpha = Math.min(1, (0.05 + 0.9 * Math.pow(a, 0.8)) * fm * (1 + 0.9 * flare));
            const wdt = (phone ? 0.5 : 0.6) + (phone ? 2.4 : 3.2) * a;
            if (a > 0.12) { g.strokeStyle = AM.rgba(dye, alpha * 0.16); g.lineWidth = wdt * 5; strokeBez(g, P); }
            g.strokeStyle = AM.rgba(dye, alpha);
            g.lineWidth = wdt;
            strokeBez(g, P);
            // particles: beads of value travelling from key to query
            if (!AM.reducedMotion && dt > 0 && fm === 1 && particles.length < 240 && Math.random() < a * 2.4 * dt) {
              particles.push({ j, k, u: 0, sp: 0.55 + Math.random() * 0.35, a });
            }
          }
        }
        // the knot: where a reversal's threads all cross
        if (r.task === 'reverse') {
          const sc = patternScores('reverse', A)[0][0];
          const kx = (G.cx(0) + G.cx(7)) / 2, ky = (G.yTop + G.yBot) / 2;
          const s = MM.smoothstep(0.35, 0.95, sc);
          if (s > 0.01) {
            const R = (phone ? 16 : 24) * (1 + 0.8 * flare);
            const grd = g.createRadialGradient(kx, ky, 0, kx, ky, R);
            grd.addColorStop(0, AM.rgba('#fff6dc', 0.5 * s));
            grd.addColorStop(0.3, AM.rgba(dye, 0.28 * s));
            grd.addColorStop(1, AM.rgba(dye, 0));
            g.fillStyle = grd;
            g.beginPath(); g.arc(kx, ky, R, 0, Math.PI * 2); g.fill();
          }
        }
        // particles
        for (let i = particles.length - 1; i >= 0; i--) {
          const p = particles[i];
          p.u += p.sp * dt;
          if (p.u >= 1 || p.j >= LEN) { particles.splice(i, 1); continue; }
          const P = threadPts(G, p.j, p.k, false);
          const pt = bez(P, 1 - p.u);
          const s = (phone ? 5 : 7) + (phone ? 8 : 11) * p.a;
          g.globalAlpha = Math.min(1, 0.25 + p.a) * Math.sin(Math.PI * p.u) * (focus < 0 || focus === p.j ? 1 : 0.15);
          g.drawImage(sprite(dye), pt.x - s / 2, pt.y - s / 2, s, s);
        }
        g.globalAlpha = 1;
        // crystallisation flare: a ring of light from the loom's centre
        if (flare > 0) {
          const cx = (G.cx(0) + G.cx(7)) / 2, cy = (G.yTop + G.yBot) / 2;
          const R = 10 + (1 - Math.sqrt(flare)) * (L.x1 - L.x0) * 0.6;
          g.strokeStyle = AM.rgba(AM.dye.weld, 0.5 * flare);
          g.lineWidth = 2 + 6 * flare;
          g.beginPath(); g.arc(cx, cy, R, 0, Math.PI * 2); g.stroke();
          g.strokeStyle = AM.rgba('#fff6dc', 0.35 * flare);
          g.lineWidth = 1;
          g.beginPath(); g.arc(cx, cy, R * 0.82, 0, Math.PI * 2); g.stroke();
        }
        g.restore();

        // top row: input digits and '>'
        for (let k = 0; k <= LEN; k++) {
          let recv = 0;
          if (focus >= 0) recv = Ah[(LEN + focus) * TN + k];
          else for (let j = 0; j < LEN; j++) recv = Math.max(recv, Ah[(LEN + j) * TN + k]);
          tile(g, G.cx(k), G.yTop, G.tile, VOCAB[seq[k]], {
            color: k === LEN ? AM.col.mist : AM.col.linen,
            stroke: recv > 0.3 ? AM.mix(AM.col.ruleStrong, dye, Math.min(1, recv)) : AM.col.ruleStrong,
            glow: recv > 0.3 ? recv * 0.7 : 0, glowColor: dye,
          });
          if (focus >= 0 && recv > 0.02) {
            D.text(g, recv.toFixed(2), G.cx(k), G.yTop - G.tile / 2 - 4, { size: phone ? 8 : 9, role: 'mono', color: dye, align: 'center' });
          }
        }
        // bottom row: the model's current greedy answer for the probe
        for (let j = 0; j < LEN; j++) {
          const want = VOCAB[seq[LEN + 1 + j]], got = v.pred ? v.pred[j] : '·';
          const ok = got === want;
          const c = ok ? AM.dye.verdigris : AM.dye.madder;
          tile(g, G.cx(j), G.yBot, G.tile, got || '·', {
            color: c,
            fill: AM.mix(AM.col.ink2, c, focus === j ? 0.2 : 0.08),
            stroke: focus === j ? AM.col.linen : AM.mix(AM.col.ruleStrong, c, 0.5),
            glow: ok ? 0.25 + 0.5 * flare : 0, glowColor: c,
          });
          if (!ok) D.text(g, want, G.cx(j) + G.tile / 2 - 3, G.yBot - G.tile / 2 + (phone ? 8 : 10), { size: phone ? 8 : 8.5, role: 'mono', color: AM.col.mist, align: 'right' });
        }
      }

      function drawMaps(g, r, A, scores) {
        const R = lay.maps, phone = lay.phone;
        const cfg = cfgOf(r.task), nL = cfg.n_layer, nH = cfg.n_head;
        const labH = phone ? 14 : 17, gx = phone ? 8 : 14, gy = phone ? 6 : 10;
        const cw = (R.x1 - R.x0 - gx * (nH - 1)) / nH, chh = (R.y1 - R.y0 - gy * (nL - 1)) / nL;
        const c = Math.max(2, Math.floor(Math.min(cw / TN, (chh - labH) / LEN) * 2) / 2);
        const multi = nL * nH > 1;
        mapRects = [];
        for (let l = 0; l < nL; l++) for (let h = 0; h < nH; h++) {
          const idx = l * nH + h, dye = headDye(idx), lut = heatLut(dye);
          const mx = R.x0 + h * (cw + gx), my = R.y0 + l * (chh + gy), gy0 = my + labH;
          const Wm = c * TN, Hm = c * LEN;
          const sel = S.head[r.task] === idx;
          D.text(g, `L${l}·H${h}`, mx, my + labH - 5, { size: phone ? 8 : 9.5, role: 'mono', color: dye, letterSpacing: '0.06em' });
          const sc = scores[l][h];
          const scTxt = (phone || multi && cw < 170) ? sc.toFixed(2) : (r.task === 'reverse' ? '7−j score ' : 'next-larger ') + sc.toFixed(2);
          D.text(g, scTxt, mx + Wm, my + labH - 5, { size: phone ? 8 : 9, role: 'mono', color: sc > 2.5 * UNIFORM[r.task] ? AM.col.linen : AM.col.mist, align: 'right' });
          const Ah = A[l][h];
          const gap = c >= 6 ? 1 : 0.5;
          for (let j = 0; j < LEN; j++) {
            const q = LEN + j;
            for (let k = 0; k < TN; k++) {
              const x = mx + k * c, y = gy0 + j * c;
              if (k > q) {
                g.fillStyle = AM.rgba(AM.col.linen, 0.07);
                g.fillRect(x + c / 2 - 0.5, y + c / 2 - 0.5, 1, 1);
                continue;
              }
              const val = Ah[q * TN + k];
              g.fillStyle = lut[Math.min(63, (Math.sqrt(Math.max(0, val)) * 63) | 0)];
              g.fillRect(x, y, c - gap, c - gap);
              if (c >= 6) { // woven sheen: alternate warp/weft highlight
                g.fillStyle = 'rgba(255,248,230,0.07)';
                if ((j + k) % 2) g.fillRect(x + (c - gap) * 0.38, y, (c - gap) * 0.24, c - gap);
                else g.fillRect(x, y + (c - gap) * 0.38, c - gap, (c - gap) * 0.24);
              }
            }
          }
          // separators: inputs | '>' | answers so far
          g.fillStyle = AM.rgba(AM.col.linen, 0.25);
          g.fillRect(mx + LEN * c - gap / 2 - 0.5, gy0 - 2, 1, Hm + 2);
          if (multi && sel) {
            g.strokeStyle = dye; g.lineWidth = 1.2;
            D.roundRect(g, mx - 3, gy0 - 3, Wm + 5 - gap, Hm + 5 - gap, 4); g.stroke();
          }
          mapRects.push({ idx, x: mx - 3, y: my, w: Wm + 6, h: labH + Hm + 4 });
        }
      }

      function drawCurves(g, r, v, t, dt) {
        const R = lay.curve, phone = lay.phone;
        const px0 = R.x0 + (phone ? 22 : 28), px1 = R.x1 - 4, py0 = R.y0 + (phone ? 20 : 24), py1 = R.y1 - (phone ? 14 : 16);
        const fs = phone ? 8 : 9;
        // legend
        const lx = R.x0;
        g.fillStyle = AM.dye.weld; g.fillRect(lx, R.y0 + 6, 12, 2);
        D.text(g, 'LOSS (LOG)', lx + 16, R.y0 + 10, { size: fs, role: 'mono', color: AM.col.linenDim, letterSpacing: '0.08em' });
        const lx2 = lx + 16 + D.measure(g, 'LOSS (LOG)', fs, 'mono') + (phone ? 10 : 16);
        g.fillStyle = AM.dye.verdigris; g.fillRect(lx2, R.y0 + 6, 12, 2);
        D.text(g, 'EXACT', lx2 + 16, R.y0 + 10, { size: fs, role: 'mono', color: AM.col.linenDim, letterSpacing: '0.08em' });

        const hist = v.hist;
        const last = hist.length ? hist[hist.length - 1].s : 0;
        const minX = r.task === 'reverse' ? 150 : 400;
        const want = NICE.find((n) => n >= Math.max(minX, v.shipped || r.done ? last : last * 1.08)) || Math.ceil(last / 1000) * 1000;
        if (r.xMax == null || dt === 0 || AM.reducedMotion || Math.abs(want - r.xMax) < 1) r.xMax = want; else r.xMax += (want - r.xMax) * (1 - Math.exp(-dt * 4));
        const xMax = r.xMax;
        const X = (s) => px0 + (s / xMax) * (px1 - px0);
        const LTOP = Math.log10(6), LBOT = -3; // headroom keeps the ln 10 line clear of the 100% line
        const Y = (L) => py0 + (LTOP - Math.log10(MM.clamp(L, 1e-3, 3.5))) / (LTOP - LBOT) * (py1 - py0);
        const YA = (a) => py1 - a * (py1 - py0);

        // grid
        g.lineWidth = 1;
        [[1, '1'], [0.1, '.1'], [0.01, '.01'], [0.001, '.001']].forEach(([val, lab]) => {
          const y = Math.round(Y(val)) + 0.5;
          g.strokeStyle = AM.rgba(AM.col.linen, 0.06);
          g.beginPath(); g.moveTo(px0, y); g.lineTo(px1, y); g.stroke();
          D.text(g, lab, px0 - 5, y + 3, { size: fs, role: 'mono', color: AM.col.mist, align: 'right' });
        });
        g.strokeStyle = AM.rgba(AM.col.linen, 0.14);
        g.beginPath(); g.moveTo(px0 + 0.5, py0); g.lineTo(px0 + 0.5, py1); g.lineTo(px1, py1 + 0.5); g.stroke();
        // ln 10: the loss of an even guess over ten digits
        const y10 = Y(LN10);
        g.save();
        g.setLineDash([3, 4]);
        g.strokeStyle = AM.rgba(AM.col.linenDim, 0.45);
        g.beginPath(); g.moveTo(px0, y10); g.lineTo(px1, y10); g.stroke();
        g.restore();
        D.text(g, phone ? 'ln 10' : 'ln 10 · even guess', px1, y10 + (phone ? 9 : 11), { size: phone ? 8 : 8.5, role: 'mono', color: AM.col.mist, align: 'right' });
        D.text(g, '100%', px1, YA(1) - 3, { size: phone ? 8 : 8.5, role: 'mono', color: AM.rgba(AM.dye.verdigris, 0.8), align: 'right' });
        D.text(g, '0', px0, py1 + (phone ? 10 : 12), { size: fs, role: 'mono', color: AM.col.mist, align: 'center' });
        D.text(g, `step ${fmtInt(xMax)}`, px1, py1 + (phone ? 10 : 12), { size: fs, role: 'mono', color: AM.col.mist, align: 'right' });

        // 90% marker
        const h90 = v.hit90;
        if (h90 != null && h90 <= xMax) {
          const x = Math.round(X(h90)) + 0.5;
          g.save();
          g.setLineDash([2, 3]);
          g.strokeStyle = AM.rgba(AM.dye.verdigris, 0.6);
          g.beginPath(); g.moveTo(x, py0); g.lineTo(x, py1); g.stroke();
          g.restore();
          const lab = `90% @ ${fmtInt(h90)}`;
          const tw = D.measure(g, lab, fs, 'mono');
          const right = x + 5 + tw < px1;
          D.text(g, lab, right ? x + 5 : x - 5, py0 + (py1 - py0) * 0.84, { size: fs, role: 'mono', color: AM.dye.verdigris, align: right ? 'left' : 'right' });
        }

        if (!hist.length) return;
        const stride = Math.max(1, Math.ceil(hist.length / 360));
        const plot = (key, map, color, wdt) => {
          g.beginPath();
          let started = false;
          for (let i = 0; i < hist.length; i += stride) {
            const p = hist[i];
            const val = p[key];
            if (!Number.isFinite(val)) continue;
            const x = X(p.s), y = map(val);
            if (!started) { g.moveTo(x, y); started = true; } else g.lineTo(x, y);
          }
          const p = hist[hist.length - 1];
          if (started && Number.isFinite(p[key])) g.lineTo(X(p.s), map(p[key]));
          if (!started) return;
          g.strokeStyle = AM.rgba(color, 0.18); g.lineWidth = wdt * 4; g.stroke();
          g.strokeStyle = color; g.lineWidth = wdt; g.stroke();
          if (Number.isFinite(p[key]) && Number.isFinite(X(p.s))) D.glowDot(g, X(p.s), map(p[key]), phone ? 2 : 2.6, color);
        };
        g.save();
        g.beginPath(); g.rect(px0 - 4, py0 - 6, px1 - px0 + 8, py1 - py0 + 10); g.clip();
        g.lineJoin = 'round';
        plot('a', YA, AM.dye.verdigris, phone ? 1.2 : 1.5);
        plot('l', Y, AM.dye.weld, phone ? 1.3 : 1.7);
        if (v.shipped) {
          for (const p of hist) {
            if (!Number.isFinite(X(p.s))) continue;
            if (Number.isFinite(p.l)) D.glowDot(g, X(p.s), Y(p.l), 1.6, AM.dye.weld, 0.8);
            if (Number.isFinite(p.a)) D.glowDot(g, X(p.s), YA(p.a), 1.4, AM.dye.verdigris, 0.7);
          }
        }
        g.restore();
      }

      function drawStage(t, dt) {
        const { g, w, h } = cv;
        if (!w || !bg) return;
        const r = cur(), v = view(r);
        smoothAttn(r, v.attn, dt);
        const A = r.disp;
        // auto-pick the head with the clearest pattern (sort), until the visitor chooses
        const scores = patternScores(r.task, A);
        if (r.task === 'sort' && !S.headManual.sort) {
          const best = scores[0][0] >= scores[0][1] ? 0 : 1;
          if (best !== S.head.sort) { S.head.sort = best; segHead.set(best); particles.length = 0; }
        }
        cv.clear();
        g.drawImage(bg, 0, 0, w, h);
        drawLoom(g, r, v, A, t, dt);
        drawMaps(g, r, A, scores);
        drawCurves(g, r, v, t, dt);
      }

      ctx.loop((t, dt) => {
        S.nowT = t;
        const r = cur();
        // the bench wants the weights being trained: ask the worker about once a second
        if (r.running && r.ctl && t - S.lastWeightsReq > 1) { S.lastWeightsReq = t; r.ctl.weights(); }
        if (S.statsDirty) { S.statsDirty = false; renderStats(); }
        if (stageVis.on) drawStage(t, dt);
        if (benchVis.on) { bench.refresh(); bench.frame(t, dt); }
      });

      // pointer + keyboard: focus one answer slot, or pick a head by clicking its map
      const slotAt = (p) => {
        if (p.y > lay.loom.y1) return -1;
        const G = loomGeom(lay.loom.x0 + 4, lay.loom.x1 - 4, 0, lay.loom.y1, lay.phone);
        const j = Math.floor((p.x - G.x0) / G.colW);
        return j >= 0 && j < LEN ? j : -1;
      };
      cv.canvas.addEventListener('pointermove', (e) => { if (e.pointerType === 'mouse') S.focus = slotAt(cv.pointer(e)); });
      cv.canvas.addEventListener('pointerleave', (e) => { if (e.pointerType === 'mouse') S.focus = -1; });
      // 'click' (not pointerdown) so a finger that starts a scroll on the canvas changes nothing
      let lastPointer = 'mouse';
      cv.canvas.addEventListener('pointerdown', (e) => { lastPointer = e.pointerType || 'mouse'; });
      cv.canvas.addEventListener('click', (e) => {
        const p = cv.pointer(e);
        const hit = mapRects.find((m) => p.x >= m.x && p.x <= m.x + m.w && p.y >= m.y && p.y <= m.y + m.h);
        if (hit && cfgOf(S.task).n_head > 1) { touch(); S.head.sort = hit.idx; S.headManual.sort = true; segHead.set(hit.idx); particles.length = 0; return; }
        const j = slotAt(p);
        if (lastPointer !== 'mouse') S.focus = j === S.focus ? -1 : j;
      });
      cv.canvas.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
          e.preventDefault();
          const d = e.key === 'ArrowRight' ? 1 : -1;
          S.focus = S.focus < 0 ? (d > 0 ? 0 : LEN - 1) : (S.focus + d + LEN) % LEN;
        } else if (e.key === 'Escape') S.focus = -1;
      });
      cv.canvas.addEventListener('blur', () => { S.focus = -1; });

      // ================================================================ training control
      function ensureCtl(r) {
        if (r.ctl) return;
        // The 200-input accuracy check costs about 3.5 training steps, so it runs at every
        // report for reverse (its jump takes ~10 steps) and every ~0.4 s of training for sort.
        const ev = r.task === 'reverse' ? { reportMs: 120, evalMs: 0 } : { reportMs: 150, evalMs: 400 };
        r.ctl = spawnTrainer(r.task, Object.assign({ seed: r.seed, delayMs: r.speed === 'slow' ? SLOW_MS : 0 }, ev), (m) => onMsg(r, m));
      }
      function onMsg(r, m) {
        if (runs[r.task] !== r) return; // a run that was reset
        if (m.type === 'ready') { r.ready = true; r.params = m.params; r.inline = r.ctl.inline; S.statsDirty = true; }
        else if (m.type === 'progress') {
          r.step = m.step; r.loss = m.loss; r.acc = m.acc; r.tokenAcc = m.tokenAcc;
          r.elapsed = m.elapsed; r.sps = m.stepsPerSec; r.attn = m.attn; r.pred = m.probePred;
          const lastH = r.hist[r.hist.length - 1];
          if (!lastH || lastH.s !== m.step) r.hist.push({ s: m.step, l: m.loss, a: m.acc, d: m.tokenAcc });
          if (r.hist.length > MAX_HIST) r.hist = r.hist.filter((_, i) => i % 2 === 0 || i === r.hist.length - 1);
          if (r.hit90 == null && m.acc >= 0.9 && r.mode === 'live') {
            r.hit90 = m.step; r.flareT = S.nowT;
            announce(`Exact accuracy passed 90% at step ${m.step}.`);
          }
          S.statsDirty = true;
        } else if (m.type === 'done') {
          r.running = false; r.done = true; r.ctl.weights();
          announce(`Training finished after ${m.step} steps.`);
          syncUI();
        } else if (m.type === 'weights') {
          r.ownNet = lib().fromWeights(m.config, m.tensors);
          r.ownStep = m.step;
          bench.onWeights(r);
        } else if (m.type === 'error') {
          r.running = false;
          if (m.fatal || !r.ready) r.failed = true;
          r.error = m.message;
          announce(r.failed ? 'Training could not start in this browser.' : 'Training stopped: ' + m.message);
          syncUI();
        }
      }
      function startRun(r) {
        if (r.mode === 'shipped') { r.mode = 'live'; segModel.set('own'); bench.setSource('own'); }
        if (r.done || r.failed) { syncUI(); return; } // back to the finished run, or nothing can train here
        ensureCtl(r);
        if (r.failed) { syncUI(); return; }
        r.ctl.set({ delayMs: r.speed === 'slow' ? SLOW_MS : 0 });
        r.ctl.start();
        r.running = true; r.started = true;
        announce(`Training ${r.task}${r.speed === 'slow' ? ' in slow motion' : ''}.`);
        syncUI();
      }
      function pauseRun(r) {
        if (!r || !r.running) return;
        r.ctl.pause();
        r.running = false;
        r.ctl.weights();
        syncUI();
      }
      function toggleTrain() {
        const r = cur();
        if (r.running) pauseRun(r); else startRun(r);
      }
      function resetRun(task) {
        const old = runs[task];
        if (old && old.ctl) old.ctl.terminate();
        runs[task] = makeRun(task, newSeed(), old ? old.speed : defaultSpeed(task));
        particles.length = 0;
        if (task === S.task) {
          segModel.set('own');
          bench.setSource('own');
          bench.onWeights(runs[task]);
          announce(`New random start, seed ${runs[task].seed}.`);
        }
        syncUI();
      }
      function setTask(task) {
        if (S.task === task && runs[task]) { syncUI(); return; }
        const prev = cur();
        if (prev && prev.running) pauseRun(prev);
        S.task = task;
        if (!runs[task]) runs[task] = makeRun(task, newSeed(), defaultSpeed(task));
        particles.length = 0;
        S.focus = -1;
        syncUI();
        bench.onTask();
      }
      function setSpeed(sp) {
        const r = cur();
        r.speed = sp;
        if (r.ctl) r.ctl.set({ delayMs: sp === 'slow' ? SLOW_MS : 0 });
        syncUI();
      }
      function loadShipped() {
        const r = cur();
        pauseRun(r);
        const m = AM.model.get(r.task);
        const mt = m.meta || {}, ta = mt.test_accuracy || {}, tr = mt.train || {};
        const snap = snapshotNet(shippedNet(r.task), r.task);
        const t90 = mt.node_timing && mt.node_timing.time_to_accuracy && mt.node_timing.time_to_accuracy['0.9'];
        r.ship = {
          shipped: true, attn: snap.attn, pred: snap.pred,
          step: tr.steps || TASKLIB.TASKS[r.task].train.total, batch: tr.batch || 32,
          loss: mt.final_train_loss, acc: ta.exact_sequence, tokenAcc: ta.digit, testN: ta.n,
          hist: (mt.loss_curve || []).map(([s, l, a]) => ({ s, l, a })),
          hit90: t90 ? t90.step : null, seed: tr.seed, params: mt.params,
        };
        r.mode = 'shipped';
        r.disp = r.disp || null;
        r.flareT = S.nowT;
        announce(`Showing the shipped ${r.task} weights.`);
        segModel.set('shipped');
        bench.setSource('shipped');
        syncUI();
      }

      // the scroll story drives the stage until the visitor takes the controls
      const touch = () => { S.autopilot = false; };
      ctx.steps(steps, (i) => {
        if (!S.autopilot) return;
        const task = i <= 4 ? 'reverse' : 'sort';
        setTask(task);
        const r = runs[task];
        if (i >= 1 && !r.done && !r.running && r.mode === 'live') startRun(r);
      });

      // don't burn CPU invisibly: pause when the chapter or the tab is hidden, resume on return
      const hide = () => { Object.values(runs).forEach((r) => { if (r && r.running) { pauseRun(r); r.resumeOnShow = true; } }); };
      const show = () => { Object.values(runs).forEach((r) => { if (r && r.resumeOnShow) { r.resumeOnShow = false; if (r === cur()) startRun(r); } }); };
      ctx.onHidden(hide);
      ctx.onVisible(show);
      document.addEventListener('visibilitychange', () => { if (document.hidden) hide(); else if (ctx.visible) show(); });

      // ================================================================ readouts
      const setStat = (k, txt, cls) => {
        const e = statEls[k];
        if (e.textContent !== txt) e.textContent = txt;
        e.className = 'lab-v' + (cls ? ' ' + cls : '');
      };
      function renderStats() {
        const r = cur(), v = view(r);
        const batch = v.batch || TASKLIB.TASKS[r.task].train.batch;
        setStat('step', fmtInt(v.step || 0));
        setStat('seen', fmtInt((v.step || 0) * batch));
        setStat('loss', fmtLoss(v.loss), 'is-weld');
        setStat('exact', pct(v.acc, v.acc > 0.995 && v.acc < 1 ? 1 : 0), v.acc >= 0.9 ? 'is-good' : '');
        setStat('digit', pct(v.tokenAcc, v.tokenAcc > 0.995 && v.tokenAcc < 1 ? 2 : 0), v.tokenAcc >= 0.9 ? 'is-good' : '');
        setStat('time', v.shipped ? '—' : fmtSecs(r.elapsed));
        if (v.shipped) {
          meta.innerHTML = `<span class="is-shipped">Shipped weights</span> · <b>${fmtInt(v.params || r.params)}</b> parameters · trained offline with this same code, seed ${v.seed ?? 1} · tested on ${fmtInt(v.testN || 10000)} inputs`;
        } else {
          const where = r.failed ? (lay.phone ? 'cannot train here' : 'this browser blocked the trainer; Load trained still works')
            : r.error ? 'training stopped'
            : !r.ctl ? 'press Train'
            : !r.ready ? 'starting…'
            : lay.phone ? (r.inline ? 'main thread' : 'Web Worker') : r.inline ? 'training on the main thread' : 'training in a Web Worker';
          meta.innerHTML = `<b>${fmtInt(r.params)}</b> parameters · seed <b>${r.seed}</b> · ${where}${r.done ? ' · finished' : ''}`;
        }
      }
      function syncUI() {
        const r = cur();
        segTask.set(S.task);
        segSpeed.set(r.speed);
        headBox.hidden = cfgOf(S.task).n_head === 1;
        segHead.set(S.head.sort);
        figTitle.textContent = `Training · ${S.task}`;
        if (r.mode === 'shipped') { btnTrain.textContent = r.done || r.failed ? 'Show mine' : r.started ? 'Resume mine' : 'Train my own'; btnTrain.disabled = false; }
        else if (r.done) { btnTrain.textContent = 'Trained'; btnTrain.disabled = true; }
        else if (r.failed) { btnTrain.textContent = 'Cannot train'; btnTrain.disabled = true; }
        else { btnTrain.textContent = r.running ? 'Pause' : r.started ? 'Resume' : 'Train'; btnTrain.disabled = false; }
        btnTrain.setAttribute('aria-pressed', String(r.running));
        btnLoad.disabled = r.mode === 'shipped';
        S.statsDirty = true;
      }

      // ================================================================ 2. the test bench
      const benchHost = el('div');
      const benchInput = el('input', { id: 'lab-bench-input', class: 'text-input lab-digits', type: 'text', inputmode: 'numeric', autocomplete: 'off', spellcheck: 'false', maxlength: '8', 'aria-label': 'Eight input digits', 'aria-describedby': 'lab-bench-read' });
      const benchRead = el('div', { id: 'lab-bench-read', class: 'lab-read', 'aria-live': 'polite' });
      const stressOut = el('span', { class: 'lab-stress-out', 'aria-live': 'polite' });
      const stressBtn = ui.button({ id: 'lab-bench-stress', label: 'Stress test · 1,000 inputs', onClick: () => bench.stress() });
      const segModel = ui.segmented({ id: 'lab-bench-model', options: [{ value: 'own', label: 'Yours' }, { value: 'shipped', label: 'Shipped' }], value: 'own', onChange: (v) => { bench.setSource(v); } });
      segModel.el.setAttribute('aria-label', 'Which weights to test');
      const segTask2 = ui.segmented({ id: 'lab-bench-task', options: [{ value: 'reverse', label: 'Reverse' }, { value: 'sort', label: 'Sort' }], value: 'reverse', onChange: (v) => { touch(); setTask(v); } });
      segTask2.el.setAttribute('aria-label', 'Task');
      const benchFig = ui.figure({
        title: 'Greedy answers, digit by digit', badge: ui.badge('live', 'Live model'), cls: 'lab-bench',
        caption: 'Live model: the weights you are training (fetched from the worker about once a second while it runs), or the shipped weights. The answer is generated greedily: at each step the most likely token is chosen and fed back in. The bar under each digit is the probability of the digit chosen. Threads show the attention of the head selected above at the step that wrote each digit; hover or tab to a digit to isolate it. The stress test scores 1,000 fresh random inputs in four batched passes: exact = all eight digits right.',
      },
      el('div', { class: 'lab-bench-top' },
        el('label', { class: 'lab-in', for: 'lab-bench-input' }, el('span', {}, 'Input · 8 digits'), benchInput),
        ui.button({ id: 'lab-bench-random', label: 'Random', onClick: () => bench.roll(false) }),
        ui.button({ id: 'lab-bench-repeats', label: 'Repeats', title: 'Random digits drawn from just two or three values', onClick: () => bench.roll(true) }),
        segTask2.el, segModel.el),
      el('div', { class: 'lab-bench-grid' },
        benchHost,
        el('div', { class: 'lab-bench-side' },
          benchRead,
          el('div', { class: 'lab-stress' },
            stressBtn,
            stressOut))));
      body0.appendChild(el('section', { class: 'ch-wide lab-sec', 'aria-labelledby': 'lab-bench-h' },
        el('div', { class: 'prose' },
          el('span', { class: 'lab-kicker' }, 'Test bench'),
          el('h3', { id: 'lab-bench-h' }, 'Ask your model'),
          el('p', { html: 'Type any eight digits or roll random ones. The model writes its answer one digit at a time, feeding each digit back in, the same way a chatbot generates text. The bar under each digit is the probability it gave that digit. Verdigris digits are right, madder ones wrong.' }),
          el('p', { html: 'Try it before training, halfway through and after. It keeps updating while the model above trains. On sorting, compare the two accuracies in the readout above the picture, or run the stress test: per digit reaches 90% long before exact does, because all eight digits must be right at once, and <span class="math">0.9⁸ ≈ 0.43</span>.' })),
        el('div', { class: 'panel' }, benchFig),
        el('div', { class: 'prose' },
          el('p', { html: 'Eight random digits have 10⁸ possible values. The model sees 32 sequences per step, so after the 600 steps of the reverse schedule it has met 19,200 of them, about 0.02%. Nearly every input you type is new to it, and once trained it still gets them right. What it learned is the rule itself.' }),
          el('p', { class: 'lab-limits', html: '<strong>Honest limits.</strong> These models only know sequences of exactly eight digits followed by <span class="math">&gt;</span>. They have 16 learned positions and nothing beyond them, so a seven- or nine-digit input means nothing to them, and the bench only accepts eight. A large language model has the same kind of edge at the end of its context window, just much further out.' }))));

      const benchCv = ctx.canvas(benchHost, { height: (w) => (isPhone(w) ? 224 : 262), label: 'Test bench: your input digits on top, the model\'s answer below with a confidence bar under each digit, and threads showing where it looked while writing each one.' });
      benchCv.canvas.tabIndex = 0;

      const bench = (() => {
        const B = { digits: TASKLIB.TASKS.reverse.probe.slice(), source: 'own', res: null, t0: -99, focus: -1, net: null, netKey: '', stale: false, stressId: 0 };
        const curNet = () => {
          const r = cur();
          if (B.source === 'shipped') return { net: shippedNet(S.task), key: 'shipped-' + S.task, label: 'shipped weights' };
          return { net: r.ownNet, key: `own-${S.task}-${r.seed}-${r.ownStep}`, label: `your model at step ${fmtInt(r.ownStep)}` };
        };
        const target = () => TASKLIB.TASKS[S.task].fn(B.digits);
        const cancelStress = () => { B.stressId++; stressBtn.disabled = false; stressOut.textContent = ''; };
        function compute(animate) {
          // announce answers the visitor asked for, not the once-a-second refresh during training
          benchRead.setAttribute('aria-live', animate ? 'polite' : 'off');
          if (B.digits.length !== LEN) { B.res = null; renderRead(); return; }
          const { net, key, label } = curNet();
          B.netKey = key; B.label = label; B.stale = false;
          B.res = decode(net, B.digits);
          if (animate) B.t0 = S.nowT;
          renderRead();
          if (!benchVis.on) frame(S.nowT, 0);
        }
        function renderRead() {
          if (!B.res) { benchRead.innerHTML = '<span class="lab-k">Answer</span><span class="lab-ans">········</span><span><span class="bad">Needs exactly 8 digits.</span> These models only ever saw length 8.</span>'; return; }
          const want = target();
          let right = 0, minP = 1;
          const ans = B.res.map((o, j) => { if (o.id === want[j]) right++; minP = Math.min(minP, o.p); return `<span class="${o.id === want[j] ? 'ok' : 'bad'}">${VOCAB[o.id]}</span>`; }).join('');
          benchRead.innerHTML = `<span class="lab-k">Answer · ${B.label}</span><span class="lab-ans">${ans}</span><span><b class="${right === LEN ? 'ok' : 'bad'}">${right} of 8</b> digits right</span><span>lowest confidence <b>${pct(minP)}</b></span><span class="lab-want">correct answer ${want.join('')}</span>`;
        }
        function frame(t, dt) {
          const { g, w, h } = benchCv;
          if (!w) return;
          benchCv.clear();
          const phone = isPhone(w);
          const G = loomGeom(4, w - 4, 0, h, phone, { topPad: phone ? 31 : 37, botPad: phone ? 46 : 52 });
          D.roundRect(g, 1, 1, w - 2, h - 2, 12);
          g.fillStyle = AM.rgba(AM.col.ink, 0.35); g.fill();
          D.text(g, 'INPUT', G.x0 + 4, phone ? 13 : 15, { size: phone ? 8 : 9, role: 'mono', color: AM.col.mist, letterSpacing: '0.14em' });
          const nH = cfgOf(S.task).n_head, hs = S.head[S.task], l = Math.floor(hs / nH), hh = hs % nH, dye = headDye(hs);
          D.text(g, `threads: L${l} · H${hh}`, G.x1 - 4, phone ? 13 : 15, { size: phone ? 8 : 9, role: 'mono', color: dye, align: 'right' });
          const want = B.digits.length === LEN ? target() : null;
          const res = B.res;
          const per = AM.reducedMotion ? 0 : 0.2;
          const shown = !res ? 0 : per === 0 ? LEN : MM.clamp(Math.floor((t - B.t0 - 0.1) / per) + 1, 0, LEN);
          // threads for each written digit (additive silk)
          if (res) {
            g.save();
            g.globalCompositeOperation = 'lighter';
            g.lineCap = 'round';
            for (let j = 0; j < shown; j++) {
              const row = res[j].attn[l][hh];
              const fm = B.focus < 0 ? (j === shown - 1 && shown < LEN ? 1 : 0.55) : B.focus === j ? 1 : 0.07;
              for (let k = 0; k < row.length; k++) {
                const a = row[k];
                if (a < 0.01) continue;
                const P = threadPts(G, j, k, true);
                g.strokeStyle = AM.rgba(dye, Math.min(1, (0.05 + 0.9 * Math.pow(a, 0.8)) * fm));
                g.lineWidth = 0.6 + (phone ? 2.2 : 3) * a;
                strokeBez(g, P);
              }
            }
            // the shuttle: a bead flies from the most-attended token to the slot being written
            if (shown < LEN) {
              const j = shown, u = MM.clamp((t - B.t0 - 0.1 - (j - 1) * per) / per, 0, 1);
              const row = res[j].attn[l][hh];
              const k = argmax(row);
              const P = threadPts(G, j, k, true);
              const pt = bez(P, 1 - MM.ease.inOut(u));
              const s = phone ? 14 : 18;
              g.drawImage(sprite(dye), pt.x - s / 2, pt.y - s / 2, s, s);
            }
            g.restore();
          }
          // input row
          const fr = res && B.focus >= 0 && B.focus < shown ? res[B.focus].attn[l][hh] : null;
          for (let k = 0; k <= LEN; k++) {
            const ch = k < LEN ? (B.digits[k] != null ? VOCAB[B.digits[k]] : '·') : '>';
            const a = fr ? fr[k] : 0;
            tile(g, G.cx(k), G.yTop, G.tile, ch, { color: k === LEN ? AM.col.mist : AM.col.linen, stroke: a > 0.3 ? dye : AM.col.ruleStrong, glow: a > 0.3 ? a * 0.7 : 0, glowColor: dye });
            if (fr && a > 0.02) D.text(g, a.toFixed(2), G.cx(k), G.yTop - G.tile / 2 - 4, { size: phone ? 8 : 9, role: 'mono', color: dye, align: 'center' });
          }
          D.text(g, 'ANSWER', G.cx(8), G.yBot + 3, { size: phone ? 8 : 9, role: 'mono', color: AM.col.mist, align: 'center', letterSpacing: phone ? '0.02em' : '0.12em', maxWidth: G.colW - 2 });
          // answer row with confidence bars
          for (let j = 0; j < LEN; j++) {
            const x = G.cx(j);
            if (!res || j >= shown) {
              g.save(); g.setLineDash([3, 3]); g.strokeStyle = AM.col.rule;
              D.roundRect(g, x - G.tile / 2, G.yBot - G.tile / 2, G.tile, G.tile, Math.min(8, G.tile * 0.22)); g.stroke(); g.restore();
              continue;
            }
            const o = res[j], ok = want && o.id === want[j], c = ok ? AM.dye.verdigris : AM.dye.madder;
            const age = t - B.t0 - 0.1 - j * per;
            const pop = per === 0 ? 1 : MM.ease.outBack(MM.clamp(age / 0.25 + 0.2, 0, 1));
            g.save();
            g.translate(x, G.yBot); g.scale(pop, pop); g.translate(-x, -G.yBot);
            tile(g, x, G.yBot, G.tile, VOCAB[o.id], { color: c, fill: AM.mix(AM.col.ink2, c, B.focus === j ? 0.22 : 0.1), stroke: B.focus === j ? AM.col.linen : AM.mix(AM.col.ruleStrong, c, 0.55), glow: 0.3, glowColor: c });
            g.restore();
            const bw = G.tile, by = G.yBot + G.tile / 2 + 7;
            g.fillStyle = AM.rgba(AM.col.linen, 0.1);
            D.roundRect(g, x - bw / 2, by, bw, 4, 2); g.fill();
            g.fillStyle = c;
            D.roundRect(g, x - bw / 2, by, Math.max(2, bw * o.p), 4, 2); g.fill();
            D.text(g, pct(o.p), x, by + (phone ? 14 : 16), { size: phone ? 8 : 9, role: 'mono', color: AM.col.linenDim, align: 'center' });
            if (!ok && want) D.text(g, `want ${want[j]}`, x, by + (phone ? 25 : 29), { size: phone ? 8 : 8.5, role: 'mono', color: AM.dye.madder, align: 'center' });
          }
        }
        benchCv.onResize(() => frame(S.nowT, 0));
        const slotAtB = (p) => {
          const w = benchCv.w, G = loomGeom(4, w - 4, 0, benchCv.h, isPhone(w));
          const j = Math.floor((p.x - G.x0) / G.colW);
          return j >= 0 && j < LEN ? j : -1;
        };
        benchCv.canvas.addEventListener('pointermove', (e) => { if (e.pointerType === 'mouse') B.focus = slotAtB(benchCv.pointer(e)); });
        benchCv.canvas.addEventListener('pointerleave', (e) => { if (e.pointerType === 'mouse') B.focus = -1; });
        let lastPointerB = 'mouse';
        benchCv.canvas.addEventListener('pointerdown', (e) => { lastPointerB = e.pointerType || 'mouse'; });
        benchCv.canvas.addEventListener('click', (e) => {
          if (lastPointerB === 'mouse') return;
          const j = slotAtB(benchCv.pointer(e));
          B.focus = j === B.focus ? -1 : j;
          if (!benchVis.on) frame(S.nowT, 0);
        });
        benchCv.canvas.addEventListener('keydown', (e) => {
          if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
            e.preventDefault();
            const d = e.key === 'ArrowRight' ? 1 : -1;
            B.focus = B.focus < 0 ? (d > 0 ? 0 : LEN - 1) : (B.focus + d + LEN) % LEN;
          } else if (e.key === 'Escape') B.focus = -1;
          if (!benchVis.on) frame(S.nowT, 0);
        });
        benchCv.canvas.addEventListener('blur', () => { B.focus = -1; });
        benchInput.value = B.digits.join('');
        benchInput.addEventListener('input', () => {
          const clean = benchInput.value.replace(/\D/g, '').slice(0, LEN);
          if (clean !== benchInput.value) benchInput.value = clean;
          B.digits = clean.split('').map(Number);
          benchInput.classList.toggle('is-bad', clean.length !== LEN);
          compute(true);
        });
        const api = {
          frame,
          roll(repeats) {
            const pool = repeats ? Array.from({ length: 2 + Math.floor(Math.random() * 2) }, () => Math.floor(Math.random() * 10)) : null;
            B.digits = Array.from({ length: LEN }, () => (pool ? pool[Math.floor(Math.random() * pool.length)] : Math.floor(Math.random() * 10)));
            benchInput.value = B.digits.join('');
            benchInput.classList.remove('is-bad');
            compute(true);
          },
          setSource(src) { B.source = src; cancelStress(); compute(true); },
          onWeights(r) {
            if (r !== cur()) return;
            segModel.el.querySelectorAll('button')[0].textContent = `Yours · step ${fmtInt(r.ownStep)}`;
            if (B.source !== 'own' || B.netKey === curNet().key) return;
            // decoding is 8 forward passes on the main thread: only while the bench can be seen
            if (benchVis.on) compute(false); else B.stale = true;
          },
          /** Called each frame while the bench is on screen: catch up on weights that arrived off-screen. */
          refresh() { if (B.stale) compute(false); },
          onTask() {
            segTask2.set(S.task);
            const src = cur().mode === 'shipped' ? 'shipped' : 'own';
            B.source = src; segModel.set(src);
            B.digits = TASKLIB.TASKS[S.task].probe.slice();
            benchInput.value = B.digits.join('');
            benchInput.classList.remove('is-bad');
            cancelStress();
            const r = cur();
            segModel.el.querySelectorAll('button')[0].textContent = `Yours · step ${fmtInt(r.ownStep)}`;
            compute(true);
          },
          stress() {
            // 1,000 inputs in four batches of 250, one per task, so the page stays responsive
            const { net, label } = curNet();
            const task = S.task, n = 1000, chunk = 250, id = ++B.stressId;
            let done = 0, seq = 0, tok = 0;
            stressBtn.disabled = true;
            stressOut.textContent = `scoring… 0 of ${fmtInt(n)}`;
            const next = () => {
              if (id !== B.stressId) return;
              const sc = stressTest(net, task, chunk);
              seq += sc.seqAcc * chunk; tok += sc.tokenAcc * chunk; done += chunk;
              if (done < n) { stressOut.textContent = `scoring… ${fmtInt(done)} of ${fmtInt(n)}`; setTimeout(next, 0); return; }
              stressBtn.disabled = false;
              const sa = seq / n, ta = tok / n;
              stressOut.innerHTML = `${label}: <b>${pct(sa, sa > 0.99 && sa < 1 ? 1 : 0)}</b> exact, ${pct(ta, ta > 0.99 && ta < 1 ? 2 : 0)} per digit, on ${fmtInt(n)} fresh random inputs`;
            };
            setTimeout(next, 0);
          },
        };
        return api;
      })();

      // per-figure visibility, so each canvas only redraws while it can be seen
      const visibility = (elm) => {
        const st = { on: true };
        if (typeof IntersectionObserver !== 'undefined') {
          st.on = false;
          new IntersectionObserver((en) => { st.on = en[en.length - 1].isIntersecting; }, { rootMargin: '60px 0px' }).observe(elm);
        }
        return st;
      };
      const stageVis = visibility(stageHost);
      const benchVis = visibility(benchHost);

      // ================================================================ key idea
      body0.appendChild(el('div', { class: 'callout' },
        el('span', { class: 'callout-label' }, 'Key idea'),
        el('p', { html: 'Nobody wrote the algorithm. Starting from random numbers, the same next-token training loop found attention patterns you can read as steps of one: <strong>look at position 7\u00a0−\u00a0j</strong> to reverse, <strong>lean toward the next larger digit</strong> to sort. Interpretability research reads trained models the same way, head by head, though in large models the patterns are far harder to read.' })));

      // resting frame
      bench.onTask();
      syncUI();
      renderStats();
      drawStage(0, 0);
    },
  });
})();
