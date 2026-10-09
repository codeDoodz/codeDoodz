/* Chapter 10 — Learning by Falling: how a language model is trained.

   Three figures:
   1. "One training step" (scrollytelling stage). A tiny causal model, drawn as
      a loom, reads five tokens and predicts the next token at every position.
      Cross-entropy hangs from each target as a red drip, blame flows back down
      the threads (backprop), one gradient step shrinks the drips, and the last
      step swaps to the published Chinchilla scaling-law fit.
   2. Cross-entropy playground. Six draggable probability bars (always
      renormalised to sum to 1), the −ln p curve with a bead, the exact logit
      gradient p − y as arrows, and a real gradient step on the logits.
   3. The loss landscape (hero). A shaded, woven 2-D terrain with analytic
      gradients. Beads fall onto it and roll downhill by gradient descent, with
      a learning-rate slider, momentum, minibatch noise and a loss sparkline. */
(() => {
  const ID = 'train';
  const MM = AM.math;
  const D = AM.draw;
  const TAU = Math.PI * 2;

  // ------------------------------------------------------------------ small helpers
  const isStacked = () => (window.matchMedia ? window.matchMedia('(max-width: 900px)').matches : window.innerWidth <= 900);
  const fmtSigned = (v, d = 2) => (v < 0 ? '−' : '+') + Math.abs(v).toFixed(d);
  const RGB = new Map();
  const rgb = (hex) => { if (!RGB.has(hex)) RGB.set(hex, AM.hexToRgb(hex)); return RGB.get(hex); };
  /** rgba() string mixing two hex colours. */
  const mixA = (h1, h2, t, a = 1) => {
    const A = rgb(h1), B = rgb(h2);
    return `rgba(${Math.round(A[0] + (B[0] - A[0]) * t)},${Math.round(A[1] + (B[1] - A[1]) * t)},${Math.round(A[2] + (B[2] - A[2]) * t)},${a})`;
  };
  const parseRgb = (s) => (String(s).match(/[\d.]+/g) || [0, 0, 0]).slice(0, 3).map(Number);

  /** Is this element (roughly) on screen? Lets each figure idle while the chapter is visible. */
  const visibility = (el, rootMargin = '80px 0px') => {
    const st = { on: true };
    if (typeof IntersectionObserver !== 'undefined') {
      st.on = false;
      new IntersectionObserver((en) => { st.on = en[en.length - 1].isIntersecting; }, { rootMargin }).observe(el);
    }
    return st;
  };

  /** Pre-rendered glow sprite per colour (particles drawn with drawImage are cheap). */
  const sprites = new Map();
  const sprite = (hex) => {
    if (sprites.has(hex)) return sprites.get(hex);
    const S = 48, c = document.createElement('canvas');
    c.width = c.height = S;
    const g = c.getContext('2d'), r = S / 2;
    const grd = g.createRadialGradient(r, r, 0, r, r, r);
    const [r0, g0, b0] = rgb(hex);
    grd.addColorStop(0, `rgba(${(r0 + 255 * 1.4) / 2.4 | 0},${(g0 + 250 * 1.4) / 2.4 | 0},${(b0 + 236 * 1.4) / 2.4 | 0},1)`);
    grd.addColorStop(0.14, AM.rgba(hex, 1));
    grd.addColorStop(0.32, AM.rgba(hex, 0.42));
    grd.addColorStop(0.62, AM.rgba(hex, 0.1));
    grd.addColorStop(1, AM.rgba(hex, 0));
    g.fillStyle = grd;
    g.fillRect(0, 0, S, S);
    sprites.set(hex, c);
    return c;
  };
  const blob = (g, hex, x, y, r, a = 1) => {
    if (a <= 0.004) return;
    g.globalAlpha = a;
    g.drawImage(sprite(hex), x - r, y - r, r * 2, r * 2);
    g.globalAlpha = 1;
  };

  /** Cubic Bézier helpers. B = [x0,y0,x1,y1,x2,y2,x3,y3]. */
  const bez = (B, t) => {
    const u = 1 - t, a = u * u * u, b = 3 * u * u * t, c = 3 * u * t * t, d = t * t * t;
    return { x: a * B[0] + b * B[2] + c * B[4] + d * B[6], y: a * B[1] + b * B[3] + c * B[5] + d * B[7] };
  };
  const bezLen = (B) => { let L = 0, p = bez(B, 0); for (let i = 1; i <= 16; i++) { const q = bez(B, i / 16); L += Math.hypot(q.x - p.x, q.y - p.y); p = q; } return L; };
  const strokeBez = (g, B) => { g.beginPath(); g.moveTo(B[0], B[1]); g.bezierCurveTo(B[2], B[3], B[4], B[5], B[6], B[7]); g.stroke(); };
  const line = (x0, y0, x1, y1) => [x0, y0, x0 + (x1 - x0) / 3, y0 + (y1 - y0) / 3, x0 + (x1 - x0) * 2 / 3, y0 + (y1 - y0) * 2 / 3, x1, y1];

  /** Tracked mono micro-label. */
  const micro = (g, s, x, y, o = {}) => D.text(g, s, x, y, { size: o.size || 9.5, role: 'mono', color: o.color || AM.col.mist, align: o.align || 'left', baseline: o.baseline || 'middle', alpha: o.alpha, letterSpacing: o.ls || '0.1em' });

  /** One gradient-descent step taken directly on the logits z = ln p:
      ∂L/∂z_i = p_i − y_i for L = −ln p_correct, so z ← z − η(p − y), then softmax. */
  const logitStep = (p, correct, eta) => MM.softmax(p.map((pi, i) => Math.log(pi) - eta * (pi - (i === correct ? 1 : 0))));

  // ------------------------------------------------------------------ toy data for the stage (TOY NUMBERS, computed)
  const SENT_IN = ['the', 'cat', 'sat', 'on', 'the'];
  const SENT_OUT = ['cat', 'sat', 'on', 'the', 'mat'];
  // probability the toy model gives to the correct next token at each position (hand-picked)
  const TOY_PC = [0.06, 0.22, 0.48, 0.7, 0.12];
  // the remaining probability is spread over five other candidates in this fixed shape (sums to 1)
  const ALT_SHAPE = [0.4, 0.25, 0.17, 0.11, 0.07];
  const toyDist = (pc) => [pc, ...ALT_SHAPE.map((s) => s * (1 - pc))];
  const ALT_WORDS = ['floor', 'sofa', 'bed', 'rug', 'roof']; // other candidates at the last position
  const P_BEFORE = TOY_PC.slice();
  const P_AFTER = TOY_PC.map((pc) => logitStep(toyDist(pc), 0, 1)[0]); // one real step, η = 1, per position
  const meanLoss = (ps) => ps.reduce((s, p) => s - Math.log(p), 0) / ps.length;
  const LOSS_BEFORE = meanLoss(P_BEFORE);
  const LOSS_AFTER = meanLoss(P_AFTER);
  // Size of the gradient at a position's logits, |p − y|. With this fixed shape for the
  // other candidates it is exactly (1 − p_correct)·√(1 + Σ shape²): worse guesses send back more.
  const ALT_NORM = Math.sqrt(1 + ALT_SHAPE.reduce((s, a) => s + a * a, 0));
  const blameOf = (pc) => (1 - pc) * ALT_NORM;

  // Chinchilla fit (Hoffmann et al. 2022, approach 3): L(N, D) = E + A/N^α + B/D^β, in nats per token.
  const CHIN = { E: 1.69, A: 406.4, B: 410.7, alpha: 0.34, beta: 0.28 };
  const chin = (N, Dt) => CHIN.E + CHIN.A / Math.pow(N, CHIN.alpha) + CHIN.B / Math.pow(Dt, CHIN.beta);

  // ------------------------------------------------------------------ chapter CSS
  AM.css(`
    #ch-${ID} .tr-badges { display: inline-flex; flex-wrap: wrap; gap: 6px; }
    #ch-${ID} .tr-stage .fig { gap: var(--space-2); }
    #ch-${ID} .tr-cap-phone { display: none; }
    @media (max-width: 900px) {
      #ch-${ID} .tr-cap-desk { display: none; }
      #ch-${ID} .tr-cap-phone { display: block; }
      #ch-${ID} .tr-stage .fig-title { font-size: 10px; }
    }
    #ch-${ID} .ctl-value, #ch-${ID} .tr-nocase { text-transform: none; }
    #ch-${ID} .step .math.block { font-size: 0.8em; }
    #ch-${ID} .tr-sec { display: grid; gap: var(--space-5); }
    #ch-${ID} .tr-sec > .prose { max-width: 68ch; }
    #ch-${ID} .tr-kicker { font-family: var(--font-mono); font-size: var(--fs-micro); letter-spacing: 0.16em; text-transform: uppercase; color: var(--weld); }

    /* cross-entropy playground */
    #ch-${ID} .tr-ce-head { display: flex; flex-wrap: wrap; align-items: end; justify-content: space-between; gap: var(--space-3) var(--space-5); }
    #ch-${ID} .tr-prompt { font-family: var(--font-display); font-style: italic; font-size: clamp(1.35rem, 1.05rem + 1.1vw, 1.95rem); line-height: 1.15; color: var(--linen); }
    #ch-${ID} .tr-prompt .tr-blank { display: inline-block; min-width: 2.6em; margin-left: 0.2em; padding: 0 0.15em; border-bottom: 2px solid var(--weld); color: var(--weld); font-style: normal; text-align: center; }
    #ch-${ID} .tr-ce-grid { display: grid; grid-template-columns: minmax(0, 1.12fr) minmax(0, 1fr); gap: var(--space-5); align-items: start; }
    #ch-${ID} .tr-bars canvas { touch-action: none; cursor: ns-resize; }
    #ch-${ID} .tr-ce-read { display: flex; flex-wrap: wrap; align-items: baseline; gap: 6px 18px; padding: var(--space-3) var(--space-4); border: 1px solid var(--rule); border-radius: var(--radius-sm); background: var(--ink); }
    #ch-${ID} .tr-ce-read .tr-lab { font-family: var(--font-mono); font-size: var(--fs-micro); letter-spacing: 0.14em; text-transform: uppercase; color: var(--mist); }
    #ch-${ID} .tr-big { font-family: var(--font-display); font-size: 2.4rem; line-height: 1; color: var(--linen); font-variant-numeric: lining-nums tabular-nums; min-width: 3.2ch; }
    #ch-${ID} .tr-big small { font-family: var(--font-mono); font-size: 11px; color: var(--mist); margin-left: 6px; letter-spacing: 0.06em; }
    #ch-${ID} .tr-formula { font-family: var(--font-mono); font-size: 12px; line-height: 1.7; color: var(--linen-dim); }
    #ch-${ID} .tr-formula b { color: var(--weld); font-weight: 500; }
    #ch-${ID} .tr-formula .tr-dim { color: var(--mist); }
    #ch-${ID} .tr-ctl-rows { display: grid; gap: var(--space-4); }
    #ch-${ID} .tr-btns { display: flex; flex-wrap: wrap; gap: var(--space-2); }
    #ch-${ID} .tr-ce-panel .seg button { text-transform: none; letter-spacing: 0.02em; font-size: 12px; }
    @media (max-width: 760px) {
      #ch-${ID} .tr-ce-grid { grid-template-columns: minmax(0, 1fr); gap: var(--space-3); }
      #ch-${ID} .tr-big { font-size: 2rem; }
      #ch-${ID} .tr-ce-panel .seg button { padding: 5px 8px; font-size: 11px; letter-spacing: 0; }
    }

    /* loss landscape */
    #ch-${ID} .tr-land canvas { cursor: crosshair; }
    #ch-${ID} .tr-land-panel { padding: var(--space-4) var(--space-4) var(--space-3); background: radial-gradient(120% 90% at 50% 40%, color-mix(in srgb, var(--ink-3) 70%, var(--ink-2)) 0%, var(--ink-2) 60%, var(--ink) 100%); }
    #ch-${ID} .tr-land-ctl { display: flex; flex-wrap: wrap; align-items: center; gap: var(--space-3) var(--space-5); }
    #ch-${ID} .tr-land-ctl .ctl-range { width: min(300px, 100%); }
    #ch-${ID} .tr-readout { display: flex; flex-wrap: wrap; align-items: center; gap: 4px 14px; min-height: 1.8em; font-family: var(--font-mono); font-size: 11.5px; color: var(--mist); font-variant-numeric: tabular-nums; }
    #ch-${ID} .tr-readout .tr-dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 6px; vertical-align: -1px; box-shadow: 0 0 8px currentColor; background: currentColor; }
    #ch-${ID} .tr-readout b { color: var(--linen); font-weight: 500; }
    #ch-${ID} .tr-hint { font-family: var(--font-mono); font-size: var(--fs-micro); letter-spacing: 0.06em; color: var(--mist); }
    #ch-${ID} .tr-hint b { color: var(--linen); font-weight: 500; }
    #ch-${ID} .tr-note { display: grid; gap: var(--space-2); align-content: start; padding: var(--space-4); border-left: 2px solid var(--c, var(--weld)); background: linear-gradient(90deg, color-mix(in srgb, var(--c, var(--weld)) 7%, transparent), transparent 70%); border-radius: 0 var(--radius-sm) var(--radius-sm) 0; }
    #ch-${ID} .tr-note h4 { margin: 0; font-family: var(--font-mono); font-size: var(--fs-micro); font-weight: 500; letter-spacing: 0.14em; text-transform: uppercase; color: var(--c, var(--weld)); }
    #ch-${ID} .tr-note p { color: var(--linen-dim); font-size: 0.95rem; line-height: 1.55; }
    #ch-${ID} .tr-handoff { max-width: var(--prose); }
  `);

  // ==================================================================== 1. the training-loop stage
  const STEP_STATES = [
    { all: 0, loss: 0, back: 0, upd: 0, chart: 0 },
    { all: 1, loss: 0, back: 0, upd: 0, chart: 0 },
    { all: 1, loss: 1, back: 0, upd: 0, chart: 0 },
    { all: 1, loss: 1, back: 1, upd: 0, chart: 0 },
    { all: 1, loss: 1, back: 0, upd: 1, chart: 0 },
    { all: 1, loss: 1, back: 0, upd: 1, chart: 1 },
  ];

  function buildLoop(ctx, host) {
    const N = SENT_IN.length, LAYERS = 4;
    const cv = ctx.canvas(host, {
      height: (w) => (isStacked() ? Math.min(w * 0.98, window.innerHeight * 0.5) : Math.min(w * 1.0, window.innerHeight * 0.7)),
      label: 'A tiny language model drawn as a loom. Five input tokens sit at the bottom, three layers of nodes are joined by threads that only run from earlier positions to later ones, and above each position is the next token it must predict. Later steps show the loss at each position, the error flowing backwards, and a scaling-law chart.',
    });
    const g = cv.g;
    const vis = visibility(cv.wrap);
    const rnd = MM.rng(23);

    // Illustrative mixing weights: for layer l and position i, a row over j ≤ i that sums to 1.
    const W = [];
    for (let l = 0; l < LAYERS - 1; l++) {
      W.push([]);
      for (let i = 0; i < N; i++) {
        const row = [];
        for (let j = 0; j <= i; j++) row.push(0.2 + rnd() * (j === i ? 1.3 : 1));
        const s = row.reduce((a, b) => a + b, 0);
        W[l].push(row.map((v) => v / s));
      }
    }
    const threads = [];
    const outs = [], ins = [];
    for (let l = 0; l < LAYERS; l++) { outs.push(Array.from({ length: N }, () => [])); ins.push(Array.from({ length: N }, () => [])); }
    for (let l = 0; l < LAYERS - 1; l++) for (let i = 0; i < N; i++) for (let j = 0; j <= i; j++) {
      const th = { l, i, j, w: W[l][i][j], B: null, len: 1, seed: rnd() };
      threads.push(th);
      outs[l][j].push(th);
      ins[l + 1][i].push(th);
    }
    const inSeg = Array.from({ length: N }, () => ({ B: null, len: 1 }));
    const outSeg = Array.from({ length: N }, () => ({ B: null, len: 1 }));

    const st = { ...STEP_STATES[0], flash: 0 };
    let target = STEP_STATES[0];
    let step = 0;
    let L = null;
    let lastT = 0;
    const parts = [];
    let spawnF = 0, spawnB = 0;

    function layout(w, h) {
      const small = w < 520;
      const fs = small ? 12 : 14.5;
      const padX = small ? 6 : Math.max(16, w * 0.045);
      const colW = (w - 2 * padX) / N;
      const xs = Array.from({ length: N }, (_, i) => padX + colW * (i + 0.5));
      const tileH = fs + 2 * Math.round(fs * 0.42);
      const yLabel = small ? 10 : 13;
      const yTgt = yLabel + 12 + tileH / 2;
      const yIn = h - tileH / 2 - (small ? 4 : 8);
      const yLoss0 = yTgt + tileH / 2 + (small ? 11 : 13); // p label
      const yDrip = yLoss0 + (small ? 9 : 11);              // drips start here
      const yTop = h * (small ? 0.5 : 0.49);
      const yBot = yIn - tileH / 2 - (small ? 16 : 24);
      const ys = Array.from({ length: LAYERS }, (_, l) => MM.lerp(yBot, yTop, l / (LAYERS - 1)));
      const lossScale = (yTop - (small ? 22 : 30) - yDrip) / 2.9; // px per nat
      const yDist = yTop - (small ? 14 : 22);                       // baseline of the mini distributions
      const distH = (yDist - yDrip) * 0.78;
      for (const th of threads) {
        const x0 = xs[th.j], y0 = ys[th.l], x1 = xs[th.i], y1 = ys[th.l + 1], dy = y0 - y1;
        th.B = th.i === th.j ? line(x0, y0, x1, y1) : [x0, y0, x0, y0 - dy * 0.62, x1, y1 + dy * 0.62, x1, y1];
        th.len = bezLen(th.B);
      }
      for (let i = 0; i < N; i++) {
        inSeg[i].B = line(xs[i], yIn - tileH / 2, xs[i], ys[0]); inSeg[i].len = yIn - tileH / 2 - ys[0];
        outSeg[i].B = line(xs[i], ys[LAYERS - 1], xs[i], yTgt + tileH / 2 + 1); outSeg[i].len = ys[LAYERS - 1] - yTgt - tileH / 2;
      }
      return { w, h, small, fs, padX, colW, xs, tileH, yLabel, yTgt, yIn, yLoss0, yDrip, ys, yTop, lossScale, yDist, distH };
    }

    // ---------------------------------------------------------------- particles
    const pickWeighted = (arr, wf) => {
      let s = 0; for (const a of arr) s += wf(a);
      let r = Math.random() * s;
      for (const a of arr) { r -= wf(a); if (r <= 0) return a; }
      return arr[arr.length - 1];
    };
    function spawnForward() {
      const j = Math.floor(Math.random() * N);
      parts.push({ mode: 'f', seg: inSeg[j], kind: 'in', t: 0, node: j, r: 4 + Math.random() * 2.5, a: 0.55 + Math.random() * 0.4 });
    }
    function spawnBackward(ps) {
      // pulses leave each position in proportion to the size of its logit gradient |p − y|
      const i = pickWeighted(MM.range(N), (k) => blameOf(ps[k]));
      const loss = -Math.log(ps[i]), blame = blameOf(ps[i]);
      // start at the bottom of this position's drip, travelling down to the top node
      const dropY = L.yDrip + loss * L.lossScale * st.loss;
      const seg = outSeg[i];
      const y0 = seg.B[1], y1 = seg.B[7];
      const t0 = MM.clamp((y0 - dropY) / (y0 - y1), 0, 1);
      parts.push({ mode: 'b', seg, kind: 'out', t: 1 - t0, node: i, r: 3.5 + 4 * blame, a: 0.45 + 0.5 * blame });
    }
    function advance(p) {
      // called when a particle reaches the end of its current segment
      if (p.mode === 'f') {
        if (p.kind === 'out') return false;
        const lNow = p.kind === 'in' ? 0 : p.seg.l + 1;
        const node = p.kind === 'in' ? p.node : p.seg.i;
        if (lNow >= LAYERS - 1) { p.kind = 'out'; p.seg = outSeg[node]; p.t = 0; p.node = node; return true; }
        let opts = outs[lNow][node];
        if (st.all < 0.5) opts = opts.filter((th) => th.i === N - 1);
        if (!opts.length) return false;
        p.seg = pickWeighted(opts, (th) => th.w); p.kind = 'th'; p.t = 0;
        return true;
      }
      // backward: walk down
      if (p.kind === 'in') return false;
      const lNow = p.kind === 'out' ? LAYERS - 1 : p.seg.l;
      const node = p.kind === 'out' ? p.node : p.seg.j;
      if (lNow <= 0) { p.kind = 'in'; p.seg = inSeg[node]; p.t = 0; return true; }
      p.seg = pickWeighted(ins[lNow][node], (th) => th.w); p.kind = 'th'; p.t = 0;
      return true;
    }
    function partPos(p) {
      // forward runs each segment start→end; backward runs it end→start
      const tt = p.mode === 'f' ? p.t : 1 - p.t;
      return bez(p.seg.B, tt);
    }

    // ---------------------------------------------------------------- drawing
    function curPs() { return P_BEFORE.map((pb, i) => MM.lerp(pb, P_AFTER[i], st.upd)); }

    function drawWeave(t, A) {
      const { w, xs, ys, fs, tileH, small } = L;
      const ps = curPs();
      g.save();
      g.globalAlpha = A;
      const fwdCol = AM.dye.weld, backCol = AM.dye.madder;
      const flash = st.flash;

      // threads
      g.lineCap = 'round';
      for (const th of threads) {
        const focus = th.i === N - 1 ? 1 : MM.lerp(0.1, 1, st.all);
        const shimmer = flash > 0.01 ? 0.5 + 0.5 * Math.sin(t * 11 + th.seed * 40) : 0;
        const a = (0.2 + 0.7 * th.w) * focus * (1 + flash * shimmer * 0.9);
        const wd = 0.7 + 2.6 * th.w + flash * shimmer * 1.2;
        const B = th.B;
        let BB = B;
        if (flash > 0.01 && th.i !== th.j) {
          // nudged weights: the thread wobbles while it settles
          const off = Math.sin(t * 7 + th.seed * 20) * flash * 5;
          BB = [B[0], B[1], B[2] + off, B[3], B[4] - off, B[5], B[6], B[7]];
        }
        // gold going forward, madder while blame flows back, a linen flicker while weights are nudged
        const F = rgb(fwdCol), Bk = rgb(backCol), Ln = rgb(AM.col.linen), fl = flash * shimmer * 0.7;
        const c = [0, 1, 2].map((q) => { const v = F[q] + (Bk[q] - F[q]) * st.back; return Math.round(v + (Ln[q] - v) * fl); });
        g.strokeStyle = `rgba(${c[0]},${c[1]},${c[2]},${Math.min(1, a * 0.16)})`;
        g.lineWidth = wd * 4;
        strokeBez(g, BB);
        g.strokeStyle = `rgba(${c[0]},${c[1]},${c[2]},${Math.min(1, a)})`;
        g.lineWidth = wd;
        strokeBez(g, BB);
      }

      // input warp (tile to first node) and output warp (top node to target)
      g.setLineDash([2, 4]);
      g.lineWidth = 1;
      for (let i = 0; i < N; i++) {
        const focus = i === N - 1 ? 1 : MM.lerp(0.15, 1, st.all);
        g.strokeStyle = AM.rgba(AM.col.linen, 0.25 * focus);
        strokeBez(g, inSeg[i].B);
        g.strokeStyle = AM.rgba(AM.dye.weld, 0.28 * focus);
        strokeBez(g, outSeg[i].B);
      }
      g.setLineDash([]);

      // nodes
      const nr = small ? 3.4 : 4.6;
      for (let l = 0; l < LAYERS; l++) for (let i = 0; i < N; i++) {
        const focus = i === N - 1 ? 1 : MM.lerp(0.2, 1, st.all);
        g.beginPath(); g.arc(xs[i], ys[l], nr, 0, TAU);
        g.fillStyle = AM.col.ink2; g.fill();
        g.lineWidth = 1.4;
        g.strokeStyle = mixA(AM.dye.weld, AM.dye.madder, st.back, 0.85 * focus);
        g.stroke();
      }

      // input tiles
      for (let i = 0; i < N; i++) {
        const focus = MM.lerp(0.75, 1, st.all);
        D.token(g, SENT_IN[i], xs[i], L.yIn, { size: fs, alpha: focus, padX: small ? 6 : 10 });
      }
      // target tiles
      for (let i = 0; i < N; i++) {
        const a = i === N - 1 ? 1 : st.all;
        if (a < 0.02) {
          D.token(g, '?', xs[i], L.yTgt, { size: fs, alpha: 0.35, stroke: AM.col.rule, color: AM.col.mist, padX: small ? 6 : 10 });
          continue;
        }
        D.token(g, SENT_OUT[i], xs[i], L.yTgt, {
          size: fs, alpha: Math.max(0.35, a), padX: small ? 6 : 10,
          stroke: AM.rgba(AM.dye.weld, 0.75), fill: AM.mix(AM.col.ink2, AM.dye.weld, 0.1),
        });
      }

      // step 1: the prediction at the last position is a full distribution (six candidates shown)
      const listA = (1 - st.all) * (1 - st.loss);
      if (listA > 0.02) {
        const dist = toyDist(ps[N - 1]).map((p, k) => ({ p, word: k ? ALT_WORDS[k - 1] : SENT_OUT[N - 1], hit: k === 0 }));
        dist.sort((a, b) => b.p - a.p);
        const rowH = small ? 15 : 21;
        const top = L.yDrip + (small ? 10 : 22);
        const labX = L.padX + (small ? 40 : 64), barX = labX + 8, barMax = xs[N - 2] + L.colW * 0.2 - barX;
        micro(g, small ? 'p(next token)' : 'p(next token | the cat sat on the)', L.padX, top - (small ? 11 : 15), { alpha: listA, size: small ? 8.5 : 9.5, ls: '0.04em', color: AM.col.linenDim });
        dist.forEach((d, k) => {
          const y = top + k * rowH + rowH / 2;
          const col = d.hit ? AM.dye.weld : AM.dye.woad;
          D.text(g, d.word, labX, y, { size: small ? 11 : 13, align: 'right', baseline: 'middle', alpha: listA, weight: d.hit ? 700 : 500, color: d.hit ? AM.dye.weld : AM.col.linenDim });
          const bl = Math.max(2, d.p * barMax * 1.6);
          g.fillStyle = AM.rgba(col, (d.hit ? 0.9 : 0.55) * listA);
          D.roundRect(g, barX, y - (small ? 3 : 4), bl, small ? 6 : 8, 3); g.fill();
          micro(g, d.p.toFixed(2), barX + bl + 6, y, { alpha: listA, size: small ? 8.5 : 9.5, ls: '0.02em', color: d.hit ? AM.dye.weld : AM.col.mist });
          if (d.hit) {
            // a gold thread from the right answer's bar up to the target tile it is scored against
            const x0 = barX + bl + (small ? 34 : 44), x1 = xs[N - 1] - 4, y1 = L.yTgt + L.tileH / 2 + 2;
            g.strokeStyle = AM.rgba(AM.dye.weld, 0.55 * listA); g.lineWidth = 1.2; g.setLineDash([3, 4]);
            g.beginPath(); g.moveTo(x0, y); g.bezierCurveTo(x1 - 30, y, x1, y + 10, x1, y1); g.stroke(); g.setLineDash([]);
          }
        });
      }
      // step 2: every position predicts a distribution (gold bar = probability of the true next token)
      const histA = st.all * (1 - st.loss);
      if (histA > 0.02) {
        for (let i = 0; i < N; i++) {
          const dist = toyDist(ps[i]).map((p, k) => ({ p, hit: k === 0 })).sort((a, b) => b.p - a.p);
          const bw = Math.max(2.5, L.colW * 0.07), gap = Math.max(1.5, L.colW * 0.025);
          const x0 = xs[i] - (dist.length * bw + (dist.length - 1) * gap) / 2;
          dist.forEach((d, k) => {
            const hh = Math.max(1.5, d.p * L.distH);
            g.fillStyle = AM.rgba(d.hit ? AM.dye.weld : AM.dye.woad, (d.hit ? 0.95 : 0.55) * histA);
            g.fillRect(x0 + k * (bw + gap), L.yDist - hh, bw, hh);
          });
          g.fillStyle = AM.rgba(AM.col.linen, 0.18 * histA);
          g.fillRect(x0 - 2, L.yDist, dist.length * (bw + gap) + 2, 1);
        }
      }

      // loss drips: −ln p hanging below each target
      if (st.loss > 0.01) {
        const la = st.loss;
        for (let i = 0; i < N; i++) {
          const p = ps[i], loss = -Math.log(p);
          const x = xs[i], y0 = L.yDrip, y1 = y0 + loss * L.lossScale * la;
          micro(g, 'p ' + p.toFixed(2), x, L.yLoss0, { alpha: la, color: AM.col.linenDim, size: small ? 8.5 : 9.5, ls: '0.04em' });
          g.strokeStyle = AM.rgba(AM.dye.madder, 0.22 * la); g.lineWidth = 6;
          g.beginPath(); g.moveTo(x, y0); g.lineTo(x, y1); g.stroke();
          g.strokeStyle = AM.rgba(AM.dye.madder, 0.9 * la); g.lineWidth = 1.8;
          g.beginPath(); g.moveTo(x, y0); g.lineTo(x, y1); g.stroke();
          // the drop: a teardrop whose size grows with the loss
          const r = (small ? 2.4 : 3) + Math.sqrt(loss) * (small ? 1.6 : 2.1);
          g.save(); g.globalAlpha *= la;
          blob(g, AM.dye.madder, x, y1 + r * 0.4, r * 3.2, 0.55);
          g.fillStyle = AM.dye.madder;
          g.beginPath();
          g.moveTo(x, y1 - r * 1.4);
          g.bezierCurveTo(x + r * 0.2, y1 - r * 0.6, x + r, y1, x + r, y1 + r * 0.4);
          g.arc(x, y1 + r * 0.4, r, 0, Math.PI);
          g.bezierCurveTo(x - r, y1, x - r * 0.2, y1 - r * 0.6, x, y1 - r * 1.4);
          g.fill();
          g.restore();
          micro(g, loss.toFixed(2), x + r + 5, y1 + r * 0.4, { alpha: la, color: AM.col.linen, size: small ? 8.5 : 10, ls: '0.02em' });
        }
        // the mean: one horizontal thread at the average drip length
        const mean = meanLoss(ps);
        const ym = L.yDrip + mean * L.lossScale * la;
        g.setLineDash([5, 5]);
        g.strokeStyle = AM.rgba(AM.dye.madder, 0.55 * la); g.lineWidth = 1;
        g.beginPath(); g.moveTo(xs[0] - L.colW * 0.46, ym); g.lineTo(xs[N - 1] + L.colW * 0.46, ym); g.stroke();
        g.setLineDash([]);
        if (!small) micro(g, 'MEAN', xs[0] - L.colW * 0.46, ym - 8, { alpha: la * 0.9, color: AM.dye.madder, size: 8.5 });
      }

      // header row
      micro(g, small ? 'TARGETS' : 'TARGET = NEXT TOKEN', L.padX, L.yLabel, { color: AM.col.mist });
      let right = '', rightCol = AM.col.linenDim;
      if (step === 0) right = 'FORWARD PASS';
      else if (step === 1) right = small ? '5 PREDICTIONS' : 'FORWARD PASS · 5 PREDICTIONS';
      else if (step === 2) { right = `MEAN LOSS ${LOSS_BEFORE.toFixed(2)}`; rightCol = AM.dye.madder; }
      else if (step === 3) { right = 'BACKWARD PASS · ∂L/∂θ'; rightCol = AM.dye.madder; }
      else { right = `MEAN LOSS ${LOSS_BEFORE.toFixed(2)} → ${meanLoss(ps).toFixed(2)}`; rightCol = AM.dye.weld; }
      micro(g, right, w - L.padX, L.yLabel, { align: 'right', color: rightCol });

      // particles
      g.globalCompositeOperation = 'lighter';
      for (const p of parts) {
        const q = partPos(p);
        const fade = p.kind === 'out' && p.mode === 'f' ? 1 - p.t : p.kind === 'in' && p.mode === 'b' ? 1 - p.t : 1;
        const col = p.mode === 'f' ? AM.dye.weld : AM.dye.madder;
        blob(g, col, q.x, q.y, p.r, p.a * fade * A);
      }
      g.globalCompositeOperation = 'source-over';
      g.restore();
    }

    function drawChart(t, A) {
      const { w, h, small } = L;
      const m = { l: small ? 34 : 46, r: small ? 46 : 70, t: small ? 40 : 46, b: small ? 40 : 50 };
      const LX0 = 9, LX1 = 13.3, LY0 = 1.6, LY1 = 3.8;
      const X = (lg) => m.l + ((lg - LX0) / (LX1 - LX0)) * (w - m.l - m.r);
      const Y = (v) => m.t + ((LY1 - v) / (LY1 - LY0)) * (h - m.t - m.b);
      g.save();
      g.globalAlpha = A;
      micro(g, 'LOSS (NATS PER TOKEN)', L.padX, L.yLabel, { color: AM.col.mist });
      micro(g, 'CHINCHILLA FIT', w - L.padX, L.yLabel, { color: AM.col.mist, align: 'right' });
      // grid
      g.lineWidth = 1;
      const xt = [[9, '1B'], [10, '10B'], [11, '100B'], [12, '1T'], [13, '10T']];
      for (const [lg, lab] of xt) {
        g.strokeStyle = AM.rgba(AM.col.linen, 0.06);
        g.beginPath(); g.moveTo(X(lg), m.t); g.lineTo(X(lg), h - m.b); g.stroke();
        micro(g, lab, X(lg), h - m.b + 13, { align: 'center', size: small ? 8.5 : 9.5, ls: '0.04em' });
      }
      for (const v of [2, 2.5, 3, 3.5]) {
        g.strokeStyle = AM.rgba(AM.col.linen, 0.06);
        g.beginPath(); g.moveTo(m.l, Y(v)); g.lineTo(w - m.r, Y(v)); g.stroke();
        micro(g, v.toFixed(1), m.l - 6, Y(v), { align: 'right', size: small ? 8.5 : 9.5, ls: '0.02em' });
      }
      micro(g, 'training tokens (log scale)', (m.l + w - m.r) / 2, h - m.b + 30, { align: 'center', size: small ? 8.5 : 9.5, ls: '0.04em', color: AM.col.linenDim });

      // fitted floor E
      g.setLineDash([3, 5]);
      g.strokeStyle = AM.rgba(AM.col.linen, 0.3);
      g.beginPath(); g.moveTo(m.l, Y(CHIN.E)); g.lineTo(w - m.r, Y(CHIN.E)); g.stroke();
      g.setLineDash([]);
      micro(g, `fitted floor E = ${CHIN.E}`, m.l + 6, Y(CHIN.E) - 8, { size: small ? 8.5 : 9.5, ls: '0.04em' });
      // markers: real token counts
      // labels on two rows so they never collide
      const marks = [[Math.log10(3e11), 'GPT-3 · 300B', 'left', m.t - 14], [Math.log10(1.5e13), small ? 'Llama 3 · 15T' : 'Llama 3 · 15T (reported)', 'right', m.t]];
      for (const [lg, lab, side, ly] of marks) {
        const x = X(lg);
        g.setLineDash([2, 4]);
        g.strokeStyle = AM.rgba(AM.dye.weld, 0.45);
        g.beginPath(); g.moveTo(x, ly - 5); g.lineTo(x, h - m.b); g.stroke();
        g.setLineDash([]);
        micro(g, lab, side === 'left' ? x + 5 : x - 5, ly, { align: side === 'left' ? 'left' : 'right', color: AM.dye.weld, size: small ? 8.5 : 9.5, ls: '0.04em' });
      }
      // curves, one per model size, each with a bead travelling along it
      const sizes = [[1e8, '100M', AM.dye.lichen], [1e9, '1B', AM.dye.woad], [1e10, '10B', AM.dye.verdigris], [1e11, '100B', AM.dye.weld]];
      const labelYs = [];
      sizes.forEach(([Np, lab, col], k) => {
        const pts = [];
        for (let s = 0; s <= 90; s++) { const lg = LX0 + (LX1 - LX0) * (s / 90); pts.push([X(lg), Y(chin(Np, Math.pow(10, lg)))]); }
        // full curve, faint
        g.strokeStyle = AM.rgba(col, 0.28); g.lineWidth = 1.2;
        g.beginPath(); pts.forEach(([x, y], s) => (s ? g.lineTo(x, y) : g.moveTo(x, y))); g.stroke();
        // training progress: a glowing trail up to the bead
        const period = 7.5;
        const u = AM.reducedMotion ? 1 : MM.clamp((((t + k * 0.9) % period) / period) * 1.15, 0, 1);
        const nU = Math.max(1, Math.round(u * 90));
        g.lineCap = 'round';
        g.strokeStyle = AM.rgba(col, 0.2); g.lineWidth = 6;
        g.beginPath(); for (let s = 0; s <= nU; s++) (s ? g.lineTo(pts[s][0], pts[s][1]) : g.moveTo(pts[s][0], pts[s][1])); g.stroke();
        g.strokeStyle = AM.rgba(col, 0.95); g.lineWidth = 2;
        g.beginPath(); for (let s = 0; s <= nU; s++) (s ? g.lineTo(pts[s][0], pts[s][1]) : g.moveTo(pts[s][0], pts[s][1])); g.stroke();
        g.globalCompositeOperation = 'lighter';
        blob(g, col, pts[nU][0], pts[nU][1], 11, 0.95 * A);
        g.globalCompositeOperation = 'source-over';
        labelYs.push({ y: pts[90][1], lab, col });
      });
      // end labels, nudged apart so they never overlap
      labelYs.sort((a, b) => a.y - b.y);
      for (let k = 1; k < labelYs.length; k++) labelYs[k].y = Math.max(labelYs[k].y, labelYs[k - 1].y + 12);
      for (const it of labelYs) micro(g, it.lab, w - m.r + 7, it.y, { color: it.col, size: small ? 8.5 : 9.5, ls: '0.04em' });
      micro(g, 'params', w - m.r + 7, labelYs[labelYs.length - 1].y + 12, { size: 8.5, ls: '0.04em' });
      g.restore();
    }

    function draw(t) {
      if (!L) return;
      cv.clear();
      const wa = 1 - st.chart;
      if (wa > 0.01) drawWeave(t, wa);
      if (st.chart > 0.01) drawChart(t, st.chart);
    }

    function tick(t, dt) {
      // ease the scene toward the active step
      for (const k of ['all', 'loss', 'back', 'chart']) st[k] += (target[k] - st[k]) * (1 - Math.exp(-dt * 4.5));
      st.upd += (target.upd - st.upd) * (1 - Math.exp(-dt * 3.6));
      st.flash = Math.max(0, st.flash - dt * 0.45);
      if (st.chart > 0.98) { parts.length = 0; frozenFor = -1; return; }
      const ps = curPs();
      if (AM.reducedMotion) { if (frozenFor !== step) freezeSnapshot(ps); return; }
      frozenFor = -1;
      // spawn: gold forward signals rise; in the backward step madder blame falls
      const fRate = step === 3 ? 6 : 34, bRate = step === 3 ? 48 : 0;
      spawnF += dt * fRate; spawnB += dt * bRate;
      while (spawnF >= 1) { spawnF -= 1; if (parts.length < 200) spawnForward(); }
      while (spawnB >= 1) { spawnB -= 1; if (parts.length < 200) spawnBackward(ps); }
      const speed = L.small ? 70 : 105; // px per second
      for (let k = parts.length - 1; k >= 0; k--) {
        const p = parts[k];
        p.t += (dt * speed) / Math.max(8, p.seg.len);
        if (p.t >= 1) { if (!advance(p)) parts.splice(k, 1); }
      }
    }

    /** Reduced motion: a still snapshot of signals caught mid-flight, instead of moving ones. */
    let frozenFor = -1;
    function freezeSnapshot(ps) {
      parts.length = 0;
      const nF = step === 3 ? 14 : 60, nB = step === 3 ? 70 : 0;
      for (let k = 0; k < nF + nB; k++) {
        if (k < nF) spawnForward(); else spawnBackward(ps);
        const p = parts[parts.length - 1];
        let dist = Math.random() * 700; // carry it a random distance along its route
        while (dist > 0) {
          const len = Math.max(8, p.seg.len), left = (1 - p.t) * len;
          if (dist < left) { p.t += dist / len; break; }
          dist -= left; p.t = 1;
          if (!advance(p)) { parts.pop(); break; }
        }
      }
      frozenFor = step;
    }

    cv.onResize((w, h) => { L = layout(w, h); frozenFor = -1; draw(lastT); });
    ctx.loop((t, dt) => { if (!vis.on || !L) return; lastT = t; tick(t, dt); draw(t); });

    return {
      setStep(i) {
        if (i === 4 && step !== 4 && !AM.reducedMotion) st.flash = 1;
        step = i;
        target = STEP_STATES[i] || STEP_STATES[0];
        if (AM.reducedMotion) Object.assign(st, target);
        if (L) draw(lastT);
      },
    };
  }

  // ==================================================================== 2. cross-entropy playground
  function buildCE(ctx, hosts) {
    // the same six candidates and starting guess as the last position of the stage above
    const WORDS = [SENT_OUT[SENT_OUT.length - 1], ...ALT_WORDS];
    const P0 = toyDist(TOY_PC[TOY_PC.length - 1]);
    const S = { p: P0.slice(), disp: P0.slice(), correct: 0, drag: -1 };
    const PMIN = 0.004;

    /** Set word k to probability v and rescale the others so everything sums to 1. */
    function setP(k, v) {
      v = MM.clamp(v, PMIN, 1 - PMIN * (WORDS.length - 1));
      const rest = 1 - S.p[k];
      const p = S.p.slice();
      for (let i = 0; i < p.length; i++) {
        if (i === k) continue;
        p[i] = rest > 1e-9 ? (p[i] * (1 - v)) / rest : (1 - v) / (p.length - 1);
      }
      p[k] = v;
      // keep every bar above the floor, then renormalise exactly
      for (let i = 0; i < p.length; i++) p[i] = Math.max(PMIN * 0.5, p[i]);
      const s = p.reduce((a, b) => a + b, 0);
      S.p = p.map((x) => x / s);
    }

    const bars = ctx.canvas(hosts.bars, {
      height: (w) => (w < 440 ? 236 : 292),
      label: 'Bar chart of a predicted probability for six candidate next words. Drag a bar to change it; the others rescale so the total stays 1. Arrows show which way one gradient step pushes each word\'s logit, and dashed ticks show each probability after that step.',
    });
    const curve = ctx.canvas(hosts.curve, {
      height: (w) => (w < 440 ? 220 : 292),
      label: 'The curve loss = −ln p for p from 0 to 1, with a bead at the probability currently given to the correct word.',
    });
    // gate on the whole panel: on phones the buttons and readout can be on screen while the bars are not
    const vis = visibility(hosts.read.closest('.panel') || hosts.bars);

    // ---------------------------------------------------------------- bars
    let BL = null;
    function barLayout(w, h) {
      const small = w < 440;
      const pad = small ? 4 : 10;
      const colW = (w - 2 * pad) / WORDS.length;
      return { w, h, small, pad, colW, bw: Math.min(small ? 26 : 40, colW * 0.42), yTop: small ? 30 : 34, yBase: h - (small ? 46 : 52) };
    }
    function drawBars() {
      if (!BL) return;
      const g = bars.g, { w, h, small, pad, colW, bw, yTop, yBase } = BL;
      bars.clear();
      const span = yBase - yTop;
      // guides
      g.lineWidth = 1;
      for (const v of [0.25, 0.5, 0.75, 1]) {
        g.strokeStyle = AM.rgba(AM.col.linen, v === 1 ? 0.1 : 0.05);
        g.setLineDash(v === 1 ? [3, 4] : []);
        g.beginPath(); g.moveTo(pad, yBase - v * span); g.lineTo(w - pad, yBase - v * span); g.stroke();
      }
      g.setLineDash([]);
      micro(g, 'p = 1', w - pad, yBase - span - 8, { align: 'right', size: 8.5, ls: '0.04em' });
      g.strokeStyle = AM.col.ruleStrong;
      g.beginPath(); g.moveTo(pad, yBase + 0.5); g.lineTo(w - pad, yBase + 0.5); g.stroke();
      // where every bar lands after one gradient step (exact: softmax of z − (p − y))
      const next = logitStep(S.disp, S.correct, 1);

      for (let i = 0; i < WORDS.length; i++) {
        const p = S.disp[i], isC = i === S.correct;
        const cx = pad + colW * (i + 0.5);
        const col = isC ? AM.dye.weld : AM.dye.woad;
        const top = yBase - p * span;
        // body: a gradient column with a few warp hairlines (the bar is woven)
        const grd = g.createLinearGradient(0, top, 0, yBase);
        grd.addColorStop(0, AM.rgba(col, isC ? 0.85 : 0.6));
        grd.addColorStop(1, AM.rgba(col, 0.1));
        g.fillStyle = grd;
        D.roundRect(g, cx - bw / 2, top, bw, Math.max(1, yBase - top), Math.min(4, bw / 4));
        g.fill();
        g.strokeStyle = AM.rgba(AM.col.linen, 0.13);
        g.beginPath();
        for (let k = 1; k < 4; k++) { const xx = cx - bw / 2 + (bw * k) / 4; g.moveTo(xx, top + 3); g.lineTo(xx, yBase); }
        g.stroke();
        // cap + handle bead
        g.strokeStyle = col; g.lineWidth = 2.2;
        g.beginPath(); g.moveTo(cx - bw / 2, top); g.lineTo(cx + bw / 2, top); g.stroke();
        D.glowDot(g, cx, top, S.drag === i ? 4.2 : 3.2, col, 1);
        // ghost tick: this bar after one step
        const yN = yBase - next[i] * span;
        if (Math.abs(yN - top) > 1.5) {
          g.save();
          g.setLineDash([3, 2]);
          g.strokeStyle = AM.rgba(AM.col.linen, 0.75); g.lineWidth = 1.3;
          g.beginPath(); g.moveTo(cx - bw / 2 - 3, yN); g.lineTo(cx + bw / 2 + 3, yN); g.stroke();
          g.restore();
        }
        // percentage (kept clear of a ghost tick that sits just above the bar)
        const labY = (yN < top && top - yN < 22 ? yN : top) - 12;
        micro(g, (p * 100).toFixed(p < 0.095 ? 1 : 0) + '%', cx, labY, { align: 'center', color: AM.col.linen, size: small ? 9 : 10, ls: '0.02em' });
        // which way a gradient step pushes this word's LOGIT: −∂L/∂z = y − p (up for the right word, down for the rest)
        const gz = p - (isC ? 1 : 0);
        const ax = cx + bw / 2 + (small ? 6 : 9);
        const K = span * 0.42;
        if (Math.abs(gz) > 0.006) {
          const len = Math.max(7, Math.abs(gz) * K);
          D.arrow(g, ax, top, ax, gz < 0 ? top - len : top + len, { color: gz < 0 ? AM.dye.verdigris : AM.dye.madder, width: 1.6, head: small ? 5 : 6 });
        }
        // word + gradient value
        D.text(g, WORDS[i], cx, yBase + (small ? 15 : 17), { size: small ? 12 : 13.5, weight: isC ? 700 : 500, align: 'center', baseline: 'middle', color: isC ? AM.dye.weld : AM.col.linenDim });
        micro(g, fmtSigned(gz), cx, yBase + (small ? 33 : 37), { align: 'center', color: gz < 0 ? AM.dye.verdigris : AM.rgba(AM.dye.madder, 1), size: small ? 8.5 : 9.5, ls: '0.02em' });
      }
    }

    // ---------------------------------------------------------------- curve
    let CL = null;
    function drawCurve() {
      if (!CL) return;
      const g = curve.g, { w, h, small } = CL;
      curve.clear();
      const m = { l: small ? 30 : 38, r: small ? 10 : 16, t: small ? 24 : 28, b: small ? 34 : 38 };
      const YMAX = 5;
      const X = (p) => m.l + p * (w - m.l - m.r);
      const Y = (v) => m.t + (1 - v / YMAX) * (h - m.t - m.b);
      // axes + ticks
      g.lineWidth = 1;
      for (const v of [1, 2, 3, 4, 5]) {
        g.strokeStyle = AM.rgba(AM.col.linen, 0.05);
        g.beginPath(); g.moveTo(m.l, Y(v)); g.lineTo(w - m.r, Y(v)); g.stroke();
        micro(g, String(v), m.l - 7, Y(v), { align: 'right', size: 8.5, ls: '0' });
      }
      g.strokeStyle = AM.col.ruleStrong;
      g.beginPath(); g.moveTo(m.l, m.t); g.lineTo(m.l, Y(0)); g.lineTo(w - m.r, Y(0)); g.stroke();
      for (const p of [0, 0.25, 0.5, 0.75, 1]) micro(g, p === 0 ? '0' : p === 1 ? '1' : String(p).replace('0.', '.'), X(p), Y(0) + 12, { align: 'center', size: 8.5, ls: '0' });
      micro(g, 'p(correct word)', (m.l + w - m.r) / 2, Y(0) + 27, { align: 'center', size: 9, ls: '0.04em', color: AM.col.linenDim });
      micro(g, 'loss = −ln p', m.l + 10, m.t - 14, { size: 9, ls: '0.04em', color: AM.col.linenDim });

      // uniform-guess reference
      const lu = Math.log(WORDS.length);
      g.setLineDash([3, 5]);
      g.strokeStyle = AM.rgba(AM.dye.lichen, 0.6);
      g.beginPath(); g.moveTo(m.l, Y(lu)); g.lineTo(w - m.r, Y(lu)); g.stroke();
      g.setLineDash([]);
      const boxes = []; // annotation boxes the bead's label must avoid
      const note = (s, x, y, align, color, size) => {
        micro(g, s, x, y, { align, color, size, ls: '0.03em' });
        const tw = D.measure(g, s, size, 'mono') * 1.04;
        const x0 = align === 'right' ? x - tw : x;
        boxes.push([x0 - 3, y - size * 0.8, x0 + tw + 3, y + size * 0.8]);
      };
      note(`uniform guess: ln 6 = ${lu.toFixed(2)}`, w - m.r, Y(lu) - 8, 'right', AM.dye.lichen, 8.5);

      // the curve, dyed madder (costly) → verdigris (cheap)
      const grd = g.createLinearGradient(X(0), 0, X(1), 0);
      grd.addColorStop(0, AM.dye.madder);
      grd.addColorStop(0.3, AM.dye.saffron);
      grd.addColorStop(0.6, AM.dye.weld);
      grd.addColorStop(1, AM.dye.verdigris);
      const pStart = Math.exp(-YMAX);
      const path = () => {
        g.beginPath();
        for (let s = 0; s <= 160; s++) {
          const p = pStart * Math.pow(1 / pStart, s / 160); // log-spaced so the steep end is smooth
          const x = X(p), y = Y(-Math.log(p));
          s ? g.lineTo(x, y) : g.moveTo(x, y);
        }
      };
      g.lineCap = 'round';
      g.globalAlpha = 0.22; g.strokeStyle = grd; g.lineWidth = 7; path(); g.stroke();
      g.globalAlpha = 1; g.lineWidth = 2.2; path(); g.stroke();

      // annotations
      note('confident and wrong', X(0.06) + 8, Y(4.55), 'left', AM.dye.madder, small ? 8.5 : 9.5);
      note('confident and right', X(1) - 2, Y(0.85), 'right', AM.dye.verdigris, small ? 8.5 : 9.5);

      // bead at the current probability of the correct word
      const pc = S.disp[S.correct], lc = -Math.log(pc);
      const bx = X(pc), by = Y(Math.min(YMAX, lc));
      g.setLineDash([2, 4]);
      g.strokeStyle = AM.rgba(AM.dye.weld, 0.55); g.lineWidth = 1;
      g.beginPath(); g.moveTo(bx, by); g.lineTo(bx, Y(0)); g.moveTo(bx, by); g.lineTo(m.l, by); g.stroke();
      g.setLineDash([]);
      D.glowDot(g, bx, by, 5, AM.dye.weld, 1);
      const lab = lc > YMAX ? `${lc.toFixed(2)} ↑` : lc.toFixed(2);
      // place the value label at the first spot around the bead that clears the annotations
      const fsz = small ? 15 : 18, tw = D.measure(g, lab, fsz, 'display') + 4;
      const spots = [[12, -12, 'left'], [-12, -12, 'right'], [12, 14, 'left'], [-12, 14, 'right'], [12, -30, 'left'], [-12, -30, 'right']];
      const hit = (b) => boxes.some((q) => b[0] < q[2] && b[2] > q[0] && b[1] < q[3] && b[3] > q[1]);
      let best = spots[0];
      for (const sp of spots) {
        const x0 = sp[2] === 'left' ? bx + sp[0] : bx + sp[0] - tw, y0 = by + sp[1] - fsz * 0.55;
        const b = [x0, y0, x0 + tw, y0 + fsz * 1.1];
        if (b[0] >= m.l + 2 && b[2] <= w - 2 && b[1] >= 0 && b[3] <= Y(0) - 2 && !hit(b)) { best = sp; break; }
      }
      D.text(g, lab, bx + best[0], by + best[1], { size: fsz, role: 'display', italic: true, color: AM.col.linen, align: best[2], baseline: 'middle' });
    }

    // ---------------------------------------------------------------- readout + controls
    const big = AM.el('span', { class: 'tr-big' });
    const formula = AM.el('span', { class: 'tr-formula' });
    // screen readers hear the result once things come to rest, not every animation frame
    const status = AM.el('span', { class: 'sr-only', role: 'status' });
    hosts.read.append(AM.el('span', { class: 'tr-lab' }, 'Loss'), big, formula, status);
    let sayTimer = 0;
    function readout() {
      const pc = S.disp[S.correct], lc = -Math.log(pc);
      big.innerHTML = `${lc.toFixed(2)}<small>nats</small>`;
      formula.innerHTML = `L = −ln p(<b>${WORDS[S.correct]}</b>) = −ln ${pc.toFixed(3)} = ${lc.toFixed(3)} <span class="tr-dim">· the other ${WORDS.length - 1} bars only matter through the sum-to-1 rule</span>`;
      if (!S.touched) return;
      clearTimeout(sayTimer);
      sayTimer = setTimeout(() => {
        const q = S.p[S.correct];
        status.textContent = `Probability of ${WORDS[S.correct]}: ${q.toFixed(2)}. Loss ${(-Math.log(q)).toFixed(2)} nats.`;
      }, 450);
    }

    const ui = AM.ui;
    const conf = ui.slider({
      id: 'tr-conf', label: 'p(correct word)', min: 0.01, max: 0.99, step: 0.01, value: S.p[S.correct],
      format: (v) => v.toFixed(2),
      onInput: (v) => { setP(S.correct, v); S.disp = S.p.slice(); redraw(); },
    });
    const ans = ui.segmented({
      id: 'tr-ans', label: 'Answer in the training text',
      options: WORDS.map((wd, i) => ({ value: i, label: wd })), value: 0,
      onChange: (v) => { S.correct = v; hosts.blank.textContent = WORDS[v]; conf.set(S.p[v]); redraw(); },
    });
    const preset = (fn) => () => { fn(); conf.set(S.p[S.correct]); };
    const btnRight = ui.button({ id: 'tr-pre-right', label: 'Confident & right', onClick: preset(() => { S.p = WORDS.map((_, i) => (i === S.correct ? 0.95 : 0.01)); }) });
    const btnUni = ui.button({ id: 'tr-pre-uni', label: 'Unsure', onClick: preset(() => { S.p = WORDS.map(() => 1 / WORDS.length); }) });
    const btnWrong = ui.button({
      id: 'tr-pre-wrong', label: 'Confident & wrong',
      onClick: preset(() => { const k = (S.correct + 1) % WORDS.length; S.p = WORDS.map((_, i) => (i === k ? 0.9 : i === S.correct ? 0.02 : 0.02)); }),
    });
    const btnReset = ui.button({ id: 'tr-pre-reset', label: 'Reset', title: 'Back to the starting guess', onClick: preset(() => { S.p = P0.slice(); }) });
    const btnStep = ui.button({
      id: 'tr-step', label: ['Gradient step ', AM.el('span', { class: 'tr-nocase' }, '(η = 1)')], kind: 'primary',
      onClick: () => { S.p = logitStep(S.p, S.correct, 1); conf.set(S.p[S.correct]); },
    });
    hosts.ctl.append(
      AM.el('div', { class: 'controls' }, ans.el, conf.el),
      AM.el('div', { class: 'tr-btns' }, btnRight, btnUni, btnWrong, btnReset, btnStep));

    // ---------------------------------------------------------------- pointer: drag bars
    const barAt = (x) => MM.clamp(Math.floor((x - BL.pad) / BL.colW), 0, WORDS.length - 1);
    const valAt = (y) => (BL.yBase - y) / (BL.yBase - BL.yTop);
    bars.canvas.addEventListener('pointerdown', (e) => {
      if (!BL) return;
      const q = bars.pointer(e);
      S.drag = barAt(q.x);
      bars.canvas.setPointerCapture(e.pointerId);
      setP(S.drag, valAt(q.y)); S.disp = S.p.slice(); conf.set(S.p[S.correct]); redraw();
      e.preventDefault();
    });
    bars.canvas.addEventListener('pointermove', (e) => {
      if (S.drag < 0) return;
      const q = bars.pointer(e);
      setP(S.drag, valAt(q.y)); S.disp = S.p.slice(); conf.set(S.p[S.correct]); redraw();
    });
    const end = () => { if (S.drag >= 0) { S.drag = -1; redraw(); } };
    bars.canvas.addEventListener('pointerup', end);
    bars.canvas.addEventListener('pointercancel', end);

    function redraw() { S.touched = true; drawBars(); drawCurve(); readout(); }
    bars.onResize((w, h) => { BL = barLayout(w, h); drawBars(); });
    curve.onResize((w, h) => { CL = { w, h, small: w < 440 }; drawCurve(); });
    readout();

    // animate presets and gradient steps: displayed bars glide to their new values
    ctx.loop((t, dt) => {
      if (!vis.on || S.drag >= 0) return;
      let moved = 0;
      const k = AM.reducedMotion ? 1 : 1 - Math.exp(-dt * 9);
      for (let i = 0; i < S.p.length; i++) { const d = S.p[i] - S.disp[i]; if (Math.abs(d) > 1e-5) { S.disp[i] += d * k; moved = 1; } else S.disp[i] = S.p[i]; }
      if (moved) redraw();
    });
  }

  // ==================================================================== 3. the loss landscape
  // A made-up function of two parameters: a bowl, a long diagonal ravine (the
  // global minimum), two smaller wells, a hill, a ridge and gentle ripples.
  const TERRAIN = {
    R: 3.2,
    bowl: 0.06, quart: 0.006,
    rip: [0.12, 1.9, 0.5, 1.6], // a · sin(kx·x + φ) · cos(ky·y)
    // [amplitude, cx, cy, σu, σv, rotation]; negative amplitude = valley
    bumps: [
      [-2.0, 1.15, -0.75, 1.05, 0.32, 0.62],
      [-1.15, -1.55, 1.15, 0.55, 0.55, 0],
      [-0.95, -1.0, -1.95, 0.42, 0.42, 0],
      [1.15, 0.05, 0.55, 0.62, 0.62, 0],
      [0.7, 1.6, 1.6, 0.5, 0.7, 0.3],
    ],
  };
  const LOSS_OFF = 2.1; // shift so the displayed loss is positive (min ≈ 0.28)
  function terrainF(x, y) {
    const T = TERRAIN, r2 = x * x + y * y;
    let z = T.bowl * r2 + T.quart * r2 * r2;
    z += T.rip[0] * Math.sin(T.rip[1] * x + T.rip[2]) * Math.cos(T.rip[3] * y);
    for (const [A, cx, cy, su, sv, th] of T.bumps) {
      const c = Math.cos(th), s = Math.sin(th), dx = x - cx, dy = y - cy;
      const u = dx * c + dy * s, v = -dx * s + dy * c;
      z += A * Math.exp(-((u * u) / (2 * su * su) + (v * v) / (2 * sv * sv)));
    }
    return z;
  }
  /** Exact analytic gradient of terrainF (checked against finite differences to ~1e-9). */
  function terrainG(x, y) {
    const T = TERRAIN, r2 = x * x + y * y;
    let gx = 2 * T.bowl * x + 4 * T.quart * r2 * x;
    let gy = 2 * T.bowl * y + 4 * T.quart * r2 * y;
    gx += T.rip[0] * T.rip[1] * Math.cos(T.rip[1] * x + T.rip[2]) * Math.cos(T.rip[3] * y);
    gy -= T.rip[0] * T.rip[3] * Math.sin(T.rip[1] * x + T.rip[2]) * Math.sin(T.rip[3] * y);
    for (const [A, cx, cy, su, sv, th] of T.bumps) {
      const c = Math.cos(th), s = Math.sin(th), dx = x - cx, dy = y - cy;
      const u = dx * c + dy * s, v = -dx * s + dy * c;
      const gb = A * Math.exp(-((u * u) / (2 * su * su) + (v * v) / (2 * sv * sv)));
      const du = u / (su * su), dv = v / (sv * sv);
      // chain rule through the rotation: ∂u/∂x = c, ∂v/∂x = −s, ∂u/∂y = s, ∂v/∂y = c
      gx -= gb * (du * c - dv * s);
      gy -= gb * (du * s + dv * c);
    }
    return [gx, gy];
  }

  const LR_MIN = 0.002, LR_MAX = 1.5, LR_DEFAULT = 0.05;
  const s2lr = (s) => LR_MIN * Math.pow(LR_MAX / LR_MIN, s);
  const lr2s = (lr) => Math.log(lr / LR_MIN) / Math.log(LR_MAX / LR_MIN);
  const fmtLR = (lr) => (lr < 0.01 ? lr.toFixed(4) : lr < 0.1 ? lr.toFixed(3) : lr.toFixed(2));
  const BETA = 0.9, NOISE_SD = 1.0, STEPS_PER_SEC = 14, MAX_STEPS = 600;
  const BEAD_DYES = ['weld', 'verdigris', 'cochineal', 'woad', 'saffron', 'lichen'];
  const INTRO = [[-1.9, 2.3], [2.2, 1.9], [-0.3, -2.6]]; // three starts, three different valleys at the default η

  function buildLandscape(ctx, hosts) {
    const R = TERRAIN.R;       // the cloth is a disc of radius R in parameter space
    const HZ = 0.66;           // vertical exaggeration of the loss axis
    const phoneBand = 74;      // on narrow screens the sparkline gets its own band above the cloth
    const cv = ctx.canvas(hosts.canvas, {
      height: (w) => (w < 640 ? Math.round(w * 0.72) + phoneBand : Math.min(Math.round(w * 0.54), 600)),
      label: 'A woven, shaded loss surface over two parameters, drawn as a round cloth with contour lines. Glowing beads fall onto it and roll downhill by gradient descent, leaving luminous trails. A small chart plots each bead\'s loss against its step number.',
    });
    const g = cv.g;
    const cache = document.createElement('canvas');
    const cg = cache.getContext('2d');
    const vis = visibility(cv.wrap, '40px 0px');
    const inDisc = (x, y, f = 1) => x * x + y * y <= R * R * f * f;

    let fMin = Infinity, fMax = -Infinity;
    for (let i = 0; i <= 120; i++) for (let j = 0; j <= 120; j++) {
      const x = -R + (2 * R * i) / 120, y = -R + (2 * R * j) / 120;
      if (!inDisc(x, y)) continue;
      const z = terrainF(x, y);
      if (z < fMin) fMin = z; if (z > fMax) fMax = z;
    }

    const opt = { lr: LR_DEFAULT, mom: false, noise: false };
    const beads = [];
    const splashes = [];
    const sparks = [];
    const queue = [];
    let clock = 0;
    let dyeIdx = 0;
    let P = null;              // projection + grid (rebuilt on resize)
    let hover = null;
    let introDone = false;
    let lastT = 0;
    let starts = INTRO.map(([x, y], k) => ({ x, y, col: AM.dye[BEAD_DYES[k]] }));

    // ---------------------------------------------------------------- projection
    function makeProj(w, h) {
      const small = w < 640;
      const n = small ? 44 : 64;
      const az = -0.74, el = 0.8, persp = 0.045;
      const ca = Math.cos(az), sa = Math.sin(az), ce = Math.cos(el), se = Math.sin(el);
      const raw = (x, y, z) => {
        const Z = z * HZ;
        const xr = x * ca - y * sa, yr = x * sa + y * ca;
        const depth = yr * ce - Z * se;          // larger = farther from the camera
        const k = 1 / (1 + depth * persp);
        return [xr * k, -(yr * se + Z * ce) * k, depth];
      };
      // grid of heights and raw projections; fit the disc's bounding box into the canvas
      const H = [], RP = [];
      let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
      for (let i = 0; i <= n; i++) {
        H.push([]); RP.push([]);
        for (let j = 0; j <= n; j++) {
          const x = -R + (2 * R * i) / n, y = -R + (2 * R * j) / n, z = terrainF(x, y);
          const r = raw(x, y, z);
          H[i].push(z); RP[i].push(r);
          if (!inDisc(x, y, 1.02)) continue;
          if (r[0] < x0) x0 = r[0]; if (r[0] > x1) x1 = r[0]; if (r[1] < y0) y0 = r[1]; if (r[1] > y1) y1 = r[1];
        }
      }
      const mx = small ? 4 : w * 0.17, mt = small ? phoneBand : h * 0.07, mb = small ? 8 : h * 0.05;
      const s = Math.min((w - 2 * mx) / (x1 - x0), (h - mt - mb) / (y1 - y0));
      const ox = (w - (x1 - x0) * s) / 2 - x0 * s;
      const oy = mt + ((h - mt - mb) - (y1 - y0) * s) / 2 - y0 * s;
      const project = (x, y, z) => { const r = raw(x, y, z); return { X: ox + r[0] * s, Y: oy + r[1] * s, d: r[2] }; };
      const SP = RP.map((row) => row.map((r) => ({ X: ox + r[0] * s, Y: oy + r[1] * s, d: r[2] })));
      return { w, h, small, n, H, SP, project, s, ver: Math.random(), box: { x0: ox + x0 * s, x1: ox + x1 * s, y0: oy + y0 * s, y1: oy + y1 * s } };
    }

    // ---------------------------------------------------------------- static surface (cached)
    const seqRGB = (t) => parseRgb(AM.color.seq(t));
    const tOf = (z) => MM.clamp((z - fMin) / (fMax - fMin), 0, 1);
    function buildCache() {
      const { w, h, n, H, SP, project } = P;
      const dpr = cv.dpr;
      cache.width = Math.round(w * dpr); cache.height = Math.round(h * dpr);
      cg.setTransform(dpr, 0, 0, dpr, 0, 0);
      cg.clearRect(0, 0, w, h);
      const ink = rgb(AM.col.ink), ink3 = rgb(AM.col.ink3), wc = rgb(AM.dye.woad);
      const Ld = (() => { const v = [-0.6, 0.35, 0.72]; const m = Math.hypot(...v); return v.map((a) => a / m); })();
      const K = 18;
      const levels = Array.from({ length: K }, (_, k) => fMin + ((k + 0.6) * (fMax - fMin)) / (K + 0.4));
      const levCol = levels.map((lv) => seqRGB(0.14 + 0.86 * tOf(lv)));
      const step = (2 * R) / n;

      // painter's algorithm: draw cells from far to near so nearer cloth hides what is behind it
      const cells = [];
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        const xc = -R + (i + 0.5) * step, yc = -R + (j + 0.5) * step;
        if (!inDisc(xc, yc, 1.0)) continue;
        cells.push({ i, j, xc, yc, d: (SP[i][j].d + SP[i + 1][j].d + SP[i][j + 1].d + SP[i + 1][j + 1].d) / 4 });
      }
      cells.sort((a, b) => b.d - a.d);
      cg.lineCap = 'round';
      cg.lineJoin = 'round';
      const seg = (p, q) => { cg.moveTo(p.X, p.Y); cg.lineTo(q.X, q.Y); };
      for (const { i, j, xc, yc } of cells) {
        const a = SP[i][j], b = SP[i + 1][j], c = SP[i + 1][j + 1], d = SP[i][j + 1];
        const zc = (H[i][j] + H[i + 1][j] + H[i][j + 1] + H[i + 1][j + 1]) / 4;
        const [gx, gy] = terrainG(xc, yc);
        const nx = -HZ * gx, ny = -HZ * gy, nn = Math.hypot(nx, ny, 1);
        const shade = MM.clamp((nx * Ld[0] + ny * Ld[1] + Ld[2]) / nn, 0, 1);
        const t = tOf(zc);
        // soft selvedge: the cloth fades out toward the rim of the disc
        const fade = MM.smoothstep(R, R * 0.8, Math.hypot(xc, yc));
        const tint = seqRGB(t);
        const sh = shade * shade;
        const k = 0.05 + 0.34 * sh;
        const fc = [0, 1, 2].map((q) => { const base = ink[q] + (ink3[q] - ink[q]) * (0.2 + 0.8 * sh); return Math.round(base + (tint[q] - base) * k); });
        cg.fillStyle = `rgba(${fc[0]},${fc[1]},${fc[2]},${fade})`;
        cg.beginPath(); cg.moveTo(a.X, a.Y); cg.lineTo(b.X, b.Y); cg.lineTo(c.X, c.Y); cg.lineTo(d.X, d.Y); cg.closePath();
        cg.fill();
        cg.strokeStyle = cg.fillStyle; cg.lineWidth = 0.8; cg.stroke(); // close hairline seams between cells
        // warp (along x) and weft (along y) threads, lit by the same light
        const tc = seqRGB(0.15 + 0.85 * t);
        const lit = 0.35 + 0.65 * shade;
        cg.lineWidth = 0.75;
        cg.strokeStyle = `rgba(${tc[0]},${tc[1]},${tc[2]},${0.46 * lit * fade})`;
        cg.beginPath(); seg(a, b); cg.stroke();
        cg.strokeStyle = `rgba(${(tc[0] + wc[0]) >> 1},${(tc[1] + wc[1]) >> 1},${(tc[2] + wc[2]) >> 1},${0.32 * lit * fade})`;
        cg.beginPath(); seg(a, d); cg.stroke();
        // contour threads through this cell (marching squares), lifted to their own height
        const v = [H[i][j], H[i + 1][j], H[i + 1][j + 1], H[i][j + 1]];
        const lo = Math.min(v[0], v[1], v[2], v[3]), hi = Math.max(v[0], v[1], v[2], v[3]);
        const px = [-R + i * step, -R + (i + 1) * step, -R + (i + 1) * step, -R + i * step];
        const py = [-R + j * step, -R + j * step, -R + (j + 1) * step, -R + (j + 1) * step];
        for (let k2 = 0; k2 < K; k2++) {
          const lv = levels[k2];
          if (lv <= lo || lv >= hi) continue;
          const pts = [];
          for (let q = 0; q < 4; q++) {
            const q2 = (q + 1) % 4, v1 = v[q], v2 = v[q2];
            if ((v1 - lv) * (v2 - lv) < 0) {
              const u = (lv - v1) / (v2 - v1);
              pts.push(project(px[q] + (px[q2] - px[q]) * u, py[q] + (py[q2] - py[q]) * u, lv));
            }
          }
          if (pts.length < 2) continue;
          const lc = levCol[k2], major = k2 % 3 === 1;
          const al = (major ? 0.9 : 0.5) * fade * (0.45 + 0.55 * lit);
          const path = () => { cg.beginPath(); seg(pts[0], pts[1]); if (pts.length === 4) seg(pts[2], pts[3]); };
          if (major) { cg.strokeStyle = `rgba(${lc[0]},${lc[1]},${lc[2]},${al * 0.2})`; cg.lineWidth = 4.5; path(); cg.stroke(); }
          cg.strokeStyle = `rgba(${lc[0]},${lc[1]},${lc[2]},${al})`; cg.lineWidth = major ? 1.5 : 0.9;
          path(); cg.stroke();
        }
      }
      // selvedge: two fine gold rings around the rim, draped at the surface height
      for (const [rr, a, dash] of [[R * 0.985, 0.4, [5, 4]], [R * 1.035, 0.18, []]]) {
        cg.setLineDash(dash);
        cg.strokeStyle = AM.rgba(AM.dye.weld, a); cg.lineWidth = 1;
        cg.beginPath();
        for (let k = 0; k <= 160; k++) {
          const ang = (k / 160) * TAU, x = Math.cos(ang) * rr, y = Math.sin(ang) * rr;
          const p = project(x, y, terrainF(Math.cos(ang) * R * 0.985, Math.sin(ang) * R * 0.985));
          k ? cg.lineTo(p.X, p.Y) : cg.moveTo(p.X, p.Y);
        }
        cg.stroke();
      }
      cg.setLineDash([]);
    }

    // ---------------------------------------------------------------- picking: screen point → parameters
    function pick(X, Y) {
      if (!P) return null;
      const { n, SP } = P;
      let best = null, bd = Infinity;
      for (let i = 0; i <= n; i++) for (let j = 0; j <= n; j++) {
        const x = -R + (2 * R * i) / n, y = -R + (2 * R * j) / n;
        if (!inDisc(x, y)) continue;
        const p = SP[i][j];
        const dd = (p.X - X) ** 2 + (p.Y - Y) ** 2 + p.d * 0.5; // slight preference for nearer cloth
        if (dd < bd) { bd = dd; best = [x, y]; }
      }
      if (!best) return null;
      let [x, y] = best;
      // refine with a few Gauss–Newton steps on the screen-space distance
      for (let it = 0; it < 6; it++) {
        const p = P.project(x, y, terrainF(x, y));
        const e = 1e-3;
        const px = P.project(x + e, y, terrainF(x + e, y)), py = P.project(x, y + e, terrainF(x, y + e));
        const a = (px.X - p.X) / e, b = (py.X - p.X) / e, c = (px.Y - p.Y) / e, d = (py.Y - p.Y) / e;
        const det = a * d - b * c;
        if (Math.abs(det) < 1e-6) break;
        const rx = X - p.X, ry = Y - p.Y;
        x += MM.clamp((d * rx - b * ry) / det, -0.3, 0.3);
        y += MM.clamp((-c * rx + a * ry) / det, -0.3, 0.3);
      }
      const p = P.project(x, y, terrainF(x, y));
      if (!inDisc(x, y, 0.96) || Math.hypot(p.X - X, p.Y - Y) > 24) return null;
      return { x, y };
    }

    // ---------------------------------------------------------------- beads
    function makeBead(x, y, falling = true, colour = null) {
      const col = colour || AM.dye[BEAD_DYES[dyeIdx % BEAD_DYES.length]];
      dyeIdx++;
      const b = { x, y, vx: 0, vy: 0, col, k: 0, acc: 0, still: 0, gn: 0, state: falling ? 'falling' : 'rolling', fallT: 0, path: [{ x, y, L: terrainF(x, y) + LOSS_OFF }], id: dyeIdx };
      beads.push(b);
      while (beads.length > 6) beads.shift();
      return b;
    }
    function stepBead(b) {
      const [tx, ty] = terrainG(b.x, b.y);
      let gx = tx, gy = ty;
      if (opt.noise) { gx += NOISE_SD * MM.randn(); gy += NOISE_SD * MM.randn(); } // minibatch noise
      const eta = opt.lr;
      if (opt.mom) { b.vx = BETA * b.vx - eta * gx; b.vy = BETA * b.vy - eta * gy; } // v ← βv − η∇L
      else { b.vx = -eta * gx; b.vy = -eta * gy; }                                   // plain step −η∇L
      const nx = b.x + b.vx, ny = b.y + b.vy;
      b.k++;
      if (!Number.isFinite(nx) || !Number.isFinite(ny) || !inDisc(nx, ny)) {
        // flew off the cloth: clip the last jump at the rim (solve |p + s·v| = R) and stop
        const dx = Number.isFinite(nx) ? nx - b.x : 0, dy = Number.isFinite(ny) ? ny - b.y : 0;
        const A = dx * dx + dy * dy, B = 2 * (b.x * dx + b.y * dy), C = b.x * b.x + b.y * b.y - R * R;
        const s = A > 1e-12 ? MM.clamp((-B + Math.sqrt(Math.max(0, B * B - 4 * A * C))) / (2 * A), 0, 1) : 0;
        b.x += dx * s; b.y += dy * s;
        b.path.push({ x: b.x, y: b.y, L: terrainF(b.x, b.y) + LOSS_OFF });
        b.state = 'diverged'; b.flare = 1;
        return;
      }
      const moved = Math.hypot(nx - b.x, ny - b.y);
      b.x = nx; b.y = ny;
      b.path.push({ x: b.x, y: b.y, L: terrainF(b.x, b.y) + LOSS_OFF });
      b.gn = Math.hypot(tx, ty);
      b.still = moved < 2e-4 && b.gn < 4e-3 ? b.still + 1 : 0;
      if (b.still > 12 || b.k >= MAX_STEPS) {
        b.state = 'settled';
        if (b.k < MAX_STEPS && P) splashes.push({ x: b.x, y: b.y, t: 0, col: b.col }); // a ripple where it comes to rest
      }
    }
    const pathPt = (q) => {
      if (q._v !== P.ver) { const pp = P.project(q.x, q.y, q.L - LOSS_OFF); q._X = pp.X; q._Y = pp.Y; q._v = P.ver; }
      return q;
    };
    /** Where to draw a bead now: glide between its last two steps, hopping in proportion to the jump. */
    function beadNow(b) {
      const n = b.path.length;
      if (b.state !== 'rolling' || n < 2) { const q = b.path[n - 1]; return { x: q.x, y: q.y, z: q.L - LOSS_OFF }; }
      const a = b.path[n - 2], c = b.path[n - 1];
      const u = MM.ease.inOut(MM.clamp(b.acc, 0, 1));
      const x = MM.lerp(a.x, c.x, u), y = MM.lerp(a.y, c.y, u);
      const hop = Math.hypot(c.x - a.x, c.y - a.y) * 0.55 * 4 * u * (1 - u);
      return { x, y, z: Math.max(terrainF(x, y), MM.lerp(a.L, c.L, u) - LOSS_OFF) + hop };
    }
    function dropBead(x, y, colour = null, remember = true) {
      const b = makeBead(x, y, !AM.reducedMotion, colour);
      if (AM.reducedMotion) splashes.push({ x, y, t: 0, col: b.col });
      // remember the last three drops so Replay can send the same beads down again
      if (remember) {
        starts.push({ x, y, col: b.col });
        while (starts.length > 3) starts.shift();
      }
      return b;
    }

    // ---------------------------------------------------------------- dust: faint grains drifting downhill (the gradient field)
    const DUST_N = 160;
    const dust = [];
    const spawnDust = (d) => {
      const r = R * 0.94 * Math.sqrt(Math.random()), a = Math.random() * TAU;
      d.x = Math.cos(a) * r; d.y = Math.sin(a) * r; d.age = 0; d.life = 1.6 + Math.random() * 2.4;
      return d;
    };
    for (let i = 0; i < DUST_N; i++) { const d = spawnDust({}); d.age = Math.random() * d.life; dust.push(d); }

    // ---------------------------------------------------------------- per-frame drawing
    function draw() {
      if (!P) return;
      const { w, h, small } = P;
      cv.clear();
      g.drawImage(cache, 0, 0, w, h);

      // silk sheen: a soft band of light drifts slowly across the cloth (only where the cloth is)
      if (!AM.reducedMotion) {
        const bx0 = P.box.x0, bx1 = P.box.x1, span = bx1 - bx0;
        const u = ((lastT * 0.045) % 1.5) - 0.25;
        const cx = bx0 + span * u, half = span * 0.22;
        const sh = g.createLinearGradient(cx - half, P.box.y0, cx + half, P.box.y1);
        sh.addColorStop(0, AM.rgba(AM.col.linen, 0));
        sh.addColorStop(0.5, AM.rgba('#fff3d0', 0.075));
        sh.addColorStop(1, AM.rgba(AM.col.linen, 0));
        g.globalCompositeOperation = 'source-atop';
        g.fillStyle = sh;
        g.fillRect(bx0 - 4, P.box.y0 - 4, span + 8, P.box.y1 - P.box.y0 + 8);
        g.globalCompositeOperation = 'source-over';
      }

      // dust
      g.globalCompositeOperation = 'lighter';
      const nDust = small ? 80 : DUST_N;
      for (let i = 0; i < nDust; i++) {
        const d = dust[i];
        const p = P.project(d.x, d.y, terrainF(d.x, d.y));
        const a = Math.sin(Math.PI * MM.clamp(d.age / d.life, 0, 1)) * 0.26;
        blob(g, AM.col.linen, p.X, p.Y, small ? 2.6 : 3.2, a);
      }
      g.globalCompositeOperation = 'source-over';

      // splash rings: circles in parameter space, draped over the cloth
      for (const s of splashes) {
        const u = s.t / 0.9;
        const rr = 0.05 + 0.34 * MM.ease.out(u);
        g.strokeStyle = AM.rgba(s.col || AM.dye.weld, 0.75 * (1 - u)); g.lineWidth = 1.3;
        g.beginPath();
        for (let k = 0; k <= 32; k++) {
          const a = (k / 32) * TAU, x = s.x + Math.cos(a) * rr, y = s.y + Math.sin(a) * rr;
          const p = P.project(x, y, terrainF(x, y));
          k ? g.lineTo(p.X, p.Y) : g.moveTo(p.X, p.Y);
        }
        g.stroke();
      }

      // sparks shed by rolling beads
      g.globalCompositeOperation = 'lighter';
      for (const s of sparks) {
        const p = P.project(s.x, s.y, s.z);
        const u = s.t / s.life;
        blob(g, s.col, p.X, p.Y - s.lift * u, 4.5 * (1 - u) + 1.2, 0.7 * (1 - u));
      }
      g.globalCompositeOperation = 'source-over';

      // trails
      g.lineCap = 'round'; g.lineJoin = 'round';
      for (const b of beads) {
        if (b.state === 'falling') continue;
        const now = beadNow(b);
        const pn = P.project(now.x, now.y, now.z);
        const nPts = b.state === 'rolling' ? b.path.length - 1 : b.path.length;
        const trace = () => {
          g.beginPath();
          for (let k = 0; k < nPts; k++) { const q = pathPt(b.path[k]); k ? g.lineTo(q._X, q._Y) : g.moveTo(q._X, q._Y); }
          if (b.state === 'rolling') g.lineTo(pn.X, pn.Y);
        };
        g.strokeStyle = AM.rgba(b.col, 0.16); g.lineWidth = 7; trace(); g.stroke();
        g.strokeStyle = AM.rgba(b.col, 0.92); g.lineWidth = 1.7; trace(); g.stroke();
        // a dot at every step: the spacing shows the step size
        g.fillStyle = AM.rgba(b.col, 0.85);
        for (let k = 0; k < nPts; k++) { const q = pathPt(b.path[k]); g.beginPath(); g.arc(q._X, q._Y, 1.5, 0, TAU); g.fill(); }
      }

      // beads
      const flags = []; // diverged beads; their labels are placed after all beads are drawn
      for (const b of beads) {
        if (b.state === 'falling') {
          const q = pathPt(b.path[0]);
          const u = MM.clamp(b.fallT, 0, 1);
          const y = MM.lerp(q._Y - h * 0.5, q._Y, u * u); // gravity: distance ∝ t²
          // a shadow gathers on the cloth as the bead approaches
          g.fillStyle = AM.rgba(AM.col.ink, 0.55 * u);
          g.beginPath(); g.ellipse(q._X, q._Y + 2, 9 * u + 2, 3.5 * u + 1, 0, 0, TAU); g.fill();
          const tail = 8 + 26 * u;
          const lg = g.createLinearGradient(0, y - tail, 0, y);
          lg.addColorStop(0, AM.rgba(b.col, 0)); lg.addColorStop(1, AM.rgba(b.col, 0.55));
          g.strokeStyle = lg; g.lineWidth = 2.5;
          g.beginPath(); g.moveTo(q._X, y - tail); g.lineTo(q._X, y - 3); g.stroke();
          g.globalCompositeOperation = 'lighter';
          blob(g, b.col, q._X, y, 15, 0.95);
          g.globalCompositeOperation = 'source-over';
          continue;
        }
        const now = beadNow(b);
        const p = P.project(now.x, now.y, now.z);
        g.globalCompositeOperation = 'lighter';
        blob(g, b.col, p.X, p.Y, b.state === 'settled' ? 13 + (AM.reducedMotion ? 0 : 2 * Math.sin(lastT * 2.4 + b.id)) : 16, 0.95);
        g.globalCompositeOperation = 'source-over';
        const grd = g.createRadialGradient(p.X - 1.4, p.Y - 1.6, 0.5, p.X, p.Y, 5);
        grd.addColorStop(0, '#fffaf0'); grd.addColorStop(0.45, b.col); grd.addColorStop(1, AM.mix(b.col, AM.col.ink, 0.45));
        g.fillStyle = grd;
        g.beginPath(); g.arc(p.X, p.Y, 4.6, 0, TAU); g.fill();
        if (b.state === 'diverged') {
          const fl = b.flare || 0;
          g.globalCompositeOperation = 'lighter';
          blob(g, AM.dye.madder, p.X, p.Y, 18 + 26 * fl, 0.85);
          g.globalCompositeOperation = 'source-over';
          flags.push(p);
        }
      }
      // "flew off" labels: newest first, kept inside the canvas, never stacked on top of each other
      const flagBoxes = [];
      for (let k = flags.length - 1; k >= 0; k--) {
        const p = flags[k];
        const lx = MM.clamp(p.X, 40, w - 40), ly = Math.max(12, p.Y - 31);
        const box = [lx - 38, ly, lx + 38, ly + 20];
        if (flagBoxes.some((q) => box[0] < q[2] && box[2] > q[0] && box[1] < q[3] && box[3] > q[1])) continue;
        flagBoxes.push(box);
        g.fillStyle = AM.rgba(AM.col.ink, 0.85);
        D.roundRect(g, box[0], box[1], 76, 20, 10); g.fill();
        g.strokeStyle = AM.rgba(AM.dye.madder, 0.6); g.lineWidth = 1; g.stroke();
        micro(g, 'FLEW OFF', lx, ly + 10, { align: 'center', color: AM.dye.madder, size: 9 });
      }

      // the downhill arrow (−∇L) at the newest rolling bead
      const live = beads.filter((b) => b.state === 'rolling');
      if (live.length) {
        const b = live[live.length - 1];
        const now = beadNow(b);
        const [gx, gy] = terrainG(now.x, now.y);
        const gn = Math.hypot(gx, gy);
        if (gn > 0.02) {
          const len = 0.5, tx = now.x - (gx / gn) * len, ty = now.y - (gy / gn) * len;
          const p0 = P.project(now.x, now.y, terrainF(now.x, now.y)), p1 = P.project(tx, ty, terrainF(tx, ty));
          D.arrow(g, p0.X, p0.Y, p1.X, p1.Y, { color: AM.rgba(AM.col.linen, 0.85), width: 1.4, head: 6 });
          micro(g, '−∇L', p1.X + (p1.X >= p0.X ? 6 : -6), p1.Y - 6, { align: p1.X >= p0.X ? 'left' : 'right', color: AM.col.linen, size: 9, ls: '0.04em' });
        }
      }

      // hover marker (mouse only)
      if (hover) {
        const p = P.project(hover.x, hover.y, terrainF(hover.x, hover.y));
        g.strokeStyle = AM.rgba(AM.col.linen, 0.7); g.lineWidth = 1;
        g.beginPath(); g.ellipse(p.X, p.Y, 8, 4, 0, 0, TAU); g.stroke();
        g.setLineDash([2, 3]);
        g.beginPath(); g.moveTo(p.X, p.Y - 6); g.lineTo(p.X, p.Y - 34); g.stroke();
        g.setLineDash([]);
        micro(g, `loss ${(terrainF(hover.x, hover.y) + LOSS_OFF).toFixed(2)}`, p.X + 8, p.Y - 30, { color: AM.col.linen, size: 9, ls: '0.04em' });
      }

      drawSpark();
      drawKey();
    }

    // loss vs step, one line per bead
    function drawSpark() {
      const { w, small } = P;
      const bw = small ? w - 8 : MM.clamp(w * 0.19, 172, 214), bh = small ? phoneBand - 14 : 96, x0 = small ? 4 : 12, y0 = small ? 4 : 12;
      g.fillStyle = AM.rgba(AM.col.ink, 0.72);
      D.roundRect(g, x0, y0, bw, bh, 8); g.fill();
      g.strokeStyle = AM.col.rule; g.lineWidth = 1; g.stroke();
      micro(g, 'LOSS vs STEP', x0 + 10, y0 + 12, { size: 9 });
      const ix = x0 + 10, iy = y0 + 22, iw = bw - 20, ih = bh - 32;
      const kMax = Math.max(80, ...beads.map((b) => b.path.length - 1));
      micro(g, `0–${kMax}`, x0 + bw - 10, y0 + 12, { align: 'right', size: 9, ls: '0.02em' });
      const YMAX = LOSS_OFF + fMax + 0.2;
      g.strokeStyle = AM.rgba(AM.col.linen, 0.08);
      g.beginPath(); g.moveTo(ix, iy + ih); g.lineTo(ix + iw, iy + ih); g.stroke();
      // the lowest loss on this cloth, for reference
      const yBest = iy + ih * (1 - (LOSS_OFF + fMin) / YMAX);
      g.setLineDash([2, 3]); g.strokeStyle = AM.rgba(AM.col.linen, 0.18);
      g.beginPath(); g.moveTo(ix, yBest); g.lineTo(ix + iw, yBest); g.stroke(); g.setLineDash([]);
      for (const b of beads) {
        if (b.path.length < 2) continue;
        g.strokeStyle = AM.rgba(b.col, 0.95); g.lineWidth = 1.3;
        g.beginPath();
        const n = b.path.length, stride = Math.max(1, Math.floor(n / 240));
        for (let k = 0; k < n; k += stride) {
          const x = ix + (k / kMax) * iw, y = iy + ih * (1 - MM.clamp(b.path[k].L / YMAX, 0, 1));
          k ? g.lineTo(x, y) : g.moveTo(x, y);
        }
        g.stroke();
        // a glowing head at the newest step
        const ex = ix + ((n - 1) / kMax) * iw, ey = iy + ih * (1 - MM.clamp(b.path[n - 1].L / YMAX, 0, 1));
        g.globalCompositeOperation = 'lighter';
        blob(g, b.col, ex, ey, b.state === 'rolling' ? 7 : 5, 0.9);
        g.globalCompositeOperation = 'source-over';
      }
      if (!beads.length) micro(g, 'drop a bead', ix + iw / 2, iy + ih / 2, { align: 'center', size: 9 });
    }

    // colour key and hint (desktop only: the right-hand margin)
    function drawKey() {
      const { w, h, small } = P;
      if (small) return;
      const x = w - 26, y0 = h * 0.3, y1 = h * 0.7;
      const grd = g.createLinearGradient(0, y1, 0, y0);
      for (let k = 0; k <= 8; k++) grd.addColorStop(k / 8, AM.color.seq(0.15 + 0.85 * (k / 8)));
      g.fillStyle = grd;
      D.roundRect(g, x, y0, 5, y1 - y0, 2.5); g.fill();
      micro(g, `${(LOSS_OFF + fMax).toFixed(1)} high loss`, x - 8, y0 + 4, { align: 'right', size: 8.5, ls: '0.04em' });
      micro(g, `${(LOSS_OFF + fMin).toFixed(1)} low loss`, x - 8, y1 - 4, { align: 'right', size: 8.5, ls: '0.04em' });
      micro(g, 'click the cloth to drop a bead', w - 14, 20, { align: 'right', size: 9, ls: '0.06em' });
    }

    // ---------------------------------------------------------------- readout
    let lastRead = '';
    function readout() {
      const b = beads[beads.length - 1];
      let html;
      if (!b) html = '<span>No beads yet. Click or tap the surface.</span>';
      else {
        const q = b.path[b.path.length - 1];
        const [gx, gy] = terrainG(q.x, q.y);
        const state = { falling: 'falling', rolling: 'rolling', settled: b.k >= MAX_STEPS ? `stopped after ${MAX_STEPS} steps` : 'settled in a valley', diverged: 'flew off: the steps were too big' }[b.state];
        html = `<span style="color:${b.col}"><span class="tr-dot"></span>newest bead</span><span>step <b>${b.k}</b></span><span>loss <b>${q.L.toFixed(3)}</b></span><span>|∇L| <b>${Math.hypot(gx, gy).toFixed(3)}</b></span><span>${state}</span>`;
      }
      if (html !== lastRead) { hosts.read.innerHTML = html; lastRead = html; }
    }

    // ---------------------------------------------------------------- simulation tick
    function tick(dt) {
      clock += dt;
      while (queue.length && queue[0].at <= clock) { const q = queue.shift(); dropBead(q.x, q.y, q.col, !q.replay); }
      for (const b of beads) {
        if (b.state === 'falling') {
          b.fallT += dt / 0.6;
          if (b.fallT >= 1) { b.state = 'rolling'; b.acc = 0; splashes.push({ x: b.x, y: b.y, t: 0, col: b.col }); }
          continue;
        }
        if (b.state === 'diverged') { b.flare = Math.max(0, (b.flare || 0) - dt * 0.8); continue; }
        if (b.state !== 'rolling') continue;
        b.acc += dt * STEPS_PER_SEC;
        while (b.acc >= 1 && b.state === 'rolling') { b.acc -= 1; stepBead(b); }
        // a moving bead sheds a few sparks that settle on the cloth behind it
        b.spark = AM.reducedMotion ? 0 : (b.spark || 0) + dt * 22;
        while (b.spark >= 1) {
          b.spark -= 1;
          if (sparks.length < 220) { const q = beadNow(b); sparks.push({ x: q.x + (Math.random() - 0.5) * 0.06, y: q.y + (Math.random() - 0.5) * 0.06, z: q.z, t: 0, life: 0.6 + Math.random() * 0.6, col: b.col, lift: Math.random() * 10 }); }
        }
      }
      for (let i = sparks.length - 1; i >= 0; i--) { sparks[i].t += dt; if (sparks[i].t > sparks[i].life) sparks.splice(i, 1); }
      for (let i = splashes.length - 1; i >= 0; i--) { splashes[i].t += dt; if (splashes[i].t > 0.9) splashes.splice(i, 1); }
      if (!AM.reducedMotion) for (const d of dust) {
        const [gx, gy] = terrainG(d.x, d.y);
        const gn = Math.hypot(gx, gy) || 1, sp = Math.min(gn, 1.6) * 0.32;
        d.x -= (gx / gn) * sp * dt; d.y -= (gy / gn) * sp * dt; d.age += dt;
        // grains that have reached a valley floor are recycled, so they never pile up
        if (d.age > d.life || gn < 0.06) spawnDust(d);
      }
    }

    function startIntro() {
      beads.length = 0; splashes.length = 0; sparks.length = 0; queue.length = 0;
      dyeIdx = 0;
      INTRO.forEach(([x, y], k) => queue.push({ at: clock + 0.25 + k * 0.55, x, y }));
    }

    // static first frame: the three intro beads, already partway down
    INTRO.forEach(([x, y]) => { const b = makeBead(x, y, false); for (let k = 0; k < 160 && b.state === 'rolling'; k++) stepBead(b); });
    dyeIdx = 0;
    readout();

    // play the drop once, the first time the figure is properly on screen
    if (typeof IntersectionObserver !== 'undefined') {
      const io = new IntersectionObserver((en) => {
        if (en.some((e) => e.isIntersecting) && !introDone) { introDone = true; io.disconnect(); if (!AM.reducedMotion) startIntro(); }
      }, { threshold: 0.45 });
      io.observe(cv.wrap);
    }

    // ---------------------------------------------------------------- pointer
    let downAt = null;
    cv.canvas.addEventListener('pointerdown', (e) => { downAt = cv.pointer(e); });
    cv.canvas.addEventListener('pointerup', (e) => {
      if (!downAt) return;
      const q = cv.pointer(e);
      if (Math.hypot(q.x - downAt.x, q.y - downAt.y) < 10) {
        const p = pick(q.x, q.y);
        if (p) { introDone = true; dropBead(p.x, p.y); }
      }
      downAt = null;
    });
    cv.canvas.addEventListener('pointercancel', () => { downAt = null; });
    cv.canvas.addEventListener('pointermove', (e) => {
      if (e.pointerType !== 'mouse') return;
      const q = cv.pointer(e);
      hover = pick(q.x, q.y);
    });
    cv.canvas.addEventListener('pointerleave', () => { hover = null; });

    // ---------------------------------------------------------------- controls
    const ui = AM.ui;
    const lr = ui.slider({ id: 'tr-lr', label: 'Learning rate', min: 0, max: 1, step: 0.001, value: lr2s(LR_DEFAULT), format: (s) => 'η = ' + fmtLR(s2lr(s)), onInput: (s) => { opt.lr = s2lr(s); } });
    const mom = ui.toggle({ id: 'tr-mom', label: 'Momentum', checked: false, onChange: (b) => { opt.mom = b; for (const bd of beads) { bd.vx = 0; bd.vy = 0; } } });
    const noise = ui.toggle({ id: 'tr-noise', label: 'Minibatch noise', checked: false, onChange: (b) => { opt.noise = b; } });
    const rain = ui.button({
      id: 'tr-rain', label: 'Drop three',
      onClick: () => {
        introDone = true;
        for (let k = 0; k < 3; k++) {
          // random starting points on higher ground
          let x, y, tries = 0;
          do {
            const r = R * 0.88 * Math.sqrt(Math.random()), a = Math.random() * TAU;
            x = Math.cos(a) * r; y = Math.sin(a) * r; tries++;
          } while (terrainF(x, y) < 0.3 && tries < 40);
          queue.push({ at: clock + 0.05 + k * 0.4, x, y });
        }
      },
    });
    const replay = ui.button({
      id: 'tr-replay', label: 'Replay', title: 'Drop the last three beads again from the same spots, with the current settings',
      onClick: () => {
        introDone = true;
        beads.length = 0; queue.length = 0; splashes.length = 0; sparks.length = 0;
        starts.forEach((st, k) => queue.push({ at: clock + 0.05 + k * 0.3, x: st.x, y: st.y, col: st.col, replay: true }));
        readout(); draw();
      },
    });
    const clear = ui.button({ id: 'tr-clear', label: 'Clear', onClick: () => { beads.length = 0; queue.length = 0; splashes.length = 0; sparks.length = 0; dyeIdx = 0; introDone = true; readout(); draw(); } });
    hosts.ctl.append(lr.el, mom.el, noise.el, AM.el('div', { class: 'tr-btns' }, rain, replay, clear));

    // The cached cloth takes ~100 ms to weave, so while a window is being dragged wider or
    // narrower the old cache is stretched to fit and rebuilt once the size settles.
    let cacheTimer = 0, cacheReady = false;
    cv.onResize((w, h) => {
      P = makeProj(w, h);
      clearTimeout(cacheTimer);
      if (!cacheReady) { buildCache(); cacheReady = true; draw(); return; }
      draw();
      cacheTimer = setTimeout(() => { if (P) { buildCache(); draw(); } }, 160);
    });
    ctx.loop((t, dt) => {
      if (!vis.on || !P) return;
      lastT = t;
      tick(dt);
      draw();
      readout();
    });
  }

  // ==================================================================== chapter
  AM.chapter({
    id: ID,
    num: 10,
    kicker: 'Training',
    title: 'Learning by <em>Falling</em>',
    lede: 'A new model is a pile of random numbers. It learns by guessing the next token, measuring how wrong the guess was, and nudging every number a little way downhill. Then it does that again, hundreds of thousands of times.',
    where: 'train',
    mount(root, ctx) {
      ctx.header();
      const el = ctx.el;
      const ui = AM.ui;
      const body = el('div', { class: 'ch-body' });
      root.appendChild(body);

      // ---------------------------------------------------------------- 1. scrollytelling: one training step
      const loopHost = el('div');
      const badgeBox = el('span', { class: 'tr-badges' });
      const figTitle = el('span', { class: 'fig-title' }, 'One training step');
      const capText = 'An illustration of a tiny three-layer model reading five tokens. Threads only run from earlier positions to later ones, as the causal mask allows. The probabilities are toy numbers over six candidate words; each loss and mean is computed from them, and the "after" values come from one real gradient step (η = 1) applied to each position\'s logits. Backward pulses leave each position in proportion to the size of its logit gradient, |p − y|. The final curves are the scaling-law formula fitted by Hoffmann et al. (2022), extended past the data it was fitted on; they are not measured training runs. A 2024 replication refit the same data and got somewhat different constants, so treat the exact values loosely.';
      const stage = el('div', { class: 'ch-stage tr-stage' },
        el('figure', { class: 'fig' },
          el('div', { class: 'fig-top' }, figTitle, badgeBox),
          loopHost,
          el('figcaption', { class: 'tr-cap-desk' }, capText)));

      const step = (label, h3, ...html) => el('div', { class: 'step' },
        el('span', { class: 'step-label' }, label),
        el('h3', {}, h3),
        ...html.map((s) => (s.startsWith('<div') ? el('div', { html: s }) : el('p', { html: s }))));

      const steps = [
        step('1 · The task', 'Guess the next token',
          'Pretraining has one job. Show the model some text, hide what comes next, and ask for a probability for every token in its vocabulary.',
          'Nobody has to label anything. The text is its own answer key: after <em>the cat sat on the</em>, the next word is right there in the data. That is why a model can learn from a large slice of the internet.'),
        step('2 · Every position', 'One sentence, many examples',
          'Shift the text left by one and every position gets a target: after <em>the</em> comes <em>cat</em>, after <em>cat</em> comes <em>sat</em>, and so on.',
          'The <span class="term">causal mask</span> lets position i see only tokens up to i, so no position can peek at its own answer. A single forward pass makes a prediction at every position at once: each small histogram is one position\'s guess, with the true next token in gold. These five tokens are five training examples. A model with a 4,096-token context gets 4,096 from every sequence it reads.'),
        step('3 · The loss', 'Score the surprise',
          'At each position, look up the probability the model gave to the token that really came next. Call it p. The loss at that position is <span class="math">−ln p</span> (the natural log), the <span class="term">cross-entropy</span>.',
          '<div class="math block">L = −(1/T) Σ<sub>t</sub> ln p<sub>θ</sub>(x<sub>t+1</sub> | x<sub>≤t</sub>)</div>',
          `A sure, correct guess costs almost nothing. A confident miss costs a lot. The loss for one training step is the average over every position of every sequence in the <span class="term">minibatch</span>. Here the five red drips average to <strong>${LOSS_BEFORE.toFixed(2)}</strong> nats.`),
        step('4 · Backpropagation', 'Send the blame backwards',
          'Now ask, for every parameter in the model: if it changed by a tiny amount, how much would the loss change? That number is the parameter\'s <span class="term">gradient</span>, <span class="math">∂L/∂θ</span>.',
          '<span class="term">Backpropagation</span> computes all of them in one sweep, using the chain rule from the loss back through every layer and every attention thread. Positions that guessed badly send back more blame. The whole sweep costs only about twice as much as the forward pass, whether the model has a thousand parameters or a trillion.'),
        step('5 · The update', 'Take a small step downhill',
          '<div class="math block">θ ← θ − η · ∂L/∂θ</div>',
          `Every parameter moves a little against its gradient. The step size η is the <span class="term">learning rate</span>. After this step the right answers are more likely, and the mean loss falls from <strong>${LOSS_BEFORE.toFixed(2)}</strong> to <strong>${LOSS_AFTER.toFixed(2)}</strong>. Our toy step is far bigger than a real one, so that you can see it. In a real run one step barely moves the loss, and it falls over many thousands of steps.`,
          'Real training uses an optimiser such as <span class="term">AdamW</span>, which sizes each parameter\'s step using running averages of its recent gradients and their squares. The learning rate follows a schedule: it warms up over the first steps, then slowly decays.'),
        step('6 · Repeat', 'Then do it again, for weeks or months',
          'Every step uses a fresh minibatch, often millions of tokens. GPT-3 trained on about 300 billion tokens in 2020. Meta reported more than 15 trillion for Llama 3 in 2024.',
          'The loss falls smoothly and predictably. Bigger models trained on more data reach lower loss, along curves close to power laws. Labs use these <span class="term">scaling laws</span> to plan training runs before they start.',
          'This pretraining makes a model that continues text. Two shorter rounds turn it into an assistant: <span class="term">supervised fine-tuning</span> on example conversations, then learning from human feedback on which of two answers people prefer.'),
      ];
      const prose = el('div', { class: 'ch-prose' }, steps);
      body.appendChild(el('div', { class: 'ch-split' }, stage, prose));
      body.appendChild(el('p', { class: 'caption tr-cap-phone' }, capText));

      const loop = buildLoop(ctx, loopHost);
      const BADGES = [['illustration', 'toy'], ['illustration', 'toy'], ['illustration', 'toy'], ['illustration', 'toy'], ['illustration', 'toy'], ['illustration']];
      ctx.steps(steps, (i) => {
        loop.setStep(i);
        badgeBox.replaceChildren(...BADGES[i].map((k) => ui.badge(k)));
        figTitle.textContent = i === 5 ? 'Loss vs. data and model size' : 'One training step';
      });

      // ---------------------------------------------------------------- 2. cross-entropy playground
      const blank = el('span', { class: 'tr-blank' }, 'mat');
      const barsHost = el('div', { class: 'tr-bars' });
      const curveHost = el('div', { class: 'tr-curve' });
      const ceRead = el('div', { class: 'tr-ce-read' });
      const ceCtl = el('div', { class: 'tr-ctl-rows' });
      body.appendChild(el('section', { class: 'ch-wide tr-sec', 'aria-labelledby': 'tr-ce-h' },
        el('div', { class: 'prose' },
          el('span', { class: 'tr-kicker' }, 'The loss up close'),
          el('h3', { id: 'tr-ce-h' }, 'What does a wrong guess cost?'),
          el('p', { html: 'Below is the guess you saw at the last position above, for the word after <em>the cat sat on the</em>. The training text says the answer is <strong>mat</strong>; pick another word to change it. Drag the bars or use the slider. However you move them, the probabilities still sum to 1.' }),
          el('p', { html: 'The loss is <span class="math">−ln p(correct)</span>, so only the bar on the right answer counts directly. As that probability falls toward 0, the loss climbs without limit. That is why confident mistakes are so expensive.' }),
          el('p', { html: 'The arrows show which way one gradient step pushes each word\'s logit. That gradient has a tidy form, <span class="math">∂L/∂z<sub>i</sub> = p<sub>i</sub> − y<sub>i</sub></span>, where y is 1 for the right word and 0 for the rest. So a step raises the right word\'s logit and lowers every other logit in proportion to its probability. The dashed ticks mark where each bar will land. Press <strong>Gradient step</strong> a few times to watch the loss fall.' })),
        el('div', { class: 'panel tr-ce-panel' },
          ui.figure({
            title: 'Cross-entropy at one position', badge: 'toy',
            caption: 'Toy numbers: six made-up candidates and starting probabilities, the same as the last position of the figure above (a real vocabulary has tens of thousands of tokens or more). Every value shown is computed exactly from the bars. Losses are in nats (natural log). The numbers under the words are ∂L/∂z = p − y. <strong>Gradient step</strong> applies z ← z − η(p − y) with η = 1 to the logits z = ln p, then takes the softmax; the dashed ticks show that result in advance.',
          },
          el('div', { class: 'tr-ce-head' },
            el('div', { class: 'tr-prompt', 'aria-label': 'Prompt' }, 'the cat sat on the', blank),
            ui.legend([{ color: AM.dye.verdigris, label: 'logit pushed up' }, { color: AM.dye.madder, label: 'logit pushed down' }, { color: AM.col.linen, label: 'after one step' }])),
          el('div', { class: 'tr-ce-grid' }, barsHost, curveHost),
          ceRead,
          ceCtl))));
      buildCE(ctx, { bars: barsHost, curve: curveHost, read: ceRead, ctl: ceCtl, blank });

      // ---------------------------------------------------------------- 3. the loss landscape
      const landHost = el('div', { class: 'tr-land' });
      const landCtl = el('div', { class: 'tr-land-ctl' });
      const landRead = el('div', { class: 'tr-readout', 'aria-live': 'off' });
      body.appendChild(el('section', { class: 'ch-wide tr-sec', 'aria-labelledby': 'tr-land-h' },
        el('div', { class: 'prose' },
          el('span', { class: 'tr-kicker' }, 'Gradient descent'),
          el('h3', { id: 'tr-land-h' }, 'A bead rolling downhill'),
          el('p', { html: 'Picture the loss as a landscape. Every possible setting of the parameters is a point on the ground, and the height there is the loss. Training starts at a random point and walks downhill, one gradient step at a time. The gradient points uphill, so each step goes the opposite way: <span class="math">θ ← θ − η∇L</span>.' }),
          el('p', { html: 'This surface is made up, with only two parameters so that it can be drawn. A real model\'s landscape has billions of dimensions. Click or tap the cloth to drop beads; where a bead starts decides which valley it ends in. Then change the learning rate and press <strong>Replay</strong> to send the same beads down again.' })),
        el('div', { class: 'panel tr-land-panel' },
          ui.figure({
            title: 'The loss landscape', badge: 'illustration',
            caption: 'Illustration: a made-up function of two parameters, standing in for a landscape with billions of dimensions. The descent itself is exact for this function. Each step uses the true analytic gradient and applies θ ← θ − η∇L, or with momentum v ← 0.9v − η∇L, θ ← θ + v. Minibatch noise adds random error to each gradient. Dots along a trail mark single steps; the faint drifting grains follow the downhill direction everywhere.',
          },
          landHost, landCtl, landRead)),
        el('div', { class: 'grid-2' },
          el('div', { class: 'tr-note', style: '--c: var(--weld)' },
            el('h4', {}, 'Learning rate'),
            el('p', { html: 'Too small and the bead crawls: at η = 0.003 it needs hundreds of steps to reach the bottom of a valley. Too big and it overshoots, bouncing between the valley walls or flying off the map. Real runs choose η carefully and lower it as training goes on.' })),
          el('div', { class: 'tr-note', style: '--c: var(--verdigris)' },
            el('h4', {}, 'Momentum and minibatch noise'),
            el('p', { html: 'With momentum the bead keeps a running velocity, so it rolls through small bumps and speeds along narrow valleys. Minibatch noise copies real training, where each gradient comes from a random sample of the data and is slightly off: <span class="term">stochastic gradient descent</span>. The jitter can even shake a bead out of a shallow dip.' }))),
        // hand-off to the Lab
        el('p', { class: 'caption tr-handoff', html: 'Want to see a loss curve fall for real? In the <a href="#ch-lab">next chapter</a> a small transformer trains from scratch in your browser, using this same loop.' })));
      buildLandscape(ctx, { canvas: landHost, ctl: landCtl, read: landRead });


      // ---------------------------------------------------------------- key idea
      body.appendChild(el('div', { class: 'callout' },
        el('span', { class: 'callout-label' }, 'Key idea'),
        el('p', { html: 'Training repeats one loop. Predict the next token at every position, score each guess with <strong>−ln p</strong>, use <strong>backpropagation</strong> to get the gradient of every parameter, and take a small step <strong>downhill</strong>. Run that loop for hundreds of thousands of steps over trillions of tokens, and numbers that started out random end up encoding grammar, facts and skills.' })));
    },
  });
})();
