/* Chapter 07 — The River Thread: the residual stream and LayerNorm.

   Two figures:
   1. "The river" (scrollytelling stage): one token's residual stream rising
      through three pre-LN blocks. Sublayers draw a copy off through a LayerNorm
      gate and pour their output back in by addition. A switch replaces the adds
      with overwrites. The strip on the left is a real (toy) 16-d computation.
   2. LayerNorm stepper: exact LayerNorm / RMSNorm on a 12-d toy vector, step by
      step, with scale/shift invariance checks. */
(() => {
  const ID = 'residual';
  const MM = AM.math;
  const D = AM.draw;
  const EPS = 1e-5;
  const D_MODEL = 16;
  const TOY_SEED = 7; // picked so the two stories (add vs replace) read clearly; numbers are still computed

  // ------------------------------------------------------------------ small helpers
  const SUBS = '₀₁₂₃₄₅₆₇₈₉';
  const sub = (n) => String(n).replace(/\d/g, (c) => SUBS[+c]);
  const SUPS = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
  const sup = (n) => String(n).replace(/[-0-9]/g, (c) => SUPS[c]);
  const mean = (a) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i]; return s / a.length; };
  const stdPop = (a) => { const m = mean(a); let v = 0; for (let i = 0; i < a.length; i++) v += (a[i] - m) * (a[i] - m); return Math.sqrt(v / a.length); };
  const rand = (a, b) => a + Math.random() * (b - a);
  const isStacked = () => (window.matchMedia ? window.matchMedia('(max-width: 900px)').matches : window.innerWidth <= 900);
  /** Side-by-side layout: the tallest canvas that keeps the whole sticky stage (title, controls,
      caption) on screen below its 6vh top offset. host holds only the canvas. Never below 380px. */
  const deskFit = (host) => {
    const stage = host && host.closest('.ch-stage');
    if (!stage) return Infinity;
    const other = Math.max(0, stage.offsetHeight - host.offsetHeight);
    return Math.max(380, window.innerHeight * 0.94 - other - 16);
  };
  /** The side-by-side stage canvas depends on the window's height, which a width observer misses. */
  const fitOnTallnessChange = (cv) => {
    let ih = window.innerHeight;
    window.addEventListener('resize', () => {
      if (window.innerHeight === ih) return;
      ih = window.innerHeight;
      if (!isStacked()) cv.resize();
    }, { passive: true });
  };

  /** Mantissa/exponent with one decimal, carrying 9.96 → 1.0×10¹ correctly. */
  const sci = (v) => {
    let e = Math.floor(Math.log10(Math.abs(v)));
    let m = v / Math.pow(10, e);
    if (Math.abs(+m.toFixed(1)) >= 10) { m /= 10; e += 1; }
    return { m: m.toFixed(1), e };
  };
  const plainRange = (v) => Math.abs(v) >= 0.00995 && Math.abs(v) < 999.5;
  /** Compact magnitude: 4.82, 0.0371, or 3.1×10⁻⁷ for tiny / huge values. */
  const fmtMag = (v) => {
    if (!Number.isFinite(v)) return '∞';
    if (v === 0) return '0';
    if (plainRange(v)) return v.toPrecision(3);
    const { m, e } = sci(v);
    return `${m}×10${sup(e)}`;
  };
  /** Signed fixed-point with a real minus sign. */
  const fmtS = (v, d = 2) => (v < 0 ? '−' : '') + Math.abs(v).toFixed(d);
  /** One decimal without the leading zero (−.5, .7, 1.4), for tight columns on phones. */
  const fmtTight = (v) => (Math.abs(v) < 0.05 ? '0' : (v < 0 ? '−' : '') + Math.abs(v).toFixed(1).replace(/^0\./, '.'));

  /** Is this element (roughly) on screen? Lets each figure idle while the chapter is visible. */
  const visibility = (el) => {
    const st = { on: true };
    if (typeof IntersectionObserver !== 'undefined') {
      st.on = false;
      new IntersectionObserver((en) => { st.on = en[en.length - 1].isIntersecting; }, { rootMargin: '80px 0px' }).observe(el);
    }
    return st;
  };

  /** Pre-rendered glow sprite per colour (particles are drawn with drawImage, which is cheap). */
  const sprites = new Map();
  const sprite = (hex) => {
    if (sprites.has(hex)) return sprites.get(hex);
    const S = 48, c = document.createElement('canvas');
    c.width = c.height = S;
    const g = c.getContext('2d'), r = S / 2;
    const grd = g.createRadialGradient(r, r, 0, r, r, r);
    const [r0, g0, b0] = AM.hexToRgb(hex);
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

  /** Cubic Bézier helpers. B = [x0,y0,x1,y1,x2,y2,x3,y3]. */
  const bez = (B, t) => {
    const u = 1 - t, a = u * u * u, b = 3 * u * u * t, c = 3 * u * t * t, d = t * t * t;
    return { x: a * B[0] + b * B[2] + c * B[4] + d * B[6], y: a * B[1] + b * B[3] + c * B[5] + d * B[7] };
  };
  const bezLen = (B) => { let L = 0, p = bez(B, 0); for (let i = 1; i <= 20; i++) { const q = bez(B, i / 20); L += Math.hypot(q.x - p.x, q.y - p.y); p = q; } return L; };
  const strokeBez = (g, B) => { g.beginPath(); g.moveTo(B[0], B[1]); g.bezierCurveTo(B[2], B[3], B[4], B[5], B[6], B[7]); g.stroke(); };
  /** S-shaped Bézier from (x0,y0) to (x3,y3) leaving and arriving horizontally. */
  const sCurve = (x0, y0, x3, y3) => { const mx = (x0 + x3) / 2; return [x0, y0, mx, y0, mx, y3, x3, y3]; };

  // ------------------------------------------------------------------ toy residual stream (TOY NUMBERS, computed)
  const randMat = (r, rows, cols, sd) => Array.from({ length: rows }, () => Array.from({ length: cols }, () => MM.randn(r) * sd));
  const matVec = (W, x) => W.map((row) => { let s = 0; for (let j = 0; j < row.length; j++) s += row[j] * x[j]; return s; });

  /** A 16-d embedding x₀ for one token, plus six small random sublayers:
      even k = "attention" (output projection of a blend of LN(x) and a fixed
      context vector that stands in for other tokens), odd k = MLP (16→32→16, GELU). */
  function makeToy(seed) {
    const r = MM.rng(seed);
    const x0 = Array.from({ length: D_MODEL }, () => MM.randn(r));
    const subs = [];
    for (let k = 0; k < 6; k++) {
      if (k % 2 === 0) {
        const A = randMat(r, D_MODEL, D_MODEL, 0.55 / Math.sqrt(D_MODEL));
        const context = MM.layerNorm(Array.from({ length: D_MODEL }, () => MM.randn(r)));
        subs.push((u) => matVec(A, u.map((v, i) => 0.5 * v + 0.5 * context[i])));
      } else {
        const B1 = randMat(r, 32, D_MODEL, 1 / Math.sqrt(D_MODEL));
        const B2 = randMat(r, D_MODEL, 32, 0.6 / Math.sqrt(32));
        subs.push((u) => matVec(B2, matVec(B1, u).map(MM.gelu)));
      }
    }
    return { x0, subs };
  }

  /** Run the stream. on[k]: sublayer k is present. replace: x ← F(LN(x)) instead of x ← x + F(LN(x)). */
  function runToy(toy, on, replace) {
    const xs = [toy.x0.slice()], ws = [];
    for (let k = 0; k < 6; k++) {
      const x = xs[k];
      const w = toy.subs[k](MM.layerNorm(x, EPS)); // every sublayer reads LN(x)
      ws.push(w);
      xs.push(!on[k] ? x.slice() : replace ? w : x.map((v, i) => v + w[i]));
    }
    return { xs, ws, cos: xs.map((x) => MM.cosine(x, toy.x0)) };
  }

  // ------------------------------------------------------------------ chapter CSS
  AM.css(`
    #ch-${ID} .rs-badges { display: inline-flex; flex-wrap: wrap; gap: 6px; }
    #ch-${ID} .rs-stage .fig { gap: var(--space-2); }
    #ch-${ID} .rs-stage .controls { gap: var(--space-2) var(--space-4); }
    #ch-${ID} .rs-hint { font-family: var(--font-mono); font-size: var(--fs-micro); letter-spacing: 0.06em; color: var(--mist); }
    #ch-${ID} .rs-hint b { color: var(--linen); font-weight: 500; }
    #ch-${ID} .rs-cap-phone { display: none; }
    #ch-${ID} .rs-short { display: none; }
    @media (max-width: 900px) {
      #ch-${ID} .rs-cap-desk, #ch-${ID} .rs-hint { display: none; }
      #ch-${ID} .rs-cap-phone { display: block; }
      #ch-${ID} .rs-stage .fig-top { min-height: 0; }
    }
    @media (max-width: 520px) {
      #ch-${ID} .rs-stage .rs-long { display: none; }
      #ch-${ID} .rs-stage .rs-short { display: inline; }
    }
    @media (min-width: 901px) and (max-height: 860px) {
      #ch-${ID} .rs-cap-desk { display: none; }
      #ch-${ID} .rs-cap-phone { display: block; }
    }
    /* keep Greek letters lower-case inside the shared upper-case widgets */
    #ch-${ID} .ctl-label, #ch-${ID} .seg button { text-transform: none; }
    #ch-${ID} .greek { text-transform: none; letter-spacing: 0; }
    #ch-${ID} .step .math.block { font-size: 0.82em; }
    #ch-${ID} .rs-sec { display: grid; gap: var(--space-5); }
    #ch-${ID} .rs-sec > .prose { max-width: 68ch; }
    #ch-${ID} .rs-kicker { font-family: var(--font-mono); font-size: var(--fs-micro); letter-spacing: 0.16em; text-transform: uppercase; color: var(--weld); }

    /* LayerNorm stepper */
    #ch-${ID} .ln-grid { display: grid; grid-template-columns: minmax(0, 1fr) minmax(230px, 280px); gap: var(--space-5); align-items: start; }
    #ch-${ID} .ln-left { display: grid; gap: var(--space-4); min-width: 0; }
    #ch-${ID} .ln-canvas canvas { cursor: ns-resize; }
    #ch-${ID} .ln-read { display: grid; gap: var(--space-4); align-content: start; }
    #ch-${ID} .ln-formula { font-family: var(--font-mono); font-size: 11.5px; line-height: 1.9; color: var(--mist); padding: var(--space-3); border: 1px solid var(--rule); border-radius: var(--radius-sm); background: var(--ink); }
    #ch-${ID} .ln-formula span { transition: color 0.25s, text-shadow 0.25s; }
    #ch-${ID} .ln-formula span.is-on { color: var(--weld); text-shadow: 0 0 12px color-mix(in srgb, var(--weld) 45%, transparent); }
    #ch-${ID} .ln-formula span.is-done { color: var(--linen); }
    #ch-${ID} .ln-dl { display: grid; grid-template-columns: auto 1fr; gap: 7px 14px; margin: 0; font-family: var(--font-mono); font-size: 11.5px; }
    #ch-${ID} .ln-dl dt { color: var(--mist); font-size: 10.5px; letter-spacing: 0.06em; align-self: center; }
    #ch-${ID} .ln-dl dd { margin: 0; color: var(--linen); text-align: right; font-variant-numeric: tabular-nums; }
    #ch-${ID} .ln-explain { font-size: var(--fs-small); line-height: 1.55; color: var(--linen-dim); padding-left: var(--space-3); border-left: 2px solid color-mix(in srgb, var(--weld) 55%, transparent); }
    #ch-${ID} .ln-verdict { font-family: var(--font-mono); font-size: 11px; line-height: 1.55; padding: var(--space-3); border-radius: var(--radius-sm); border: 1px solid var(--rule); color: var(--linen-dim); }
    #ch-${ID} .ln-verdict b { display: block; font-weight: 500; letter-spacing: 0.08em; text-transform: uppercase; font-size: 10px; margin-bottom: 2px; }
    #ch-${ID} .ln-verdict.is-same { border-color: color-mix(in srgb, var(--verdigris) 45%, var(--rule)); }
    #ch-${ID} .ln-verdict.is-same b { color: var(--verdigris); }
    #ch-${ID} .ln-verdict.is-diff { border-color: color-mix(in srgb, var(--madder) 45%, var(--rule)); }
    #ch-${ID} .ln-verdict.is-diff b { color: var(--madder); }
    #ch-${ID} .ln-controls { display: grid; gap: var(--space-4); }
    #ch-${ID} .ln-controls .controls { gap: var(--space-3) var(--space-5); }
    @media (max-width: 760px) {
      /* phone: interleave so the step explanation sits right under the bars */
      #ch-${ID} .ln-grid { grid-template-columns: minmax(0, 1fr); gap: var(--space-4); }
      #ch-${ID} .ln-left, #ch-${ID} .ln-read { display: contents; }
      #ch-${ID} .ln-cv { order: 1; }
      #ch-${ID} .ln-explain { order: 2; }
      #ch-${ID} .ln-controls { order: 3; }
      #ch-${ID} .ln-verdict { order: 4; }
      #ch-${ID} .ln-formula { order: 5; }
      #ch-${ID} .ln-dl { order: 6; }
    }

  `);

  // ================================================================== FIGURE 1 — the river
  const DYE_ORDER = ['woad', 'verdigris', 'cochineal', 'saffron', 'lichen', 'madder'];
  const RIVER_STEPS = [
    { read: 0, write: 0, focus: -1 },
    { read: 1, write: 0, focus: 0, gate: 1 },
    { read: 2, write: 2, focus: [0, 1], plus: 1 },
    { read: 6, write: 6, focus: -1, bus: 1 },
    { read: 6, write: 6, focus: -1, replace: 1 },
  ];

  function buildRiver(ctx, host, toy) {
    const GOLD = AM.dye.weld;
    const DYES = DYE_ORDER.map((k) => AM.dye[k]);
    const COLS = [GOLD, ...DYES];               // particle colour index: 0 = the token itself, 1 + k = sublayer k
    const RGB = COLS.map(AM.hexToRgb);
    const kindOf = (k) => (k % 2 === 0 ? 'attention' : 'MLP');
    const blockOf = (k) => Math.floor(k / 2) + 1;

    const st = {
      read: new Array(6).fill(0), write: new Array(6).fill(0), focus: new Array(6).fill(0),
      pulse: new Array(6).fill(0), cacc: new Array(6).fill(0), rip: new Array(6).fill(0), flash: new Array(6).fill(0),
      bus: 0, replace: 0, gate: 0, plus: 0, acc: 0, dam: 0,
      tgt: { read: new Array(6).fill(0), write: new Array(6).fill(0), focus: new Array(6).fill(0), bus: 0, replace: 0, gate: 0, plus: 0 },
      strip: Array.from({ length: 7 }, () => new Array(D_MODEL).fill(0)),
      cos: new Array(7).fill(1),
      model: null,
    };
    const parts = [];   // particles in the river
    const wparts = [];  // particles travelling through a workshop (offtake → box → return)
    const ripples = [];
    let Lo = null, bg = null, primed = false;

    const cv = ctx.canvas(host, {
      label: 'Illustration of one token\'s residual stream as a rising river of gold light. Six workshops on its bank (attention and MLP sublayers of three blocks) draw a copy off through a LayerNorm gate and pour coloured dye back in by addition. A strip on the left shows the 16 numbers of the stream at each level.',
      height: (w) => (isStacked() ? Math.round(Math.min(w * 1.12, Math.max(330, window.innerHeight * 0.47)) - (w < 520 ? 30 : 0)) : Math.round(Math.min(w * 1.2, Math.max(520, window.innerHeight * 0.7), deskFit(host)))),
    });
    const vis = visibility(cv.wrap);
    fitOnTallnessChange(cv);

    // ---------------------------------------------------------------- toy numbers behind the strip
    function recompute() {
      const on = st.tgt.write.map((v) => v > 0.5);
      st.model = runToy(toy, on, st.tgt.replace > 0.5);
    }

    // ---------------------------------------------------------------- layout
    function layout(w, h) {
      const narrow = w < 520;
      const pad = narrow ? 2 : 8;
      const cw = narrow ? 3.6 : 6.4;                 // strip column per dimension
      const stripX = pad;
      const stripW = cw * D_MODEL;
      const cosX = stripX + stripW + (narrow ? 4 : 8);
      const cosW = narrow ? 24 : 34;
      const riverX = Math.round(cosX + cosW + (narrow ? 34 : 60));
      const topY = narrow ? 40 : 56;                 // river mouth
      const tileY = h - (narrow ? 17 : 22);
      const srcY = tileY - (narrow ? 15 : 19);       // river source, just above the token tile
      const M0 = srcY - (narrow ? 4 : 8);
      const s = (M0 - topY) / 6.62;                  // height of one sublayer
      const M = MM.range(7).map((k) => M0 - k * s);  // M[k]: level where x_k begins (⊕ of sublayer k-1)
      const hw0 = narrow ? 6 : 10, hwInc = narrow ? 1.7 : 2.8;
      const boxX = riverX + (narrow ? 64 : 106);
      const boxW = w - pad - boxX;
      const bank = riverX + hw0;
      const ws = MM.range(6).map((k) => {
        const top = M[k] - 0.9 * s, bot = M[k] - 0.17 * s;
        const R = M[k] - 0.07 * s;
        const entryY = M[k] - 0.38 * s, exitY = M[k] - 0.7 * s;
        const off = sCurve(bank, R, boxX, entryY);
        const ret = sCurve(boxX, exitY, bank + 1, M[k + 1]);
        const gate = bez(off, 0.5);
        return { k, top, bot, R, entryY, exitY, off, ret, gate, offLen: bezLen(off), retLen: bezLen(ret), mergeY: M[k + 1] };
      });
      return {
        w, h, narrow, pad, cw, stripX, stripW, cosX, cosW, riverX, topY, tileY, srcY, s, M, hw0, hwInc, boxX, boxW, ws,
        endY: topY - (narrow ? 4 : 8),
        sp: Math.max(40, h * 0.105),                 // river speed px/s
        chanSp: Math.max(46, h * 0.12),
        goldRate: narrow ? 17 : 25,
        copyRate: narrow ? 2.6 : 3.4,
        pr: narrow ? 1.55 : 2.0,                     // particle radius
      };
    }

    /** Half-width of the river at height y: wider above every tributary that adds to it. */
    function hw(y) {
      let v = Lo.hw0;
      const open = 1 - st.replace;
      if (open <= 0.001) return v;
      for (let k = 0; k < 6; k++) {
        if (st.write[k] < 0.001) continue;
        v += Lo.hwInc * st.write[k] * open * MM.smoothstep(Lo.M[k + 1] + 5, Lo.M[k + 1] - 5, y);
      }
      return v;
    }

    /** Mixture colour (RGB) of the river in segment s, for the faint water body. */
    function mixture(s) {
      // adding: gold plus a share of every dye poured in below
      let r = RGB[0][0], gg = RGB[0][1], b = RGB[0][2], wsum = 1;
      for (let k = 0; k < s; k++) {
        const wt = 0.6 * st.write[k];
        r += RGB[k + 1][0] * wt; gg += RGB[k + 1][1] * wt; b += RGB[k + 1][2] * wt; wsum += wt;
      }
      const add = [r / wsum, gg / wsum, b / wsum];
      // replacing: only the most recent writer's dye survives
      let last = 0;
      for (let k = 0; k < s; k++) if (st.tgt.write[k] > 0.5) last = k + 1;
      const rep = RGB[last];
      const m = st.replace;
      return [add[0] + (rep[0] - add[0]) * m, add[1] + (rep[1] - add[1]) * m, add[2] + (rep[2] - add[2]) * m];
    }

    // ---------------------------------------------------------------- particles
    function spawnGold(y) {
      parts.push({ y, lane: rand(-0.95, 0.95), lt: rand(-0.95, 0.95), col: 0, sp: Lo.sp * rand(0.82, 1.18), size: rand(0.65, 1.3), ph: rand(0, 6.28) });
    }
    /** Colour of a copy taken at offtake k: copy a real river particle near that level. */
    function copyColour(k) {
      const R = Lo.ws[k].R;
      if (st.bus > 0.5 && k === 4 && st.write[0] > 0.5 && Math.random() < 0.55) return 1; // block 3 reads block 1's dye
      const near = [];
      for (let i = 0; i < parts.length; i++) if (Math.abs(parts[i].y - R) < 14) near.push(parts[i].col);
      return near.length ? near[(Math.random() * near.length) | 0] : 0;
    }

    function update(dt) {
      const a = 1 - Math.exp(-dt * 3.0);
      const T = st.tgt;
      for (let k = 0; k < 6; k++) {
        st.read[k] += (T.read[k] - st.read[k]) * a;
        st.write[k] += (T.write[k] - st.write[k]) * a;
        st.focus[k] += (T.focus[k] - st.focus[k]) * a;
        st.pulse[k] *= Math.exp(-dt * 2.6);
        st.flash[k] *= Math.exp(-dt * 1.5);
        st.rip[k] = Math.max(0, st.rip[k] - dt);
      }
      st.bus += (T.bus - st.bus) * a;
      st.gate += (T.gate - st.gate) * a;
      st.plus += (T.plus - st.plus) * a;
      st.replace += (T.replace - st.replace) * (1 - Math.exp(-dt * 2.4));
      st.dam *= Math.exp(-dt * 2.2);

      // strip + cosine readouts glide toward the computed values
      const b = 1 - Math.exp(-dt * 4.5);
      const xs = st.model.xs, cs = st.model.cos;
      for (let s = 0; s < 7; s++) {
        for (let i = 0; i < D_MODEL; i++) st.strip[s][i] += (xs[s][i] - st.strip[s][i]) * b;
        st.cos[s] += (cs[s] - st.cos[s]) * b;
      }

      // gold token water rises from the source
      st.acc += dt * Lo.goldRate;
      while (st.acc >= 1) { st.acc -= 1; if (parts.length < 520) spawnGold(Lo.srcY + rand(-2, 4)); }

      // copies drawn off by reading workshops (only while the residual path is open)
      for (let k = 0; k < 6; k++) {
        const rate = Lo.copyRate * st.read[k] * (1 - st.replace);
        st.cacc[k] += dt * rate;
        while (st.cacc[k] >= 1) {
          st.cacc[k] -= 1;
          if (wparts.length < 160) wparts.push({ k, ph: 0, t: 0, v: rand(0.85, 1.2), col: copyColour(k), size: rand(0.6, 1.4), div: false });
        }
      }

      // river particles
      for (let i = parts.length - 1; i >= 0; i--) {
        const p = parts[i];
        const y0 = p.y;
        p.y -= p.sp * dt;
        p.lane += (p.lt - p.lane) * Math.min(1, dt * 0.8);
        if (p.y < Lo.endY) { parts[i] = parts[parts.length - 1]; parts.pop(); continue; }
        // with the residual path cut, the whole river is diverted into each workshop
        if (st.replace > 0.01) {
          for (let k = 0; k < 6; k++) {
            const R = Lo.ws[k].R;
            if (y0 > R && p.y <= R && T.write[k] > 0.5 && st.read[k] > 0.5 && Math.random() < st.replace) {
              wparts.push({ k, ph: 0, t: 0, v: rand(0.85, 1.2), col: p.col, size: p.size, div: true });
              parts[i] = parts[parts.length - 1]; parts.pop();
              break;
            }
          }
        }
      }

      // workshop particles
      for (let i = wparts.length - 1; i >= 0; i--) {
        const q = wparts[i], W = Lo.ws[q.k];
        let kill = false;
        if (q.ph === 0) {
          q.t += (dt * Lo.chanSp * q.v) / W.offLen;
          if (q.t >= 1) { q.ph = 1; q.t = 0; st.pulse[q.k] = Math.min(1, st.pulse[q.k] + 0.22); }
        } else if (q.ph === 1) {
          q.t += (dt * q.v) / 0.4;
          if (q.t >= 1) {
            if (T.write[q.k] < 0.5) kill = true;      // reading only: absorbed
            else { q.ph = 2; q.t = 0; q.col = q.k + 1; }
          }
        } else {
          q.t += (dt * Lo.chanSp * q.v) / W.retLen;
          if (q.t >= 1) {
            if (q.burst === 1) { st.pulse[q.k] = 1; st.flash[q.k] = 1; }
            // the add: dye joins the river at the bank and slowly mixes across it
            const y = W.mergeY;
            const lane0 = Math.min(1, Lo.hw0 / hw(y));
            parts.push({ y, lane: lane0, lt: rand(-0.9, 0.95), col: q.col, sp: Lo.sp * rand(0.85, 1.12), size: rand(0.7, 1.25), ph: rand(0, 6.28) });
            if (st.rip[q.k] <= 0 || q.burst === 1) { ripples.push({ x: Lo.riverX + hw(y), y, t: 0, col: COLS[q.k + 1], big: q.burst === 1 || (st.plus > 0.5 && st.focus[q.k] > 0.5) }); st.rip[q.k] = 0.55; }
            kill = true;
          }
        }
        if (kill) { wparts[i] = wparts[wparts.length - 1]; wparts.pop(); }
      }

      for (let i = ripples.length - 1; i >= 0; i--) { ripples[i].t += dt / 1.1; if (ripples[i].t >= 1) ripples.splice(i, 1); }
    }

    /** A burst of dye from workshop k: the first pour after it switches on. */
    function pour(k) {
      const n = Lo.narrow ? 9 : 13;
      for (let i = 0; i < n; i++) wparts.push({ k, ph: 2, t: -0.25 - i * 0.07, v: 1, col: k + 1, size: 1, div: false, burst: i === 0 ? 1 : 2 });
    }

    /** Fill the river so the very first frame already looks alive. */
    function prime() {
      parts.length = 0; wparts.length = 0; ripples.length = 0;
      for (let i = 0; i < 360; i++) update(1 / 30);
    }

    // ---------------------------------------------------------------- static background (cached)
    function buildBg() {
      const { w, h } = Lo;
      bg = document.createElement('canvas');
      bg.width = Math.round(w * cv.dpr); bg.height = Math.round(h * cv.dpr);
      const g = bg.getContext('2d');
      g.setTransform(cv.dpr, 0, 0, cv.dpr, 0, 0);
      // a soft vertical haze behind the river, like light on water
      const grd = g.createLinearGradient(Lo.riverX - 90, 0, Lo.riverX + 90, 0);
      grd.addColorStop(0, AM.rgba(GOLD, 0));
      grd.addColorStop(0.5, AM.rgba(GOLD, 0.045));
      grd.addColorStop(1, AM.rgba(GOLD, 0));
      g.fillStyle = grd;
      g.fillRect(Lo.riverX - 90, Lo.endY, 180, Lo.srcY - Lo.endY + 10);
      // faint warp threads behind the workshops
      D.weave(g, Lo.boxX - 6, Lo.topY - 6, Lo.boxW + 12, Lo.M[0] - Lo.topY + 6, { step: 7, alpha: 0.025 });
    }

    // ---------------------------------------------------------------- drawing
    function drawStrip(g) {
      const { stripX, cw, cosX, cosW, M, topY, narrow } = Lo;
      const vmax = 2.6;
      for (let s = 0; s < 7; s++) {
        const yBot = M[s], yTop = s === 6 ? topY : M[s + 1];
        const hgt = yBot - yTop - 2;
        for (let i = 0; i < D_MODEL; i++) {
          g.fillStyle = AM.color.div(st.strip[s][i], vmax);
          g.fillRect(stripX + i * cw, yTop + 1, cw - (narrow ? 0.7 : 1.2), hgt);
        }
        // cosine similarity with the original embedding x₀ (how much of the token survives)
        const c = st.cos[s];
        const cy = (yTop + yBot) / 2;
        D.text(g, fmtS(c, 2), cosX + cosW, cy + 3, { role: 'mono', size: narrow ? 8 : 9.5, align: 'right', color: c > 0.5 ? GOLD : AM.col.mist, alpha: 0.45 + 0.55 * Math.max(0, Math.min(1, c)) });
        // a little gold gauge under the number
        const gw = (cosW - 2) * Math.max(0, Math.min(1, c));
        g.fillStyle = AM.rgba(GOLD, 0.55);
        g.fillRect(cosX + cosW - gw, cy + (narrow ? 6 : 7), gw, 1.5);
      }
      // the add, made visible: at each seam (level of the ⊕), the vector that sublayer k
      // wrote, F(LN(x)), as a small barcode in its dye. Band k+1 = band k + this barcode.
      // With the residual path cut nothing is added, so the barcodes fade out.
      const bh = narrow ? 14 : 22;
      for (let k = 0; k < 6; k++) {
        const a = st.write[k] * (1 - st.replace);
        const dye = COLS[k + 1];
        const yS = M[k + 1];
        const yTop = k === 5 ? topY : M[k + 2];
        if (st.flash[k] > 0.02) {
          const f = st.flash[k];
          g.save();
          g.fillStyle = AM.rgba(dye, 0.16 * f);
          g.fillRect(stripX - 1, yTop, Lo.stripW + 1, yS - yTop);
          g.strokeStyle = AM.rgba(dye, 0.85 * f);
          g.lineWidth = 1.2;
          g.strokeRect(stripX - 1.5, yTop + 0.5, Lo.stripW + 2, yS - yTop - 1);
          g.restore();
        }
        if (a < 0.02) continue;
        g.fillStyle = AM.rgba(AM.col.ink, 0.82 * a);
        g.fillRect(stripX - 1, yS - bh / 2 - 1, Lo.stripW + 1, bh + 2);
        g.fillStyle = AM.rgba(dye, 0.35 * a);
        g.fillRect(stripX, yS - 0.5, Lo.stripW - 1, 1);
        const wv = st.model.ws[k];
        g.fillStyle = AM.rgba(dye, 0.95 * a);
        for (let i = 0; i < D_MODEL; i++) {
          const t = MM.clamp(wv[i], -1, 1) * st.write[k];   // ±1 fills the seam band
          const hh = Math.max(0.8, Math.abs(t) * (bh / 2));
          g.fillRect(stripX + i * cw + (narrow ? 0.3 : 0.8), t >= 0 ? yS - hh : yS, cw - (narrow ? 1.3 : 2.4), hh);
        }
        g.fillStyle = AM.rgba(dye, 0.9 * st.write[k]);
        g.fillRect(stripX + Lo.stripW + 1, yS - 1, narrow ? 2.5 : 4, 2);
      }
      const hy = topY - (narrow ? 9 : 12);
      D.text(g, 'STREAM x', stripX, hy, { role: 'mono', size: 8.5, color: AM.col.mist, letterSpacing: '0.08em' });
      D.text(g, 'COS', cosX + cosW, hy, { role: 'mono', size: 8.5, color: AM.col.mist, align: 'right', letterSpacing: '0.08em' });
      D.text(g, narrow ? 'x' + sub(0) : 'x' + sub(0) + ' · 16 dims', stripX, Lo.M[0] + (narrow ? 11 : 14), { role: 'mono', size: narrow ? 8 : 8.5, color: AM.col.mist });
      if (!narrow) D.text(g, 'vs x' + sub(0), cosX + cosW, Lo.M[0] + 14, { role: 'mono', size: 8.5, color: AM.col.mist, align: 'right' });
    }

    function drawChannels(g) {
      g.save();
      g.lineCap = 'round';
      for (let k = 0; k < 6; k++) {
        const W = Lo.ws[k];
        const r = st.read[k], wr = st.write[k];
        // ghost outline of the path while the workshop is idle
        if (r < 0.98) {
          g.setLineDash([2, 4]);
          g.strokeStyle = AM.rgba(AM.col.ruleStrong, 0.7 * (1 - r));
          g.lineWidth = 1;
          strokeBez(g, W.off); strokeBez(g, W.ret);
          g.setLineDash([]);
        }
        if (r > 0.01) {
          const mix = mixture(k);
          const col = `rgba(${mix[0] | 0},${mix[1] | 0},${mix[2] | 0},`;
          g.strokeStyle = col + 0.09 * r + ')'; g.lineWidth = Lo.narrow ? 6 : 9; strokeBez(g, W.off);
          g.strokeStyle = col + 0.4 * r + ')'; g.lineWidth = 1.1; strokeBez(g, W.off);
        }
        if (wr > 0.01) {
          const dye = COLS[k + 1];
          g.strokeStyle = AM.rgba(dye, 0.1 * wr); g.lineWidth = Lo.narrow ? 6 : 9; strokeBez(g, W.ret);
          g.strokeStyle = AM.rgba(dye, 0.5 * wr); g.lineWidth = 1.2; strokeBez(g, W.ret);
        }
      }
      g.restore();
    }

    function drawRiverBody(g) {
      const { riverX, endY, srcY, M, topY } = Lo;
      // vertical gradient of mixture colours, one stop per segment
      const grd = g.createLinearGradient(0, endY, 0, srcY);
      for (let s = 0; s < 7; s++) {
        const yc = s === 6 ? (topY + M[6]) / 2 : (M[s] + M[s + 1]) / 2;
        const c = mixture(s);
        grd.addColorStop(MM.clamp((yc - endY) / (srcY - endY), 0, 1), `rgba(${c[0] | 0},${c[1] | 0},${c[2] | 0},1)`);
      }
      const ys = [];
      for (let y = endY; y < srcY; y += 5) ys.push(y);
      ys.push(srcY);
      const path = (extra) => {
        g.beginPath();
        ys.forEach((y, i) => { const x = riverX - hw(y) - extra; if (i) g.lineTo(x, y); else g.moveTo(x, y); });
        for (let i = ys.length - 1; i >= 0; i--) g.lineTo(riverX + hw(ys[i]) + extra, ys[i]);
        g.closePath();
      };
      const k = 1 - 0.75 * st.replace;
      g.save();
      g.fillStyle = grd;
      g.globalAlpha = 0.06 * k; path(10); g.fill();
      g.globalAlpha = 0.08 * k; path(4); g.fill();
      g.globalAlpha = 0.14 * k; path(0); g.fill();
      g.restore();
    }

    /** The braid: one silk strand per contributor. Gold from the embedding, plus a
        dye strand from every sublayer that has written. Strands are cut at the
        dams when the residual path is removed. */
    function drawStrands(g, t) {
      const { riverX, endY, srcY, hw0, narrow } = Lo;
      const k0 = (2 * Math.PI) / (narrow ? 95 : 130);
      const step = narrow ? 3 : 4;
      g.save();
      g.lineCap = 'round';
      g.lineJoin = 'round';
      for (let j = 0; j < 7; j++) {
        const pres = j === 0 ? 1 : st.write[j - 1];
        if (pres < 0.01) continue;
        const y0 = j === 0 ? srcY : Lo.M[j];
        let cut = endY;
        for (let k = j; k < 6; k++) if (st.tgt.write[k] > 0.5) { cut = Lo.ws[k].R; break; }
        const amp = 0.5 + 0.38 * (((j * 0.618) % 1));
        const ph = j * 2.1;
        const xAt = (y) => {
          const braid = riverX + hw(y) * amp * Math.sin(y * k0 + ph + t * 1.5);
          if (j === 0) return braid;
          const e = MM.smoothstep(y0, y0 - 26, y);       // enter from the bank, then join the braid
          return riverX + hw0 + (braid - riverX - hw0) * e;
        };
        const seg = (ya, yb, alpha) => {
          if (alpha < 0.01 || ya - yb < 2) return;
          g.beginPath();
          for (let y = ya, i = 0; y >= yb; y -= step, i++) { const x = xAt(y); if (i) g.lineTo(x, y); else g.moveTo(x, y); }
          const col = COLS[j];
          g.strokeStyle = AM.rgba(col, 0.13 * alpha); g.lineWidth = narrow ? 3.5 : 4.5; g.stroke();
          g.strokeStyle = AM.rgba(col, 0.78 * alpha); g.lineWidth = narrow ? 0.9 : 1.15; g.stroke();
        };
        const m = st.replace;
        if (m < 0.01 || cut <= endY) seg(y0, endY, pres);
        else { seg(y0, cut, pres); seg(cut, endY, pres * (1 - m)); }
      }
      g.restore();
    }

    function fadeMouth(g) {
      const { riverX, endY, topY } = Lo;
      const fade = g.createLinearGradient(0, endY - 2, 0, topY + 16);
      fade.addColorStop(0, 'rgba(0,0,0,1)');
      fade.addColorStop(1, 'rgba(0,0,0,0)');
      g.save();
      g.globalCompositeOperation = 'destination-out';
      g.fillStyle = fade;
      const half = Lo.hw0 + 6 * Lo.hwInc + 16;
      g.fillRect(riverX - half, endY - 6, half * 2, topY + 16 - endY + 6);
      g.restore();
    }

    function drawParticles(g, t) {
      const { riverX, endY, srcY, pr } = Lo;
      g.save();
      g.globalCompositeOperation = 'lighter';
      for (let i = 0; i < parts.length; i++) {
        const p = parts[i];
        const x = riverX + p.lane * hw(p.y) + Math.sin(t * 1.3 + p.ph) * 0.7;
        const fadeTop = MM.clamp((p.y - endY) / 34, 0, 1), fadeBot = MM.clamp((srcY + 4 - p.y) / 10, 0, 1);
        const r = pr * p.size * (0.85 + 0.15 * Math.sin(t * 3 + p.ph * 2));
        g.globalAlpha = 0.62 * fadeTop * fadeBot;
        const S = r * 7;
        g.drawImage(sprite(COLS[p.col]), x - S / 2, p.y - S / 2, S, S);
      }
      for (let i = 0; i < wparts.length; i++) {
        const q = wparts[i], W = Lo.ws[q.k];
        if (q.ph === 1 || q.t < 0) continue;
        const pt = q.ph === 0 ? bez(W.off, q.t) : bez(W.ret, q.t);
        // before the LayerNorm gate sizes vary; after it they are uniform (normalised)
        const r = q.ph === 0 && q.t < 0.5 ? pr * q.size : pr * (q.burst ? 1.15 : 0.95);
        const fade = q.ph === 0 ? MM.clamp((1 - q.t) / 0.06, 0, 1) : MM.clamp(q.t / 0.06, 0, 1);
        g.globalAlpha = 0.7 * fade;
        const S = r * 7;
        g.drawImage(sprite(COLS[q.col]), pt.x - S / 2, pt.y - S / 2, S, S);
      }
      g.restore();
    }

    function drawRipples(g) {
      g.save();
      for (const rp of ripples) {
        const e = MM.ease.out(rp.t);
        g.strokeStyle = AM.rgba(rp.col, (rp.big ? 0.7 : 0.4) * (1 - rp.t));
        g.lineWidth = rp.big ? 1.4 : 1;
        g.beginPath();
        g.ellipse(rp.x - 2, rp.y, 3 + e * (rp.big ? 22 : 12), 1.5 + e * (rp.big ? 7 : 4), 0, 0, Math.PI * 2);
        g.stroke();
      }
      g.restore();
    }

    function drawWorkshops(g, t) {
      const { boxX, boxW, narrow } = Lo;
      for (let k = 0; k < 6; k++) {
        const W = Lo.ws[k];
        const dye = COLS[k + 1];
        const on = Math.max(st.read[k], st.write[k]);
        const f = st.focus[k];
        const bh = W.bot - W.top;
        // box
        g.save();
        if (on > 0.05) { g.shadowColor = AM.rgba(dye, 0.35 * on + 0.35 * f + 0.4 * st.pulse[k]); g.shadowBlur = 10 + 14 * f; }
        D.roundRect(g, boxX, W.top, boxW, bh, narrow ? 6 : 8);
        g.fillStyle = AM.mix(AM.col.ink, AM.col.ink2, 0.4 + 0.6 * on);
        g.fill();
        g.shadowBlur = 0;
        if (on < 0.05) { g.setLineDash([3, 4]); g.strokeStyle = AM.rgba(AM.col.ruleStrong, 0.9); }
        else g.strokeStyle = AM.rgba(dye, MM.clamp(0.28 + 0.45 * on + 0.3 * f + 0.4 * st.pulse[k], 0, 1));
        g.lineWidth = 1 + 0.6 * f;
        g.stroke();
        g.setLineDash([]);
        // label
        const la = 0.35 + 0.65 * on;
        const name = narrow ? (k % 2 === 0 ? 'ATTN' : 'MLP') : kindOf(k).toUpperCase();
        const labelW = narrow ? 44 : 92;
        if (narrow) {
          D.text(g, name, boxX + 7, W.top + bh / 2 - 1, { role: 'mono', size: 8.5, color: on > 0.5 ? AM.col.linen : AM.col.mist, alpha: la, letterSpacing: '0.06em' });
          D.text(g, 'block ' + blockOf(k), boxX + 7, W.top + bh / 2 + 9.5, { role: 'mono', size: 8.5, color: on > 0.5 ? dye : AM.col.mist, alpha: la });
        } else {
          D.text(g, name, boxX + 12, W.top + bh / 2 - 2, { role: 'mono', size: 9.5, color: on > 0.5 ? AM.col.linen : AM.col.mist, alpha: la, letterSpacing: '0.1em' });
          D.text(g, 'block ' + blockOf(k), boxX + 12, W.top + bh / 2 + 12, { role: 'mono', size: 8.5, color: on > 0.5 ? dye : AM.col.mist, alpha: la });
        }
        // the write vector F(LN(x)) as a barcode, in this sublayer's dye
        const bx = boxX + labelW, bw = boxW - labelW - (narrow ? 6 : 12);
        if (bw > 20) {
          g.fillStyle = AM.rgba(AM.col.linen, 0.07);
          g.fillRect(bx, W.top + bh / 2 - 0.5, bw, 1);
          D.vectorBars(g, bx, W.top + 5, bw, bh - 10, st.model.ws[k], { max: 1.6, pos: dye, neg: dye, alpha: 0.18 + 0.82 * st.write[k] + 0.15 * st.read[k] * (1 - st.write[k]) });
        }
        g.restore();

        // LayerNorm gate on the offtake
        const gx = W.gate.x, gy = W.gate.y;
        const gw = narrow ? 20 : 22, gh = narrow ? 13 : 14;
        const ga = 0.3 + 0.7 * st.read[k];
        g.save();
        if (st.gate > 0.02 && st.focus[k] > 0.5) {
          const pulse = 0.5 + 0.5 * Math.sin(t * 4);
          g.strokeStyle = AM.rgba(GOLD, 0.5 * st.gate * (0.4 + 0.6 * pulse));
          g.lineWidth = 1.2;
          D.roundRect(g, gx - gw / 2 - 4 - 2 * pulse, gy - gh / 2 - 4 - 2 * pulse, gw + 8 + 4 * pulse, gh + 8 + 4 * pulse, 7);
          g.stroke();
        }
        D.roundRect(g, gx - gw / 2, gy - gh / 2, gw, gh, 4);
        g.fillStyle = AM.col.ink;
        g.fill();
        g.strokeStyle = AM.rgba(st.read[k] > 0.5 ? GOLD : AM.col.ruleStrong, st.read[k] > 0.5 ? 0.85 * ga : 0.9);
        g.lineWidth = 1;
        g.stroke();
        D.text(g, 'LN', gx, gy + 0.5, { role: 'mono', size: narrow ? 8.5 : 8, align: 'center', baseline: 'middle', color: st.read[k] > 0.5 ? GOLD : AM.col.mist, alpha: ga });
        g.restore();

        // the add: ⊕ where the return channel meets the river
        const px = W.ret[6] + (narrow ? 1 : 2), py = W.ret[7];
        const pa = st.write[k];
        const rr = narrow ? 4.2 : 5.5;
        g.save();
        g.beginPath(); g.arc(px, py, rr, 0, Math.PI * 2);
        g.fillStyle = AM.col.ink; g.fill();
        g.strokeStyle = pa > 0.05 ? AM.rgba(dye, 0.35 + 0.65 * pa) : AM.rgba(AM.col.ruleStrong, 0.9);
        g.lineWidth = 1.1; g.stroke();
        g.strokeStyle = pa > 0.05 ? AM.rgba(AM.col.linen, 0.4 + 0.6 * pa) : AM.rgba(AM.col.mist, 0.6);
        g.beginPath(); g.moveTo(px - rr * 0.5, py); g.lineTo(px + rr * 0.5, py); g.moveTo(px, py - rr * 0.5); g.lineTo(px, py + rr * 0.5); g.stroke();
        g.restore();
      }
    }

    /** Step 5: a thread from where block 1's attention wrote to where block 3's attention reads. */
    function drawBus(g, t) {
      const a = st.bus * (1 - st.replace);   // with the stream cut there is no shared bus to read from
      if (a < 0.02) return;
      const y1 = Lo.M[1], y2 = Lo.ws[4].R;
      const x1 = Lo.riverX - hw(y1) - 4, x2 = Lo.riverX - hw(y2) - 4;
      const lift = Lo.narrow ? 13 : 24;
      const bend = lift / Math.max(1, y1 - y2);
      g.save();
      g.globalAlpha = 0.9;
      D.thread(g, x1, y1, x2, y2, { color: AM.col.ink, width: 6, alpha: 0.75 * a, bend, glow: false });
      g.restore();
      D.thread(g, x1, y1, x2, y2, { color: COLS[1], width: Lo.narrow ? 1.8 : 2.3, alpha: a, bend });
      const tip = D.threadPoint(x1, y1, x2, y2, 0.97, { bend });
      D.arrow(g, tip.x, tip.y + 6, x2, y2 + 0.5, { color: AM.rgba(COLS[1], a), width: 1.4, head: 6 });
      for (let i = 0; i < 3; i++) {
        const u = (t * 0.32 + i / 3) % 1;
        const p = D.threadPoint(x1, y1, x2, y2, u, { bend });
        D.glowDot(g, p.x, p.y, Lo.narrow ? 1.6 : 2.2, COLS[1], a * Math.sin(Math.PI * u));
      }
      // mark both ends
      g.save();
      g.strokeStyle = AM.rgba(COLS[1], 0.8 * a);
      g.lineWidth = 1;
      g.beginPath(); g.arc(x1, y1, 3, 0, Math.PI * 2); g.stroke();
      g.restore();
    }

    function drawLabels(g) {
      const { riverX, tileY, narrow, w, pad } = Lo;
      // token at the source
      const tile = D.token(g, 'crown', riverX, tileY, { size: narrow ? 12 : 14, selected: true });
      D.text(g, narrow ? 'embedding' : 'token embedding → x' + sub(0), tile.x + tile.w + 8, tileY + 3, { role: 'mono', size: 8.5, color: AM.col.mist, letterSpacing: '0.04em' });
      // the mouth
      D.text(g, narrow ? '↑ to unembedding' : '↑ to final LN + unembedding', riverX, narrow ? 13 : 16, { role: 'mono', size: 8.5, align: 'center', color: AM.col.linenDim, letterSpacing: '0.04em' });
      // the update rule currently in force
      const m = st.replace;
      const fs = narrow ? 8.5 : 10.5;
      const fy = narrow ? 13 : 16;
      if (m < 0.98) D.text(g, 'x ← x + F(LN(x))', w - pad, fy, { role: 'mono', size: fs, align: 'right', color: GOLD, alpha: 1 - m });
      if (m > 0.02) D.text(g, 'x ← F(LN(x))', w - pad, fy, { role: 'mono', size: fs, align: 'right', color: AM.dye.madder, alpha: m });
      D.text(g, m > 0.5 ? 'residual path cut' : 'residual: add', w - pad, fy + (narrow ? 12 : 15), { role: 'mono', size: narrow ? 8.5 : 8, align: 'right', color: AM.col.mist, letterSpacing: '0.08em' });
      // dam bars where the river is cut
      if (m > 0.02) {
        g.save();
        for (let k = 0; k < 6; k++) {
          if (st.tgt.write[k] < 0.5) continue;
          const y = Lo.ws[k].R - (narrow ? 3 : 4);
          const half = hw(y) + 4;
          if (st.dam > 0.02) {
            g.strokeStyle = AM.rgba(AM.dye.madder, 0.3 * st.dam);
            g.lineWidth = 8;
            g.beginPath(); g.moveTo(riverX - half - 4, y); g.lineTo(riverX + half, y); g.stroke();
          }
          g.strokeStyle = AM.rgba(AM.dye.madder, 0.85 * m);
          g.lineWidth = 2;
          g.beginPath(); g.moveTo(riverX - half, y); g.lineTo(riverX + half - 4, y); g.stroke();
        }
        g.restore();
      }
    }

    function frame(t) {
      if (!Lo || !st.model) return;
      const g = cv.g;
      cv.clear();
      if (bg) g.drawImage(bg, 0, 0, Lo.w, Lo.h);
      drawStrip(g);
      drawChannels(g);
      drawRiverBody(g);
      drawStrands(g, t);
      drawBus(g, t);
      drawParticles(g, t);
      fadeMouth(g);
      drawRipples(g);
      drawWorkshops(g, t);
      drawLabels(g);
    }

    cv.onResize((w, h) => {
      Lo = layout(w, h);
      buildBg();
      if (!st.model) recompute();
      if (!primed) {
        // snap animated values to the current step so the first frame is already settled
        for (let k = 0; k < 6; k++) { st.read[k] = st.tgt.read[k]; st.write[k] = st.tgt.write[k]; st.focus[k] = st.tgt.focus[k]; }
        st.replace = st.tgt.replace; st.bus = st.tgt.bus; st.gate = st.tgt.gate; st.plus = st.tgt.plus;
        st.strip = st.model.xs.map((x) => x.slice()); st.cos = st.model.cos.slice();
      }
      prime();
      primed = true;
      frame(performance.now() / 1000);
    });

    ctx.loop((t, dt) => {
      if (!vis.on || !Lo) return;
      update(dt);
      frame(t);
    });

    return {
      setStep(i) {
        const S = RIVER_STEPS[i] || RIVER_STEPS[0];
        if (primed && Lo) {
          for (let k = 0; k < Math.min(S.write, 6); k++) if (st.tgt.write[k] < 0.5) pour(k);
          if (S.replace && st.tgt.replace < 0.5) st.dam = 1;
        }
        for (let k = 0; k < 6; k++) {
          st.tgt.read[k] = k < S.read ? 1 : 0;
          st.tgt.write[k] = k < S.write ? 1 : 0;
          st.tgt.focus[k] = (Array.isArray(S.focus) ? S.focus.includes(k) : k === S.focus) ? 1 : 0;
        }
        st.tgt.bus = S.bus ? 1 : 0;
        st.tgt.gate = S.gate ? 1 : 0;
        st.tgt.plus = S.plus ? 1 : 0;
        st.tgt.replace = S.replace ? 1 : 0;
        recompute();
        if (!vis.on && Lo) frame(performance.now() / 1000);
      },
      setReplace(on) {
        if (on && st.tgt.replace < 0.5) st.dam = 1;
        st.tgt.replace = on ? 1 : 0;
        recompute();
      },
      get replace() { return st.tgt.replace > 0.5; },
    };
  }

  // ================================================================== FIGURE 2 — LayerNorm stepper
  const LN_BASE = [2.6, 0.3, 3.4, 1.1, -0.7, 2.2, 3.9, 0.8, 1.7, -0.3, 2.9, 1.4];
  const LN_GAMMA = [1.35, 0.7, 1.1, 0.55, 1.25, 0.9, 1.5, 0.8, 1.2, 0.65, 1.0, 1.4];
  const LN_BETA = [0.3, -0.45, 0.0, 0.55, -0.2, 0.35, -0.5, 0.1, 0.45, -0.3, 0.2, -0.35];
  const LN_N = LN_BASE.length;
  const YMIN = -3.5, YMAX = 5.5;

  /** Exact LayerNorm / RMSNorm, returning every intermediate stage. */
  function normStages(x, kind) {
    const mu = mean(x);
    let v = 0; for (const a of x) v += (a - mu) * (a - mu); v /= x.length;   // population variance, as LayerNorm uses
    const sd = Math.sqrt(v + EPS);
    let ms = 0; for (const a of x) ms += a * a; ms /= x.length;
    const rms = Math.sqrt(ms + EPS);
    if (kind === 'rms') {
      const hat = x.map((a) => a / rms);
      return { mu, sigma: Math.sqrt(v), sd, rms, stages: [x.slice(), x.slice(), hat, hat.map((a, i) => LN_GAMMA[i] * a)] };
    }
    const cen = x.map((a) => a - mu);
    const hat = cen.map((a) => a / sd);
    return { mu, sigma: Math.sqrt(v), sd, rms, stages: [x.slice(), cen, hat, hat.map((a, i) => LN_GAMMA[i] * a + LN_BETA[i])] };
  }

  function buildLayerNorm(ctx, parts) {
    const { canvasHost, readHost, ctlHost, verdictHost } = parts;
    const S = { base: LN_BASE.slice(), stage: 0, kind: 'ln', u: 0, shift: 0, sweep: null, drag: -1 };
    const scaleOf = () => Math.pow(10, S.u);
    const disp = new Array(LN_N).fill(0);
    const ghost = new Array(LN_N).fill(0);
    const band = { c: 0, h: 0, a: 0 };
    let cur = null, Lo = null, primed = false;
    let x10Btn = null;
    const sweepTarget = () => (S.sweep ? S.sweep.to : S.u);

    const cv = ctx.canvas(canvasHost, {
      label: 'Bar chart of a 12-dimensional vector going through LayerNorm one step at a time: subtract the mean, divide by the standard deviation, then apply a learned per-dimension scale and shift.',
      height: (w) => (w < 520 ? 300 : 340),
    });
    cv.wrap.classList.add('ln-canvas');
    const vis = visibility(cv.wrap);

    // ---------------------------------------------------------------- readouts (DOM)
    const el = AM.el;
    const fLN = el('div', { class: 'ln-formula', 'aria-live': 'polite' });
    const ddA = el('dd'), ddB = el('dd'), ddEps = el('dd', {}, '0.00001'), ddStage = el('dd');
    const dtA = el('dt'), dtB = el('dt');
    const verdict = el('div', { class: 'ln-verdict' });
    const explain = el('p', { class: 'ln-explain' });
    readHost.append(fLN, el('dl', { class: 'ln-dl' }, dtA, ddA, dtB, ddB, el('dt', {}, 'ε'), ddEps, el('dt', {}, 'this stage'), ddStage), explain);
    verdictHost.append(verdict);

    const STAGE_NAMES = {
      ln: ['x', 'x − μ', 'x̂ = (x − μ) / √(σ² + ε)', 'y = γ ⊙ x̂ + β'],
      rms: ['x', 'x  (no centring)', 'x̂ = x / √(mean(x²) + ε)', 'y = γ ⊙ x̂'],
    };
    const EXPLAIN = {
      ln: [
        'The input: one token\'s vector. Its mean μ is the dashed gold line, and the shaded band is μ ± σ.',
        'Subtract μ from every number. The vector now averages to zero; its spread σ is unchanged.',
        'Divide by √(σ² + ε). The spread becomes 1 (up to the tiny ε), whatever it was before.',
        'Multiply each dimension by its learned γ and add its learned β. Training tunes these, so the model can re-stretch any dimension that needs it.',
      ],
      rms: [
        'The input. RMSNorm measures the root-mean-square, rms = √(mean(x²) + ε), shown as the band ± rms around zero.',
        'RMSNorm skips the centring step, so the mean stays wherever it was.',
        'Divide by rms. The root-mean-square of the vector becomes 1.',
        'Multiply each dimension by its learned γ. There is no β.',
      ],
    };
    const STAGE_SHORT = { ln: ['x', 'x − μ', 'x̂ = (x−μ)/σ', 'γ ⊙ x̂ + β'], rms: ['x', 'x (no centring)', 'x̂ = x / rms', 'γ ⊙ x̂'] };

    function formulaHTML() {
      const s = S.stage;
      const cls = (i) => (s === i ? 'is-on' : s > i ? 'is-done' : '');
      if (S.kind === 'rms') {
        return `RMSNorm(x) =<br><span class="${cls(3)}">γ ⊙</span> <span class="${cls(0)}">x</span> <span class="${cls(2)}">/ √(mean(x²) + ε)</span>`;
      }
      return `LN(x) =<br><span class="${cls(3)}">γ ⊙</span> <span class="${cls(1)}">(<span class="${s === 0 ? 'is-on' : ''}">x</span> − μ)</span> <span class="${cls(2)}">/ √(σ² + ε)</span> <span class="${cls(3)}">+ β</span>`;
    }

    // ---------------------------------------------------------------- math
    function recompute() {
      const sc = scaleOf();
      const input = S.base.map((v) => v * sc + S.shift);
      cur = normStages(input, S.kind);
      cur.input = input;
      const ref = normStages(S.base, S.kind);   // same vector at ×1, +0
      let dmax = 0;
      for (let i = 0; i < LN_N; i++) dmax = Math.max(dmax, Math.abs(cur.stages[3][i] - ref.stages[3][i]));
      cur.dmax = dmax;
      refresh();
    }

    function refresh() {
      fLN.innerHTML = formulaHTML();
      explain.textContent = EXPLAIN[S.kind][S.stage];
      if (S.kind === 'rms') {
        dtA.textContent = 'rms'; ddA.textContent = cur.rms.toFixed(3);
        dtB.textContent = 'mean μ'; ddB.textContent = fmtS(cur.mu, 3) + ' (unused)';
      } else {
        dtA.textContent = 'mean μ'; ddA.textContent = fmtS(cur.mu, 3);
        dtB.textContent = 'std σ'; ddB.textContent = cur.sigma.toFixed(3);
      }
      const v = cur.stages[S.stage];
      const m = mean(v), sd = stdPop(v);
      if (S.kind === 'rms' && S.stage === 2) {
        let ms = 0; for (const a of v) ms += a * a;
        ddStage.textContent = 'rms ' + Math.sqrt(ms / v.length).toFixed(3);
      } else ddStage.textContent = `mean ${fmtS(m, 3)} · std ${sd.toFixed(3)}`;

      if (x10Btn) x10Btn.textContent = sweepTarget() > 0.5 ? 'Back to ×1' : 'Scale input ×10';
      const sc = scaleOf();
      const isRef = Math.abs(S.u) < 1e-9 && Math.abs(S.shift) < 1e-9;
      const same = cur.dmax < 1e-3;
      verdict.className = 'ln-verdict ' + (isRef ? '' : same ? 'is-same' : 'is-diff');
      const what = `input ×${sc < 1 ? sc.toFixed(2) : sc.toPrecision(3)}${Math.abs(S.shift) > 1e-9 ? ` ${S.shift < 0 ? '−' : '+'} ${Math.abs(S.shift).toFixed(1)}` : ''}`;
      const shifted = Math.abs(S.shift) > 1e-9;
      // where a tiny non-zero change comes from: ε (only matters when the scale changes) or float rounding
      const why = cur.dmax === 0 ? '' : cur.dmax < 1e-9 ? ' (floating-point rounding)' : ' (from ε)';
      if (isRef) verdict.innerHTML = '<b>Invariance check</b>Scale or shift the input with the sliders. The output y is compared with the output for the unscaled, unshifted input.';
      else if (same) verdict.innerHTML = `<b>Output unchanged</b>${what}: largest change in y is ${fmtMag(cur.dmax)}${why}.`;
      else if (S.kind === 'rms' && shifted) verdict.innerHTML = `<b>Output changed</b>${what}: largest change in y is ${cur.dmax.toFixed(3)}. RMSNorm does not subtract the mean, so a shift gets through.`;
      else verdict.innerHTML = `<b>Output changed</b>${what}: largest change in y is ${fmtMag(cur.dmax)}. At this scale ε is no longer tiny next to ${S.kind === 'rms' ? 'mean(x²)' : 'σ²'}, so it starts to matter.`;
    }

    // ---------------------------------------------------------------- layout & drawing
    function layout(w, h) {
      const narrow = w < 520;
      const ax = narrow ? 28 : 40, ar = narrow ? 24 : 40;
      const ay = 34, rows = narrow ? 32 : 34;
      const ah = h - ay - rows - 14;
      return { w, h, narrow, ax, ay, aw: w - ax - ar, ah, rowsY: ay + ah + 14 };
    }
    const yOf = (v) => Lo.ay + ((YMAX - v) / (YMAX - YMIN)) * Lo.ah;
    const vOf = (y) => YMAX - ((y - Lo.ay) / Lo.ah) * (YMAX - YMIN);

    function bandTarget() {
      const s = S.stage;
      if (S.kind === 'rms') {
        if (s <= 1) return { c: 0, h: cur.rms, a: 1 };
        if (s === 2) return { c: 0, h: 1, a: 1 };
        return { c: 0, h: 1, a: 0 };
      }
      if (s === 0) return { c: cur.mu, h: cur.sigma, a: 1 };
      if (s === 1) return { c: 0, h: cur.sigma, a: 1 };
      if (s === 2) return { c: 0, h: cur.sigma / cur.sd, a: 1 };
      return { c: 0, h: 1, a: 0 };
    }

    function draw() {
      if (!Lo || !cur) return;
      const g = cv.g;
      const { narrow, ax, aw, ah, rowsY } = Lo;
      cv.clear();
      const colW = aw / LN_N;
      const barW = Math.min(34, colW * 0.56);

      // grid + axis
      g.save();
      for (let v = -2; v <= 4; v += 2) {
        const y = yOf(v);
        g.strokeStyle = AM.rgba(AM.col.linen, v === 0 ? 0.22 : 0.06);
        g.lineWidth = 1;
        g.beginPath(); g.moveTo(ax, Math.round(y) + 0.5); g.lineTo(ax + aw, Math.round(y) + 0.5); g.stroke();
        D.text(g, fmtS(v, 0), ax - 8, y + 3, { role: 'mono', size: narrow ? 8 : 9, align: 'right', color: AM.col.mist });
      }
      g.restore();

      // σ band and mean line
      if (band.a > 0.01) {
        const y1 = yOf(band.c + band.h), y2 = yOf(band.c - band.h);
        g.save();
        g.globalAlpha = band.a;
        g.fillStyle = AM.rgba(AM.dye.weld, 0.06);
        g.fillRect(ax, y1, aw, y2 - y1);
        g.setLineDash([2, 3]);
        g.strokeStyle = AM.rgba(AM.dye.weld, 0.35);
        g.beginPath(); g.moveTo(ax, y1); g.lineTo(ax + aw, y1); g.moveTo(ax, y2); g.lineTo(ax + aw, y2); g.stroke();
        g.setLineDash([6, 4]);
        g.strokeStyle = AM.rgba(AM.dye.weld, 0.85);
        g.lineWidth = 1.3;
        const yc = yOf(band.c);
        g.beginPath(); g.moveTo(ax, yc); g.lineTo(ax + aw, yc); g.stroke();
        g.setLineDash([]);
        const isRms = S.kind === 'rms';
        const lx = ax + aw + 5;
        D.text(g, isRms ? '+rms' : '+σ', lx, y1 + 3, { role: 'mono', size: narrow ? 8 : 9, color: AM.dye.weld, alpha: 0.8 });
        D.text(g, isRms ? '−rms' : '−σ', lx, y2 + 3, { role: 'mono', size: narrow ? 8 : 9, color: AM.dye.weld, alpha: 0.8 });
        if (!isRms && Math.abs(yc - y1) > 10 && Math.abs(yc - y2) > 10) D.text(g, 'μ', lx, yc + 3, { role: 'mono', size: narrow ? 9 : 10, color: AM.dye.weld });
        g.restore();
      }

      // bars
      const y0 = yOf(0);
      for (let i = 0; i < LN_N; i++) {
        const cx = ax + colW * (i + 0.5);
        const v = disp[i];
        const vc = MM.clamp(v, YMIN, YMAX);
        const yv = yOf(vc);
        const col = v >= 0 ? AM.dye.woad : AM.dye.madder;
        const top = Math.min(yv, y0), hgt = Math.max(1, Math.abs(yv - y0));
        g.save();
        const grd = g.createLinearGradient(0, top, 0, top + hgt);
        grd.addColorStop(v >= 0 ? 0 : 1, AM.rgba(col, 0.95));
        grd.addColorStop(v >= 0 ? 1 : 0, AM.rgba(col, 0.35));
        g.fillStyle = grd;
        g.shadowColor = AM.rgba(col, 0.45);
        g.shadowBlur = S.drag === i ? 16 : 6;
        D.roundRect(g, cx - barW / 2, top, barW, hgt, 3);
        g.fill();
        g.restore();
        // clipped: chevron at the edge
        if (v > YMAX || v < YMIN) {
          const yy = v > YMAX ? Lo.ay - 1 : Lo.ay + ah + 1;
          const dir = v > YMAX ? -1 : 1;
          g.save();
          g.fillStyle = AM.col.linen;
          g.beginPath(); g.moveTo(cx - 5, yy - dir * 1); g.lineTo(cx + 5, yy - dir * 1); g.lineTo(cx, yy + dir * 5); g.closePath(); g.fill();
          g.restore();
        }
        // ghost outline of the input (what you drag)
        if (S.stage > 0) {
          const gv = MM.clamp(ghost[i], YMIN, YMAX), gy = yOf(gv);
          g.save();
          g.setLineDash([2, 2]);
          g.strokeStyle = AM.rgba(AM.col.linen, S.drag === i ? 0.75 : 0.28);
          g.lineWidth = 1;
          g.strokeRect(cx - barW / 2 - 0.5, Math.min(gy, y0), barW + 1, Math.max(1, Math.abs(gy - y0)));
          g.restore();
        }
        // value label
        const txt = narrow ? fmtS(v, 1) : fmtS(v, 2);
        const ly = v >= 0 ? Math.max(Lo.ay + 8, yv - 5) : Math.min(Lo.ay + ah - 2, yv + 12);
        D.text(g, txt, cx, ly, { role: 'mono', size: 8.5, align: 'center', color: AM.col.linenDim, alpha: 0.9 });
      }

      // γ and β rows
      const rowA = S.stage === 3 ? 1 : 0.62;
      const rh = narrow ? 14 : 15;
      D.text(g, 'γ', ax - 8, rowsY + 9, { role: 'mono', size: narrow ? 9 : 10, align: 'right', color: AM.dye.weld, alpha: rowA });
      if (S.kind === 'ln') D.text(g, 'β', ax - 8, rowsY + rh + 9, { role: 'mono', size: narrow ? 9 : 10, align: 'right', color: AM.dye.weld, alpha: rowA });
      for (let i = 0; i < LN_N; i++) {
        const cx = ax + colW * (i + 0.5);
        D.text(g, narrow ? fmtTight(LN_GAMMA[i]) : LN_GAMMA[i].toFixed(2), cx, rowsY + 9, { role: 'mono', size: 8.5, align: 'center', color: AM.col.linenDim, alpha: rowA });
        if (S.kind === 'ln') D.text(g, narrow ? fmtTight(LN_BETA[i]) : fmtS(LN_BETA[i], 2), cx, rowsY + rh + 9, { role: 'mono', size: 8.5, align: 'center', color: AM.col.linenDim, alpha: rowA });
      }
      if (S.kind === 'rms') D.text(g, 'no β in RMSNorm', ax + aw / 2, rowsY + rh + 9, { role: 'mono', size: 8.5, align: 'center', color: AM.col.mist, alpha: rowA });

      // stage title
      const names = narrow ? STAGE_SHORT[S.kind] : STAGE_NAMES[S.kind];
      D.text(g, `${S.stage + 1}/4`, ax, 16, { role: 'mono', size: narrow ? 8.5 : 9.5, color: AM.col.mist });
      D.text(g, names[S.stage], ax + (narrow ? 30 : 36), 16, { role: 'mono', size: narrow ? 10 : 12, color: AM.dye.weld });
    }

    function tick(dt) {
      if (!cur) return;
      // animated sweep of the scale slider (the "×10" button)
      if (S.sweep) {
        S.sweep.t = Math.min(1, S.sweep.t + dt / 1.1);
        S.u = MM.lerp(S.sweep.from, S.sweep.to, MM.ease.inOut(S.sweep.t));
        scaleSlider.set(S.u);
        recompute();
        if (S.sweep.t >= 1) S.sweep = null;
      }
      const a = 1 - Math.exp(-dt * 6);
      const tgt = cur.stages[S.stage];
      for (let i = 0; i < LN_N; i++) { disp[i] += (tgt[i] - disp[i]) * a; ghost[i] += (cur.input[i] - ghost[i]) * Math.min(1, a * 2); }
      const bt = bandTarget();
      band.c += (bt.c - band.c) * a; band.h += (bt.h - band.h) * a; band.a += (bt.a - band.a) * a;
    }

    function snap() {
      const tgt = cur.stages[S.stage];
      for (let i = 0; i < LN_N; i++) { disp[i] = tgt[i]; ghost[i] = cur.input[i]; }
      Object.assign(band, bandTarget());
    }

    // ---------------------------------------------------------------- interaction: click / drag a bar to set the input
    const pick = (ev) => {
      const p = cv.pointer(ev);
      const i = Math.floor((p.x - Lo.ax) / (Lo.aw / LN_N));
      return { i: MM.clamp(i, 0, LN_N - 1), v: MM.clamp(vOf(p.y), YMIN - 0.5, YMAX + 0.5) };
    };
    const setFrom = (ev) => {
      const { i, v } = pick(ev);
      S.drag = i;
      S.base[i] = MM.clamp((v - S.shift) / scaleOf(), -9, 9);
      recompute();
    };
    cv.canvas.addEventListener('pointerdown', (ev) => {
      if (!Lo) return;
      const p = cv.pointer(ev);
      if (p.x < Lo.ax - 6 || p.y < Lo.ay - 12 || p.y > Lo.ay + Lo.ah + 12) return;
      S.sweep = null;
      try { cv.canvas.setPointerCapture(ev.pointerId); } catch (e) { /* ignore */ }
      setFrom(ev);
    });
    cv.canvas.addEventListener('pointermove', (ev) => { if (S.drag >= 0 && cv.canvas.hasPointerCapture && cv.canvas.hasPointerCapture(ev.pointerId)) setFrom(ev); });
    const endDrag = () => { S.drag = -1; };
    cv.canvas.addEventListener('pointerup', endDrag);
    cv.canvas.addEventListener('pointercancel', endDrag);

    // ---------------------------------------------------------------- controls
    const ui = AM.ui;
    const stageSeg = ui.segmented({
      id: 'ln-stage', label: 'STEP',
      options: [{ value: 0, label: 'x' }, { value: 1, label: '− μ' }, { value: 2, label: '÷ σ' }, { value: 3, label: '× γ + β' }],
      value: 0, onChange: (v) => { S.stage = v; refresh(); },
    });
    const nextBtn = ui.button({ id: 'ln-next', label: 'Next step ▸', kind: 'primary', onClick: () => { S.stage = (S.stage + 1) % 4; stageSeg.set(S.stage); refresh(); } });
    const kindSeg = ui.segmented({
      id: 'ln-kind', label: 'NORM',
      options: [{ value: 'ln', label: 'LayerNorm' }, { value: 'rms', label: 'RMSNorm' }],
      value: 'ln', onChange: (v) => { S.kind = v; relabel(); recompute(); },
    });
    const scaleSlider = ui.slider({
      id: 'ln-scale', label: 'SCALE INPUT', min: -1, max: 1, step: 0.01, value: 0,
      format: (u) => { const s = Math.pow(10, u); return '×' + (s < 1 ? s.toFixed(2) : s.toPrecision(3)); },
      onInput: (u) => { S.sweep = null; S.u = u; recompute(); },
    });
    const shiftSlider = ui.slider({
      id: 'ln-shift', label: 'SHIFT INPUT', min: -3, max: 3, step: 0.1, value: 0,
      format: (v) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(1),
      onInput: (v) => { S.shift = v; recompute(); },
    });
    x10Btn = ui.button({ id: 'ln-x10', label: 'Scale input ×10', onClick: () => { S.sweep = { from: S.u, to: sweepTarget() > 0.5 ? 0 : 1, t: 0 }; if (cur) refresh(); } });
    const resetBtn = ui.button({
      id: 'ln-reset', label: 'Reset',
      onClick: () => { S.base = LN_BASE.slice(); S.u = 0; S.shift = 0; S.sweep = null; scaleSlider.set(0); shiftSlider.set(0); recompute(); },
    });
    const randBtn = ui.button({
      id: 'ln-random', label: 'Random input',
      onClick: () => { const r = MM.rng((Math.random() * 1e9) | 0); S.base = S.base.map(() => +(1.2 + 1.6 * MM.randn(r)).toFixed(2)); recompute(); },
    });
    function relabel() {
      const b = stageSeg.el.querySelectorAll('button');
      const L = S.kind === 'rms' ? ['x', '(skip)', '÷ rms', '× γ'] : ['x', '− μ', '÷ σ', '× γ + β'];
      b.forEach((btn, i) => { btn.textContent = L[i]; });
    }
    ctlHost.append(
      el('div', { class: 'controls' }, stageSeg.el, nextBtn, kindSeg.el),
      el('div', { class: 'controls' }, scaleSlider.el, x10Btn, shiftSlider.el, randBtn, resetBtn),
    );

    cv.onResize((w, h) => {
      Lo = layout(w, h);
      if (!cur) recompute();
      if (!primed) { snap(); primed = true; }
      draw();
    });
    ctx.loop((t, dt) => { if (!vis.on) return; tick(dt); draw(); });
  }

  // ================================================================== chapter
  AM.chapter({
    id: ID,
    num: 7,
    kicker: 'Residual stream & LayerNorm',
    title: 'The River <em>Thread</em>',
    lede: 'Each token carries one vector from the bottom of the stack to the top. Every sublayer reads it, computes a small edit, and adds the edit back in. Nothing is overwritten.',
    where: 'resid',
    mount(root, ctx) {
      ctx.header();
      const el = ctx.el;
      const ui = AM.ui;
      const toy = makeToy(TOY_SEED);
      // numbers quoted in the prose come from the same toy computation the strip shows
      const allOn = new Array(6).fill(true);
      const cosAdd = runToy(toy, allOn, false).cos[6];
      const cosRep = runToy(toy, allOn, true).cos[6];

      const body = el('div', { class: 'ch-body' });
      root.appendChild(body);

      // ---------------------------------------------------------------- 1. scrollytelling river
      const riverHost = el('div');
      let river = null;
      const hint = el('span', { class: 'rs-hint' });
      const setHint = (residOn) => { hint.innerHTML = residOn ? 'flip to make sublayers <b>replace</b> the stream' : 'flip back to <b>add</b> to the stream'; };
      setHint(true);
      const toggle = ui.toggle({ id: 'rs-resid-toggle', label: 'Residual connections', checked: true, onChange: (b) => { setHint(b); if (river) river.setReplace(!b); } });
      const capText = `The river is an illustration. The strip on the left is a toy 16-dimensional stream: each column is one dimension of x (blue positive, red negative), each band one level of the stack, and the numbers beside it are the cosine similarity with the original embedding x${sub(0)}. Each sublayer's barcode is the vector it writes, F(LN(x)), from small random weights; the same barcode appears on the strip at the level where it is added. The attention sublayers also blend in a fixed vector that stands in for the other tokens. All of it is computed here.`;
      const stage = el('div', { class: 'ch-stage rs-stage' },
        el('figure', { class: 'fig' },
          el('div', { class: 'fig-top' },
            el('span', { class: 'fig-title' }, el('span', { class: 'rs-long' }, 'One token, three blocks'), el('span', { class: 'rs-short' }, 'Three blocks')),
            el('span', { class: 'rs-badges' }, ui.badge('illustration'), ui.badge('toy'))),
          riverHost,
          el('div', { class: 'controls' }, toggle.el, hint),
          el('figcaption', { class: 'rs-cap-desk' }, capText)));

      const step = (label, h3, ...html) => el('div', { class: 'step' },
        el('span', { class: 'step-label' }, label),
        el('h3', {}, h3),
        ...html.map((s) => (s.startsWith('<div') ? el('div', { html: s }) : el('p', { html: s }))));

      const steps = [
        step('1 · The stream', 'One river per token',
          'After embedding, every token position holds a vector of <span class="math">d<sub>model</sub></span> numbers. Follow the one for <em>crown</em>: the gold light rising from the bottom of the picture.',
          'This vector is the <span class="term">residual stream</span>. It runs unbroken from the embedding to the top of the model, where it is turned into scores for the next token. Every other part of the transformer hangs off it.'),
        step('2 · Read', 'Reading through a LayerNorm gate',
          'A sublayer starts by taking a copy of the stream and passing it through <span class="term">LayerNorm</span>, which rescales the copy to a standard size. The sublayer works on that copy.',
          'Reading takes nothing away. The river flows on past the gate untouched.'),
        step('3 · Write', 'Writing by addition',
          'Each sublayer returns a vector the same size as the stream, and that vector is <strong>added</strong> to it. Attention writes first, then the MLP:',
          '<div class="math block">x ← x + Attention(LN(x))<br>x ← x + MLP(LN(x))</div>',
          'The pair makes one <span class="term">transformer block</span>. Putting LayerNorm in front of each sublayer like this is called <span class="term">pre-LN</span>; GPT-2 and most models since are built this way.',
          'Watch the strip on the left, from bottom to top. Each barcode at a seam is the vector that sublayer added. It nudges each of the 16 numbers a little, and everything that was there before is still there.'),
        step('4 · A shared bus', 'A channel every layer can read',
          'Stack three blocks and the stream carries a running sum: the token\'s own embedding plus every edit written so far. A sublayer high up can read what one far below wrote. The blue thread marks the third attention sublayer reading the dye the first one poured in.',
          'Interpretability researchers think of the residual stream as a communication channel. Layers talk to each other by writing to it and reading from it.'),
        step('5 · Without the residual', 'Now replace instead of add',
          'Each sublayer now overwrites the stream: <span class="math">x ← F(LN(x))</span>. The switch under the picture does the same thing at any step.',
          `The gold is gone after the first sublayer. In the toy stream, the cosine similarity between the top vector and the original embedding falls from <strong>${fmtS(cosAdd, 2)}</strong> to <strong>${fmtS(cosRep, 2)}</strong>. Other random weights tell the same story. With the residual path the similarity usually lands between 0.5 and 0.85; without it, it scatters around zero, mostly within ±0.3.`,
          'Every layer would now have to carry forward everything useful by itself. The straight path also helps training: the learning signal can flow back down it to the earliest layers without fading (chapter 10).'),
      ];
      const prose = el('div', { class: 'ch-prose' }, steps);
      body.appendChild(el('div', { class: 'ch-split' }, stage, prose));
      body.appendChild(el('p', { class: 'caption rs-cap-phone' }, capText));

      river = buildRiver(ctx, riverHost, toy);
      ctx.steps(steps, (i) => {
        river.setStep(i);
        const add = !(RIVER_STEPS[i] && RIVER_STEPS[i].replace);
        toggle.set(add);
        setHint(add);
      });

      // ---------------------------------------------------------------- 2. LayerNorm stepper
      const lnCanvas = el('div', { class: 'ln-cv' });
      const lnRead = el('div', { class: 'ln-read' });
      const lnCtl = el('div', { class: 'ln-controls' });
      const lnSec = el('section', { class: 'ch-wide rs-sec', 'aria-labelledby': 'rs-ln-h' },
        el('div', { class: 'prose' },
          el('span', { class: 'rs-kicker' }, 'The gate'),
          el('h3', { id: 'rs-ln-h' }, 'LayerNorm, one step at a time'),
          el('p', { html: 'LayerNorm works on one token\'s vector at a time. It subtracts the vector\'s mean, divides by its standard deviation, then applies a learned scale <span class="math">γ</span> and shift <span class="math">β</span> to each dimension:' }),
          el('div', { class: 'math block' }, 'LN(x) = γ ⊙ (x − μ) / √(σ² + ε) + β'),
          el('p', { html: 'Here μ and σ are the mean and standard deviation of the d numbers in <span class="math">x</span>, and ε is a tiny constant that prevents division by zero. The result has a stable range whatever the stream looks like, and it does not depend on the stream\'s overall scale. Step through it, drag the bars, then press <strong>Scale input ×10</strong> while step 4 is showing.' }),
          el('p', { html: 'Many recent models, LLaMA among them, use a cheaper cousin called <span class="term">RMSNorm</span>. It skips the mean and the shift β; switch the stepper to it and a shift of the input now gets through. Where the norm sits matters too. The 2017 original normalised after the add (post-LN), which is harder to train than the pre-LN layout above.' })),
        el('div', { class: 'panel' },
          ui.figure({ title: 'LayerNorm stepper · 12 numbers', badge: 'toy', caption: 'A hand-picked 12-number vector with hand-picked γ and β (a freshly initialised model starts at γ = 1, β = 0). Every number shown is computed exactly, with ε = 10⁻⁵ and the population variance (divide by d), as LayerNorm uses. Click, tap or drag a bar to change the input; the dashed outline is the input while later steps are shown.' },
            el('div', { class: 'ln-grid' }, el('div', { class: 'ln-left' }, lnCanvas, lnCtl), lnRead))));
      body.appendChild(lnSec);
      buildLayerNorm(ctx, { canvasHost: lnCanvas, readHost: lnRead, ctlHost: lnCtl, verdictHost: lnRead });

      // ---------------------------------------------------------------- key idea
      body.appendChild(el('div', { class: 'callout' },
        el('span', { class: 'callout-label' }, 'Key idea'),
        el('p', { html: 'The <strong>residual stream</strong> is the backbone of the transformer: one vector per token, carried from the embedding to the output. Sublayers read it through <strong>LayerNorm</strong> and write to it by <strong>addition</strong>, so every edit accumulates and any layer can read what an earlier one wrote.' })));
    },
  });
})();
