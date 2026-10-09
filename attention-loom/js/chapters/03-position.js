/* The Attention Loom · Chapter 03 · Positional encoding · "The Clockwork of Order".
 *
 * Teaches: self-attention on its own is permutation-equivariant (it receives
 * a bag of tokens); the fix is to ADD a position vector to every embedding;
 * the 2017 sinusoidal encoding read as a bank of clock hands turning at
 * geometrically spaced speeds; why that encodes relative offsets (the dot
 * product of two encodings depends only on their offset, and a shift is a
 * rotation); learned absolute positions, read live from the tiny model.
 * RoPE gets one forward pointer here; its lab lives at the end of chapter 4,
 * where queries and keys are known.
 *
 * Data, all computed in this file:
 *  - Bag stage: a TOY one-head attention layer (d = 8, seeded random
 *    embeddings and W_Q / W_K / W_V, no causal mask). Weights and outputs are
 *    recomputed every frame. The bag is a 2-D PCA shadow of the 8-d inputs.
 *  - Clockwork: the exact sinusoidal formula. Similarity = cosine of exact
 *    encodings (every encoding has length √(d_model/2)).
 *  - Learned table (live): the tiny model's own position rows, wpe[t], read
 *    through its forward pass.
 */
(() => {
  const AM = window.AM;
  const DR = AM.draw;
  const { clamp, lerp } = AM.math;
  const EASE = AM.math.ease;
  const ID = 'position';
  const TAU = Math.PI * 2;
  const BASE = 10000;

  // ======================================================================
  // 0. Small helpers
  // ======================================================================

  const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
  /** Frame-rate independent smoothing toward a target (rate = speed · dt). */
  const approach = (x, target, rate) => x + (target - x) * (1 - Math.exp(-rate));
  const hex2 = (n) => n.toString(16).padStart(2, '0');
  /** Mix two hex colours → hex (so AM.rgba can add alpha). */
  const mixHex = (a, b, t) => {
    const A = AM.hexToRgb(a), B = AM.hexToRgb(b);
    return '#' + [0, 1, 2].map((i) => hex2(Math.round(A[i] + (B[i] - A[i]) * t))).join('');
  };
  /** Number with a typographic minus. */
  const fmt = (x, n = 2) => { const s = Math.abs(x).toFixed(n); return (x < 0 && Number(s) !== 0 ? '−' : '') + s; };
  /** Canvas text with a subscript, drawn by hand (the mono face has no subscript digits). Subscripts never drop below 9px. */
  function subText(g, base, subscript, x, y, o = {}) {
    const size = o.size || 10, role = o.role || 'mono', ss = Math.max(9, size * 0.72);
    const bw = DR.measure(g, base, size, role, o.weight);
    const sw = DR.measure(g, subscript, ss, role, o.weight);
    const x0 = o.align === 'center' ? x - (bw + sw) / 2 : o.align === 'right' ? x - bw - sw : x;
    DR.text(g, base, x0, y, { ...o, size, role, align: 'left' });
    DR.text(g, subscript, x0 + bw + 0.5, y + size * 0.28, { ...o, size: ss, role, align: 'left' });
  }
  const thousands = (n) => String(Math.round(n)).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  /**
   * Is this element near the viewport? The chapter is long, and AM runs every
   * loop while any of it is on screen, so each figure checks its own canvas
   * and skips frames while it is out of sight.
   */
  function inView(node, margin = '80px') {
    const s = { on: true };
    if (typeof IntersectionObserver !== 'undefined') {
      new IntersectionObserver((es) => { for (const e of es) s.on = e.isIntersecting; }, { rootMargin: `${margin} 0px ${margin} 0px` }).observe(node);
    }
    return s;
  }

  // ======================================================================
  // 1. The sinusoidal encoding (Vaswani et al., 2017)
  // ======================================================================

  /** Angular speed of sin/cos pair i, in radians per position: ω_i = 1 / 10000^(2i/d). */
  const omega = (i, d) => Math.pow(BASE, (-2 * i) / d);

  /** PE(pos, 2i) = sin(pos·ω_i), PE(pos, 2i+1) = cos(pos·ω_i). (pos may be fractional mid-animation.) */
  function sinusoid(pos, d) {
    const v = new Float64Array(d);
    for (let i = 0; i < d / 2; i++) {
      const a = pos * omega(i, d);
      v[2 * i] = Math.sin(a);
      v[2 * i + 1] = Math.cos(a);
    }
    return v;
  }

  /** Positions per full turn of a hand (2π / ω), short label. */
  const periodLabel = (T) => (T < 10 ? T.toFixed(1) : T < 1000 ? String(Math.round(T)) : T < 9950 ? (T / 1000).toFixed(1) + 'k' : Math.round(T / 1000) + 'k');

  // ======================================================================
  // 2. Clock-face drawing (shared by every figure)
  // ======================================================================

  /** Dial face: soft disc, rim, optional ticks and crosshair. */
  function dialFace(g, cx, cy, r, o = {}) {
    g.save();
    const grd = g.createRadialGradient(cx - r * 0.25, cy - r * 0.35, r * 0.05, cx, cy, r);
    grd.addColorStop(0, AM.col.ink3);
    grd.addColorStop(1, AM.col.ink2);
    g.fillStyle = grd;
    g.globalAlpha = o.alpha ?? 1;
    g.beginPath(); g.arc(cx, cy, r, 0, TAU); g.fill();
    g.lineWidth = o.rimWidth || 1;
    g.strokeStyle = o.rim || AM.col.ruleStrong;
    g.stroke();
    if (o.cross) {
      g.strokeStyle = AM.rgba(AM.col.linen, 0.07);
      g.beginPath(); g.moveTo(cx - r, cy); g.lineTo(cx + r, cy); g.moveTo(cx, cy - r); g.lineTo(cx, cy + r); g.stroke();
    }
    const ticks = o.ticks ?? (r > 13 ? 12 : 0);
    if (ticks) {
      g.strokeStyle = AM.rgba(AM.col.linen, 0.22);
      g.beginPath();
      for (let k = 0; k < ticks; k++) {
        const a = (k / ticks) * TAU, long = k % (ticks / 4) === 0;
        const r0 = r * (long ? 0.8 : 0.88);
        g.moveTo(cx + Math.cos(a) * r0, cy + Math.sin(a) * r0);
        g.lineTo(cx + Math.cos(a) * r * 0.96, cy + Math.sin(a) * r * 0.96);
      }
      g.stroke();
    }
    g.restore();
  }

  /** A glowing clock hand at math angle `ang` (counter-clockwise from 3 o'clock). Returns the tip. */
  function hand(g, cx, cy, len, ang, color, o = {}) {
    const tx = cx + Math.cos(ang) * len, ty = cy - Math.sin(ang) * len;
    g.save();
    g.globalAlpha = o.alpha ?? 1;
    g.lineCap = 'round';
    g.strokeStyle = AM.rgba(color, 0.22);
    g.lineWidth = (o.width || 1.6) * 3.2;
    g.beginPath(); g.moveTo(cx, cy); g.lineTo(tx, ty); g.stroke();
    g.strokeStyle = color;
    g.lineWidth = o.width || 1.6;
    g.beginPath(); g.moveTo(cx, cy); g.lineTo(tx, ty); g.stroke();
    g.restore();
    if (o.bead !== false) DR.glowDot(g, tx, ty, o.bead || 2, color, o.alpha ?? 1);
    return { x: tx, y: ty };
  }

  /** Radar-style sweep sector behind a moving hand, fading from the hand backwards. */
  function sweep(g, cx, cy, r, th, thOld, color, alpha = 1) {
    let L = th - thOld;
    if (Math.abs(L) < 0.015) return;
    L = clamp(L, -TAU * 0.985, TAU * 0.985);
    const n = Math.min(18, 4 + Math.ceil(Math.abs(L) * 3));
    g.save();
    for (let s = 0; s < n; s++) {
      const a0 = th - (L * s) / n, a1 = th - (L * (s + 1)) / n;
      g.fillStyle = AM.rgba(color, alpha * 0.24 * (1 - s / n));
      g.beginPath(); g.moveTo(cx, cy);
      g.arc(cx, cy, r, -a0, -a1, L < 0);
      g.closePath(); g.fill();
    }
    g.restore();
  }

  // ======================================================================
  // 3. Toy attention for the bag stage
  // ======================================================================

  const BAG_D = 8;
  const WORDS = ['dog', 'bites', 'man'];
  const WORD_DYE = ['verdigris', 'weld', 'cochineal'];
  const ORDER_A = [0, 1, 2]; // seat of each word in "dog bites man"
  const ORDER_B = [2, 1, 0]; // seat of each word in "man bites dog"

  /** Top-k eigenvectors of a small symmetric matrix (power iteration + deflation). */
  function topEig(C, k) {
    const n = C.length, out = [];
    let M = C.map((r) => r.slice());
    for (let c = 0; c < k; c++) {
      let v = Array.from({ length: n }, (_, i) => 1 + 0.37 * i - 0.05 * i * i);
      let lam = 0;
      for (let it = 0; it < 500; it++) {
        const w = AM.math.vecMat(v, M); // M symmetric: v·M = M·v
        const nn = Math.hypot(...w) || 1;
        v = w.map((x) => x / nn);
        lam = nn;
      }
      out.push(v);
      M = M.map((row, i) => row.map((x, j) => x - lam * v[i] * v[j]));
    }
    return out;
  }

  function buildBag() {
    const r = AM.math.rng(3);
    const randn = () => AM.math.randn(r);
    const E = WORDS.map(() => Array.from({ length: BAG_D }, () => randn() * 0.8));
    const mat = (s) => Array.from({ length: BAG_D }, () => Array.from({ length: BAG_D }, () => randn() * s));
    const WQ = mat(0.55), WK = mat(0.55), WV = mat(0.5);
    const scale = 1 / Math.sqrt(BAG_D);

    /** x_w = e_w + s · PE(seat_w). s ∈ [0,1] fades the position vectors in. */
    const inputs = (seats, s) => E.map((e, w) => {
      const p = sinusoid(seats[w], BAG_D);
      return e.map((x, j) => x + s * p[j]);
    });

    /** One attention head, no mask: A = softmax(Q·Kᵀ / √d), O = A·V. Rows of X are in sequence order. */
    const attend = (X) => {
      const Q = X.map((x) => AM.math.vecMat(x, WQ));
      const K = X.map((x) => AM.math.vecMat(x, WK));
      const V = X.map((x) => AM.math.vecMat(x, WV));
      const A = Q.map((q) => AM.math.softmax(K.map((k) => dot(q, k) * scale)));
      const O = A.map((a) => V[0].map((_, j) => a.reduce((s, wt, t) => s + wt * V[t][j], 0)));
      return { A, O };
    };

    /** Feed the words to attention in seat order, then index the results by word. */
    const run = (seats, s) => {
      const order = [0, 1, 2].sort((a, b) => seats[a] - seats[b]); // word at each sequence slot
      const X = inputs(seats, s);
      const res = attend(order.map((w) => X[w]));
      const A = WORDS.map(() => new Array(3)), O = new Array(3);
      order.forEach((w, qi) => {
        O[w] = res.O[qi];
        order.forEach((w2, ki) => { A[w][w2] = res.A[qi][ki]; });
      });
      return { X, A, O };
    };

    // The 2-D shadow: top two principal axes of every vector the bag can show.
    const pts = [];
    E.forEach((e) => {
      pts.push(e);
      for (let t = 0; t < 3; t++) { const p = sinusoid(t, BAG_D); pts.push(e.map((x, j) => x + p[j])); }
    });
    const mu = Array.from({ length: BAG_D }, (_, j) => pts.reduce((s, p) => s + p[j], 0) / pts.length);
    const C = Array.from({ length: BAG_D }, (_, a) => Array.from({ length: BAG_D }, (_, b) =>
      pts.reduce((s, p) => s + (p[a] - mu[a]) * (p[b] - mu[b]), 0) / pts.length));
    const [u1, u2] = topEig(C, 2);
    const proj = (x) => { const c = x.map((v, j) => v - mu[j]); return [dot(c, u1), dot(c, u2)]; };
    const ext = [0, 0];
    pts.forEach((p) => { const q = proj(p); ext[0] = Math.max(ext[0], Math.abs(q[0])); ext[1] = Math.max(ext[1], Math.abs(q[1])); });

    // Rest states: both sentences, without (s = 0) and with (s = 1) positions.
    const rest = {};
    for (const s of [0, 1]) {
      const a = run(ORDER_A, s), b = run(ORDER_B, s);
      rest[s] = { a, b, diff: WORDS.map((_, w) => Math.max(...a.O[w].map((x, j) => Math.abs(x - b.O[w][j])))) };
    }
    let oMax = 0;
    for (const s of [0, 1]) for (const k of ['a', 'b']) rest[s][k].O.forEach((o) => o.forEach((x) => { oMax = Math.max(oMax, Math.abs(x)); }));
    let xMax = 0;
    pts.forEach((p) => p.forEach((x) => { xMax = Math.max(xMax, Math.abs(x)); }));
    return { E, run, proj, ext, rest, oMax, xMax };
  }

  // ======================================================================
  // 4. Figure A: two sentences, one bag (the sticky stage)
  // ======================================================================

  function mountBag(ctx) {
    const { el } = ctx;
    const B = buildBag();
    const st = { step: 0, clock: 0, u: 0, posMix: 0, ghostMix: 0, outMix: 0.35, G: null };
    const HOLD = 1.8, GLIDE = 1.3, CYC = 2 * (HOLD + GLIDE);
    const dyes = WORD_DYE.map((k) => AM.dye[k]);

    const cv = ctx.canvas(null, {
      // sticky stage: keep it (plus its readout) inside short laptop screens too
      height: (w) => (w < 520 ? 330 : window.innerWidth <= 900 ? 400 : clamp(Math.round(w * 0.86), 380, Math.max(380, Math.min(540, window.innerHeight - 330)))),
      label: 'The words dog, bites and man swap places between the sentences “dog bites man” and “man bites dog”. Threads run from each word into an oval bag where attention compares the vectors, and on to each word’s output. Without position vectors the bag looks the same for both orders; with them, the beads in the bag move and the outputs change.',
    });
    const readout = el('p', { class: 'pos-readout' });
    const fig = AM.ui.figure({
      title: 'Two sentences, one bag', badge: 'toy', cls: 'pos-bag-fig',
      caption: 'Toy numbers: three random 8-number embeddings pass through one attention step with random weights, the kind chapter 4 opens up, all computed live. Here every word sees every other word. The bag is a 2-D shadow (PCA) of the 8-number input vectors; threads inside it are attention weights, thicker for heavier weights. Bars are vectors, blue positive and red negative.',
    }, cv.wrap, readout);

    /** Swap phase: 0 = "dog bites man", 1 = "man bites dog". */
    const swapU = (t) => {
      const p = t % CYC;
      if (p < HOLD) return 0;
      if (p < HOLD + GLIDE) return EASE.inOut((p - HOLD) / GLIDE);
      if (p < 2 * HOLD + GLIDE) return 1;
      return 1 - EASE.inOut((p - 2 * HOLD - GLIDE) / GLIDE);
    };

    function geom(w, h) {
      const small = w < 520;
      const G = { small, w, h };
      G.slotX = [w * 0.2, w * 0.5, w * 0.8];
      G.sentY = small ? 22 : 34;
      G.seatY = small ? 50 : 72;
      G.tokY = small ? 84 : 116;
      G.tokSize = small ? 13 : 16;
      G.barW = small ? 66 : 104;
      G.barH = small ? 18 : 28;
      G.barY = G.tokY + (small ? 18 : 24);
      G.outH = small ? 18 : 28;
      G.outY = h - (small ? 20 : 30);
      const top = G.barY + G.barH + (small ? 14 : 24);
      const bot = G.outY - G.outH / 2 - (small ? 14 : 24);
      G.bag = { cx: w / 2, cy: (top + bot) / 2, rx: Math.min(w * 0.46, 290), ry: (bot - top) / 2 };
      return G;
    }

    /** 2-D bag coordinates of an 8-d vector. */
    const bead = (G, x) => {
      const p = B.proj(x), b = G.bag;
      return { x: b.cx + (p[0] / B.ext[0]) * b.rx * 0.72, y: b.cy - (p[1] / B.ext[1]) * b.ry * 0.6 };
    };

    /** A vector chip: dark rounded panel with the vector's bars. */
    function chip(g, cx, y, w, h, vec, max, alpha = 1) {
      g.save();
      DR.roundRect(g, cx - w / 2 - 5, y - 4, w + 10, h + 8, 6);
      g.fillStyle = AM.rgba(AM.col.ink, 0.88); g.fill();
      g.strokeStyle = AM.rgba(AM.col.linen, 0.1); g.lineWidth = 1; g.stroke();
      g.fillStyle = AM.rgba(AM.col.linen, 0.1); g.fillRect(cx - w / 2, y + h / 2 - 0.5, w, 1);
      g.restore();
      DR.vectorBars(g, cx - w / 2, y, w, h, vec, { max, alpha });
    }

    /** Outline of a vector's bars (same geometry as AM.draw.vectorBars). */
    function barOutline(g, x, y, w, h, vec, max, alpha) {
      const n = vec.length, bw = w / n, mid = y + h / 2;
      g.save();
      g.strokeStyle = AM.rgba(AM.col.linen, 0.75 * alpha);
      g.lineWidth = 1;
      for (let i = 0; i < n; i++) {
        const t = clamp(vec[i] / max, -1, 1), hh = Math.max(1, Math.abs(t) * (h / 2));
        g.strokeRect(x + i * bw + bw * 0.12 + 0.5, (t >= 0 ? mid - hh : mid) + 0.5, Math.max(1, bw * 0.76) - 1, hh - 1);
      }
      g.restore();
    }

    /** A seat's position stamp: a tiny four-hand clock showing PE(seat) for d = 8. */
    function stamp(g, cx, cy, r, seat, alpha) {
      dialFace(g, cx, cy, r, { ticks: 0, alpha, rim: AM.rgba(AM.dye.weld, 0.55 * alpha) });
      for (let i = BAG_D / 2 - 1; i >= 0; i--) {
        const col = mixHex(AM.dye.weld, AM.dye.woad, i / (BAG_D / 2 - 1));
        hand(g, cx, cy, r * (0.88 - i * 0.15), seat * omega(i, BAG_D), col, { width: 1.2, alpha, bead: false });
      }
    }

    /** The basket never changes between frames: paint it once per size into its own layer. */
    let bagLayer = null;
    function renderBag() {
      const G = st.G;
      if (!G || !cv.w) return;
      const b = G.bag;
      bagLayer = bagLayer || document.createElement('canvas');
      bagLayer.width = Math.max(1, Math.round(cv.w * cv.dpr)); bagLayer.height = Math.max(1, Math.round(cv.h * cv.dpr));
      const g = bagLayer.getContext('2d');
      g.setTransform(cv.dpr, 0, 0, cv.dpr, 0, 0);
      g.save();
      g.beginPath(); g.ellipse(b.cx, b.cy, b.rx, b.ry, 0, 0, TAU);
      const bg = g.createRadialGradient(b.cx, b.cy - b.ry * 0.4, 4, b.cx, b.cy, b.rx);
      bg.addColorStop(0, AM.rgba(AM.col.ink3, 0.95));
      bg.addColorStop(1, AM.rgba(AM.col.ink2, 0.9));
      g.fillStyle = bg; g.fill();
      g.clip();
      DR.weave(g, b.cx - b.rx, b.cy - b.ry, b.rx * 2, b.ry * 2, { step: 7, alpha: 0.045 });
      g.restore();
      g.save();
      g.beginPath(); g.ellipse(b.cx, b.cy, b.rx, b.ry, 0, 0, TAU);
      g.strokeStyle = AM.rgba(AM.dye.weld, 0.07); g.lineWidth = 9; g.stroke();
      g.strokeStyle = AM.rgba(AM.dye.weld, 0.3); g.lineWidth = 1; g.stroke();
      // braided rim: short slanted stitches all the way round, like a basket's edge
      const nSt = Math.round((Math.PI * (b.rx + b.ry)) / 10);
      g.strokeStyle = AM.rgba(AM.dye.weld, 0.17); g.lineWidth = 1.1; g.lineCap = 'round';
      g.beginPath();
      for (let k = 0; k < nSt; k++) {
        const t = (k / nSt) * TAU, x = b.cx + Math.cos(t) * b.rx, y = b.cy + Math.sin(t) * b.ry;
        const tx = -Math.sin(t) * b.rx, ty = Math.cos(t) * b.ry, tl = Math.hypot(tx, ty); // tangent
        const nx = Math.cos(t) * b.ry, ny = Math.sin(t) * b.rx, nl = Math.hypot(nx, ny);  // normal
        const ux = (tx / tl) * 2.6 + (nx / nl) * 3, uy = (ty / tl) * 2.6 + (ny / nl) * 3;
        g.moveTo(x - ux, y - uy); g.lineTo(x + ux, y + uy);
      }
      g.stroke();
      g.restore();
    }

    function draw() {
      const G = st.G;
      if (!G || !cv.w) return;
      const g = cv.g, { w } = cv;
      cv.clear();
      const u = st.u, s = st.posMix;
      const seats = WORDS.map((_, wi) => lerp(ORDER_A[wi], ORDER_B[wi], u));
      const R = B.run(seats, s);
      const other = B.run(u < 0.5 ? ORDER_B : ORDER_A, s); // the same words in the other sentence
      const xOfSeat = (q) => lerp(G.slotX[0], G.slotX[2], q / 2);
      const arc = G.small ? 26 : 38;

      // ---- the sentence as written: its words trade places like the tokens below
      // (dog arcs over, man ducks under), so the two sentences never overprint
      const fs = G.small ? 21 : 29;
      g.save(); g.font = AM.font(fs, 'display', 400, true);
      const wW = WORDS.map((wd) => g.measureText(wd).width), spW = g.measureText(' ').width * 1.4; // italic overhang eats into a plain space
      g.restore();
      const wordLeft = (order) => {
        const bySeat = [0, 1, 2].map((q) => order.indexOf(q));
        const total = wW[0] + wW[1] + wW[2] + 2 * spW;
        const left = new Array(3);
        let x = w / 2 - total / 2;
        bySeat.forEach((wi) => { left[wi] = x; x += wW[wi] + spW; });
        return { left, right: w / 2 + total / 2 };
      };
      const LA = wordLeft(ORDER_A), LB = wordLeft(ORDER_B);
      const lift = Math.sin(Math.PI * u);
      WORDS.forEach((wd, wi) => {
        let y = G.sentY + 8, al = 1;
        if (wi === 0) y -= lift * fs * 0.55;
        if (wi === 1) al = 1 - 0.4 * lift;
        if (wi === 2) { y += lift * fs * 0.5; al = 1 - 0.6 * lift; }
        DR.text(g, wd, lerp(LA.left[wi], LB.left[wi], u), y, { size: fs, role: 'display', italic: true, weight: 400, alpha: al });
      });
      // the gloss fades out, then the other one fades in (never both at once)
      const gx = Math.max(LA.right, LB.right) + 10;
      const gA = 1 - AM.math.smoothstep(0.15, 0.42, u), gB = AM.math.smoothstep(0.58, 0.85, u);
      if (gA > 0.01) DR.text(g, 'not news', gx, G.sentY + 4, { size: 9, role: 'mono', color: AM.col.mist, alpha: gA * 0.9, letterSpacing: '0.08em' });
      if (gB > 0.01) DR.text(g, 'news', gx, G.sentY + 4, { size: 9, role: 'mono', color: AM.col.mist, alpha: gB * 0.9, letterSpacing: '0.08em' });

      // ---- seats (positions) with their clock stamps
      for (let q = 0; q < 3; q++) {
        const x = G.slotX[q];
        const label = 'pos ' + q;
        const lw = DR.measure(g, label, 10, 'mono');
        const r = G.small ? 8 : 10.5;
        const groupW = lw + (r * 2 + 7) * s;
        const lx = x - groupW / 2;
        DR.text(g, label, lx, G.seatY + 3.5, { size: 10, role: 'mono', color: s > 0.5 ? AM.dye.weld : AM.col.mist, alpha: 0.55 + 0.45 * s });
        if (s > 0.01) stamp(g, lx + lw + 7 + r, G.seatY, r, q, s);
        // a faint peg line from the seat label down to the token row
        g.save(); g.strokeStyle = AM.rgba(AM.col.linen, 0.06); g.setLineDash([2, 4]);
        g.beginPath(); g.moveTo(x, G.seatY + 10); g.lineTo(x, G.tokY - 14); g.stroke(); g.restore();
      }

      // ---- bag (woven basket, cached)
      const b = G.bag;
      if (bagLayer) g.drawImage(bagLayer, 0, 0, cv.w, cv.h);
      DR.text(g, 'WHAT ATTENTION RECEIVES', b.cx, b.cy + b.ry - (G.small ? 9 : 13), { size: G.small ? 8.5 : 9, role: 'mono', color: AM.col.mist, align: 'center', letterSpacing: '0.14em' });

      const beads = R.X.map((x) => bead(G, x));
      const ghosts = B.E.map((e) => bead(G, e));

      // ---- tokens: positions along the row (dog arcs over, man ducks under)
      const tok = WORDS.map((_, wi) => {
        const x = xOfSeat(seats[wi]);
        let y = G.tokY, sc = 1, al = 1;
        if (wi === 0) { y -= lift * arc * 1.3; sc = 1 - 0.1 * lift; }
        if (wi === 1) al = 1 - 0.3 * lift;
        if (wi === 2) { y += lift * arc * 0.35; sc = 1 - 0.25 * lift; al = 1 - 0.55 * lift; }
        return { x, y, sc, al };
      });

      // threads: token → bead (in) and bead → output (out)
      WORDS.forEach((_, wi) => {
        const t = tok[wi], bd = beads[wi];
        DR.thread(g, t.x, t.y + (G.barY - G.tokY) + G.barH + 3, bd.x, bd.y, { color: dyes[wi], width: 1.3, alpha: 0.5 * t.al, bend: 0.1 });
        DR.thread(g, bd.x, bd.y, t.x, G.outY - G.outH / 2 - 5, { color: dyes[wi], width: 1.1, alpha: (0.18 + 0.32 * st.outMix) * t.al, bend: -0.1 });
      });

      // attention threads inside the bag: key → query, width = weight; values flow along them
      const bend = 0.3;
      for (let qi = 0; qi < 3; qi++) for (let ki = 0; ki < 3; ki++) {
        if (qi === ki) continue;
        const a = R.A[qi][ki], p0 = beads[ki], p1 = beads[qi];
        DR.thread(g, p0.x, p0.y, p1.x, p1.y, { color: dyes[qi], width: 0.5 + a * 5.5, alpha: 0.18 + a * 0.7, bend });
        const n = Math.round(a * 7);
        for (let k = 0; k < n; k++) {
          const tt = (st.clock * 0.3 + k / n + qi * 0.17 + ki * 0.31) % 1;
          const pt = DR.threadPoint(p0.x, p0.y, p1.x, p1.y, tt, { bend });
          DR.glowDot(g, pt.x, pt.y, 1.5, dyes[qi], 0.55 + 0.45 * Math.sin(Math.PI * tt));
        }
      }
      // self-attention: a halo around each bead
      for (let qi = 0; qi < 3; qi++) {
        const a = R.A[qi][qi], p = beads[qi];
        g.save(); g.strokeStyle = AM.rgba(dyes[qi], 0.2 + 0.6 * a); g.lineWidth = 0.6 + a * 4;
        g.beginPath(); g.arc(p.x, p.y, (G.small ? 9 : 12) + a * 4, 0, TAU); g.stroke(); g.restore();
      }

      // ghosts (bare embeddings) and the position arrow p added to each
      if (st.ghostMix > 0.01) {
        WORDS.forEach((_, wi) => {
          const gh = ghosts[wi], bd = beads[wi], a = st.ghostMix;
          g.save(); g.setLineDash([2, 3]); g.strokeStyle = AM.rgba(AM.col.linen, 0.7 * a); g.lineWidth = 1;
          g.beginPath(); g.arc(gh.x, gh.y, 5, 0, TAU); g.stroke(); g.restore();
          if (Math.hypot(bd.x - gh.x, bd.y - gh.y) > 10) {
            DR.arrow(g, gh.x, gh.y, bd.x, bd.y, { color: AM.dye.weld, width: 1.4, head: 7, alpha: 0.9 * a, dash: [3, 3] });
            subText(g, '+p', String(Math.round(seats[wi])), (gh.x + bd.x) / 2 + 4, (gh.y + bd.y) / 2 - 7, { size: 11.5, color: AM.dye.weld, alpha: a });
          }
          DR.text(g, 'e', gh.x - 9, gh.y + 3, { size: 10, role: 'mono', color: AM.col.linenDim, align: 'right', alpha: 0.8 * a });
        });
      }

      // beads + labels
      WORDS.forEach((word, wi) => {
        const p = beads[wi];
        DR.glowDot(g, p.x, p.y, G.small ? 4.5 : 5.5, dyes[wi]);
        DR.text(g, word, p.x + (G.small ? 10 : 13), p.y - (G.small ? 8 : 10), { size: G.small ? 11 : 12.5, weight: 600, color: AM.col.linen });
      });

      // ---- token tiles + their input vectors (x = e, or e + p)
      const drawOrder = [2, 1, 0]; // man passes behind, dog passes in front
      for (const wi of drawOrder) {
        const t = tok[wi];
        g.save(); g.globalAlpha = t.al;
        DR.token(g, WORDS[wi], t.x, t.y, { size: G.tokSize * t.sc, underline: dyes[wi] });
        chip(g, t.x, t.y + (G.barY - G.tokY), G.barW * t.sc, G.barH * t.sc, R.X[wi], B.xMax * 0.8);
        g.restore();
      }
      if (!G.small) {
        const lx = G.slotX[0] - G.barW / 2 - 12;
        DR.text(g, 'x =', lx, G.barY + G.barH / 2 - 3, { size: 9.5, role: 'mono', color: AM.col.mist, align: 'right' });
        DR.text(g, s > 0.5 ? 'e + p' : 'e', lx, G.barY + G.barH / 2 + 10, { size: 9.5, role: 'mono', color: s > 0.5 ? AM.dye.weld : AM.col.mist, align: 'right' });
      }

      // ---- outputs: current sentence (bars) vs the other sentence (outline)
      WORDS.forEach((_, wi) => {
        const t = tok[wi], y = G.outY - G.outH / 2;
        g.save(); g.globalAlpha = t.al;
        chip(g, t.x, y, G.barW, G.outH, R.O[wi], B.oMax * 0.85, 0.4 + 0.6 * st.outMix);
        if (st.outMix > 0.5) barOutline(g, t.x - G.barW / 2, y, G.barW, G.outH, other.O[wi], B.oMax * 0.85, (st.outMix - 0.5) * 2);
        g.restore();
      });
      if (!G.small) DR.text(g, 'out', G.slotX[0] - G.barW / 2 - 12, G.outY + 3.5, { size: 9.5, role: 'mono', color: AM.col.mist, align: 'right' });
    }

    function setReadout() {
      const r0 = B.rest[0], r1 = B.rest[1];
      const d3 = (arr) => WORDS.map((wd, i) => `${wd} <b>${arr[i].toFixed(3)}</b>`).join(' · ');
      let html;
      if (st.step === 0) {
        html = `How much dog attends to man: <b>${r0.a.A[0][2].toFixed(3)}</b> in “dog bites man”, <b>${r0.b.A[0][2].toFixed(3)}</b> in “man bites dog”. Same three vectors, same weights.`;
      } else if (st.step === 1) {
        html = `Largest output difference between the sentences: ${d3(r0.diff)}. Each output fills the other sentence’s outline exactly.`;
      } else {
        html = `With positions added, the outputs differ by up to ${d3(r1.diff)}.`;
      }
      readout.innerHTML = html;
    }

    function setStep(i) {
      st.step = i;
      setReadout();
    }

    cv.onResize((w, h) => { st.G = geom(w, h); renderBag(); draw(); });
    const seen = inView(cv.wrap);
    ctx.loop((t, dt) => {
      if (!seen.on) return; // off-screen: no time passes and nothing is drawn
      st.clock += dt;
      // reduced motion: the loop runs at ~2 fps, so swap in one clean cut instead of gliding
      st.u = AM.reducedMotion ? Math.floor(st.clock / (CYC / 2)) % 2 : swapU(st.clock);
      st.posMix = approach(st.posMix, st.step >= 2 ? 1 : 0, dt * 3.2);
      st.ghostMix = approach(st.ghostMix, st.step >= 3 ? 1 : 0, dt * 3.2);
      st.outMix = approach(st.outMix, st.step >= 1 ? 1 : 0.35, dt * 3.2);
      draw();
    });
    setReadout();
    return { el: fig, setStep, B };
  }

  // ======================================================================
  // 5. Figure B: the clockwork (hero) — dials, encoding table, similarity
  // ======================================================================

  function mountClockwork(ctx) {
    const { el } = ctx;
    const N = 64; // positions shown
    const st = {
      d: window.innerWidth < 640 ? 16 : 32,
      pos: 0, anim: 0, from: 0, to: 0, k: 1, dur: 0.3, easeFn: EASE.out,
      playing: !AM.reducedMotion, hold: 0, clock: 0,
      hover: null, dragging: false, swept: false, sweeping: false,
      wake: new Float32Array(N), hist: [], waves: [], lastWave: -1, boost: 0, shown: -1, G: null, lastAnim: 0, readoutHtml: '',
      dirty: true, idleFrames: 0,
    };
    const TICK_MOVE = 0.3, TICK_HOLD = 0.34;

    let T = null; // tables for the current d
    function buildTables(d) {
      const P = Array.from({ length: N }, (_, p) => sinusoid(p, d));
      // cosine similarity; every encoding has length √(d/2) because sin² + cos² = 1 for each pair
      const S = P.map((a) => P.map((bb) => dot(a, bb) / (d / 2)));
      let sMin = 1;
      S.forEach((row) => row.forEach((v) => { if (v < sMin) sMin = v; }));
      const cols = Array.from({ length: d / 2 }, (_, i) => mixHex(AM.dye.weld, AM.dye.woad, i / (d / 2 - 1)));
      return { d, P, S, sMin, cols };
    }
    T = buildTables(st.d);

    /**
     * Dial train for a given d: each dial sits over its two columns. When the
     * dials would be too small in one row they are staggered into two rows
     * (even hands up, odd hands down), like the wheels of a clock movement.
     */
    const dialTrain = (hw, d, wide) => {
      const colW = hw / d, pitch = 2 * colW, rMax = wide ? 34 : 30;
      if (pitch * 0.47 >= rMax * 0.8) { const r = Math.min(rMax, pitch * 0.47); return { colW, pitch, r, v: 0, stagger: false, span: 2 * r }; }
      const r = Math.max(3, Math.min(rMax, pitch - 3.5));
      const v = Math.sqrt(Math.max(0, (2 * r + 6) ** 2 - pitch ** 2)); // keep neighbours in the two rows apart
      return { colW, pitch, r, v, stagger: true, span: v + 2 * r };
    };

    const geomClock = (w) => {
      const wide = w >= 760;
      const G = { wide, w };
      if (wide) {
        G.sw = Math.round(clamp(w * 0.29, 240, 330));
        const gap = Math.round(clamp(w * 0.045, 34, 56));
        G.hx = 34;
        G.hw = w - G.sw - gap - G.hx;
      } else {
        G.hx = 26; G.hw = w - G.hx - 12; // right margin leaves room for the last dial
      }
      // reserve room for the tallest dial train among the d options
      const span = Math.max(...[16, 32, 64].map((d) => dialTrain(G.hw, d, wide).span));
      G.bandTop = 24; G.bandH = span + 34;
      G.barTop = G.bandTop + G.bandH + 22; G.barH = wide ? 30 : 24;
      G.hy = G.barTop + G.barH + 10;
      if (wide) {
        G.hh = G.sw;
        G.sx = w - G.sw; G.sy = G.hy;
        G.profTop = G.bandTop - 18; G.profH = G.barTop + G.barH - G.profTop;
        G.h = G.hy + G.hh + 40;
      } else {
        G.hh = 200;
        G.sw = Math.min(G.hw, 252); G.sx = G.hx + (G.hw - G.sw) / 2 + 4;
        G.profTop = G.hy + G.hh + 42; G.profH = 66;
        G.sy = G.profTop + G.profH + 12;
        G.h = G.sy + G.sw + 34;
      }
      return G;
    };
    /** Centre of dial i in the current train. */
    const dialPos = (G, tr, i) => {
      const top = G.bandTop + 17 + (G.bandH - 34 - tr.span) / 2;
      return { x: G.hx + (2 * i + 1) * tr.colW, y: top + tr.r + (tr.stagger && i % 2 ? tr.v : 0) };
    };

    const cv = ctx.canvas(null, {
      height: (w) => geomClock(w).h,
      label: 'A row of clock dials, one per sine and cosine pair of the positional encoding, turning at geometrically spaced speeds as a position marker moves. Below them, the encoding as a heat map of positions by dimensions. Beside it, the cosine similarity between every pair of positions, brightest along the diagonal.',
    });
    cv.canvas.style.cursor = 'crosshair';

    // offscreen caches for the two static maps
    const heatC = document.createElement('canvas');
    const simC = document.createElement('canvas');

    function renderCaches() {
      const G = st.G, dpr = cv.dpr;
      if (!G) return;
      const d = T.d, colW = G.hw / d, rowH = G.hh / N;
      // --- the encoding table: one warp thread per dimension, one row per position
      heatC.width = Math.max(1, Math.round(G.hw * dpr)); heatC.height = Math.max(1, Math.round(G.hh * dpr));
      let g = heatC.getContext('2d');
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      g.fillStyle = AM.col.ink; g.fillRect(0, 0, G.hw, G.hh);
      const gap = colW > 7 ? 1 : 0.5;
      for (let p = 0; p < N; p++) for (let j = 0; j < d; j++) {
        g.fillStyle = AM.color.seq((T.P[p][j] + 1) / 2);
        g.fillRect(j * colW + gap / 2, p * rowH, colW - gap, rowH + 0.6);
      }
      // weave: alternate floats, so the table reads as cloth
      g.fillStyle = 'rgba(0,0,0,0.05)';
      for (let p = 0; p < N; p++) for (let j = 0; j < d; j++) if ((p + j) % 2 === 0) g.fillRect(j * colW + gap / 2, p * rowH, colW - gap, rowH);
      // round-thread sheen on every column
      const sh = g.createLinearGradient(0, 0, colW, 0);
      sh.addColorStop(0, 'rgba(0,0,0,0.26)');
      sh.addColorStop(0.42, 'rgba(255,248,230,0.07)');
      sh.addColorStop(0.58, 'rgba(255,248,230,0.07)');
      sh.addColorStop(1, 'rgba(0,0,0,0.26)');
      g.fillStyle = sh;
      for (let j = 0; j < d; j++) { g.save(); g.translate(j * colW, 0); g.fillRect(gap / 2, 0, colW - gap, G.hh); g.restore(); }

      // --- similarity matrix
      const c = G.sw / N;
      simC.width = Math.max(1, Math.round(G.sw * dpr)); simC.height = simC.width;
      g = simC.getContext('2d');
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
        g.fillStyle = AM.color.heat((T.S[i][j] - T.sMin) / (1 - T.sMin));
        g.fillRect(j * c, i * c, c + 0.4, c + 0.4);
      }
    }

    // ---------- motion
    const setTarget = (p, dur, fn) => {
      st.from = st.anim; st.to = p; st.k = 0; st.dur = dur; st.easeFn = fn; st.pos = p; st.hold = 0;
    };
    const jumpTo = (p) => setTarget(clamp(Math.round(p), 0, N - 1), 0.42, EASE.out);
    function startSweep() {
      st.sweeping = true;
      setTarget(N - 1, 4.4, EASE.inOut);
    }
    function trailFrom() {
      const cut = st.clock - 0.26;
      for (const h of st.hist) if (h.t >= cut) return h.a;
      return st.anim;
    }

    // ---------- controls
    const playBtn = AM.ui.button({ id: 'pos-clock-play', label: st.playing ? 'Pause' : 'Play', onClick: () => setPlaying(!st.playing) });
    playBtn.setAttribute('aria-pressed', String(st.playing));
    function setPlaying(on) {
      st.playing = on;
      playBtn.textContent = on ? 'Pause' : 'Play';
      playBtn.setAttribute('aria-pressed', String(on));
      if (!on && st.sweeping) {
        // pausing mid wind-up: settle on the nearest whole position instead of finishing the sweep
        st.sweeping = false;
        setTarget(clamp(Math.round(st.anim), 0, N - 1), 0.25, EASE.out);
      }
    }
    const slider = AM.ui.slider({
      id: 'pos-clock-pos', label: 'Position', min: 0, max: N - 1, step: 1, value: 0, format: (v) => String(v),
      onInput: (v) => { setPlaying(false); jumpTo(v); },
    });
    const seg = AM.ui.segmented({
      id: 'pos-clock-d', label: 'd_model', value: st.d,
      options: [16, 32, 64].map((v) => ({ value: v, label: String(v) })),
      onChange: (v) => { st.d = v; st.hover = null; T = buildTables(v); renderCaches(); updateReadout(true); st.dirty = true; draw(); },
    });
    seg.el.classList.add('pos-lc');
    { const lab = seg.el.querySelector('.ctl-label'); if (lab) lab.innerHTML = 'd<sub>model</sub>'; }
    { const grp = seg.el.querySelector('.seg'); if (grp) grp.setAttribute('aria-label', 'd_model, the width of each position vector'); }
    const BITS = 6;
    const bitEls = Array.from({ length: BITS }, () => el('i', { 'aria-hidden': 'true' }));
    const bitNum = el('span', { class: 'pos-bits-num' });
    const bits = el('div', { class: 'pos-bits', role: 'img' }, el('span', { class: 'ctl-label' }, 'Binary'), el('span', { class: 'pos-bits-row' }, bitEls), bitNum);
    const readout = el('p', { class: 'pos-readout pos-clock-readout' });
    const controls = el('div', { class: 'controls pos-ctrls' }, playBtn, slider.el, seg.el, bits);

    const fig = AM.ui.figure({
      title: 'The position clockwork', badge: AM.ui.badge('toy', 'Exact formula'), cls: 'pos-clock-fig',
      caption: `Exact formula for positions 0–63. Each dial drives the two columns beneath it; gold hands are fast, blue ones slow. The tall map is the encoding table, one row per position (cream = +1, blue = 0, ink = −1). The square map is the cosine similarity between every pair of positions, colour stretched from its lowest value to 1. Drag or tap a map to move the shuttle; hover or tap any cell to read its formula.`,
    }, cv.wrap, controls, readout);

    // ---------- readout
    function updateReadout(force) {
      const d = T.d, h = st.hover, p = st.shown;
      let html;
      if (h && h.type === 'heat') {
        const i = Math.floor(h.j / 2), fn = h.j % 2 ? 'cos' : 'sin', ang = h.p * omega(i, d);
        html = `PE(${h.p}, ${h.j}) = ${fn}(${h.p} / 10000<sup>${2 * i}/${d}</sup>) = ${fn}(${fmt(ang, 3)}) = <b>${fmt(T.P[h.p][h.j], 3)}</b> <span class="pos-dim">· dims ${2 * i} and ${2 * i + 1} are hand ${i}</span>`;
      } else if (h && h.type === 'sim') {
        const k = h.i - h.j, raw = dot(T.P[h.i], T.P[h.j]);
        html = `PE(${h.i})·PE(${h.j}) = Σ<sub>i</sub> cos((${h.i} − ${h.j})·ω<sub>i</sub>) = <b>${fmt(raw, 3)}</b>, cosine <b>${fmt(T.S[h.i][h.j], 3)}</b> <span class="pos-dim">· the same for every pair ${Math.abs(k)} apart (the lit diagonal)</span>`;
      } else if (h && h.type === 'dial') {
        const w_ = omega(h.i, d), per = TAU / w_;
        html = `Hand ${h.i} (dims ${2 * h.i}, ${2 * h.i + 1}) turns ω = 10000<sup>−${2 * h.i}/${d}</sup> = <b>${w_ < 0.01 ? w_.toExponential(2) : fmt(w_, 4)}</b> rad per position: one full turn every <b>${per < 100 ? per.toFixed(1) : thousands(per)}</b> positions. At position ${p} it points at ${fmt((p * w_) % TAU, 3)} rad.`;
      } else {
        html = `Position <b>${p}</b>: hand 0 has turned ${p} rad, so PE(${p}, 0) = sin(${p}) = <b>${fmt(Math.sin(p), 3)}</b> and PE(${p}, 1) = cos(${p}) = <b>${fmt(Math.cos(p), 3)}</b>. <span class="pos-dim">Hover or tap the dials and maps to read more.</span>`;
      }
      if (force || html !== st.readoutHtml) { st.readoutHtml = html; readout.innerHTML = html; }
    }
    function syncUI() {
      const shown = clamp(Math.round(st.anim), 0, N - 1);
      if (shown === st.shown) return;
      if (st.shown >= 0 && st.clock - st.lastWave > 0.18) { st.waves.push(st.clock); st.lastWave = st.clock; }
      st.shown = shown;
      if (Math.round(slider.get()) !== shown && !st.draggingSlider) slider.set(shown);
      const b = shown.toString(2).padStart(BITS, '0');
      bitEls.forEach((e, i) => e.classList.toggle('on', b[i] === '1'));
      bitNum.textContent = `${shown} = ${b}₂`;
      bits.setAttribute('aria-label', `${shown} in binary is ${b}`);
      updateReadout();
    }
    slider.input.addEventListener('pointerdown', () => { st.draggingSlider = true; });
    window.addEventListener('pointerup', () => { st.draggingSlider = false; });

    // ---------- drawing
    function draw() {
      const G = st.G;
      if (!G || !cv.w) return;
      const g = cv.g;
      cv.clear();
      const d = T.d, nP = d / 2, colW = G.hw / d, rowH = G.hh / N, cell = G.sw / N;
      const a = clamp(st.anim, -0.45, N - 0.55);
      const aOld = clamp(trailFrom(), -0.45, N - 0.55);
      const hv = st.hover;

      // ---- header (the longest wording that fits; mono at 9px with 0.1em tracking)
      const room = (G.wide ? G.hw : G.w - G.hx) - 4;
      const header = ['ONE DIAL PER SIN/COS PAIR · NUMBER = POSITIONS PER TURN', 'ONE DIAL PER SIN/COS PAIR · POSITIONS PER TURN', 'ONE DIAL PER PAIR · POSITIONS PER TURN']
        .find((s) => DR.measure(g, s, 9, 'mono') + 0.9 * s.length <= room) || 'DIALS · POSITIONS PER TURN';
      DR.text(g, header, G.hx, 10, { size: 9, role: 'mono', color: AM.col.mist, letterSpacing: '0.1em' });

      // ---- dial train + the silk threads each dial hangs into its two columns
      const tr = dialTrain(G.hw, d, G.wide), r = tr.r;
      const forkY = G.barTop - 13;
      const rowGap = tr.stagger ? 2 * tr.pitch : tr.pitch;          // spacing between labels in one row
      const labelStride = Math.max(1, Math.ceil(30 / rowGap));
      const labelBelow = (i) => !tr.stagger || i % 2 === 1;
      const showLabel = (i) => Math.floor(i / (tr.stagger ? 2 : 1)) % labelStride === 0;
      const threadStart = (i, p) => p.y + r + (labelBelow(i) && showLabel(i) ? 15 : 2);
      g.save();
      g.lineWidth = 1;
      for (let i = 0; i < nP; i++) {
        const p = dialPos(G, tr, i);
        g.strokeStyle = AM.rgba(T.cols[i], 0.34);
        g.beginPath();
        g.moveTo(p.x, threadStart(i, p)); g.lineTo(p.x, forkY);
        // fork: sin to the left column, cos to the right
        g.moveTo(p.x, forkY); g.quadraticCurveTo(p.x - colW / 2, forkY + 4, p.x - colW / 2, G.barTop - 2);
        g.moveTo(p.x, forkY); g.quadraticCurveTo(p.x + colW / 2, forkY + 4, p.x + colW / 2, G.barTop - 2);
        g.stroke();
      }
      g.restore();
      // glints: each tick sends a bead of light down every thread into the cloth
      for (const wv of st.waves) {
        const tt = (st.clock - wv) / 0.6;
        if (tt < 0 || tt > 1) continue;
        for (let i = 0; i < nP; i++) {
          const p = dialPos(G, tr, i), y0 = threadStart(i, p), col = T.cols[i];
          const al = 0.95 * (1 - tt * 0.5);
          if (tt < 0.72) {
            const yy = lerp(y0, forkY, tt / 0.72);
            g.fillStyle = AM.rgba(col, 0.25 * al); g.beginPath(); g.arc(p.x, yy, 3.6, 0, TAU); g.fill();
            g.fillStyle = AM.rgba('#fff8e6', al); g.beginPath(); g.arc(p.x, yy, 1.3, 0, TAU); g.fill();
          } else {
            const f = (tt - 0.72) / 0.28;
            for (const sgn of [-1, 1]) {
              const xx = p.x + sgn * (colW / 2) * f, yy = lerp(forkY, G.barTop - 2, f);
              g.fillStyle = AM.rgba(col, 0.25 * al); g.beginPath(); g.arc(xx, yy, 3.2, 0, TAU); g.fill();
              g.fillStyle = AM.rgba('#fff8e6', al); g.beginPath(); g.arc(xx, yy, 1.2, 0, TAU); g.fill();
            }
          }
        }
      }
      for (let i = 0; i < nP; i++) {
        const { x: cx, y: cy } = dialPos(G, tr, i);
        const w_ = omega(i, d), th = a * w_, thOld = aOld * w_, col = T.cols[i];
        const hot = hv && ((hv.type === 'dial' && hv.i === i) || (hv.type === 'heat' && Math.floor(hv.j / 2) === i));
        if (r >= 7) {
          dialFace(g, cx, cy, r, { cross: r > 12, rim: hot ? AM.dye.weld : AM.rgba(col, 0.55), rimWidth: hot ? 1.6 : 1, ticks: r > 22 ? 12 : r > 12 ? 4 : 0 });
          sweep(g, cx, cy, r * 0.86, th, thOld, col);
          if (r > 14) {
            // the hand's shadows on the axes: cos (across) and sin (up) are the pair's two numbers
            const tipX = cx + Math.cos(th) * r * 0.86, tipY = cy - Math.sin(th) * r * 0.86;
            g.save(); g.setLineDash([1.5, 2.5]); g.strokeStyle = AM.rgba(AM.col.linen, 0.3); g.lineWidth = 1;
            g.beginPath(); g.moveTo(tipX, tipY); g.lineTo(tipX, cy); g.moveTo(tipX, tipY); g.lineTo(cx, tipY); g.stroke(); g.restore();
            g.fillStyle = AM.rgba(AM.col.linen, 0.85);
            g.beginPath(); g.arc(tipX, cy, 1.7, 0, TAU); g.fill();
            g.beginPath(); g.arc(cx, tipY, 1.7, 0, TAU); g.fill();
          }
          hand(g, cx, cy, r * 0.86, th, col, { width: r > 14 ? 1.8 : 1.3, bead: r > 14 ? 2.3 : 1.6 });
          g.fillStyle = AM.col.ink3; g.beginPath(); g.arc(cx, cy, r > 14 ? 2.2 : 1.4, 0, TAU); g.fill();
        } else {
          sweep(g, cx, cy, r, th, thOld, col, 0.8);
          dialFace(g, cx, cy, r, { ticks: 0, rim: AM.rgba(col, 0.4) });
          hand(g, cx, cy, r, th, col, { width: 1.1, bead: 1.2 });
        }
        if (showLabel(i)) {
          const ly = labelBelow(i) ? cy + r + 11 : cy - r - 5;
          DR.text(g, periodLabel(TAU / w_), cx, ly, { size: G.wide ? 9.5 : 8.5, role: 'mono', color: hot ? AM.dye.weld : AM.col.linenDim, align: 'center' });
        }
      }

      // ---- the current row as bars, sitting right above its columns
      const mid = G.barTop + G.barH / 2;
      g.fillStyle = AM.rgba(AM.col.linen, 0.1); g.fillRect(G.hx, mid - 0.5, G.hw, 1);
      for (let j = 0; j < d; j++) {
        const i = j >> 1, v = j % 2 ? Math.cos(a * omega(i, d)) : Math.sin(a * omega(i, d));
        const bw = Math.max(1, colW * 0.62), x = G.hx + j * colW + (colW - bw) / 2, hh = Math.max(1, Math.abs(v) * (G.barH / 2 - 1));
        g.fillStyle = AM.color.seq((v + 1) / 2);
        g.fillRect(x, v >= 0 ? mid - hh : mid, bw, hh);
        g.strokeStyle = AM.rgba(AM.col.linen, 0.22); g.lineWidth = 1;
        g.strokeRect(x + 0.5, (v >= 0 ? mid - hh : mid) + 0.5, Math.max(0, bw - 1), Math.max(0, hh - 1));
      }
      if (colW >= 14) {
        // name the two threads of the first hand where they fork
        DR.text(g, 'sin', G.hx + colW * 0.5 - 5, G.barTop - 6, { size: 8.5, role: 'mono', color: AM.col.mist, align: 'right' });
        DR.text(g, 'cos', G.hx + colW * 1.5 + 5, G.barTop - 6, { size: 8.5, role: 'mono', color: AM.col.mist, align: 'left' });
      }

      // ---- encoding table
      g.drawImage(heatC, G.hx, G.hy, G.hw, G.hh);
      g.strokeStyle = AM.col.rule; g.lineWidth = 1; g.strokeRect(G.hx - 0.5, G.hy - 0.5, G.hw + 1, G.hh + 1);
      // ---- similarity matrix
      g.drawImage(simC, G.sx, G.sy, G.sw, G.sw);
      g.strokeRect(G.sx - 0.5, G.sy - 0.5, G.sw + 1, G.sw + 1);

      // ---- wake: freshly woven rows glow, then settle
      g.save();
      g.globalCompositeOperation = 'lighter';
      for (let p = 0; p < N; p++) {
        const I = st.wake[p];
        if (I < 0.02) continue;
        g.fillStyle = AM.rgba(AM.dye.weld, (0.24 + 0.14 * st.boost) * I);
        g.fillRect(G.hx, G.hy + p * rowH, G.hw, rowH);
        g.fillStyle = AM.rgba(AM.dye.weld, (0.14 + 0.1 * st.boost) * I);
        g.fillRect(G.sx, G.sy + p * cell, G.sw, cell);
      }
      g.restore();

      // ---- scanline (the shuttle) across both maps
      const rp = clamp(Math.round(a), 0, N - 1);
      const ys = G.hy + (a + 0.5) * rowH, yS = G.sy + (a + 0.5) * cell;
      g.save();
      g.fillStyle = AM.rgba(AM.dye.weld, 0.12 + 0.12 * st.boost);
      g.fillRect(G.hx, ys - rowH * (1.3 + st.boost), G.hw, rowH * (2.6 + 2 * st.boost));
      g.fillRect(G.sx, yS - cell * (1.3 + st.boost), G.sw, cell * (2.6 + 2 * st.boost));
      g.strokeStyle = AM.rgba(AM.dye.weld, 0.95); g.lineWidth = 1.2;
      g.beginPath(); g.moveTo(G.hx, ys); g.lineTo(G.hx + G.hw, ys); g.moveTo(G.sx, yS); g.lineTo(G.sx + G.sw, yS); g.stroke();
      g.strokeStyle = AM.rgba(AM.col.linen, 0.85); g.lineWidth = 1;
      g.strokeRect(G.hx - 0.5, G.hy + rp * rowH - 0.5, G.hw + 1, rowH + 1);
      // the matching column of the (symmetric) similarity matrix
      const xS = G.sx + (a + 0.5) * cell;
      g.strokeStyle = AM.rgba(AM.dye.weld, 0.35); g.setLineDash([2, 3]);
      g.beginPath(); g.moveTo(xS, G.sy); g.lineTo(xS, G.sy + G.sw); g.stroke();
      g.restore();
      DR.glowDot(g, G.hx - 5, ys, 2.4, AM.dye.weld);
      DR.glowDot(g, G.sx + (a + 0.5) * cell, yS, 2.2, '#fff4d6');

      // ---- position axis (left of the table)
      [0, 16, 32, 48, 63].forEach((p) => {
        const y = G.hy + (p + 0.5) * rowH;
        if (Math.abs(y - ys) < 11) return;
        DR.text(g, String(p), G.hx - 9, y + 3, { size: 9, role: 'mono', color: AM.col.mist, align: 'right' });
      });
      DR.text(g, String(rp), G.hx - 9, ys + 3.5, { size: 10, role: 'mono', color: AM.dye.weld, align: 'right', weight: 500 });
      DR.text(g, 'pos', G.hx - 9, G.hy - 4, { size: 8.5, role: 'mono', color: AM.col.mist, align: 'right' });
      // dimension axis (below the table)
      const by = G.hy + G.hh + 14;
      DR.text(g, 'dim 0', G.hx, by, { size: 9, role: 'mono', color: AM.col.mist });
      DR.text(g, `dim ${d - 1}`, G.hx + G.hw, by, { size: 9, role: 'mono', color: AM.col.mist, align: 'right' });
      DR.text(g, G.wide ? '← fast hands · slow hands →' : '← fast · slow →', G.hx + G.hw / 2, by, { size: 9, role: 'mono', color: AM.col.linenDim, align: 'center' });
      if (G.wide) DR.text(g, 'cream = +1 · blue = 0 · ink = −1', G.hx + G.hw / 2, by + 15, { size: 8.5, role: 'mono', color: AM.col.mist, align: 'center' });
      // similarity axes
      const sb = G.sy + G.sw + 14;
      DR.text(g, '0', G.sx, sb, { size: 9, role: 'mono', color: AM.col.mist });
      DR.text(g, '63', G.sx + G.sw, sb, { size: 9, role: 'mono', color: AM.col.mist, align: 'right' });
      DR.text(g, 'position j →', G.sx + G.sw / 2, sb, { size: 9, role: 'mono', color: AM.col.linenDim, align: 'center' });
      DR.text(g, `cosine of PE(i), PE(j) · ${fmt(T.sMin, 2)} to 1`, G.sx + G.sw / 2, sb + 15, { size: 8.5, role: 'mono', color: AM.col.mist, align: 'center' });
      if (!G.wide) DR.text(g, 'i ↓', G.sx - 6, G.sy + 10, { size: 9, role: 'mono', color: AM.col.mist, align: 'right' });

      // ---- hover marks
      if (hv && hv.type === 'heat') {
        g.save(); g.strokeStyle = AM.rgba(AM.col.linen, 0.95); g.lineWidth = 1.2;
        g.strokeRect(G.hx + hv.j * colW - 0.5, G.hy + hv.p * rowH - 0.5, colW + 1, rowH + 1);
        g.strokeStyle = AM.rgba(AM.col.linen, 0.25);
        g.strokeRect(G.hx + hv.j * colW - 0.5, G.hy - 0.5, colW + 1, G.hh + 1);
        g.restore();
      }
      if (hv && hv.type === 'sim') {
        // every cell on this diagonal has the same offset, hence the same value
        const k = hv.j - hv.i;
        const i0 = Math.max(0, -k), i1 = Math.min(N, N - k);
        g.save();
        g.strokeStyle = AM.rgba('#fff4d6', 0.25); g.lineWidth = cell * 2.2; g.lineCap = 'round';
        g.beginPath(); g.moveTo(G.sx + (i0 + k + 0.5) * cell, G.sy + (i0 + 0.5) * cell); g.lineTo(G.sx + (i1 - 1 + k + 0.5) * cell, G.sy + (i1 - 0.5) * cell); g.stroke();
        g.strokeStyle = AM.rgba('#fff4d6', 0.9); g.lineWidth = 1;
        g.stroke();
        g.strokeStyle = AM.col.linen; g.lineCap = 'butt';
        g.strokeRect(G.sx + hv.j * cell - 1, G.sy + hv.i * cell - 1, cell + 2, cell + 2);
        g.restore();
      }

      // ---- profile: similarity of the current position to every other one
      const px0 = G.sx, px1 = G.sx + G.sw, pTop = G.profTop + 22, pBot = G.profTop + G.profH;
      DR.text(g, `POS ${rp} · SIMILARITY TO EVERY POSITION`, px0, G.profTop + 10, { size: 9, role: 'mono', color: AM.col.mist, letterSpacing: '0.08em' });
      g.save();
      g.strokeStyle = AM.rgba(AM.col.linen, 0.08); g.lineWidth = 1;
      g.beginPath(); g.moveTo(px0, pBot + 0.5); g.lineTo(px1, pBot + 0.5); g.moveTo(px0, pTop + 0.5); g.lineTo(px1, pTop + 0.5); g.stroke();
      g.restore();
      DR.text(g, '1', px0 - 5, pTop + 3, { size: 8.5, role: 'mono', color: AM.col.mist, align: 'right' });
      DR.text(g, fmt(T.sMin, 2), px0 - 5, pBot + 3, { size: 8.5, role: 'mono', color: AM.col.mist, align: 'right' });
      const yOf = (v) => pBot - clamp((v - T.sMin) / (1 - T.sMin), -0.08, 1) * (pBot - pTop);
      const pts = [];
      for (let j = 0; j < N; j++) {
        let s = 0;
        for (let i = 0; i < nP; i++) s += Math.cos((a - j) * omega(i, d)); // PE(a)·PE(j) = Σ cos((a − j)ω_i)
        pts.push({ x: px0 + (j + 0.5) * cell, y: yOf(s / nP) });
      }
      g.save();
      const fill = g.createLinearGradient(0, pTop, 0, pBot);
      fill.addColorStop(0, AM.rgba(AM.dye.weld, 0.3)); fill.addColorStop(1, AM.rgba(AM.dye.weld, 0));
      g.beginPath(); g.moveTo(pts[0].x, pBot);
      pts.forEach((pt) => g.lineTo(pt.x, pt.y));
      g.lineTo(pts[N - 1].x, pBot); g.closePath();
      g.fillStyle = fill; g.fill();
      g.lineJoin = 'round';
      g.beginPath(); pts.forEach((pt, j) => (j ? g.lineTo(pt.x, pt.y) : g.moveTo(pt.x, pt.y)));
      g.strokeStyle = AM.rgba(AM.dye.weld, 0.22); g.lineWidth = 5; g.stroke();
      g.strokeStyle = AM.dye.weld; g.lineWidth = 1.5; g.stroke();
      g.restore();
      DR.glowDot(g, px0 + (a + 0.5) * cell, yOf(1), 2.6, '#fff4d6');
    }

    // ---------- pointer
    function hit(pt) {
      const G = st.G;
      if (!G) return null;
      const d = T.d, colW = G.hw / d, rowH = G.hh / N, cell = G.sw / N;
      if (pt.x >= G.hx && pt.x < G.hx + G.hw && pt.y >= G.hy && pt.y < G.hy + G.hh)
        return { type: 'heat', p: clamp(Math.floor((pt.y - G.hy) / rowH), 0, N - 1), j: clamp(Math.floor((pt.x - G.hx) / colW), 0, d - 1) };
      if (pt.x >= G.sx && pt.x < G.sx + G.sw && pt.y >= G.sy && pt.y < G.sy + G.sw)
        return { type: 'sim', i: clamp(Math.floor((pt.y - G.sy) / cell), 0, N - 1), j: clamp(Math.floor((pt.x - G.sx) / cell), 0, N - 1) };
      if (pt.x >= G.hx && pt.x < G.hx + G.hw && pt.y >= G.bandTop && pt.y < G.barTop + G.barH)
        return { type: 'dial', i: clamp(Math.floor((pt.x - G.hx) / (2 * colW)), 0, d / 2 - 1) };
      return null;
    }
    const rowOf = (h) => (h.type === 'heat' ? h.p : h.i);
    cv.canvas.addEventListener('pointerdown', (e) => {
      const h = hit(cv.pointer(e));
      st.hover = h;
      if (h && (h.type === 'heat' || h.type === 'sim')) {
        setPlaying(false); jumpTo(rowOf(h)); st.dragging = true;
        try { cv.canvas.setPointerCapture(e.pointerId); } catch (_) { /* ignore */ }
      }
      updateReadout(); draw();
    });
    const sameHit = (a, b) => (!a && !b) || (a && b && a.type === b.type && a.i === b.i && a.j === b.j && a.p === b.p);
    cv.canvas.addEventListener('pointermove', (e) => {
      const h = hit(cv.pointer(e));
      if (!sameHit(h, st.hover)) st.dirty = true;
      st.hover = h;
      if (st.dragging && h && (h.type === 'heat' || h.type === 'sim') && rowOf(h) !== st.pos) jumpTo(rowOf(h));
      updateReadout();
    });
    const endDrag = () => { st.dragging = false; };
    cv.canvas.addEventListener('pointerup', endDrag);
    cv.canvas.addEventListener('pointercancel', endDrag);
    cv.canvas.addEventListener('pointerleave', () => { if (!st.dragging) { st.hover = null; updateReadout(); draw(); } });

    // ---------- the wind-up: the first time the figure is in view, sweep the shuttle down the cloth
    if (typeof IntersectionObserver !== 'undefined') {
      const io = new IntersectionObserver((ens) => {
        for (const en of ens) {
          if (!en.isIntersecting || st.swept) continue;
          st.swept = true;
          io.disconnect();
          if (!AM.reducedMotion && st.playing) startSweep();
        }
      }, { threshold: 0.4 });
      io.observe(cv.wrap);
    }

    cv.onResize((w) => { st.G = geomClock(w); renderCaches(); draw(); });
    const seen = inView(cv.wrap);
    ctx.loop((t, dt) => {
      // off-screen: the clockwork stops (no ticks, no redraws) and picks up where it left off
      if (!seen.on) { st.dirty = true; return; }
      st.clock += dt;
      if (st.k < 1) {
        st.k = Math.min(1, st.k + dt / st.dur);
        st.anim = lerp(st.from, st.to, st.easeFn(st.k));
      } else {
        if (st.sweeping) st.sweeping = false;
        if (st.playing) {
          st.hold += dt;
          if (st.hold >= TICK_HOLD) {
            if (st.pos >= N - 1) setTarget(0, 1.5, EASE.inOut); // rewind: every hand spins back home
            else setTarget(st.pos + 1, TICK_MOVE, EASE.outBack); // one tick of the clockwork
          }
        }
      }
      // rows crossed this frame get woven (lit), then the glow decays
      const lo = Math.max(0, Math.round(Math.min(st.lastAnim, st.anim))), hi = Math.min(N - 1, Math.round(Math.max(st.lastAnim, st.anim)));
      st.boost = approach(st.boost, st.sweeping ? 1 : 0, dt * 4);
      const decay = Math.exp(-dt * (st.sweeping ? 0.75 : 2.2));
      for (let p = 0; p < N; p++) st.wake[p] *= decay;
      if (Math.abs(st.anim - st.lastAnim) > 1e-4) for (let p = lo; p <= hi; p++) st.wake[p] = 1;
      st.lastAnim = st.anim;
      st.hist.push({ t: st.clock, a: st.anim });
      while (st.hist.length && st.hist[0].t < st.clock - 0.5) st.hist.shift();
      syncUI();
      while (st.waves.length && st.clock - st.waves[0] > 0.7) st.waves.shift();
      // paused and settled: skip redrawing an unchanged frame (hover, resize and d changes set dirty)
      let glow = st.boost > 0.01 || st.waves.length > 0 || st.k < 1 || st.playing || Math.abs(trailFrom() - st.anim) > 1e-3;
      if (!glow) for (let p = 0; p < N; p++) if (st.wake[p] > 0.02) { glow = true; break; }
      if (glow || st.dirty || st.idleFrames < 3) draw();
      st.idleFrames = glow || st.dirty ? 0 : st.idleFrames + 1;
      st.dirty = false;
    });
    syncUI();
    updateReadout(true);
    return { el: fig };
  }

  // ======================================================================
  // 6. Figure C (live model): the tiny transformer's learned position table
  // ======================================================================

  /**
   * The real tiny model ('tinyworld') learns one position vector per seat.
   * We read them with its forward pass: feed one word n_ctx times, so the
   * first residual vector at seat t is wte[word] + wpe[t]; subtracting the
   * word's own embedding row leaves exactly wpe[t]. Cosine similarity of those
   * rows is set beside the sinusoids for the same seats and width, both raw
   * (neither side is mean-centred), so the comparison is like for like.
   * Returns null (and the page simply omits the figure) when no model ships.
   */
  function mountLearned(ctx) {
    let ok = false;
    try { ok = !!(AM.model && AM.model.ready && typeof AM.model.get === 'function' && AM.model.list().includes('tinyworld')); } catch (_) { ok = false; }
    if (!ok) return null;
    const { el } = ctx;
    const st = { L: null, S: null, N: 0, d: 0, hover: null, stats: null };
    const cv = ctx.canvas(null, {
      height: (w) => Math.round(Math.max(120, (w - 36 - 18) / 2) + 76),
      label: 'Two square maps of cosine similarity between positions. Left: the position vectors the tiny live model learned in training. Right: sinusoidal encodings for the same positions and width. Both are brightest along the diagonal.',
    });
    cv.canvas.style.cursor = 'crosshair';
    const readout = el('p', { class: 'pos-readout' }, 'Reading the tiny model’s position table…');
    const fig = AM.ui.figure({
      title: 'What the tiny model learned', badge: 'live', cls: 'pos-live-fig',
      caption: 'Live model: the tiny transformer on this page feeds one word through every seat, and its first residual vectors, minus the word’s own embedding, give its learned position vectors. Left: cosine similarity between every pair of them. Right: sinusoids for the same seats and width. One colour scale for both, from −1 (ink) to 1 (cream). Its table started as faint sinusoids and was reshaped by training. Hover or tap a cell to compare.',
    }, cv.wrap, readout);

    const geom = (w) => {
      const lm = 22, gap = 18, top = 24;
      const s = Math.max(120, Math.floor((w - lm - gap - 14) / 2));
      return { lm, gap, top, s, x0: lm, x1: lm + s + gap, y0: top, barY: top + s + 22 };
    };

    function compute() {
      const m = AM.model.get('tinyworld');
      const N = m.config.n_ctx, d = m.config.d_model;
      let id = typeof m.tokenId === 'function' ? m.tokenId('the') : -1;
      if (id < 0) id = Math.min(2, m.vocab.length - 1);
      const R = m.run(new Array(N).fill(id), { capture: true }).resid[0];
      // resid[0][t] = wte[id] + wpe[t]: take away the word's own embedding row and wpe[t] is left
      const wte = m._net && m._net.params && m._net.params.wte && m._net.params.wte.data;
      if (!wte || wte.length < (id + 1) * d) throw new Error('no embedding table');
      const P = R.map((x) => Array.from(x, (v, j) => v - wte[id * d + j]));
      const cos = (a, b) => dot(a, b) / (Math.sqrt(dot(a, a) * dot(b, b)) || 1);
      st.L = P.map((a) => P.map((b) => cos(a, b)));
      const PE = Array.from({ length: N }, (_, p) => sinusoid(p, d));
      st.S = PE.map((a) => PE.map((b) => dot(a, b) / (d / 2)));
      st.N = N; st.d = d;
      // a few facts for the readout, all read off the two matrices
      const near = (M) => { let s = 0; for (let i = 0; i + 1 < N; i++) s += M[i][i + 1]; return s / (N - 1); };
      let lo = { v: 2, i: 0, j: 0 }, sLo = 2;
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
        if (st.L[i][j] < lo.v) lo = { v: st.L[i][j], i, j };
        if (st.S[i][j] < sLo) sLo = st.S[i][j];
      }
      st.stats = { nearL: near(st.L), nearS: near(st.S), lo, sLo };
    }

    function setReadout() {
      if (!st.L) return;
      const h = st.hover, N = st.N;
      if (h) {
        readout.innerHTML = `Positions ${h.i} and ${h.j} (${Math.abs(h.i - h.j)} apart): learned <b>${fmt(st.L[h.i][h.j], 3)}</b> · sinusoid <b>${fmt(st.S[h.i][h.j], 3)}</b>`;
      } else {
        const s = st.stats, a = Math.min(s.lo.i, s.lo.j), b = Math.max(s.lo.i, s.lo.j);
        readout.innerHTML = `${N} positions × ${st.d} numbers. Next-door positions average <b>${fmt(s.nearL, 2)}</b> learned and <b>${fmt(s.nearS, 2)}</b> sinusoid. `
          + `The least similar learned pair is ${a} and ${b} at <b>${fmt(s.lo.v, 2)}</b> (about ${Math.round((Math.acos(clamp(s.lo.v, -1, 1)) * 180) / Math.PI)}° apart); the sinusoids never drop below <b>${fmt(s.sLo, 2)}</b> here.`;
      }
    }

    function map(g, x, y, s, M, label) {
      const N = M.length, c = s / N;
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
        g.fillStyle = AM.color.heat((M[i][j] + 1) / 2);
        g.fillRect(x + j * c, y + i * c, c + 0.4, c + 0.4);
      }
      g.strokeStyle = AM.col.rule; g.lineWidth = 1; g.strokeRect(x - 0.5, y - 0.5, s + 1, s + 1);
      DR.text(g, label, x, y - 9, { size: 9, role: 'mono', color: AM.col.linenDim, letterSpacing: '0.08em' });
      DR.text(g, '0', x - 5, y + c / 2 + 3, { size: 8.5, role: 'mono', color: AM.col.mist, align: 'right' });
      DR.text(g, String(N - 1), x - 5, y + s - c / 2 + 3, { size: 8.5, role: 'mono', color: AM.col.mist, align: 'right' });
    }

    function draw() {
      if (!cv.w) return;
      const g = cv.g, G = geom(cv.w);
      cv.clear();
      if (!st.L) {
        DR.text(g, 'loading the tiny model…', cv.w / 2, cv.h / 2, { size: 10, role: 'mono', color: AM.col.mist, align: 'center' });
        return;
      }
      const N = st.N, c = G.s / N;
      map(g, G.x0, G.y0, G.s, st.L, G.s > 170 ? 'LEARNED · TINY MODEL' : 'LEARNED');
      map(g, G.x1, G.y0, G.s, st.S, G.s > 170 ? `SINUSOIDS · ${st.d} DIMS` : 'SINUSOIDS');
      if (st.hover) {
        const { i, j } = st.hover;
        for (const x of [G.x0, G.x1]) {
          g.save();
          g.strokeStyle = AM.rgba(AM.col.linen, 0.3); g.lineWidth = 1;
          g.strokeRect(x - 0.5, G.y0 + i * c - 0.5, G.s + 1, c + 1);
          g.strokeStyle = AM.col.linen; g.lineWidth = 1.4;
          g.strokeRect(x + j * c - 1, G.y0 + i * c - 1, c + 2, c + 2);
          g.restore();
        }
      }
      // shared colour scale
      const bx = G.x0, bw = G.x1 + G.s - G.x0, by = G.barY;
      for (let k = 0; k < 64; k++) { g.fillStyle = AM.color.heat(k / 63); g.fillRect(bx + (k / 64) * bw, by, bw / 64 + 0.5, 5); }
      DR.text(g, '−1', bx, by + 18, { size: 8.5, role: 'mono', color: AM.col.mist });
      DR.text(g, '0', bx + bw / 2, by + 18, { size: 8.5, role: 'mono', color: AM.col.mist, align: 'center' });
      DR.text(g, '1', bx + bw, by + 18, { size: 8.5, role: 'mono', color: AM.col.mist, align: 'right' });
      const scaleLabel = 'one colour scale for both';
      if (DR.measure(g, scaleLabel, 8.5, 'mono') + 48 < bw / 2) DR.text(g, scaleLabel, bx + bw / 4, by + 18, { size: 8.5, role: 'mono', color: AM.col.linenDim, align: 'center' });
    }

    const hit = (pt) => {
      if (!st.L) return null;
      const G = geom(cv.w), N = st.N;
      for (const x of [G.x0, G.x1]) {
        if (pt.x >= x && pt.x < x + G.s && pt.y >= G.y0 && pt.y < G.y0 + G.s) {
          return { i: clamp(Math.floor(((pt.y - G.y0) / G.s) * N), 0, N - 1), j: clamp(Math.floor(((pt.x - x) / G.s) * N), 0, N - 1) };
        }
      }
      return null;
    };
    const onPt = (e) => {
      const h = hit(cv.pointer(e));
      if ((h && st.hover && h.i === st.hover.i && h.j === st.hover.j) || (!h && !st.hover)) return;
      st.hover = h; setReadout(); draw();
    };
    cv.canvas.addEventListener('pointermove', onPt);
    cv.canvas.addEventListener('pointerdown', onPt);
    cv.canvas.addEventListener('pointerleave', () => { if (st.hover) { st.hover = null; setReadout(); draw(); } });

    cv.onResize(draw);
    AM.model.ready.then(() => { compute(); setReadout(); draw(); }).catch(() => { fig.remove(); });
    return fig;
  }

  // ======================================================================
  // 7. Chapter-scoped styles
  // ======================================================================

  const CSS = `
    #ch-position .pos-readout { margin: 0; font-family: var(--font-mono); font-size: 11.5px; line-height: 1.7; color: var(--linen-dim); min-height: 3.4em; font-variant-numeric: tabular-nums; }
    #ch-position .pos-readout b { color: var(--weld); font-weight: 500; }
    #ch-position .pos-readout sup, #ch-position .pos-readout sub { font-size: 0.72em; line-height: 0; }
    #ch-position .pos-readout .pos-dim { color: var(--mist); }
    #ch-position .pos-clock-readout { min-height: 3.6em; }
    #ch-position .pos-ctrls { gap: var(--space-4) var(--space-6); align-items: end; }
    #ch-position .pos-bits { display: grid; gap: 6px; }
    #ch-position .pos-bits-row { display: inline-flex; gap: 5px; align-items: center; height: 22px; }
    #ch-position .pos-bits i { display: inline-block; width: 11px; height: 11px; border-radius: 50%; border: 1px solid var(--rule-strong); background: var(--ink); }
    #ch-position .pos-bits i.on { background: var(--weld); border-color: var(--weld); box-shadow: 0 0 9px color-mix(in srgb, var(--weld) 65%, transparent); }
    #ch-position .pos-bits-num { font-family: var(--font-mono); font-size: var(--fs-micro); color: var(--linen-dim); font-variant-numeric: tabular-nums; min-width: 11ch; }
    #ch-position .pos-bits { grid-template-columns: auto auto; grid-template-areas: 'l l' 'r n'; column-gap: 12px; align-items: center; }
    #ch-position .pos-bits .ctl-label { grid-area: l; }
    #ch-position .pos-bits-row { grid-area: r; }
    #ch-position .pos-bits-num { grid-area: n; }
    #ch-position .pos-clock-sec { display: grid; gap: var(--space-6); }
    #ch-position .pos-notes { align-items: start; }
    #ch-position .pos-notes .prose, #ch-position .pos-learned .prose { max-width: none; }
    #ch-position .pos-learned { align-items: start; }
    #ch-position .math.block { font-size: 0.82em; }
    #ch-position .step .math.block { margin-block: 2px; }
    #ch-position .pos-formula { display: grid; gap: 4px; }
    #ch-position .prose > *, #ch-position .step > * { min-width: 0; }
    #ch-position .pos-lc .ctl-label { text-transform: none; letter-spacing: 0.06em; }
    #ch-position .ctl-label sub { font-size: 0.8em; line-height: 0; }
    @media (max-width: 900px) {
      #ch-position .pos-bag-fig figcaption { display: none; }
      #ch-position .pos-bag-fig .pos-readout { font-size: 10.5px; line-height: 1.55; min-height: 4.7em; }
      #ch-position .pos-bag-fig { gap: var(--space-2); }
    }
    @media (max-width: 520px) {
      #ch-position .pos-ctrls { gap: var(--space-4); }
      #ch-position .pos-ctrls .ctl-range { width: 100%; }
      #ch-position .pos-readout { font-size: 10.5px; }
    }
  `;

  // ======================================================================
  // 8. The chapter
  // ======================================================================

  AM.chapter({
    id: ID,
    num: 3,
    kicker: 'Positional encoding',
    title: 'The Clockwork of <em>Order</em>',
    lede: 'Attention cannot tell first from last on its own, so every position gets its own signature, a set of turning clock hands, added to its vector.',
    where: 'pos',

    mount(root, ctx) {
      ctx.header();
      AM.css(CSS);
      const { el } = ctx;
      const P = (html) => el('p', { html });
      const step = (label, title, ...kids) => el('div', { class: 'step' }, el('span', { class: 'step-label' }, label), el('h3', {}, title), ...kids);

      // ---- scrollytelling: the bag
      const bag = mountBag(ctx);
      const r1 = bag.B.rest[1];
      const steps = [
        step('01 · The problem', 'Same words, opposite news',
          P('“Dog bites man” is not news. “Man bites dog” is. The words are identical; only the order differs, and the order carries the meaning.'),
          P('<span class="term">Self-attention</span>, the step where tokens read from one another (chapter 4 builds it piece by piece), compares tokens through dot products of their vectors. Nothing in that arithmetic records who came first. Hand it three word vectors and it receives a set, a <span class="term">bag of words</span>. Watch the threads cross as the order flips. The bag does not change.')),
        step('02 · Equivariance', 'Shuffle in, shuffle out',
          P('Each word’s output is a blend of what every word offers, weighted by how well each pair of vectors matches. Reorder the words and every pair still matches just as well, so every output simply moves with its word.'),
          P('This property is called <span class="term">permutation equivariance</span>: shuffle the inputs and the outputs come out shuffled the same way. Below the bag, each output sits exactly inside the outline of the same word’s output from the other sentence.'),
          P('GPT-style models also stop each token from seeing the words after it. That leaks a little order information, and models trained with no positions at all learn to exploit it. In practice, models are almost always given positions explicitly.')),
        step('03 · The fix', 'Stamp every seat',
          P('Give each position <span class="math">t</span> its own vector <span class="math">p<sub>t</sub></span>, the same length as the embeddings, and add it:'),
          el('span', { class: 'math block' }, 'x', el('sub', {}, 't'), ' = e(token', el('sub', {}, 't'), ') + p', el('sub', {}, 't')),
          P(`Now “dog” in seat 0 and “dog” in seat 2 are different vectors. The beads in the bag move with every reorder, the attention weights change, and the outputs part ways: dog’s output now differs by up to <strong>${r1.diff[0].toFixed(2)}</strong> between the sentences.`),
          P('The little clocks over the seats are the position vectors. Each hand is one sin/cos pair; we open the clock up just below.')),
        step('04 · Why add?', 'Two messages in one vector',
          P('Why add the position instead of appending it as extra numbers? Adding keeps every vector at <span class="math">d<sub>model</sub></span> numbers and costs nothing.'),
          P('It works because the space is roomy. With hundreds of dimensions, token information and position information can sit in nearly separate directions, and the layers learn to read each one.'),
          P('In the bag, each dashed ring is the bare embedding and the gold arrow is the position vector added to it. The shadow is a linear projection, so it keeps the sum honest: the arrow for a seat is the same whichever word sits there.')),
      ];
      const stage = el('div', { class: 'ch-stage' }, bag.el);
      const split = el('div', { class: 'ch-split' }, stage, el('div', { class: 'ch-prose' }, steps));
      ctx.steps(steps, (i) => bag.setStep(i));

      // ---- the clockwork
      const lastPeriod = thousands(Math.round(TAU / omega(15, 32) / 1000) * 1000);
      const clockIntro = el('div', { class: 'prose pos-intro' },
        el('h3', {}, 'A clock for every position'),
        P('So what should the position vectors be? The original Transformer (Vaswani et al., 2017) used a fixed recipe of sines and cosines, with nothing to learn:'),
        el('div', { class: 'math block pos-formula', html:
          '<span>PE(pos, 2i) = sin(pos · ω<sub>i</sub>)</span><span>PE(pos, 2i+1) = cos(pos · ω<sub>i</sub>)</span><span>ω<sub>i</sub> = 1 / 10000<sup>2i/d<sub>model</sub></sup></span>' }),
        P('Read each sin/cos pair as the tip of a clock hand: cos is how far it points right, sin how far up. One step forward turns hand <span class="math">i</span> by <span class="math">ω<sub>i</sub></span> radians, wherever it starts. Hand 0 turns one radian per step and goes round every 6.3 positions. Each hand after it is slower by the same factor, down to the last, which needs about ' + lastPeriod + ' positions for one turn when <span class="math">d<sub>model</sub> = 32</span>.'),
        P('Fast hands tell neighbours apart; slow hands tell distant positions apart. A clock does the same with its second, minute and hour hands, and a binary counter with its bits: the lowest flips every step, each higher bit half as often.'));
      const clock = mountClockwork(ctx);
      const learned = mountLearned(ctx); // null when the page ships without the live model
      const learnedP = P('The alternative is to learn the position vectors like word embeddings, one trainable row per position. GPT-2 learns 1,024 of them, and the tiny live model on this page (chapter 5 introduces it properly) learns its own 32. The 2017 paper found the two approaches worked about equally well, though a learned table has nothing to offer past the longest position seen in training.');
      const notes = el('div', { class: 'grid-2 pos-notes' },
        el('div', { class: 'prose' },
          el('h3', {}, 'Nearby positions look alike'),
          P('Multiply two encodings. Each pair contributes'),
          el('span', { class: 'math block' }, 'sin a · sin b + cos a · cos b = cos(a − b)'),
          P('so the whole dot product is'),
          el('span', { class: 'math block' }, 'PE(a) · PE(b) = Σ', el('sub', {}, 'i'), ' cos((a − b) · ω', el('sub', {}, 'i'), ')'),
          P('Only the offset <span class="math">a − b</span> appears. Neighbours score high. Farther apart, the score drops and ripples as the fast hands drift in and out of step. That is the glowing band in the square map and the wavy curve above it. Every diagonal of the map holds a single value.')),
        el('div', { class: 'prose' },
          el('h3', {}, 'A shift is a rotation'),
          P('Moving <span class="math">k</span> places forward turns every hand by <span class="math">k · ω<sub>i</sub></span>, wherever it started. So <span class="math">PE(pos + k)</span> is a fixed rotation of <span class="math">PE(pos)</span>, the same linear map for every <span class="math">pos</span>. The authors chose sinusoids hoping this would make relative positions easy to learn.'),
          // today's most common scheme turns the same clocks inside attention: one pointer ahead, no detour here
          P('Most open models today, including Llama, Mistral and Qwen, build on this rotation with <span class="term">rotary position embeddings</span> (RoPE). Nothing is added to the token vectors. The hands turn inside attention instead, and the end of chapter 4 shows how.'),
          learned ? null : learnedP));
      // with the live model on the page, learned positions get their own row: prose beside the live figure
      const learnedRow = learned ? el('div', { class: 'grid-2 pos-learned' },
        el('div', { class: 'prose' },
          el('h3', {}, 'Or learn the positions'),
          learnedP,
          P('The tiny model’s own table, read live in the figure, also keeps neighbours alike: the bright diagonal. Farther apart it is freer than the sinusoids, and some pairs end up pointing well apart, the dark patches. Nobody designed that pattern; it came out of training.')),
        learned) : null;
      const clockSec = el('div', { class: 'ch-wide pos-clock-sec' }, clockIntro, clock.el, notes, learnedRow);

      const callout = el('div', { class: 'callout' },
        el('span', { class: 'callout-label' }, 'Key idea'),
        P('Attention treats its input as a bag, so order has to be supplied. The original Transformer adds a bank of clock hands, turning at geometrically spaced speeds, to every token vector. GPT-2 adds a learned table instead. RoPE adds nothing to the vectors: it rotates queries and keys inside attention (chapter 4).'));

      root.appendChild(el('div', { class: 'ch-body' }, split, clockSec, callout));
    },
  });
})();
