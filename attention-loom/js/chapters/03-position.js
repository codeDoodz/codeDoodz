/* The Attention Loom · Chapter 03 · Positional encoding · "The Clockwork of Order".
 *
 * Teaches: self-attention on its own is permutation-equivariant (it receives
 * a bag of tokens); the fix is to ADD a position vector to every embedding;
 * the 2017 sinusoidal encoding read as a bank of clock hands turning at
 * geometrically spaced speeds; why that encodes relative offsets (the dot
 * product of two encodings depends only on their offset, and a shift is a
 * rotation); learned absolute positions; RoPE (rotate q and k) and ALiBi.
 *
 * Data, all computed in this file:
 *  - Bag stage: a TOY one-head attention layer (d = 8, seeded random
 *    embeddings and W_Q / W_K / W_V, no causal mask). Weights and outputs are
 *    recomputed every frame. The bag is a 2-D PCA shadow of the 8-d inputs.
 *  - Dial anatomy + clockwork: the exact sinusoidal formula. Similarity =
 *    cosine of exact encodings (every encoding has length √(d/2)).
 *  - RoPE lab: toy q and k (d = 8), rotated exactly with θ_i = 10000^(−2i/d).
 *  - ALiBi: exact softmax of the distance penalty for three head slopes.
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
  /** Canvas text with a subscript, drawn by hand (the mono face has no subscript digits). */
  function subText(g, base, subscript, x, y, o = {}) {
    const size = o.size || 10, role = o.role || 'mono';
    const bw = DR.measure(g, base, size, role, o.weight);
    const sw = DR.measure(g, subscript, size * 0.72, role, o.weight);
    const x0 = o.align === 'center' ? x - (bw + sw) / 2 : o.align === 'right' ? x - bw - sw : x;
    DR.text(g, base, x0, y, { ...o, size, role, align: 'left' });
    DR.text(g, subscript, x0 + bw + 0.5, y + size * 0.28, { ...o, size: size * 0.72, role, align: 'left' });
  }
  const thousands = (n) => String(Math.round(n)).replace(/\B(?=(\d{3})+(?!\d))/g, ',');

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

  /** Arrow from the centre of a dial (math-angle convention). */
  function dialArrow(g, cx, cy, len, ang, color, o = {}) {
    const tx = cx + Math.cos(ang) * len, ty = cy - Math.sin(ang) * len;
    g.save();
    g.strokeStyle = AM.rgba(color, 0.2 * (o.alpha ?? 1));
    g.lineWidth = (o.width || 2) * 3;
    g.lineCap = 'round';
    if (!o.dash) { g.beginPath(); g.moveTo(cx, cy); g.lineTo(tx, ty); g.stroke(); }
    g.restore();
    DR.arrow(g, cx, cy, tx, ty, { color, width: o.width || 2, head: o.head || 9, alpha: o.alpha ?? 1, dash: o.dash });
    return { x: tx, y: ty };
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
      caption: 'Toy numbers: three random 8-number embeddings and one attention head with random W<sub>Q</sub>, W<sub>K</sub>, W<sub>V</sub> and no causal mask, all computed live. The bag is a 2-D shadow (PCA) of the 8-d input vectors; threads inside it are attention weights, thicker for heavier weights. Bars are vectors, blue positive and red negative.',
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

      // ---- the sentence as written
      const sA = 'dog bites man', sB = 'man bites dog';
      const fs = G.small ? 21 : 29;
      const mixT = AM.math.smoothstep(0.35, 0.65, u);
      [[sA, 1 - mixT, 'not news'], [sB, mixT, 'news']].forEach(([txt, a, gloss]) => {
        if (a < 0.01) return;
        DR.text(g, txt, w / 2, G.sentY + 8, { size: fs, role: 'display', italic: true, weight: 400, align: 'center', alpha: a });
        const tw = DR.measure(g, txt, fs, 'display', 400);
        DR.text(g, gloss, w / 2 + tw / 2 + 10, G.sentY + 4, { size: 9, role: 'mono', color: AM.col.mist, alpha: a * 0.9, letterSpacing: '0.08em' });
      });

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

      // ---- bag (woven basket)
      const b = G.bag;
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
      DR.text(g, 'WHAT ATTENTION RECEIVES', b.cx, b.cy + b.ry - (G.small ? 9 : 13), { size: G.small ? 8 : 9, role: 'mono', color: AM.col.mist, align: 'center', letterSpacing: '0.14em' });

      const beads = R.X.map((x) => bead(G, x));
      const ghosts = B.E.map((e) => bead(G, e));

      // ---- tokens: positions along the row (dog arcs over, man ducks under)
      const tok = WORDS.map((_, wi) => {
        const x = xOfSeat(seats[wi]);
        const lift = Math.sin(Math.PI * u);
        let y = G.tokY, sc = 1, al = 1;
        if (wi === 0) y -= lift * arc;
        if (wi === 2) { y += lift * arc * 0.3; sc = 1 - 0.18 * lift; al = 1 - 0.45 * lift; }
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
            subText(g, '+p', String(Math.round(seats[wi])), (gh.x + bd.x) / 2 + 4, (gh.y + bd.y) / 2 - 7, { size: 10.5, color: AM.dye.weld, alpha: a });
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
        html = `Attention weight dog → man: <b>${r0.a.A[0][2].toFixed(3)}</b> in “dog bites man”, <b>${r0.b.A[0][2].toFixed(3)}</b> in “man bites dog”. Same three vectors, same weights.`;
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

    cv.onResize((w, h) => { st.G = geom(w, h); draw(); });
    ctx.loop((t, dt) => {
      st.clock += dt;
      st.u = swapU(st.clock);
      st.posMix = approach(st.posMix, st.step >= 2 ? 1 : 0, dt * 3.2);
      st.ghostMix = approach(st.ghostMix, st.step >= 3 ? 1 : 0, dt * 3.2);
      st.outMix = approach(st.outMix, st.step >= 1 ? 1 : 0.35, dt * 3.2);
      draw();
    });
    setReadout();
    return { el: fig, setStep, B };
  }

  // ======================================================================
  // 5a. Figure B0: how to read one dial
  // ======================================================================

  function mountAnatomy(ctx) {
    const LAST = 12;                       // the dial ticks through positions 0…12, then rewinds
    const st = { pos: 0, anim: 0, from: 0, k: 1, dur: 0.4, fn: EASE.outBack, hold: 0, playing: true };
    const SIN = AM.dye.cochineal, COS = AM.dye.verdigris;
    const cv = ctx.canvas(null, {
      height: (w) => (w < 440 ? Math.round(Math.min(w * 0.33, 120) * 2 + 156) : clamp(Math.round(w * 0.74), 300, 380)),
      label: 'One clock hand of the positional encoding. Its tip has two coordinates: how far across is the cosine value and how far up is the sine value. Beads on the rim mark where the hand pointed at earlier positions, each one radian further round.',
    });

    function draw() {
      if (!cv.w) return;
      const g = cv.g, w = cv.w, h = cv.h;
      cv.clear();
      const narrow = w < 440;
      const R = narrow ? Math.min(w * 0.33, 120) : Math.min(h * 0.36, w * 0.27, 128);
      const cx = narrow ? w / 2 : R + 34, cy = narrow ? R + 34 : h / 2;
      const a = st.anim, th = a; // hand 0: ω₀ = 1 radian per position
      const pNow = Math.round(clamp(a, 0, LAST));

      dialFace(g, cx, cy, R, { ticks: 24, cross: true, rim: AM.rgba(AM.dye.weld, 0.4), rimWidth: 1.2 });
      g.save(); g.strokeStyle = AM.rgba(AM.dye.weld, 0.07); g.lineWidth = 9;
      g.beginPath(); g.arc(cx, cy, R + 5, 0, TAU); g.stroke(); g.restore();

      // rim beads: where the hand pointed at each earlier position, one radian apart
      for (let k = 0; k <= Math.min(LAST, Math.floor(a + 1e-6)); k++) {
        const recent = 1 - clamp((a - k) / (LAST + 1), 0, 1) * 0.6;
        const bx = cx + Math.cos(k) * R, by = cy - Math.sin(k) * R;
        DR.glowDot(g, bx, by, 2.2, AM.dye.weld, recent);
        DR.text(g, String(k), cx + Math.cos(k) * (R + 15), cy - Math.sin(k) * (R + 15) + 3.5, { size: 9.5, role: 'mono', color: AM.col.linenDim, align: 'center', alpha: recent });
      }
      // the angle turned so far (beyond whole turns)
      const t = ((th % TAU) + TAU) % TAU;
      if (t > 0.02) {
        g.save(); g.strokeStyle = AM.rgba(AM.dye.weld, 0.6); g.lineWidth = 1.2;
        g.beginPath(); g.arc(cx, cy, R * 0.22, 0, -t, true); g.stroke(); g.restore();
      }
      // projections: cos across, sin up
      const tipX = cx + Math.cos(th) * R * 0.9, tipY = cy - Math.sin(th) * R * 0.9;
      g.save();
      g.setLineDash([2, 3]); g.lineWidth = 1;
      g.strokeStyle = AM.rgba(COS, 0.7); g.beginPath(); g.moveTo(tipX, tipY); g.lineTo(tipX, cy); g.stroke();
      g.strokeStyle = AM.rgba(SIN, 0.7); g.beginPath(); g.moveTo(tipX, tipY); g.lineTo(cx, tipY); g.stroke();
      g.setLineDash([]); g.lineCap = 'round'; g.lineWidth = 3.5;
      g.strokeStyle = AM.rgba(COS, 0.9); g.beginPath(); g.moveTo(cx, cy); g.lineTo(tipX, cy); g.stroke();
      g.strokeStyle = AM.rgba(SIN, 0.9); g.beginPath(); g.moveTo(cx, cy); g.lineTo(cx, tipY); g.stroke();
      g.restore();
      hand(g, cx, cy, R * 0.9, th, AM.dye.weld, { width: 2.2, bead: 3 });
      DR.glowDot(g, cx, cy, 2.4, AM.col.linen);
      DR.text(g, 'cos', tipX + (Math.cos(th) >= 0 ? 6 : -6), cy + 14, { size: 10, role: 'mono', color: COS, align: Math.cos(th) >= 0 ? 'left' : 'right' });
      DR.text(g, 'sin', cx + (Math.cos(th) >= 0 ? -7 : 7), tipY + 4, { size: 10, role: 'mono', color: SIN, align: Math.cos(th) >= 0 ? 'right' : 'left' });

      // the two numbers the hand stands for
      const tx = narrow ? 8 : cx + R + 44, ty = narrow ? cy + R + 46 : cy - 40;
      const s = Math.sin(pNow), c = Math.cos(pNow);
      DR.text(g, `POSITION ${pNow}`, tx, ty, { size: 9.5, role: 'mono', color: AM.col.mist, letterSpacing: '0.12em' });
      DR.text(g, `PE(${pNow}, 0) = sin(${pNow})`, tx, ty + 24, { size: 11, role: 'mono', color: SIN });
      DR.text(g, `= ${fmt(s, 3)}`, tx + (narrow ? 178 : 0), ty + (narrow ? 24 : 42), { size: narrow ? 11 : 15, role: 'mono', color: AM.col.linen, weight: 500 });
      DR.text(g, `PE(${pNow}, 1) = cos(${pNow})`, tx, ty + (narrow ? 44 : 70), { size: 11, role: 'mono', color: COS });
      DR.text(g, `= ${fmt(c, 3)}`, tx + (narrow ? 178 : 0), ty + (narrow ? 44 : 88), { size: narrow ? 11 : 15, role: 'mono', color: AM.col.linen, weight: 500 });
      if (!narrow) DR.text(g, 'one step = one radian further', tx, ty + 116, { size: 9.5, role: 'mono', color: AM.col.mist });
    }

    cv.onResize(draw);
    ctx.loop((t, dt) => {
      if (st.k < 1) {
        st.k = Math.min(1, st.k + dt / st.dur);
        st.anim = lerp(st.from, st.pos, st.fn(st.k));
        draw();
      } else if (st.playing) {
        st.hold += dt;
        if (st.hold > 1.1) {
          st.hold = 0; st.from = st.anim; st.k = 0;
          if (st.pos >= LAST) { st.pos = 0; st.dur = 1.6; st.fn = EASE.inOut; }
          else { st.pos += 1; st.dur = 0.45; st.fn = EASE.outBack; }
        }
      }
    });
    return AM.ui.figure({
      title: 'How to read one dial', badge: AM.ui.badge('toy', 'Exact formula'), cls: 'pos-anat-fig',
      caption: 'Hand 0 (ω<sub>0</sub> = 1). The hand’s tip sits at (cos, sin) of the angle it has turned, pos × ω. Beads mark the earlier positions: every step turns the hand by the same angle, wherever it starts.',
    }, cv.wrap);
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
      if (!on) st.sweeping = false;
    }
    const slider = AM.ui.slider({
      id: 'pos-clock-pos', label: 'Position', min: 0, max: N - 1, step: 1, value: 0, format: (v) => String(v),
      onInput: (v) => { setPlaying(false); jumpTo(v); },
    });
    const seg = AM.ui.segmented({
      id: 'pos-clock-d', label: 'd_model', value: st.d,
      options: [16, 32, 64].map((v) => ({ value: v, label: String(v) })),
      onChange: (v) => { st.d = v; T = buildTables(v); renderCaches(); updateReadout(true); draw(); },
    });
    seg.el.classList.add('pos-lc');
    const BITS = 6;
    const bitEls = Array.from({ length: BITS }, () => el('i', { 'aria-hidden': 'true' }));
    const bitNum = el('span', { class: 'pos-bits-num' });
    const bits = el('div', { class: 'pos-bits', role: 'img' }, el('span', { class: 'ctl-label' }, 'Binary'), el('span', { class: 'pos-bits-row' }, bitEls), bitNum);
    const readout = el('p', { class: 'pos-readout pos-clock-readout' });
    const controls = el('div', { class: 'controls pos-ctrls' }, playBtn, slider.el, seg.el, bits);

    const fig = AM.ui.figure({
      title: 'The position clockwork', badge: AM.ui.badge('toy', 'Exact formula'), cls: 'pos-clock-fig',
      caption: `Exact formula for positions 0–63. Each dial drives the two columns beneath it; gold hands are fast, blue ones slow. The tall map is the encoding table, one row per position (cream = +1, ink = −1). The square map is the cosine similarity between every pair of positions, colour stretched from its lowest value to 1. Drag or tap a map to move the shuttle; hover or tap any cell to read its formula.`,
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
        const w_ = omega(h.i, d);
        html = `Hand ${h.i} (dims ${2 * h.i}, ${2 * h.i + 1}) turns ω = 10000<sup>−${2 * h.i}/${d}</sup> = <b>${w_ < 0.01 ? w_.toExponential(2) : fmt(w_, 4)}</b> rad per position: one full turn every <b>${thousands(TAU / w_)}</b> positions. At position ${p} it points at ${fmt((p * w_) % TAU, 3)} rad.`;
      } else {
        html = `Position <b>${p}</b>: hand 0 has turned ${p} rad, so PE(${p}, 0) = sin(${p}) = <b>${fmt(Math.sin(p), 3)}</b> and PE(${p}, 1) = cos(${p}) = <b>${fmt(Math.cos(p), 3)}</b>. <span class="pos-dim">Hover or tap the dials and maps to read more.</span>`;
      }
      if (force || html !== st.readoutHtml) { st.readoutHtml = html; readout.innerHTML = html; }
    }
    function syncUI() {
      const shown = clamp(Math.round(st.anim), 0, N - 1);
      if (shown === st.shown) return;
      if (st.shown >= 0 && st.clock - st.lastWave > 0.18) { st.waves.push(st.clock); st.lastWave = st.clock; }
      while (st.waves.length && st.clock - st.waves[0] > 0.7) st.waves.shift();
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

      // ---- header
      DR.text(g, G.wide ? 'ONE DIAL PER SIN/COS PAIR · NUMBER = POSITIONS PER TURN' : 'ONE DIAL PER SIN/COS PAIR · POSITIONS PER TURN', G.hx, 10, { size: 9, role: 'mono', color: AM.col.mist, letterSpacing: '0.1em' });

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
        DR.text(g, 'sin', G.hx + colW * 0.5 - 5, G.barTop - 6, { size: 8, role: 'mono', color: AM.col.mist, align: 'right' });
        DR.text(g, 'cos', G.hx + colW * 1.5 + 5, G.barTop - 6, { size: 8, role: 'mono', color: AM.col.mist, align: 'left' });
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
      if (G.wide) DR.text(g, 'cream = +1 · ink = −1', G.hx + G.hw / 2, by + 15, { size: 8.5, role: 'mono', color: AM.col.mist, align: 'center' });
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
    cv.canvas.addEventListener('pointermove', (e) => {
      const h = hit(cv.pointer(e));
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
    ctx.loop((t, dt) => {
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
      draw();
    });
    syncUI();
    updateReadout(true);
    return { el: fig };
  }

  // ======================================================================
  // 6. Figure C: the RoPE lab
  // ======================================================================

  function mountRope(ctx) {
    const { el } = ctx;
    const D = 8, NP = D / 2, MAXP = 30;
    const theta = (i) => Math.pow(BASE, (-2 * i) / D); // RoPE speeds: the same clock speeds as the sinusoids
    // toy query and key contents (before any rotation)
    const Q = [0.92, 0.42, 0.68, -0.32, 0.4, 0.62, -0.5, 0.3];
    const K = [0.78, -0.34, 0.48, 0.6, -0.36, 0.5, 0.44, 0.36];

    /** RoPE: rotate pair i of v by the angle pos·θ_i. */
    const rope = (v, pos) => {
      const out = new Array(D);
      for (let i = 0; i < NP; i++) {
        const ang = pos * theta(i), c = Math.cos(ang), s = Math.sin(ang);
        out[2 * i] = v[2 * i] * c - v[2 * i + 1] * s;
        out[2 * i + 1] = v[2 * i] * s + v[2 * i + 1] * c;
      }
      return out;
    };
    /** The attention logit before the 1/√d scale: RoPE(q, m) · RoPE(k, n). */
    const score = (m, n) => dot(rope(Q, m), rope(K, n));
    const pairScore = (m, n, i) => { const q = rope(Q, m), k = rope(K, n); return q[2 * i] * k[2 * i] + q[2 * i + 1] * k[2 * i + 1]; };
    const pairNorm = (v, i) => Math.hypot(v[2 * i], v[2 * i + 1]);
    const normMax = Math.max(...[0, 1, 2, 3].flatMap((i) => [pairNorm(Q, i), pairNorm(K, i)]));
    // the score as a function of the offset alone: S(Δ) = RoPE(q, Δ)·RoPE(k, 0)
    const curve = Array.from({ length: 2 * MAXP + 1 }, (_, i) => score(i - MAXP, 0));
    const cMin = Math.min(...curve), cMax = Math.max(...curve);
    const thLabel = (t) => (t >= 0.1 ? t.toFixed(1) : t.toFixed(3));

    const st = { m: 7, n: 3, am: 7, an: 3, pair: 0, msg: '', dirty: true, flash: 0 };

    function geom(w) {
      const wide = w >= 440;
      const G = { w, wide };
      if (wide) {
        G.R = Math.min(w * 0.24, 132);
        G.cx = G.R + 10; G.cy = 40 + G.R;
        const x0 = G.cx + G.R + 26, aw = w - x0;
        G.mr = Math.min(aw / 5.2, 30);
        G.mx = [0, 1, 2, 3].map((i) => x0 + aw * (i % 2 ? 0.74 : 0.26));
        G.my = [0, 1, 2, 3].map((i) => G.cy + (i < 2 ? -0.42 : 0.48) * G.R);
        G.miniTitle = { x: x0 + aw / 2, y: G.cy - G.R + 2 };
        G.cTop = G.cy + G.R + 58; G.cH = 118;
      } else {
        G.R = Math.min(w * 0.3, 112);
        G.cx = w / 2; G.cy = 40 + G.R;
        G.mr = Math.min(w / 11, 24);
        const sp = Math.min(w / 4.2, 100);
        G.mx = [0, 1, 2, 3].map((i) => w / 2 + (i - 1.5) * sp);
        G.my = [0, 1, 2, 3].map(() => G.cy + G.R + 36 + G.mr);
        G.miniTitle = { x: w / 2, y: G.cy + G.R + 24 };
        G.cTop = G.my[0] + G.mr + 66; G.cH = 104;
      }
      G.cBot = G.cTop + G.cH;
      G.cx0 = 40; G.cx1 = w - 10;
      G.h = G.cBot + 24;
      return G;
    }

    const cv = ctx.canvas(null, {
      height: (w) => geom(w).h,
      label: 'A dial with a gold query arrow and a blue key arrow, each rotated in proportion to its position, beside four smaller dials for the four pairs of dimensions, above a curve of the dot product against the offset between the two positions.',
    });
    const readout = el('p', { class: 'pos-readout pos-rope-readout' });

    const mS = AM.ui.slider({ id: 'pos-rope-m', label: 'query position m', min: 0, max: MAXP, step: 1, value: st.m, format: String, onInput: (v) => set(v, st.n, 'm') });
    const nS = AM.ui.slider({ id: 'pos-rope-n', label: 'key position n', min: 0, max: MAXP, step: 1, value: st.n, format: String, onInput: (v) => set(st.m, v, 'n') });
    mS.el.classList.add('pos-lc'); nS.el.classList.add('pos-lc');
    const back = AM.ui.button({ id: 'pos-rope-back', label: '← Both −1', onClick: () => set(st.m - 1, st.n - 1, 'both') });
    const fwd = AM.ui.button({ id: 'pos-rope-fwd', label: 'Both +1 →', kind: 'primary', onClick: () => set(st.m + 1, st.n + 1, 'both') });
    const pairSeg = AM.ui.segmented({
      id: 'pos-rope-pair', label: 'Big dial shows pair', value: 0,
      options: [0, 1, 2, 3].map((i) => ({ value: i, label: String(i) })),
      onChange: (v) => { st.pair = v; st.dirty = true; },
    });

    function set(m, n, why) {
      m = Math.round(m); n = Math.round(n);
      if (why === 'both' && (m < 0 || n < 0 || m > MAXP || n > MAXP)) {
        // one of them is at the edge: refuse rather than silently change the offset
        st.msg = 'edge';
        updateReadout(); return;
      }
      m = clamp(m, 0, MAXP); n = clamp(n, 0, MAXP);
      const before = score(st.m, st.n), offBefore = st.m - st.n;
      st.m = m; st.n = n;
      mS.set(m); nS.set(n);
      const after = score(m, n);
      if (why === 'both') { st.msg = Math.abs(after - before) < 1e-9 ? 'kept' : 'moved'; st.flash = 1; }
      else st.msg = m - n === offBefore ? 'sameoff' : 'changed';
      st.dirty = true;
      updateReadout();
    }

    function updateReadout() {
      const s = score(st.m, st.n), off = st.m - st.n;
      const parts = [0, 1, 2, 3].map((i) => fmt(pairScore(st.m, st.n, i), 3)).join(' + ').replace(/\+ −/g, '− ');
      let line2;
      if (st.msg === 'kept') line2 = `Both moved by one: the offset is still ${off} and the score is still <b>${fmt(s, 4)}</b>.`;
      else if (st.msg === 'changed') line2 = `The offset changed to ${off}, so the score changed too.`;
      else if (st.msg === 'sameoff') line2 = 'Same offset, same score.';
      else if (st.msg === 'edge') line2 = 'One position is at the end of the range; move the other one first.';
      else line2 = 'Press “Both +1” a few times and watch the score.';
      readout.innerHTML = `q at m = ${st.m}, k at n = ${st.n} · offset m − n = <b>${off}</b><br>q·k = ${parts} = <b>${fmt(s, 4)}</b><br><span class="pos-dim">${line2}</span>`;
    }

    function draw() {
      if (!cv.w) return;
      const G = geom(cv.w), g = cv.g;
      cv.clear();
      const i = st.pair, th = theta(i);
      const qa0 = Math.atan2(Q[2 * i + 1], Q[2 * i]), ka0 = Math.atan2(K[2 * i + 1], K[2 * i]);
      const qa = qa0 + st.am * th, ka = ka0 + st.an * th;
      const qL = (G.R * 0.9 * pairNorm(Q, i)) / normMax, kL = (G.R * 0.9 * pairNorm(K, i)) / normMax;

      // ---- header
      subText(g, 'PAIR ' + i + ' · θ', String(i), 0, 11, { size: 9, color: AM.col.mist });
      DR.text(g, `= ${thLabel(th)} rad per position` + (G.wide ? ' · dashed: before rotation' : ''), DR.measure(g, 'PAIR ' + i + ' · θ', 9, 'mono') + 9, 11, { size: 9, role: 'mono', color: AM.col.mist });
      if (!G.wide) DR.text(g, 'dashed: before rotation', 0, 25, { size: 9, role: 'mono', color: AM.col.mist });

      // ---- big dial
      dialFace(g, G.cx, G.cy, G.R, { ticks: 24, cross: true, rim: AM.rgba(AM.dye.weld, 0.4), rimWidth: 1.2 });
      g.save(); g.strokeStyle = AM.rgba(AM.dye.weld, 0.07); g.lineWidth = 9;
      g.beginPath(); g.arc(G.cx, G.cy, G.R + 5, 0, TAU); g.stroke(); g.restore();
      // the wedge between the rotated query and key: its angle is all the score depends on
      const diff = Math.atan2(Math.sin(qa - ka), Math.cos(qa - ka));
      g.save();
      g.fillStyle = AM.rgba(AM.dye.weld, 0.13);
      g.beginPath(); g.moveTo(G.cx, G.cy); g.arc(G.cx, G.cy, G.R * 0.4, -ka, -(ka + diff), diff > 0); g.closePath(); g.fill();
      g.strokeStyle = AM.rgba('#fff4d6', 0.6); g.lineWidth = 1;
      g.beginPath(); g.arc(G.cx, G.cy, G.R * 0.4, -ka, -(ka + diff), diff > 0); g.stroke();
      g.restore();
      // rotation arcs: how far position has turned each vector (the part beyond whole turns)
      const rotArc = (a0, turn, rr, col) => {
        const t = turn % TAU;
        if (t < 0.01) return;
        g.save(); g.strokeStyle = AM.rgba(col, 0.55); g.lineWidth = 1.2; g.setLineDash([2, 3]);
        g.beginPath(); g.arc(G.cx, G.cy, rr, -a0, -(a0 + t), true); g.stroke(); g.restore();
      };
      rotArc(qa0, st.am * th, G.R * 0.6, AM.dye.weld);
      rotArc(ka0, st.an * th, G.R * 0.68, AM.dye.woad);
      dialArrow(g, G.cx, G.cy, qL, qa0, AM.dye.weld, { alpha: 0.32, width: 1.2, dash: [3, 3], head: 7 });
      dialArrow(g, G.cx, G.cy, kL, ka0, AM.dye.woad, { alpha: 0.32, width: 1.2, dash: [3, 3], head: 7 });
      const kt = dialArrow(g, G.cx, G.cy, kL, ka, AM.dye.woad, { width: 2.4, head: 11 });
      const qt = dialArrow(g, G.cx, G.cy, qL, qa, AM.dye.weld, { width: 2.4, head: 11 });
      DR.glowDot(g, G.cx, G.cy, 2.6, AM.col.linen);
      const lab = (pt, ang, txt, col) => {
        DR.text(g, txt, pt.x + Math.cos(ang) * 17, pt.y - Math.sin(ang) * 17 + 4, { size: 12, role: 'mono', color: col, align: 'center', weight: 500 });
      };
      lab(qt, qa, 'q', AM.dye.weld);
      lab(kt, ka, 'k', AM.dye.woad);

      // ---- the four pairs, each turning at its own speed
      DR.text(g, 'EACH PAIR ADDS ITS PART OF q·k', G.miniTitle.x, G.miniTitle.y, { size: 8.5, role: 'mono', color: AM.col.mist, align: 'center', letterSpacing: '0.08em' });
      for (let p = 0; p < NP; p++) {
        const cx = G.mx[p], cy = G.my[p], r = G.mr, tp = theta(p);
        const sel = p === st.pair;
        if (sel) { g.save(); g.fillStyle = AM.rgba(AM.dye.weld, 0.08); g.beginPath(); g.arc(cx, cy, r + 7, 0, TAU); g.fill(); g.restore(); }
        dialFace(g, cx, cy, r, { ticks: 12, rim: sel ? AM.dye.weld : AM.col.ruleStrong, rimWidth: sel ? 1.5 : 1 });
        const qa1 = Math.atan2(Q[2 * p + 1], Q[2 * p]) + st.am * tp, ka1 = Math.atan2(K[2 * p + 1], K[2 * p]) + st.an * tp;
        hand(g, cx, cy, (r * 0.86 * pairNorm(K, p)) / normMax, ka1, AM.dye.woad, { width: 1.7, bead: 1.6 });
        hand(g, cx, cy, (r * 0.86 * pairNorm(Q, p)) / normMax, qa1, AM.dye.weld, { width: 1.7, bead: 1.6 });
        const ps = pairScore(st.m, st.n, p);
        subText(g, 'θ', String(p), cx - 4, cy + r + 14, { size: 9, color: sel ? AM.dye.weld : AM.col.mist, align: 'right' });
        DR.text(g, '=' + thLabel(tp), cx - 3, cy + r + 14, { size: 9, role: 'mono', color: sel ? AM.dye.weld : AM.col.mist });
        DR.text(g, (ps >= 0 ? '+' : '') + fmt(ps, 3), cx, cy + r + 28, { size: 10.5, role: 'mono', color: AM.col.linen, align: 'center' });
      }

      // ---- score vs offset: a function of m − n only
      const xOf = (dd) => G.cx0 + ((dd + MAXP) / (2 * MAXP)) * (G.cx1 - G.cx0);
      const pad = (cMax - cMin) * 0.12;
      const yOf = (v) => G.cBot - ((v - (cMin - pad)) / (cMax - cMin + 2 * pad)) * (G.cBot - G.cTop);
      DR.text(g, 'SCORE FOR EVERY OFFSET', G.cx0, G.cTop - 14, { size: 8.5, role: 'mono', color: AM.col.mist, letterSpacing: '0.1em' });
      g.save();
      g.strokeStyle = AM.rgba(AM.col.linen, 0.1); g.lineWidth = 1;
      g.beginPath(); g.moveTo(G.cx0, Math.round(yOf(0)) + 0.5); g.lineTo(G.cx1, Math.round(yOf(0)) + 0.5); g.stroke();
      g.beginPath(); g.moveTo(Math.round(xOf(0)) + 0.5, G.cTop); g.lineTo(Math.round(xOf(0)) + 0.5, G.cBot); g.stroke();
      const fill = g.createLinearGradient(0, G.cTop, 0, G.cBot);
      fill.addColorStop(0, AM.rgba(AM.dye.lichen, 0.22)); fill.addColorStop(1, AM.rgba(AM.dye.lichen, 0));
      g.beginPath(); g.moveTo(xOf(-MAXP), G.cBot);
      curve.forEach((v, k) => g.lineTo(xOf(k - MAXP), yOf(v)));
      g.lineTo(xOf(MAXP), G.cBot); g.closePath(); g.fillStyle = fill; g.fill();
      g.lineJoin = 'round';
      g.beginPath(); curve.forEach((v, k) => (k ? g.lineTo(xOf(k - MAXP), yOf(v)) : g.moveTo(xOf(k - MAXP), yOf(v))));
      g.strokeStyle = AM.rgba(AM.dye.lichen, 0.25); g.lineWidth = 5; g.stroke();
      g.strokeStyle = AM.dye.lichen; g.lineWidth = 1.5; g.stroke();
      g.fillStyle = AM.rgba(AM.dye.lichen, 0.85);
      curve.forEach((v, k) => { g.beginPath(); g.arc(xOf(k - MAXP), yOf(v), 1.7, 0, TAU); g.fill(); });
      g.restore();
      DR.text(g, 'q·k', G.cx0 - 7, G.cTop + 4, { size: 9.5, role: 'mono', color: AM.col.mist, align: 'right' });
      DR.text(g, '0', G.cx0 - 7, yOf(0) + 3, { size: 9, role: 'mono', color: AM.col.mist, align: 'right' });
      DR.text(g, `−${MAXP}`, G.cx0, G.cBot + 15, { size: 9, role: 'mono', color: AM.col.mist });
      DR.text(g, `+${MAXP}`, G.cx1, G.cBot + 15, { size: 9, role: 'mono', color: AM.col.mist, align: 'right' });
      DR.text(g, 'offset m − n', xOf(0), G.cBot + 15, { size: 9, role: 'mono', color: AM.col.linenDim, align: 'center' });
      // the current (m, n): computed from both positions, it lands on the offset-only curve
      const ao = st.am - st.an, sNow = score(st.am, st.an);
      const mx = xOf(ao), my = yOf(sNow);
      g.save(); g.strokeStyle = AM.rgba(AM.dye.weld, 0.45); g.setLineDash([2, 3]);
      g.beginPath(); g.moveTo(mx, G.cTop); g.lineTo(mx, G.cBot); g.stroke(); g.restore();
      if (st.flash > 0.01) {
        g.save(); g.strokeStyle = AM.rgba(AM.dye.weld, 0.8 * st.flash); g.lineWidth = 1.5;
        g.beginPath(); g.arc(mx, my, 6 + 18 * (1 - st.flash), 0, TAU); g.stroke(); g.restore();
      }
      DR.glowDot(g, mx, my, 3.4, AM.dye.weld);
    }

    // a tap on a small dial shows that pair in the big dial
    cv.canvas.addEventListener('pointerdown', (e) => {
      const p = cv.pointer(e), G = geom(cv.w);
      G.mx.forEach((x, k) => { if (Math.hypot(p.x - x, p.y - G.my[k]) < G.mr + 8) { st.pair = k; pairSeg.set(k); st.dirty = true; } });
    });

    const controls = el('div', { class: 'controls pos-ctrls pos-rope-ctrls' }, mS.el, nS.el, el('div', { class: 'pos-btns' }, back, fwd), pairSeg.el);
    const fig = AM.ui.figure({
      title: 'Rotary embeddings, one dial per pair', badge: AM.ui.badge('toy', 'Exact formula'), cls: 'pos-rope-fig',
      caption: 'Toy query and key (d = 8, four pairs); the rotation is the exact RoPE formula with θ<sub>i</sub> = 10000<sup>−2i/8</sup>. The score is the raw dot product q·k, before the 1/√d scale and softmax. The purple curve is computed from the offset alone, as RoPE(q, Δ)·RoPE(k, 0); the gold bead is computed from both positions and always lands on it.',
    }, cv.wrap, controls, readout);

    cv.onResize(() => { st.dirty = true; draw(); });
    ctx.loop((t, dt) => {
      const tm = approach(st.am, st.m, dt * 7), tn = approach(st.an, st.n, dt * 7);
      const moving = Math.abs(tm - st.m) > 1e-3 || Math.abs(tn - st.n) > 1e-3;
      st.am = moving ? tm : st.m; st.an = moving ? tn : st.n;
      if (st.flash > 0) st.flash = Math.max(0, st.flash - dt * 1.4);
      if (moving || st.dirty || st.flash > 0) { draw(); st.dirty = false; }
    });
    updateReadout();
    return { el: fig };
  }

  // ======================================================================
  // 7. Figure D: ALiBi, a distance penalty with one slope per head
  // ======================================================================

  function mountAlibi(ctx) {
    const T = 16, q = T - 1; // a query at position 15 looking back over keys 0…15 (causal)
    // ALiBi with 8 heads gives head h the slope 2^(−h); we show heads 1, 3 and 5
    const heads = [{ h: 1, s: 1 / 2, label: '1/2' }, { h: 3, s: 1 / 8, label: '1/8' }, { h: 5, s: 1 / 32, label: '1/32' }];
    // equal content scores (all 0), so the weights are softmax(−s · distance) alone
    const W = heads.map((hd) => AM.math.softmax(Array.from({ length: T }, (_, j) => -hd.s * (q - j))));
    const cv = ctx.canvas(null, {
      height: (w) => (w < 420 ? 176 : 188),
      label: 'Three rows of attention weights for a query at position 15 over keys 0 to 15, one row per ALiBi slope. The steepest slope puts almost all weight on the nearest keys; the shallowest spreads it widely.',
    });

    function draw() {
      if (!cv.w) return;
      const g = cv.g, w = cv.w, h = cv.h;
      cv.clear();
      const small = w < 420;
      const lx = small ? 58 : 70, x0 = lx + 10, x1 = w - (small ? 40 : 50);
      const top = 10, rowH = (h - top - 30) / heads.length, cw = (x1 - x0) / T;
      heads.forEach((hd, r) => {
        const col = AM.headColor(r), base = top + (r + 1) * rowH - 6, hMax = rowH - 14;
        DR.text(g, 'slope', lx, base - hMax / 2 - 2, { size: 9, role: 'mono', color: AM.col.mist, align: 'right' });
        DR.text(g, hd.label, lx, base - hMax / 2 + 11, { size: 10.5, role: 'mono', color: col, align: 'right', weight: 500 });
        g.save(); g.strokeStyle = AM.rgba(AM.col.linen, 0.08); g.lineWidth = 1;
        g.beginPath(); g.moveTo(x0, base + 0.5); g.lineTo(x1, base + 0.5); g.stroke(); g.restore();
        for (let j = 0; j < T; j++) {
          const v = W[r][j] / W[r][q], x = x0 + (j + 0.5) * cw, y = base - v * hMax; // each row scaled to its largest weight
          g.save();
          g.strokeStyle = AM.rgba(col, 0.2); g.lineWidth = 4;
          g.beginPath(); g.moveTo(x, base); g.lineTo(x, y); g.stroke();
          g.strokeStyle = AM.rgba(col, 0.85); g.lineWidth = 1.4;
          g.beginPath(); g.moveTo(x, base); g.lineTo(x, y); g.stroke();
          g.restore();
          DR.glowDot(g, x, y, 1.6 + 1.4 * v, col, 0.5 + 0.5 * v);
        }
        DR.text(g, fmt(W[r][q], 2), x1 + 8, base - 2, { size: 10, role: 'mono', color: AM.col.linen });
      });
      const ay = h - 9;
      DR.text(g, 'key 0', x0, ay, { size: 9, role: 'mono', color: AM.col.mist });
      DR.text(g, 'key 15 = the query', x1, ay, { size: 9, role: 'mono', color: AM.col.mist, align: 'right' });
      if (!small) DR.text(g, '← farther', (x0 + x1) / 2, ay, { size: 9, role: 'mono', color: AM.col.linenDim, align: 'center' });
      DR.text(g, 'on 15', x1 + 8, top + 2, { size: 8.5, role: 'mono', color: AM.col.mist });
    }
    cv.onResize(draw);
    return AM.ui.figure({
      title: 'ALiBi · one penalty, three slopes', badge: AM.ui.badge('toy', 'Exact formula'), cls: 'pos-alibi-fig',
      caption: 'A query at position 15 looks back over keys 0–15. Every key gets the same content score here, so the weights are softmax(−s · distance) alone. Each row is scaled to its largest weight; the number on the right is that weight, on the query’s own position. A steep slope keeps attention local; a shallow one spreads it out. With 8 heads, ALiBi gives head h the slope 2<sup>−h</sup>; these are heads 1, 3 and 5.',
    }, cv.wrap);
  }

  // ======================================================================
  // 8. Chapter-scoped styles
  // ======================================================================

  const CSS = `
    #ch-position .pos-readout { margin: 0; font-family: var(--font-mono); font-size: 11.5px; line-height: 1.7; color: var(--linen-dim); min-height: 3.4em; font-variant-numeric: tabular-nums; }
    #ch-position .pos-readout b { color: var(--weld); font-weight: 500; }
    #ch-position .pos-readout sup, #ch-position .pos-readout sub { font-size: 0.72em; line-height: 0; }
    #ch-position .pos-readout .pos-dim { color: var(--mist); }
    #ch-position .pos-clock-readout { min-height: 3.6em; }
    #ch-position .pos-ctrls { gap: var(--space-4) var(--space-6); align-items: end; }
    #ch-position .pos-btns { display: flex; gap: var(--space-2); flex-wrap: wrap; }
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
    #ch-position .pos-notes .prose, #ch-position .pos-intro .prose { max-width: none; }
    #ch-position .pos-intro { align-items: center; }
    #ch-position .math.block { font-size: 0.82em; }
    #ch-position .step .math.block { margin-block: 2px; }
    #ch-position .pos-formula { display: grid; gap: 4px; }
    #ch-position .pos-rope { align-items: start; }
    #ch-position .pos-rope .prose { padding-top: var(--space-2); }
    #ch-position .prose > *, #ch-position .step > * { min-width: 0; }
    #ch-position .pos-lc .ctl-label { text-transform: none; letter-spacing: 0.06em; }
    #ch-position .pos-rope-ctrls { gap: var(--space-4) var(--space-5); }
    #ch-position .pos-rope-ctrls .ctl-range { width: min(220px, 100%); }
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
  // 9. The chapter
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
          P('Self-attention compares tokens through dot products of their vectors, and nothing in that arithmetic records who came first. Hand it three word vectors and it receives a set, a <span class="term">bag of words</span>. Watch the threads cross as the order flips. The bag does not change.')),
        step('02 · Equivariance', 'Shuffle in, shuffle out',
          P('Each output is a weighted mix of the value vectors, with weights from <span class="math">softmax(q·k / √d)</span>. Reorder the inputs and every weight travels with its token, so the outputs are the same vectors in the new order.'),
          P('This property is called <span class="term">permutation equivariance</span>: <span class="math">Attn(PX) = P · Attn(X)</span> for any reordering <span class="math">P</span>. Below the bag, each output sits exactly inside the outline of the same word’s output from the other sentence.'),
          P('GPT-style models also use a causal mask, so each token sees only the ones before it. That leaks a little order information, and models trained with no positions at all learn to exploit it. In practice, models are almost always given positions explicitly.')),
        step('03 · The fix', 'Stamp every seat',
          P('Give each position <span class="math">t</span> its own vector <span class="math">p<sub>t</sub></span>, the same length as the embeddings, and add it:'),
          el('span', { class: 'math block' }, 'x', el('sub', {}, 't'), ' = e(token', el('sub', {}, 't'), ') + p', el('sub', {}, 't')),
          P(`Now “dog” in seat 0 and “dog” in seat 2 are different vectors. The beads in the bag move with every reorder, the attention weights change, and the outputs part ways: dog’s output now differs by up to <strong>${r1.diff[0].toFixed(2)}</strong> between the sentences.`),
          P('The little clocks over the seats are the position vectors. Each hand is one sin/cos pair; we open the clock up just below.')),
        step('04 · Why add?', 'Adding, not appending',
          P('Why add the position instead of appending it as extra numbers? Adding keeps every vector at <span class="math">d_model</span> numbers and costs nothing.'),
          P('It works because the space is roomy. With hundreds of dimensions, token information and position information can sit in nearly separate directions, and the layers learn to read each one.'),
          P('In the bag, each dashed ring is the bare embedding and the gold arrow is the position vector added to it. The shadow is a linear projection, so it keeps the sum honest: the arrow for a seat is the same whichever word sits there.')),
      ];
      const stage = el('div', { class: 'ch-stage' }, bag.el);
      const split = el('div', { class: 'ch-split' }, stage, el('div', { class: 'ch-prose' }, steps));
      ctx.steps(steps, (i) => bag.setStep(i));

      // ---- the clockwork
      const lastPeriod = thousands(Math.round(TAU / omega(15, 32) / 1000) * 1000);
      const clockIntro = el('div', { class: 'prose' },
        el('h3', {}, 'A clock for every position'),
        P('So what should the position vectors be? The original Transformer (Vaswani et al., 2017) used a fixed recipe of sines and cosines, with nothing to learn:'),
        el('div', { class: 'math block pos-formula', html:
          '<span>PE(pos, 2i) = sin(pos · ω<sub>i</sub>)</span><span>PE(pos, 2i+1) = cos(pos · ω<sub>i</sub>)</span><span>ω<sub>i</sub> = 1 / 10000<sup>2i/d</sup></span>' }),
        P('Read each sin/cos pair as the tip of a clock hand: cos is how far it points right, sin how far up. One step forward turns hand <span class="math">i</span> by <span class="math">ω<sub>i</sub></span> radians. Hand 0 turns one radian per step and goes round every 6.3 positions. Each hand after it is slower by the same factor, down to the last, which needs about ' + lastPeriod + ' positions for one turn when <span class="math">d = 32</span>.'),
        P('Fast hands tell neighbours apart; slow hands tell distant positions apart. A clock does the same with its second, minute and hour hands, and a binary counter with its bits: the lowest flips every step, each higher bit half as often.'));
      const clock = mountClockwork(ctx);
      const notes = el('div', { class: 'grid-2 pos-notes' },
        el('div', { class: 'prose' },
          el('h3', {}, 'Nearby positions look alike'),
          P('Multiply two encodings. Each pair contributes'),
          el('span', { class: 'math block' }, 'sin a · sin b + cos a · cos b = cos(a − b)'),
          P('so the whole dot product is'),
          el('span', { class: 'math block' }, 'PE(a) · PE(b) = Σ', el('sub', {}, 'i'), ' cos((a − b) · ω', el('sub', {}, 'i'), ')'),
          P('Only the offset <span class="math">a − b</span> appears. Neighbours score high and the score fades with distance: that is the glowing band in the square map. Every diagonal of that map holds a single value.')),
        el('div', { class: 'prose' },
          el('h3', {}, 'A shift is a rotation'),
          P('Moving <span class="math">k</span> places forward turns every hand by <span class="math">k · ω<sub>i</sub></span>, wherever it started. So <span class="math">PE(pos + k)</span> is a fixed rotation of <span class="math">PE(pos)</span>, the same linear map for every <span class="math">pos</span>. The authors chose sinusoids hoping this would make relative positions easy to learn.'),
          P('The alternative is to learn the position vectors like word embeddings, one trainable row per position. GPT-2 learns 1,024 of them, and the tiny live model on this page learns its own too. The 2017 paper found the two approaches worked about equally well, though a learned table has nothing to offer past the longest position seen in training.')));
      const introGrid = el('div', { class: 'grid-2 pos-intro' }, clockIntro, mountAnatomy(ctx));
      const clockSec = el('div', { class: 'ch-wide pos-clock-sec' }, introGrid, clock.el, notes);

      // ---- RoPE
      const rope = mountRope(ctx);
      const ropeProse = el('div', { class: 'prose' },
        el('h3', {}, 'Rotate the query and key instead'),
        P('Most open models today, including Llama, Mistral and Qwen, use <span class="term">rotary position embeddings</span> (RoPE; Su et al., 2021). Nothing is added to the token vectors. Inside every attention layer, each query and key is split into pairs, and pair <span class="math">i</span> is rotated by its position times <span class="math">θ<sub>i</sub> = 10000<sup>−2i/d</sup></span>, the same clock speeds as before.'),
        P('A rotation keeps lengths. Turning the query by <span class="math">m·θ<sub>i</sub></span> and the key by <span class="math">n·θ<sub>i</sub></span> changes the angle between them by <span class="math">(m − n)·θ<sub>i</sub></span>, so the score <span class="math">q·k</span> depends on the two contents and on the offset <span class="math">m − n</span>, never on <span class="math">m</span> or <span class="math">n</span> alone. Slide both together and the score holds still.'),
        P('<span class="term">ALiBi</span> (Press et al., 2021) is simpler still. It adds no vectors at all; it subtracts a penalty proportional to distance from every attention score, <span class="math">score − s·(m − n)</span>, with a different slope <span class="math">s</span> for each head, so far-away tokens are down-weighted, some heads steeply and some gently.'),
        mountAlibi(ctx));
      const ropeGrid = el('div', { class: 'grid-2 pos-rope' }, ropeProse, rope.el);

      const callout = el('div', { class: 'callout' },
        el('span', { class: 'callout-label' }, 'Key idea'),
        P('Attention treats its input as a bag, so order has to be written into the vectors themselves. The original Transformer adds a bank of clock hands turning at geometrically spaced speeds. GPT-2 learns its position vectors. RoPE rotates queries and keys so that attention scores depend on how far apart two tokens are.'));

      root.appendChild(el('div', { class: 'ch-body' }, split, clockSec, ropeGrid, callout));
    },
  });
})();
