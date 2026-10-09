/* Chapter 04 — Self-attention: "Threads of Attention".
   The centrepiece. A sticky "loom" stage walks through q/k/v, scores, the
   causal mask, softmax, the weighted sum of values, the full attention matrix
   (woven by a shuttle) and the matrix formula. Below it, a free-play workbench,
   a small experiment on why scores are divided by √d_k, and the RoPE lab
   (position applied to queries and keys; it lives here rather than in
   chapter 3 because it needs q, k and q·k).

   The loom and workbench numbers come from the toy model defined at the top
   of this file: hand-built embeddings (d_model = 8) and hand-built W_Q, W_K,
   W_V (d_k = 4) and W_O. Every score, weight and output vector shown is
   computed from them. The RoPE lab uses its own toy q and k (d_k = 8). */
(() => {
  const ID = 'attention';
  const M = AM.math;

  // ====================================================================
  // 1. The toy model
  // ====================================================================

  // Embedding features. Seven named meaning features plus one position number
  // (a deliberate simplification of chapter 3's positional encoding).
  const FEATS = ['creature', 'thing', 'pronoun', 'action', 'determiner', 'link', 'quality', 'position'];
  const FEAT_SHORT = ['crea', 'thing', 'pron', 'act', 'det', 'link', 'qual', 'pos'];
  // Value dimensions (what a token hands over when chosen).
  const VDIMS = ['creature', 'thing', 'action', 'grammar'];

  // Hand-built token embeddings (the first 7 features; position is appended).
  // quality: + describes a creature (tired), − describes a thing (soft).
  const LEX = {
    the: [0, 0, 0, 0, 1, 0, 0],
    cat: [1, 0, 0, 0, 0, 0, 0],
    sat: [0, 0, 0, 1, 0, 0, 0],
    on: [0, 0, 0, 0, 0, 1, 0],
    mat: [0, 1, 0, 0, 0, 0, 0],
    because: [0, 0, 0, 0.2, 0, 1, 0],
    it: [0, 0, 1, 0, 0, 0, 0],
    was: [0, 0, 0, 0.8, 0, 0, 0],
    tired: [0, 0, 0, 0, 0, 0, 1],
    soft: [0, 0, 0, 0, 0, 0, -1],
  };
  const BASE = ['the', 'cat', 'sat', 'on', 'the', 'mat', 'because', 'it', 'was'];
  const sentence = (end) => BASE.concat(end);
  /** x_i = [meaning features…, position i/10] (d_model = 8). */
  const embed = (toks) => toks.map((t, i) => LEX[t].concat(Math.round(i) / 10));

  // Rows = embedding features (8), columns = the 4 query/key/value dimensions.
  // Query columns read as "what am I looking for": [creature, thing, action, determiner-or-link nearby].
  const W_Q = [
    [0, 0, 0, 1.5], // creature → looks for its determiner
    [0, 0, 0, 2.2], // thing → looks for its determiner / preposition
    [3.6, 1.6, 0, 0], // pronoun → looks for an entity, prefers a creature
    [2.0, 0, 0, 0], // action → looks for its subject
    [0, 0, 0, 0.6], // determiner → looks for nearby function words
    [0, 0, 2.0, 0], // link → looks for the action it attaches to
    [2.4, -2.4, 0, 0], // quality → creature (+) or thing (−)
    [0, 0, 0, 0], // position
  ];
  // Key columns read as "what I advertise": [creature, thing, action, function word + recency].
  const W_K = [
    [2, 0, 0, 0],
    [0, 2, 0, 0],
    [1.4, 0.2, 0, 0], // a pronoun advertises itself as a stand-in creature
    [0, 0, 2, 0],
    [0, 0, 0, 1.5],
    [0, 0, 0, 0.8],
    [0, 0, 0, 0],
    [0, 0, 0, 2], // later tokens advertise a larger "recency" key
  ];
  // Value columns: [creature, thing, action, grammar].
  const W_V = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0.2, 0.2, 0, 0.3],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 0.6],
    [0.3, -0.3, 0, 0],
    [0, 0, 0, 0],
  ];
  // Output projection: 4 value dims back into the 8 residual-stream features.
  const W_O = [
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.5, 0.5, 0, 0],
  ];
  const DK = 4;

  /**
   * One attention head, computed exactly:
   *   Q = X·W_Q, K = X·W_K, V = X·W_V
   *   S = (Q·Kᵀ) · (1/√d_k if scaled) · mult
   *   S_masked[i][j] = −∞ for j > i (causal)
   *   W = softmax over each row,  O = W·V,  X' = X + O·W_O
   */
  function attend(X, { mask = true, scale = true, mult = 1 } = {}) {
    const Q = X.map((x) => M.vecMat(x, W_Q));
    const K = X.map((x) => M.vecMat(x, W_K));
    const V = X.map((x) => M.vecMat(x, W_V));
    const n = X.length;
    const factor = (scale ? 1 / Math.sqrt(DK) : 1) * mult;
    const dots = Q.map((q) => K.map((k) => M.dot(q, k)));
    const S = dots.map((row) => row.map((d) => d * factor));
    const Sm = S.map((row, i) => row.map((s, j) => (mask && j > i ? -Infinity : s)));
    const Wt = Sm.map((row) => M.softmax(row));
    const O = Wt.map((w) => V[0].map((_, d) => { let s = 0; for (let j = 0; j < n; j++) s += w[j] * V[j][d]; return s; }));
    const dX = O.map((o) => M.vecMat(o, W_O));
    const Xn = X.map((x, i) => M.add(x, dX[i]));
    return { X, Q, K, V, dots, S, Sm, W: Wt, O, dX, Xn, n, factor, mask };
  }

  // What each hand-built query "asks", in words (interpretation of W_Q above).
  const QUESTION = {
    the: 'which function word is near me?',
    cat: 'which “the” introduced me?',
    sat: 'who is doing this?',
    on: 'which action am I attached to?',
    mat: 'which “the” introduced me?',
    because: 'which action am I explaining?',
    it: 'what do I refer to?',
    was: 'who is this about?',
    tired: 'which creature feels this?',
    soft: 'which thing has this property?',
  };

  // ====================================================================
  // 2. Small helpers
  // ====================================================================
  const now = () => performance.now() / 1000;
  const clamp = M.clamp;
  const easeOut = M.ease.out;
  const easeInOut = M.ease.inOut;
  /** Number with a true minus sign; tiny values print as 0. */
  const fmt = (x, d = 2) => {
    if (x === -Infinity) return '−∞';
    if (Math.abs(x) < 0.5 * Math.pow(10, -d)) x = 0;
    return (x < 0 ? '−' : '') + Math.abs(x).toFixed(d);
  };
  const pct = (w) => (w >= 0.995 ? '100%' : w < 0.005 ? '0%' : Math.round(w * 100) + '%');
  const argmax = (a) => M.argmax(a);
  /** Which occurrence of a repeated word token j is (1-based), or 0 if the word appears once. */
  const occurrence = (toks, j) => (toks.filter((t) => t === toks[j]).length < 2 ? 0 : toks.slice(0, j + 1).filter((t) => t === toks[j]).length);
  const ORD = ['first', 'second', 'third', 'fourth'];
  /** “mat”, or the second “the” when a word repeats. */
  const tokPhrase = (toks, j) => { const k = occurrence(toks, j); return k ? `the ${ORD[k - 1] || k + 'th'} “${toks[j]}”` : `“${toks[j]}”`; };
  /** Short label for stats: “the (2nd)”. */
  const tokShort = (toks, j) => { const k = occurrence(toks, j); return k ? `${toks[j]} (${k}${['st', 'nd', 'rd'][k - 1] || 'th'})` : toks[j]; };

  /** Cubic control points for a thread arching upward from (x1,y) to (x2,y). Self-loops when x1≈x2. */
  function arcPts(x1, y1, x2, y2, lift) {
    if (Math.abs(x2 - x1) < 0.5) {
      const r = lift;
      return [x1 - 1, y1, x1 - r * 1.15, y1 - r * 1.9, x1 + r * 1.15, y1 - r * 1.9, x2 + 1, y2];
    }
    const dx = x2 - x1;
    return [x1, y1, x1 + dx * 0.1, y1 - lift, x1 + dx * 0.9, y2 - lift, x2, y2];
  }
  function bezAt(P, t) {
    const u = 1 - t;
    return {
      x: u * u * u * P[0] + 3 * u * u * t * P[2] + 3 * u * t * t * P[4] + t * t * t * P[6],
      y: u * u * u * P[1] + 3 * u * u * t * P[3] + 3 * u * t * t * P[5] + t * t * t * P[7],
    };
  }
  /** First fraction f of a cubic (de Casteljau split). */
  function bezPart(P, f) {
    if (f >= 1) return P;
    const L = (a, b) => a + (b - a) * f;
    const x01 = L(P[0], P[2]), y01 = L(P[1], P[3]);
    const x12 = L(P[2], P[4]), y12 = L(P[3], P[5]);
    const x23 = L(P[4], P[6]), y23 = L(P[5], P[7]);
    const x012 = L(x01, x12), y012 = L(y01, y12);
    const x123 = L(x12, x23), y123 = L(y12, y23);
    return [P[0], P[1], x01, y01, x012, y012, L(x012, x123), L(y012, y123)];
  }
  /** A silk thread: soft glow, dyed core and a thin bright sheen. */
  function silk(g, P, { color = AM.dye.weld, width = 2, alpha = 1, f = 1, sheen = true } = {}) {
    if (alpha <= 0.004 || f <= 0.001) return;
    const Q = bezPart(P, f);
    const path = () => { g.beginPath(); g.moveTo(Q[0], Q[1]); g.bezierCurveTo(Q[2], Q[3], Q[4], Q[5], Q[6], Q[7]); };
    g.save();
    g.lineCap = 'round';
    path();
    g.strokeStyle = AM.rgba(color, 0.13 * alpha);
    g.lineWidth = width * 4 + 2;
    g.stroke();
    g.strokeStyle = AM.rgba(color, alpha);
    g.lineWidth = width;
    g.stroke();
    if (sheen && width > 1.6) {
      g.strokeStyle = AM.rgba('#fff4d6', 0.35 * alpha);
      g.lineWidth = Math.max(0.6, width * 0.22);
      g.stroke();
    }
    g.restore();
  }
  /** Text with an ink halo so it reads over threads. */
  function haloText(g, str, x, y, o = {}) {
    g.save();
    g.font = AM.font(o.size || 11, o.role || 'mono', o.weight, o.italic);
    g.textAlign = o.align || 'center';
    g.textBaseline = o.baseline || 'middle';
    g.globalAlpha *= o.alpha ?? 1;
    g.lineJoin = 'round';
    if (o.halo !== false) {
      g.strokeStyle = o.halo || AM.rgba(AM.col.ink, 0.9);
      g.lineWidth = 4;
      g.strokeText(str, x, y);
    }
    g.fillStyle = o.color || AM.col.linen;
    g.fillText(str, x, y);
    g.restore();
  }
  /** Tracks whether an element is near the viewport, so a long chapter only redraws what can be seen. */
  function inView(node, margin = '120px') {
    const s = { on: true };
    if (typeof IntersectionObserver !== 'undefined') {
      new IntersectionObserver((es) => { for (const e of es) s.on = e.isIntersecting; }, { rootMargin: `${margin} 0px ${margin} 0px` }).observe(node);
    }
    return s;
  }
  /** Ink or linen, whichever reads better on an opaque background colour ('rgb(…)' or '#hex'): keeps cell numbers above 4.5:1. */
  function textOn(bg) {
    const m = String(bg).match(/\d+(\.\d+)?/g);
    const rgb = String(bg).startsWith('#') ? AM.hexToRgb(bg) : (m || [0, 0, 0]).slice(0, 3).map(Number);
    const lum = (c) => { const [r, g2, b] = c.map((v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); }); return 0.2126 * r + 0.7152 * g2 + 0.0722 * b; };
    const L = lum(rgb), Li = lum(AM.hexToRgb(AM.col.ink)), Ll = lum(AM.hexToRgb(AM.col.linen));
    const cL = (Ll + 0.05) / (L + 0.05), cI = (L + 0.05) / (Li + 0.05);
    if (Math.max(cL, cI) >= 4.5) return cL >= cI ? AM.col.linen : AM.col.ink;
    // mid-tone cells: pure white or black is what clears 4.5:1
    return 1.05 / (L + 0.05) >= (L + 0.05) / 0.05 ? '#ffffff' : '#000000';
  }
  /** Text colour for a masked (−∞) cell: madder lifted toward linen, so it reads on its madder-tinted cell. */
  const MASK_TEXT = () => AM.mix(AM.dye.madder, AM.col.linen, 0.45);
  /** Left-aligned canvas text in which every “d_k” is drawn as d with a subscript k, matching the prose. */
  function textDk(g, str, x, y, o = {}) {
    const size = o.size || 14, role = o.role || 'body', sub = Math.max(8.5, size * 0.75);
    const parts = String(str).split('d_k');
    let cx = x;
    parts.forEach((part, i) => {
      if (part) { AM.draw.text(g, part, cx, y, o); cx += AM.draw.measure(g, part, size, role, o.weight); }
      if (i < parts.length - 1) {
        AM.draw.text(g, 'd', cx, y, o); cx += AM.draw.measure(g, 'd', size, role, o.weight);
        AM.draw.text(g, 'k', cx + 0.5, y + size * 0.3, { ...o, size: sub }); cx += AM.draw.measure(g, 'k', sub, role, o.weight) + 1;
      }
    });
  }
  /** Word-wrap text into lines that fit maxW. */
  function wrapLines(g, str, maxW, font) {
    g.save(); g.font = font;
    const words = str.split(' '), lines = [];
    let line = '';
    for (const w of words) {
      const t = line ? line + ' ' + w : w;
      if (g.measureText(t).width > maxW && line) { lines.push(line); line = w; } else line = t;
    }
    if (line) lines.push(line);
    g.restore();
    return lines;
  }
  /** Mini barcode for a vector (bars around a midline). */
  function bars(g, x, y, w, h, vec, { max = 1, pos = AM.dye.woad, neg = AM.dye.madder, alpha = 1, frame = true, grow = 1, hl = -1 } = {}) {
    g.save();
    g.globalAlpha *= alpha;
    if (frame) {
      AM.draw.roundRect(g, x - 2, y - 2, w + 4, h + 4, 3);
      g.fillStyle = AM.rgba(AM.col.ink, 0.75);
      g.fill();
      g.strokeStyle = AM.rgba(AM.col.linen, 0.1);
      g.lineWidth = 1;
      g.stroke();
    }
    const n = vec.length, bw = w / n, mid = y + h / 2;
    g.fillStyle = AM.rgba(AM.col.linen, 0.14);
    g.fillRect(x, mid - 0.5, w, 1);
    for (let i = 0; i < n; i++) {
      const t = clamp(vec[i] / max, -1, 1) * grow;
      const hh = Math.abs(t) * (h / 2 - 0.5);
      g.globalAlpha = alpha * (0.45 + 0.55 * Math.min(1, Math.abs(t) * 1.4));
      g.fillStyle = t >= 0 ? pos : neg;
      if (i === hl) { g.shadowColor = t >= 0 ? pos : neg; g.shadowBlur = 10; }
      g.fillRect(x + i * bw + bw * 0.14, t >= 0 ? mid - hh : mid, Math.max(1, bw * 0.72), Math.max(0.8, hh));
      g.shadowBlur = 0;
    }
    g.restore();
  }

  // ====================================================================
  // 3. Shared "loom bench" drawing (tokens in a row, threads arching above)
  // ====================================================================

  /** Lay out tokens across [x0, x1]; shrinks the font if needed. */
  function layoutTokens(g, toks, x0, x1, size, gap) {
    const slots = AM.draw.layoutRow(g, toks, x0, x1, { size, gap });
    const fs = size * (slots[0] ? slots[0].scale : 1);
    return { slots, fs };
  }

  /** Faint vertical warp threads behind every token. */
  function drawWarp(g, slots, y0, y1, clock, alpha = 1) {
    g.save();
    slots.forEach((s, j) => {
      const a = (0.06 + 0.035 * Math.sin(clock * 1.1 + j * 0.9)) * alpha;
      const grd = g.createLinearGradient(0, y0, 0, y1);
      grd.addColorStop(0, AM.rgba(AM.col.linen, 0));
      grd.addColorStop(0.35, AM.rgba(AM.col.linen, a));
      grd.addColorStop(1, AM.rgba(AM.col.linen, a * 0.6));
      g.strokeStyle = grd;
      g.lineWidth = 1;
      g.beginPath(); g.moveTo(Math.round(s.cx) + 0.5, y0); g.lineTo(Math.round(s.cx) + 0.5, y1); g.stroke();
    });
    g.restore();
  }

  /** Thread geometry from query qi to every key j. */
  function threadPaths(slots, qi, anchorY, maxLift, loopR) {
    const xq = slots[qi].cx;
    const span = Math.max(1, slots[slots.length - 1].cx - slots[0].cx);
    return slots.map((s, j) => {
      if (j === qi) return arcPts(xq, anchorY, xq, anchorY, loopR);
      // nested arcs: the farther the key, the higher the thread climbs
      const u = Math.abs(s.cx - xq) / span;
      const lift = maxLift * (0.12 + 0.88 * Math.pow(u, 0.8));
      return arcPts(xq, anchorY, s.cx, anchorY, lift);
    });
  }

  /** Masked-future curtain: hatched madder veil to the right of the query. */
  function drawCurtain(g, slots, qi, y0, y1, w, amt, label = true, phone = false) {
    if (amt <= 0.003 || qi >= slots.length - 1) return;
    const a = slots[qi], b = slots[qi + 1];
    const edge = (a.cx + a.w / 2 + b.cx - b.w / 2) / 2;
    const x0 = M.lerp(w + 4, edge, easeOut(amt));
    g.save();
    g.beginPath(); g.rect(x0, y0, w - x0, y1 - y0); g.clip();
    const grd = g.createLinearGradient(x0, 0, w, 0);
    grd.addColorStop(0, AM.rgba(AM.dye.madder, 0.13 * amt));
    grd.addColorStop(1, AM.rgba(AM.dye.madder, 0.04 * amt));
    g.fillStyle = grd;
    g.fillRect(x0, y0, w - x0, y1 - y0);
    g.strokeStyle = AM.rgba(AM.dye.madder, 0.16 * amt);
    g.lineWidth = 1;
    g.beginPath();
    for (let k = -((y1 - y0) | 0); k < w - x0 + 10; k += 9) { g.moveTo(x0 + k, y1); g.lineTo(x0 + k + (y1 - y0), y0); }
    g.stroke();
    g.restore();
    g.save();
    g.strokeStyle = AM.rgba(AM.dye.madder, 0.75 * amt);
    g.setLineDash([3, 4]);
    g.beginPath(); g.moveTo(x0 + 0.5, y0); g.lineTo(x0 + 0.5, y1); g.stroke();
    g.restore();
    if (label) {
      const lx = x0 + 6;
      AM.draw.text(g, 'FUTURE', lx, y0 + 12, { size: phone ? 8.5 : 9.5, role: 'mono', color: AM.dye.madder, alpha: amt, letterSpacing: '0.12em' });
      AM.draw.text(g, 'masked', lx, y0 + (phone ? 24 : 26), { size: phone ? 8.5 : 9.5, role: 'mono', color: AM.dye.madder, alpha: amt * 0.8 });
    }
  }

  /** Glowing value beads flowing from keys to the query along the threads. */
  function makeBeads(cap = 180) {
    const list = [];
    let acc = [];
    return {
      list,
      clear() { list.length = 0; acc = []; },
      /** rate: beads per second for a weight of 1. */
      step(dt, weights, rate, live) {
        if (acc.length !== weights.length) acc = weights.map(() => Math.random());
        if (live) {
          weights.forEach((w, j) => {
            if (w < 0.004) return;
            acc[j] += dt * rate * w;
            while (acc[j] >= 1 && list.length < cap) {
              acc[j] -= 1;
              list.push({ j, s: 0, v: 0.42 + Math.random() * 0.12, r: 0.8 + Math.sqrt(w) * 1.7, jit: Math.random() });
            }
            if (acc[j] >= 1) acc[j] = 0;
          });
        }
        let arrived = 0;
        for (let k = list.length - 1; k >= 0; k--) {
          const b = list[k];
          b.s += dt * b.v;
          if (b.s >= 1) { arrived += weights[b.j] || 0; list.splice(k, 1); }
        }
        return arrived;
      },
    };
  }
  /** Position of a bead: rise from (sx, sy0) to the anchor, then follow the thread key → query. */
  function beadPos(b, paths, slots, riseFrom, anchorY) {
    const a = 0.22;
    if (b.s < a) {
      const u = easeInOut(b.s / a);
      return { x: slots[b.j].cx, y: M.lerp(riseFrom, anchorY, u) };
    }
    const u = (b.s - a) / (1 - a);
    return bezAt(paths[b.j], 1 - u);
  }

  // ====================================================================
  // 4. The woven attention matrix
  // ====================================================================

  /** Grid geometry centred in the canvas. */
  function gridGeom(g, w, h, n, { top, left, right, bottom, pad }) {
    const cell = Math.floor(Math.min((w - left - right - 2 * pad) / n, (h - top - bottom - 6) / n));
    const gw = cell * n;
    const gx = Math.round(left + pad + ((w - left - right - 2 * pad) - gw) / 2);
    const gy = Math.round(top + ((h - top - bottom) - gw) / 2);
    return { cell, gx, gy, gw, n };
  }

  /**
   * Draw the attention matrix as cloth. Warp threads run down each key column;
   * each query row is a weft thread whose colour shows the cell value.
   * cellFn(i,j) → {color, strength, masked, text, textColor}. reveal(i,j) ∈ [0,1].
   */
  function drawWeave(g, G, cellFn, { reveal = () => 1, alpha = 1, focus = -1, hover = null, clock = 0, showText = () => false, hatchMask = 0.35 } = {}) {
    const { cell, gx, gy, n } = G;
    g.save();
    g.globalAlpha *= alpha;
    // backing cloth
    AM.draw.roundRect(g, gx - 6, gy - 6, cell * n + 12, cell * n + 12, 8);
    g.fillStyle = AM.rgba(AM.col.ink, 0.65);
    g.fill();
    g.strokeStyle = AM.rgba(AM.col.linen, 0.07);
    g.stroke();
    const warpW = Math.max(2, cell * 0.24);
    // warp threads (keys) run the full height, even through masked cells
    for (let j = 0; j < n; j++) {
      const cx = gx + j * cell + cell / 2;
      g.fillStyle = AM.mix(AM.col.ink3, AM.dye.woad, 0.22);
      g.fillRect(cx - warpW / 2, gy - 4, warpW, cell * n + 8);
      g.fillStyle = AM.rgba(AM.col.linen, 0.06);
      g.fillRect(cx - warpW / 2, gy - 4, 1, cell * n + 8);
    }
    for (let i = 0; i < n; i++) {
      const cy = gy + i * cell + cell / 2;
      // the row's continuous weft thread, as far as the shuttle has carried it
      let reach = 0;
      for (let j = 0; j < n; j++) { if (cellFn(i, j).masked) break; const r = reveal(i, j); if (r <= 0) break; reach = j + Math.min(1, r); }
      if (reach > 0) {
        g.fillStyle = AM.rgba(AM.col.linen, 0.16);
        g.fillRect(gx - 5, cy - 0.75, reach * cell + 5, 1.5);
      }
      for (let j = 0; j < n; j++) {
        const c = cellFn(i, j);
        const x = gx + j * cell, y = gy + i * cell;
        if (c.masked) {
          if (hatchMask > 0) {
            g.strokeStyle = AM.rgba(AM.dye.madder, hatchMask * 0.5);
            g.lineWidth = 1;
            g.beginPath();
            g.moveTo(x + cell * 0.2, y + cell * 0.8); g.lineTo(x + cell * 0.8, y + cell * 0.2);
            g.stroke();
          }
          continue;
        }
        const r = reveal(i, j);
        if (r <= 0.001) continue;
        const pw = (cell - 2) * (0.35 + 0.65 * r);
        const wh = cell * 0.56;
        g.globalAlpha = alpha * Math.min(1, r * 1.5);
        AM.draw.roundRect(g, x + 1, cy - wh / 2, pw, wh, Math.min(wh / 2, cell * 0.2));
        g.fillStyle = c.color;
        g.fill();
        // sheen along the weft
        g.fillStyle = AM.rgba('#fff4d6', 0.1 + 0.25 * c.strength);
        g.fillRect(x + 3, cy - wh / 2 + 2, Math.max(0, pw - 6), Math.max(1, wh * 0.12));
        // over-under: on alternate cells the warp crosses on top (plain weave)
        if ((i + j) % 2 === 1) {
          const cx = x + cell / 2;
          g.fillStyle = AM.mix(AM.col.ink3, AM.dye.woad, 0.3);
          g.globalAlpha = alpha * 0.85 * Math.min(1, r * 1.5);
          g.fillRect(cx - warpW / 2, cy - wh / 2 - 1, warpW, wh + 2);
        }
        g.globalAlpha = alpha;
        if (c.strength > 0.22) AM.draw.glowDot(g, x + cell / 2, cy, cell * 0.13 * c.strength, c.glow || AM.dye.weld, 0.45 * c.strength * r);
      }
    }
    g.globalAlpha = alpha;
    // focus row outline
    if (focus >= 0) {
      AM.draw.roundRect(g, gx - 3, gy + focus * cell - 1, cell * n + 6, cell + 2, 5);
      g.strokeStyle = AM.rgba(AM.dye.weld, 0.85);
      g.lineWidth = 1.5;
      g.stroke();
    }
    if (hover && hover.i >= 0) {
      g.strokeStyle = AM.col.linen;
      g.lineWidth = 1.5;
      g.strokeRect(gx + hover.j * cell + 0.5, gy + hover.i * cell + 0.5, cell - 1, cell - 1);
    }
    // numbers
    const fsz = Math.max(8, Math.min(10.5, cell * 0.24));
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      if (!showText(i, j)) continue;
      const c = cellFn(i, j);
      if (!c.text || reveal(i, j) < 0.6) continue;
      const dark = c.textColor === AM.col.ink;
      haloText(g, c.text, gx + j * cell + cell / 2, gy + i * cell + cell / 2 + 0.5, { size: c.role === 'body' ? fsz * 1.25 : fsz, role: c.role || 'mono', weight: dark || c.role === 'body' ? 600 : 400, color: c.textColor || AM.col.linen, halo: dark ? false : undefined });
    }
    g.restore();
  }

  /** Row (query) and column (key) labels for the weave. */
  function drawGridLabels(g, G, toks, { rowColor, colColor, alpha = 1, phone = false, focus = -1 } = {}) {
    const { cell, gx, gy, n } = G;
    const fs = phone ? 10 : 12;
    g.save();
    g.globalAlpha *= alpha;
    for (let i = 0; i < n; i++) {
      const c = rowColor ? rowColor(i) : (i === focus ? AM.dye.weld : AM.col.linenDim);
      AM.draw.text(g, toks[i], gx - 9, gy + i * cell + cell / 2 + 0.5, { size: fs, weight: i === focus ? 700 : 500, color: c, align: 'right', baseline: 'middle' });
    }
    for (let j = 0; j < n; j++) {
      const c = colColor ? colColor(j) : AM.col.linenDim;
      g.save();
      g.translate(gx + j * cell + cell / 2 + 3, gy + n * cell + 10);
      g.rotate(-Math.PI / 4);
      AM.draw.text(g, toks[j], 0, 0, { size: fs, color: c, align: 'right', baseline: 'middle' });
      g.restore();
    }
    g.restore();
  }

  // ====================================================================
  // 5. Chapter
  // ====================================================================
  AM.chapter({
    id: ID,
    num: 4,
    kicker: 'Self-attention',
    title: 'Threads of <em>Attention</em>',
    lede: 'Every token asks a question, every token advertises a label, and information flows along the threads where the two match. This is the only place where tokens talk to each other.',
    where: 'attn',
    // exposed for tools/verification scripts (not used by other chapters)
    toy: { FEATS, LEX, W_Q, W_K, W_V, W_O, DK, embed, sentence, attend },

    mount(root, ctx) {
      ctx.header();
      const el = ctx.el;
      const ui = AM.ui;
      const D = AM.draw;

      AM.css(CSS);

      // ---- the story sentence and its exact attention
      const TOKS = sentence('tired');
      const A = attend(embed(TOKS));
      const QI = 7; // "it"
      const n = TOKS.length;
      const wIt = A.W[QI];
      const best = argmax(wIt);

      // ---- DOM skeleton
      const body = el('div', { class: 'ch-body' });
      root.appendChild(body);

      body.appendChild(el('div', { class: 'prose att-intro' },
        el('p', { html: 'Up to now every token has travelled alone. Its vector knows which word it is and where it sits, nothing more: the vector for <strong>“it”</strong> has no idea there is a cat in the sentence.' }),
        el('p', { html: '<span class="term">Self-attention</span> is the step where tokens read from one another. Chapter 3 watched it from the outside, as a bag that ignores order. This chapter opens the bag.' }),
      ));

      // ------------------------------------------------ scrollytelling split
      const split = el('div', { class: 'ch-split att-split' });
      body.appendChild(split);
      const stage = el('div', { class: 'ch-stage' });
      const prose = el('div', { class: 'ch-prose' });
      split.append(stage, prose);

      const stageFig = ui.figure({
        title: 'The loom · one head',
        badge: 'toy',
        cls: 'att-stagefig',
      });
      stage.appendChild(stageFig);
      const cv = ctx.canvas(stageFig, {
        label: 'Attention on the sentence “the cat sat on the mat because it was tired”. Tokens sit in a row like warp threads; the query “it” draws weighted threads to earlier tokens, strongest to “cat”.',
        height: (w) => (w < 520
          ? Math.round(Math.min(Math.max(360, w * 1.06), 420, Math.max(330, window.innerHeight * 0.5)))
          : Math.round(Math.min(Math.max(500, w * 0.9), 600, Math.max(460, window.innerHeight * 0.7)))),
      });
      stageFig.appendChild(el('figcaption', { class: 'att-stage-cap', html: 'Toy numbers: hand-built 8-number embeddings and weight matrices with d<sub>k</sub> = 4. Every score, weight and vector drawn here is computed from them.' }));

      const s = (k) => `<span class="math">${k}</span>`;
      const STEPS = [
        {
          label: '1 · Three vectors from one',
          html: [
            `Each token arrives as a vector <b>x</b>; here it has d<sub>model</sub> = 8 numbers. An attention <span class="term">head</span> multiplies it by three matrices and gets three shorter vectors of d<sub>k</sub> = 4 numbers:`,
            `<span class="dye-weld">query</span> ${s('q = x·W<sub>Q</sub>')}, <span class="dye-woad">key</span> ${s('k = x·W<sub>K</sub>')} and <span class="dye-verdigris">value</span> ${s('v = x·W<sub>V</sub>')}.`,
            `The query is the question a token asks. The key is the label it advertises to everyone else. The value is the content it hands over if it gets chosen.`,
            `In a real model the three matrices are learned in training, and x is normalised before it goes in (chapter 7). Here they are written by hand so the patterns are easy to read.`,
          ],
        },
        {
          label: '2 · Asking around',
          html: [
            `Take <strong>“it”</strong>. Its query roughly asks <em>“what do I refer to?”</em> To see who answers, we take the dot product of that query with every key, then divide by ${s('√d<sub>k</sub> = 2')}:`,
            `<span class="math block">score(i, j) = q<sub>i</sub> · k<sub>j</sub> / √d<sub>k</sub></span>`,
            `A large score means the key matches the question. “cat” scores highest, <strong>${fmt(A.S[QI][best])}</strong>. Dividing keeps scores from growing as the vectors get more dimensions. There is more on that below.`,
          ],
        },
        {
          label: '3 · No peeking',
          html: [
            `This is a decoder, the kind of model that writes text left to right. While it reads “it”, the words “was tired” have not been written yet.`,
            `So a <span class="term">causal mask</span> sets every score with j &gt; i to ${s('−∞')}. Each token may look at itself and at everything before it, never ahead. In training the whole sentence is there at once, and the mask is what stops a position from peeking at the words it is learning to predict.`,
          ],
        },
        {
          label: '4 · Softmax',
          html: [
            `Softmax turns the surviving scores into <span class="term">attention weights</span>:`,
            `<span class="math block">w<sub>ij</sub> = e<sup>s<sub>ij</sub></sup> / Σ<sub>j′</sub> e<sup>s<sub>ij′</sub></sup></span>`,
            `Masked keys get exactly 0, because e<sup>−∞</sup> = 0. Every other weight is positive, and the row sums to 1. “it” gives <strong>${pct(wIt[best])}</strong> of its attention to “cat”. Thread thickness is the weight.`,
          ],
        },
        {
          label: '5 · Values travel the threads',
          html: [
            `Now each earlier token hands over its value, scaled by its weight. The output for “it” is the weighted sum`,
            `<span class="math block">o<sub>i</sub> = Σ<sub>j</sub> w<sub>ij</sub> v<sub>j</sub></span>`,
            `That output goes through one more matrix, W<sub>O</sub>, and is <em>added</em> to the token’s own vector, the <span class="term">residual stream</span> (chapter 7). The “creature” feature of “it” goes from <strong>${fmt(A.X[QI][0])}</strong> to <strong>${fmt(A.Xn[QI][0])}</strong>: it now carries some cat.`,
          ],
        },
        {
          label: '6 · Every token at once',
          html: [
            `Every token does this at the same time, each with its own query. Stack the weight rows and you get the <span class="term">attention matrix</span>: rows are queries, columns are keys.`,
            `${rowStory(A, TOKS)} The mask leaves the upper triangle unwoven: bare warp, no weft.`,
          ],
        },
        {
          label: '7 · One line of algebra',
          html: [
            `Stack the queries, keys and values as rows of matrices Q, K and V. The whole head is then one line:`,
            `<span class="math block"><span class="att-nw">Attention(Q, K, V) =</span> <span class="att-nw">softmax(QK<sup>T</sup>/√d<sub>k</sub> + M)·V</span></span>`,
            `M is the mask: 0 on and below the diagonal, −∞ above it. The stage now walks through the formula one piece at a time.`,
          ],
        },
      ];
      const stepEls = STEPS.map((st) => el('div', { class: 'step' },
        el('div', { class: 'step-label' }, st.label),
        st.html.map((h) => el('p', { html: h }))));
      prose.append(...stepEls);

      // ------------------------------------------------ stage animation state
      const st = {
        step: 0,
        tEnter: now(),
        tPeel: now(),
        tBeam: -99, tThread: -99, tFlow: -99, tWeave: -99, tFormula: now(),
        tw: { score: 0, mask: 0, thread: 0, flow: 0, loom: 0, formula: 0 },
        shelf: [1, 0, 0, 0, 0],
        flash: 0,
        lastClock: now(),
      };
      const beads = makeBeads(170);

      ctx.steps(stepEls, (i) => {
        const t = now();
        const prev = st.step;
        st.step = i;
        st.tEnter = t;
        if (i === 0) st.tPeel = t;
        if (i === 1) st.tBeam = t;
        if (i === 3 && prev < 3) st.tThread = t;
        if (i === 4) { st.tFlow = t; beads.clear(); }
        if (i === 5 && prev < 5) st.tWeave = t;
        if (i === 6) st.tFormula = t;
        if (i >= 5 && prev < 0) st.tWeave = t - 10;
      });
      ctx.onVisible(() => { if (st.step === 0) st.tPeel = now(); });

      // value scales for barcodes
      const maxAbs = (Mx) => Math.max(1e-6, ...Mx.flat().map(Math.abs));
      const qMax = maxAbs(A.Q), kMax = maxAbs(A.K), vMax = maxAbs(A.V);
      const rawMax = maxAbs(A.dots);

      function age(t0) { return AM.reducedMotion ? 99 : now() - t0; }

      // ------------------------------------------------ stage drawing
      function rowLayout(w, h) {
        const g = cv.g;
        const phone = w < 520;
        const pad = phone ? 2 : 10;
        const labW = phone ? 14 : 26;
        const { slots, fs } = layoutTokens(g, TOKS, pad + labW, w - pad, phone ? 11.5 : 15, phone ? 3 : 8);
        const tokH = Math.round(fs + (phone ? 11 : 15));
        const topH = phone ? 40 : 52;
        const shelfH = phone ? 70 : 88;
        const strip = phone ? 13 : 18, sgap = phone ? 4 : 6;
        const shelfY = h - shelfH;
        const stripY0 = shelfY - (phone ? 10 : 14) - (3 * strip + 2 * sgap);
        const rowY = stripY0 - (phone ? 10 : 14) - tokH / 2;
        const anchorY = rowY - tokH / 2 - (phone ? 19 : 25);
        const labelY = rowY - tokH / 2 - (phone ? 9 : 12);
        const minW = Math.min(...slots.map((sl) => sl.w));
        const sw = Math.max(14, Math.min(minW * 0.86, phone ? 24 : 40));
        const arcTop = topH + 8;
        const maxLift = (anchorY - arcTop - 6) / 0.75;
        return { phone, pad, labW, slots, fs, tokH, topH, shelfH, shelfY, strip, sgap, stripY0, rowY, anchorY, labelY, sw, arcTop, maxLift, w, h };
      }

      function drawStage() {
        const { g, w, h } = cv;
        if (!w) return;
        const clock = now();
        const dt = Math.min(0.1, clock - st.lastClock);
        st.lastClock = clock;
        const k = 1 - Math.exp(-dt * (AM.reducedMotion ? 60 : 5.5));
        const step = st.step;
        const tgt = {
          score: step === 1 || step === 2 ? 1 : 0,
          mask: step >= 2 ? 1 : 0,
          thread: step === 3 || step === 4 ? 1 : 0,
          flow: step === 4 ? 1 : 0,
          loom: step >= 5 ? 1 : 0,
          formula: step === 6 ? 1 : 0,
        };
        for (const key in tgt) st.tw[key] += (tgt[key] - st.tw[key]) * k;
        for (let i = 0; i < 5; i++) st.shelf[i] += ((step === i ? 1 : 0) - st.shelf[i]) * k;

        cv.clear();
        const R = rowLayout(w, h);
        const loomE = easeInOut(clamp(st.tw.loom));
        const rowA = 1 - loomE;

        // background weave texture, very faint
        D.weave(g, 0, 0, w, h, { step: 7, alpha: 0.022 });

        if (rowA > 0.01) drawRowScene(g, R, rowA, clock, dt);
        if (loomE > 0.01) drawLoomScene(g, w, h, loomE, clock, R);
      }

      function drawRowScene(g, R, alpha, clock, dt) {
        const { slots, phone, w } = R;
        const tw = st.tw;
        g.save();
        g.globalAlpha = alpha;

        drawWarp(g, slots, R.arcTop, R.shelfY - 4, clock, 1);

        // --- causal curtain (behind tokens)
        drawCurtain(g, slots, QI, R.arcTop - 2, R.stripY0 + 3 * R.strip + 2 * R.sgap + 4, w, tw.mask, true, phone);

        const paths = threadPaths(slots, QI, R.anchorY, R.maxLift, phone ? 9 : 12);

        // --- beam sweep (step 2)
        const beamAge = age(st.tBeam);
        const sweep = clamp((beamAge - 0.25) / 1.5);
        const beamIdx = sweep * (TOKS.length - 1);
        if (st.step === 1 && sweep > 0 && beamAge < 2.3) {
          const fade = 1 - clamp((beamAge - 1.75) / 0.5);
          const jx = M.lerp(slots[0].cx, slots[slots.length - 1].cx, sweep);
          const xq = slots[QI].cx;
          const grd = g.createLinearGradient(xq, R.anchorY, jx, R.anchorY);
          grd.addColorStop(0, AM.rgba(AM.dye.weld, 0.32 * fade));
          grd.addColorStop(1, AM.rgba(AM.dye.weld, 0.06 * fade));
          g.fillStyle = grd;
          g.beginPath();
          g.moveTo(xq, R.rowY - R.tokH / 2);
          g.lineTo(jx - 12, R.labelY - 8);
          g.lineTo(jx + 12, R.labelY - 8);
          g.closePath();
          g.fill();
          D.glowDot(g, jx, R.labelY - 8, 3, AM.dye.weld, fade);
        }

        // --- threads (steps 4–5)
        const W = A.W[QI];
        if (tw.thread > 0.01) {
          const grow = (j) => (AM.reducedMotion ? 1 : easeOut(clamp((age(st.tThread) - 0.1 - Math.abs(j - QI) * 0.05) / 0.9)));
          const order = W.map((x, j) => j).sort((a, b) => W[a] - W[b]);
          for (const j of order) {
            if (W[j] < 0.002) continue;
            silk(g, paths[j], {
              color: AM.dye.weld,
              width: (phone ? 0.7 : 0.9) + W[j] * (phone ? 7 : 11),
              alpha: tw.thread * (0.22 + 0.78 * Math.sqrt(W[j])),
              f: grow(j),
            });
          }
        }

        // --- value beads (step 5)
        if (tw.flow > 0.01 || beads.list.length) {
          const riseFrom = R.stripY0 + 2 * (R.strip + R.sgap) + R.strip / 2;
          const arrived = beads.step(dt, W, 16, st.step === 4 && !AM.reducedMotion);
          st.flash = Math.min(1.4, st.flash * Math.exp(-dt * 3) + arrived * 1.2);
          g.save();
          g.globalCompositeOperation = 'lighter';
          for (const b of beads.list) {
            const p = beadPos(b, paths, slots, riseFrom, R.anchorY);
            D.glowDot(g, p.x, p.y, b.r * (phone ? 1.2 : 1.5), AM.dye.verdigris, Math.min(1, tw.flow * 1.4));
          }
          g.restore();
          if (AM.reducedMotion && st.step === 4) {
            // a still frame: one bead resting mid-thread on each strong thread
            W.forEach((wj, j) => { if (wj > 0.05 && j !== QI) { const p = bezAt(paths[j], 0.5); D.glowDot(g, p.x, p.y, 1 + wj * 3, AM.dye.verdigris, 1); } });
          }
        }

        // --- labels above tokens: scores (steps 2–3) and weights (steps 4–5)
        const lblSize = phone ? 9 : 10.5;
        const colBase = R.labelY - (phone ? 2 : 4);
        const colMaxH = colBase - R.arcTop - (phone ? 34 : 50);
        const sMaxIt = Math.max(1e-6, ...A.S[QI]);
        slots.forEach((sl, j) => {
          const masked = j > QI;
          const popped = st.step > 1 ? 1 : st.step === 1 ? M.smoothstep(j - 0.4, j + 0.1, beamIdx) : 0;
          if (tw.score > 0.01 && popped > 0) {
            // a glass column whose height is the score; masked columns collapse to −∞
            const val = A.S[QI][j];
            const pop = easeOut(popped);
            const mk = masked ? easeOut(clamp(tw.mask)) : 0;
            const hgt = Math.max(0, val) / sMaxIt * colMaxH * pop * (1 - mk);
            const cw = Math.min(sl.w * 0.5, phone ? 14 : 22);
            if (hgt > 0.5) {
              const grd = g.createLinearGradient(0, colBase - hgt, 0, colBase);
              grd.addColorStop(0, AM.rgba(AM.dye.weld, 0.55 * tw.score));
              grd.addColorStop(1, AM.rgba(AM.dye.weld, 0.04 * tw.score));
              g.fillStyle = grd;
              g.fillRect(sl.cx - cw / 2, colBase - hgt, cw, hgt);
              g.fillStyle = AM.rgba('#fff4d6', 0.85 * tw.score);
              g.fillRect(sl.cx - cw / 2, colBase - hgt, cw, 1.5);
            }
            const isMasked = masked && tw.mask > 0.5;
            const txt = isMasked ? '−∞' : fmt(val, phone ? 1 : 2);
            const c = isMasked ? AM.dye.madder : (val > 1.5 ? AM.dye.weld : val < 0 ? AM.mix(AM.col.mist, AM.dye.madder, 0.5) : AM.col.linenDim);
            const arrowLen = phone ? 24 : 32;
            const ly = colBase - hgt - (phone ? 7 : 9) - (1 - pop) * 6 - arrowLen * mk;
            if (hgt > 0.5 && val > 0) D.glowDot(g, sl.cx, colBase - hgt, Math.min(4, 1.5 + val), AM.dye.weld, 0.6 * tw.score * pop);
            haloText(g, txt, sl.cx, ly, { size: lblSize, color: c, alpha: tw.score * pop, weight: val > 1.5 ? 600 : 400 });
            if (mk > 0.02) {
              D.arrow(g, sl.cx, colBase - arrowLen, sl.cx, colBase - 2, { color: AM.dye.madder, width: 1.2, head: 5, alpha: tw.score * mk });
            }
          }
          const tWeight = tw.thread;
          if (tWeight > 0.01) {
            const wj = W[j];
            const txt = masked ? '0' : pct(wj);
            const c = masked ? AM.rgba(AM.dye.madder, 0.8) : wj > 0.25 ? AM.dye.weld : AM.col.linenDim;
            haloText(g, txt, sl.cx, R.labelY, { size: lblSize, color: c, alpha: tWeight, weight: wj > 0.25 ? 600 : 400 });
          }
        });

        // --- tokens
        const flashA = clamp(st.flash);
        slots.forEach((sl, j) => {
          const masked = j > QI;
          const dim = masked ? 1 - 0.62 * tw.mask : 1;
          const isQ = j === QI;
          const hot = st.step === 1 && Math.abs(beamIdx - j) < 0.5 && clamp((age(st.tBeam) - 0.25) / 1.5) < 1;
          if (isQ && flashA > 0.02) D.glowDot(g, sl.cx, R.rowY, R.tokH * 0.5, AM.dye.verdigris, 0.35 * flashA);
          D.token(g, TOKS[j], sl.cx, R.rowY, {
            size: R.fs, w: sl.w, h: R.tokH, alpha: dim, selected: isQ,
            stroke: hot ? AM.dye.woad : undefined,
            underline: isQ ? AM.dye.weld : null,
            padX: 2,
          });
        });

        // --- step 1: the vector of "it" passes through a prism and splits into q, k, v
        const prismA = M.smoothstep(0.45, 1, st.shelf[0]);
        if (prismA > 0.01) drawPrism(g, R, prismA, clock);

        // --- q / k / v strips under each token
        drawStrips(g, R, clock);

        g.restore();

        // --- top band and bottom shelf
        drawTopBand(g, R, alpha);
        drawShelf(g, R, alpha, clock);
      }

      function drawStrips(g, R, clock) {
        const { slots, strip, sgap, sw, phone } = R;
        const kinds = [
          { M: A.Q, max: qMax, color: AM.dye.weld, name: 'q' },
          { M: A.K, max: kMax, color: AM.dye.woad, name: 'k' },
          { M: A.V, max: vMax, color: AM.dye.verdigris, name: 'v' },
        ];
        const pAge = age(st.tPeel);
        const step = st.step;
        kinds.forEach((kd, r) => {
          const y = R.stripY0 + r * (strip + sgap);
          // emphasis per step: queries+keys during scoring, values during flow
          let emph = 1;
          if (step === 1 || step === 2) emph = r === 2 ? 0.3 : 1;
          if (step === 3) emph = r === 2 ? 0.35 : 0.55;
          if (step === 4) emph = r === 2 ? 1 : 0.3;
          AM.draw.text(g, kd.name, R.pad + 2, y + strip / 2 + 0.5, { size: phone ? 10 : 12, role: 'mono', weight: 500, color: kd.color, baseline: 'middle', alpha: 0.5 + 0.5 * emph });
          slots.forEach((sl, j) => {
            const delay = j * 0.06 + r * 0.14;
            const p = easeOut(clamp((pAge - delay) / 0.7));
            const yy = M.lerp(R.rowY, y, p);
            let a = p * emph;
            if (j > QI) a *= 1 - 0.6 * st.tw.mask;
            const isQ = j === QI;
            if (step >= 1 && step <= 2 && r === 0 && !isQ) a *= 0.35; // only the query's q matters here
            // light ray while peeling
            if (p < 1 && p > 0) {
              g.save();
              g.strokeStyle = AM.rgba(kd.color, 0.5 * (1 - p));
              g.lineWidth = 1.5;
              g.beginPath(); g.moveTo(sl.cx, R.rowY + R.tokH / 2); g.lineTo(sl.cx, yy); g.stroke();
              g.restore();
            }
            bars(g, sl.cx - sw / 2, yy, sw, strip, kd.M[j], { max: kd.max, pos: kd.color, neg: AM.dye.madder, alpha: a });
            if (isQ && r === 0 && step >= 1 && step <= 3) {
              g.save();
              g.strokeStyle = AM.rgba(AM.dye.weld, 0.8 * a);
              g.lineWidth = 1;
              AM.draw.roundRect(g, sl.cx - sw / 2 - 3.5, yy - 3.5, sw + 7, strip + 7, 4);
              g.stroke();
              g.restore();
            }
          });
        });
      }

      function drawTopBand(g, R, alpha) {
        const { phone, w } = R;
        const y = phone ? 16 : 20;
        // hand over rather than cross-fade, so the two labels never overlap
        const a0 = M.smoothstep(0.5, 1, st.shelf[0]);
        const aQ = M.smoothstep(0.5, 1, 1 - st.shelf[0]);
        g.save();
        g.globalAlpha = alpha;
        if (a0 > 0.01) {
          // legend: three dyes
          const items = [['query', AM.dye.weld], ['key', AM.dye.woad], ['value', AM.dye.verdigris]];
          let x = R.pad + 2;
          AM.draw.text(g, 'EACH TOKEN', x, y, { size: phone ? 8.5 : 9.5, role: 'mono', color: AM.col.mist, baseline: 'middle', alpha: a0, letterSpacing: '0.12em' });
          x += D.measure(g, 'EACH TOKEN', phone ? 8.5 : 9.5, 'mono') + (phone ? 18 : 24);
          for (const [lab, c] of items) {
            D.glowDot(g, x, y, phone ? 2.4 : 3, c, a0);
            AM.draw.text(g, lab, x + 9, y + 0.5, { size: phone ? 11 : 13, weight: 600, color: c, baseline: 'middle', alpha: a0 });
            x += D.measure(g, lab, phone ? 11 : 13, 'body', 600) + (phone ? 22 : 30);
          }
        }
        if (aQ > 0.01) {
          AM.draw.text(g, 'QUERY', R.pad + 2, y, { size: phone ? 8.5 : 9.5, role: 'mono', color: AM.col.mist, baseline: 'middle', alpha: aQ, letterSpacing: '0.14em' });
          const qx = R.pad + 2 + D.measure(g, 'QUERY', phone ? 8.5 : 9.5, 'mono') + 12;
          AM.draw.text(g, 'it', qx, y + 1, { size: phone ? 19 : 24, role: 'display', italic: true, weight: 500, color: AM.dye.weld, baseline: 'middle', alpha: aQ });
          const qx2 = qx + D.measure(g, 'it', phone ? 19 : 24, 'display') + 12;
          AM.draw.text(g, '“' + QUESTION.it + '”', qx2, y + 1, { size: phone ? 12 : 14.5, italic: true, weight: 400, color: AM.col.linenDim, baseline: 'middle', alpha: aQ, maxWidth: w - qx2 - 4 });
        }
        g.restore();
      }

      function drawShelf(g, R, alpha, clock) {
        const { phone, w, shelfY, shelfH, pad } = R;
        g.save();
        g.globalAlpha = alpha;
        // a faint selvedge rule
        g.strokeStyle = AM.rgba(AM.col.linen, 0.09);
        g.setLineDash([2, 4]);
        g.beginPath(); g.moveTo(pad, shelfY + 0.5); g.lineTo(w - pad, shelfY + 0.5); g.stroke();
        g.setLineDash([]);
        g.restore();
        const x0 = pad + 2, x1 = w - pad - 2, y0 = shelfY + 8, y1 = shelfY + shelfH - 4;
        const S = st.shelf.map((v) => M.smoothstep(0.5, 1, v));
        if (S[0] > 0.01) shelfPrism(g, R, x0, x1, y0, y1, S[0] * alpha);
        if (S[1] > 0.01) shelfDot(g, R, x0, x1, y0, y1, S[1] * alpha);
        if (S[2] > 0.01) shelfMask(g, R, x0, x1, y0, y1, S[2] * alpha);
        if (S[3] > 0.01) shelfSoftmax(g, R, x0, x1, y0, y1, S[3] * alpha);
        if (S[4] > 0.01) shelfOutput(g, R, x0, x1, y0, y1, S[4] * alpha);
      }

      const micro = (g, str, x, y, a, o = {}) => AM.draw.text(g, str, x, y, { size: o.size || 9, role: 'mono', color: o.color || AM.col.mist, alpha: a, baseline: 'middle', align: o.align || 'left', letterSpacing: '0.1em' });

      // Step 1 (stage): the vector of "it" rises as white light, enters a prism and splits
      // into three dyed beams that land as q, k and v.
      function drawPrism(g, R, a, clock) {
        const { phone, slots } = R;
        const qx = slots[QI].cx;
        const top = R.arcTop + (phone ? 4 : 10), bot = R.labelY - (phone ? 10 : 14);
        const sz = phone ? 19 : 30;
        const py = top + (bot - top) * 0.42;
        const apex = { x: qx, y: py - sz }, bl = { x: qx - sz * 1.08, y: py + sz * 0.8 }, br = { x: qx + sz * 1.08, y: py + sz * 0.8 };
        const pAge = age(st.tPeel);
        const grow = easeOut(clamp((pAge - 0.15) / 0.8));
        const grow2 = easeOut(clamp((pAge - 0.8) / 1.0));
        g.save();
        g.globalAlpha *= a;
        g.lineCap = 'round';
        // incoming white light (the embedding x of "it")
        const by0 = R.rowY - R.tokH / 2 - 3, by1 = bl.y;
        const byTop = M.lerp(by0, by1, grow);
        g.strokeStyle = AM.rgba(AM.col.linen, 0.12); g.lineWidth = 9;
        g.beginPath(); g.moveTo(qx, by0); g.lineTo(qx, byTop); g.stroke();
        g.strokeStyle = AM.rgba('#fff8e6', 0.9); g.lineWidth = 2;
        g.beginPath(); g.moveTo(qx, by0); g.lineTo(qx, byTop); g.stroke();
        if (grow >= 1) { const ph = (clock * 0.7) % 1; D.glowDot(g, qx, M.lerp(by0, by1, ph), 2, AM.col.linen, Math.sin(ph * Math.PI)); }
        // x barcode riding beside the beam
        const xbW = phone ? 50 : 78, xbH = phone ? 15 : 22;
        const xbx = qx + (phone ? 9 : 14), xby = (by0 + by1) / 2 - xbH / 2 + (phone ? 4 : 6);
        micro(g, phone ? 'x(it)' : 'x(it) · 8 numbers', xbx, xby - (phone ? 8 : 10), grow, { size: phone ? 7.5 : 9, color: AM.col.linenDim });
        bars(g, xbx, xby, xbW, xbH, A.X[QI], { max: 1, pos: AM.col.linen, alpha: grow });
        // three beams fan out of the prism's left face
        const kinds = [
          ['query', 'q', 'W_Q', A.Q[QI], qMax, AM.dye.weld],
          ['key', 'k', 'W_K', A.K[QI], kMax, AM.dye.woad],
          ['value', 'v', 'W_V', A.V[QI], vMax, AM.dye.verdigris],
        ];
        const bx = R.pad + (phone ? 4 : 8);
        const bw = phone ? 44 : 72, bh = phone ? 15 : 22;
        const numSize = phone ? 8.5 : 10.5;
        kinds.forEach(([word, sym, wname, vec, mx, c], r) => {
          const ty = top + (bot - top) * (0.06 + r * 0.33) + (phone ? 8 : 10);
          const ex = M.lerp(apex.x, bl.x, 0.36 + r * 0.2), ey = M.lerp(apex.y, bl.y, 0.36 + r * 0.2);
          const nums = `[${vec.map((v) => fmt(v, 1)).join(', ')}]`;
          const tw = D.measure(g, nums, numSize, 'mono');
          const endX = bx + bw + 8 + tw + 8, endY = ty + bh / 2;
          // dispersion inside the glass
          g.strokeStyle = AM.rgba(c, 0.55 * grow);
          g.lineWidth = 1.2;
          g.beginPath(); g.moveTo(qx, by1 - 1); g.lineTo(ex, ey); g.stroke();
          const P = [ex, ey, M.lerp(ex, endX, 0.45), ey, M.lerp(ex, endX, 0.6), endY, endX, endY];
          silk(g, P, { color: c, width: phone ? 1.8 : 2.4, alpha: 0.9, f: grow2 });
          if (grow2 >= 1) { const tt = (clock * 0.42 + r * 0.31) % 1; const p = bezAt(P, tt); D.glowDot(g, p.x, p.y, 2, c, Math.sin(tt * Math.PI)); }
          const land = clamp((grow2 - 0.7) / 0.3);
          AM.draw.text(g, phone ? `${sym} = x·${wname.replace('_', '')}` : `${word}  ${sym} = x·${wname.replace('_', '')}`, bx, ty - (phone ? 7 : 9), { size: phone ? 9 : 11, role: 'mono', color: c, alpha: land, baseline: 'middle' });
          bars(g, bx, ty, bw, bh, vec, { max: mx, pos: c, alpha: land });
          AM.draw.text(g, nums, bx + bw + 8, endY + 0.5, { size: numSize, role: 'mono', color: AM.col.linenDim, alpha: land, baseline: 'middle' });
        });
        // the glass prism
        g.beginPath(); g.moveTo(apex.x, apex.y); g.lineTo(br.x, br.y); g.lineTo(bl.x, bl.y); g.closePath();
        const grd = g.createLinearGradient(bl.x, apex.y, br.x, bl.y);
        grd.addColorStop(0, AM.rgba(AM.col.linen, 0.2));
        grd.addColorStop(0.55, AM.rgba(AM.dye.woad, 0.08));
        grd.addColorStop(1, AM.rgba(AM.col.linen, 0.05));
        g.fillStyle = grd;
        g.fill();
        g.strokeStyle = AM.rgba(AM.col.linen, 0.6);
        g.lineWidth = 1.2;
        g.lineJoin = 'round';
        g.stroke();
        g.strokeStyle = AM.rgba('#fff8e6', 0.35);
        g.beginPath(); g.moveTo(apex.x + 2, apex.y + 6); g.lineTo(br.x - 6, br.y - 3); g.stroke();
        g.restore();
      }

      // Step 1 shelf: the shapes of the multiplication, coloured by the real numbers.
      function shelfPrism(g, R, x0, x1, y0, y1, a) {
        const phone = R.phone;
        micro(g, phone ? '(1×8) · (8×4) = (1×4)' : 'one row times one matrix · (1×8) · (8×4) = (1×4)', x0, y0 + 2, a, { size: phone ? 8 : 9 });
        const c = Math.floor(Math.min(phone ? 5.5 : 6.5, (y1 - y0 - 26) / 8));
        const yy = y0 + 12;
        const cellsAt = (x, y, data, colorFn) => {
          g.save(); g.globalAlpha *= a;
          data.forEach((row, i) => row.forEach((v, j) => {
            g.fillStyle = colorFn(v);
            g.fillRect(x + j * c, y + i * c, c - 1, c - 1);
          }));
          g.restore();
        };
        const groups = [
          ['W_Q', W_Q, A.Q[QI], qMax, AM.dye.weld],
          ['W_K', W_K, A.K[QI], kMax, AM.dye.woad],
          ['W_V', W_V, A.V[QI], vMax, AM.dye.verdigris],
        ];
        const tint = (dye, mx) => (v) => (v >= 0 ? AM.mix(AM.col.ink3, dye, Math.min(1, 0.15 + v / mx)) : AM.mix(AM.col.ink3, AM.dye.madder, Math.min(1, -v / mx)));
        const xTint = (v) => (v >= 0 ? AM.mix(AM.col.ink3, AM.col.linen, Math.min(1, 0.12 + v)) : AM.mix(AM.col.ink3, AM.dye.madder, Math.min(1, -v)));
        const gw = 16 * c + (phone ? 26 : 34);
        const gap = phone ? 14 : 22;
        let x = x0;
        groups.forEach(([name, Wm, out, mx, dye]) => {
          if (x + gw > x1 + 1) return;
          const midY = yy + 4 * c;
          cellsAt(x, midY - c / 2, [A.X[QI]], xTint);
          let xx = x + 8 * c + (phone ? 4 : 6);
          AM.draw.text(g, '·', xx + 2, midY, { size: 12, color: AM.col.linenDim, alpha: a, baseline: 'middle', align: 'center' });
          xx += phone ? 7 : 10;
          cellsAt(xx, yy, Wm, (v) => AM.color.div(v, 3.6));
          AM.draw.text(g, name.replace('_', ''), xx + 2 * c, yy + 8 * c + 7, { size: phone ? 7.5 : 8.5, role: 'mono', color: dye, alpha: a, align: 'center', baseline: 'middle' });
          xx += 4 * c + (phone ? 4 : 6);
          AM.draw.text(g, '=', xx + 3, midY, { size: 11, color: AM.col.linenDim, alpha: a, baseline: 'middle', align: 'center' });
          xx += phone ? 9 : 12;
          cellsAt(xx, midY - c / 2, [out], tint(dye, mx));
          x += gw + gap;
        });
        if (x1 - x > 96) {
          const lines = wrapLines(g, 'The same three matrices are used for every token.', x1 - x, AM.font(11, 'body', 400, true));
          lines.slice(0, 3).forEach((ln, k) => AM.draw.text(g, ln, x, yy + 10 + k * 15, { size: 11, italic: true, weight: 400, color: AM.col.linenDim, alpha: a, baseline: 'middle' }));
        }
      }

      // Step 2 shelf: the dot product for the key under the beam.
      function shelfDot(g, R, x0, x1, y0, y1, a) {
        const phone = R.phone;
        const sweep = clamp((age(st.tBeam) - 0.25) / 1.5);
        const j = sweep < 1 ? Math.round(sweep * (TOKS.length - 1)) : best;
        const q = A.Q[QI], kk = A.K[j];
        const midY = (y0 + y1) / 2 + 2;
        micro(g, `score(it, ${TOKS[j]})${j === best && sweep >= 1 ? ' · the best match' : ''}`, x0, y0 + 2, a, { color: j === best && sweep >= 1 ? AM.dye.weld : AM.col.mist, size: phone ? 8 : 9 });
        const bw = phone ? 40 : 58, bh = phone ? 22 : 28;
        let x = x0;
        bars(g, x, midY - bh / 2 + 4, bw, bh, q, { max: qMax, pos: AM.dye.weld, alpha: a });
        x += bw + (phone ? 8 : 12);
        AM.draw.text(g, '·', x, midY + 5, { size: 18, color: AM.col.linenDim, alpha: a, baseline: 'middle', align: 'center' });
        x += phone ? 8 : 12;
        bars(g, x, midY - bh / 2 + 4, bw, bh, kk, { max: kMax, pos: AM.dye.woad, alpha: a });
        x += bw + (phone ? 10 : 16);
        const dotv = A.dots[QI][j];
        const sc = A.S[QI][j];
        const masked = j > QI && st.step >= 2;
        const txt = `= ${fmt(dotv)}  ÷ 2  = `;
        AM.draw.text(g, txt, x, midY + 5, { size: phone ? 10 : 12.5, role: 'mono', color: AM.col.linenDim, alpha: a, baseline: 'middle' });
        x += D.measure(g, txt, phone ? 10 : 12.5, 'mono') + 2;
        AM.draw.text(g, fmt(sc), x, midY + 5, { size: phone ? 15 : 20, role: 'display', weight: 600, color: masked ? AM.dye.madder : AM.dye.weld, alpha: a, baseline: 'middle' });
      }

      // Step 3 shelf: which positions the mask allows for query "it".
      function shelfMask(g, R, x0, x1, y0, y1, a) {
        const phone = R.phone;
        micro(g, 'mask row M[it] · 0 = allowed, −∞ = future', x0, y0 + 2, a, { size: phone ? 8 : 9 });
        const n2 = TOKS.length;
        const cw = Math.min(phone ? 32 : 46, (x1 - x0) / n2);
        const ch = phone ? 24 : 30;
        const yy = y0 + (phone ? 14 : 16);
        for (let j = 0; j < n2; j++) {
          const masked = j > QI;
          const xx = x0 + j * cw;
          AM.draw.roundRect(g, xx + 1.5, yy, cw - 3, ch, 4);
          g.save();
          g.globalAlpha = a;
          g.fillStyle = masked ? AM.rgba(AM.dye.madder, 0.18) : AM.rgba(AM.dye.weld, 0.1);
          g.fill();
          g.strokeStyle = masked ? AM.rgba(AM.dye.madder, 0.6) : AM.rgba(AM.dye.weld, 0.35);
          g.stroke();
          g.restore();
          AM.draw.text(g, masked ? '−∞' : '0', xx + cw / 2, yy + ch / 2 + 0.5, { size: phone ? 9.5 : 11, role: 'mono', color: masked ? AM.dye.madder : AM.dye.weld, alpha: a, align: 'center', baseline: 'middle' });
        }
        const visible = QI + 1, hidden = n2 - QI - 1;
        const tx = x0 + n2 * cw + 10;
        if (x1 - tx > 90) {
          AM.draw.text(g, `sees ${visible} tokens`, tx, yy + ch * 0.3, { size: phone ? 10 : 12, color: AM.dye.weld, alpha: a, baseline: 'middle' });
          AM.draw.text(g, `${hidden} hidden`, tx, yy + ch * 0.8, { size: phone ? 10 : 12, color: AM.dye.madder, alpha: a, baseline: 'middle' });
        }
      }

      // Step 4 shelf: the weights as one cloth band that sums to exactly 1.
      function shelfSoftmax(g, R, x0, x1, y0, y1, a) {
        const phone = R.phone;
        const W = A.W[QI];
        const sum = W.reduce((s2, x) => s2 + x, 0);
        micro(g, 'weights for “it” · softmax of the scores', x0, y0 + 2, a, { size: phone ? 8 : 9 });
        const sumTxt = `Σ = ${sum.toFixed(3)}`;
        AM.draw.text(g, sumTxt, x1, y0 + 2, { size: phone ? 10 : 12, role: 'mono', weight: 600, color: AM.dye.weld, alpha: a, align: 'right', baseline: 'middle' });
        const bh = phone ? 22 : 28, yy = y0 + (phone ? 14 : 16);
        const grow = easeOut(clamp((age(st.tThread) - 0.2) / 1.0));
        let x = x0;
        const total = (x1 - x0);
        g.save();
        g.globalAlpha = a;
        AM.draw.roundRect(g, x0, yy, total, bh, 5);
        g.fillStyle = AM.rgba(AM.col.ink3, 0.8);
        g.fill();
        g.clip();
        W.forEach((wj, j) => {
          const ww = total * wj * grow;
          if (ww < 0.5) { x += ww; return; }
          g.fillStyle = AM.color.heat(0.25 + 0.75 * Math.sqrt(wj));
          g.fillRect(x, yy, ww, bh);
          g.fillStyle = AM.rgba(AM.col.ink, 0.9);
          g.fillRect(x + ww - 1, yy, 1.5, bh);
          const lab = `${TOKS[j]} ${pct(wj)}`;
          const lw = D.measure(g, lab, phone ? 9 : 10.5, 'mono');
          if (lw < ww - 6) AM.draw.text(g, lab, x + ww / 2, yy + bh / 2 + 0.5, { size: phone ? 9 : 10.5, role: 'mono', color: wj > 0.3 ? AM.col.ink : AM.col.linen, align: 'center', baseline: 'middle', weight: 500 });
          x += ww;
        });
        g.restore();
      }

      // Step 5 shelf: the output o = Σ w·v, then the residual update x + o·W_O = x′.
      function shelfOutput(g, R, x0, x1, y0, y1, a) {
        const phone = R.phone;
        const fill = easeOut(clamp((age(st.tFlow) - 0.5) / 2.4));
        const o = A.O[QI];
        const bh = (y1 - y0) - (phone ? 18 : 32);
        const yy = y0 + 13, midY = yy + bh / 2;
        const lsz = phone ? 7.5 : 9;
        const ow = phone ? 34 : 56, gap0 = phone ? 12 : 26, opW = phone ? 14 : 26;
        const xw = Math.max(40, Math.min(phone ? 96 : 124, Math.floor((x1 - x0 - ow - gap0 - 2 * opW) / 3)));
        const op = (s, cx) => AM.draw.text(g, s, cx, midY, { size: phone ? 13 : 16, color: AM.col.linenDim, alpha: a, baseline: 'middle', align: 'center' });
        let x = x0;
        // o itself: four value dimensions, filling as the beads arrive
        micro(g, phone ? 'o' : 'o = Σ w·v', x, y0 + 2, a, { color: AM.dye.verdigris, size: lsz });
        bars(g, x, yy, ow, bh, o, { max: vMax, pos: AM.dye.verdigris, alpha: a, grow: fill, hl: 0 });
        if (!phone) AM.draw.text(g, `crea ${fmt(o[0] * fill)}`, x, yy + bh + 11, { size: 9, role: 'mono', color: AM.dye.verdigris, alpha: a * 0.9, baseline: 'middle' });
        x += ow + gap0;
        g.save();
        g.globalAlpha = a;
        g.strokeStyle = AM.rgba(AM.col.linen, 0.14);
        g.beginPath(); g.moveTo(Math.round(x - gap0 / 2) + 0.5, yy - 4); g.lineTo(Math.round(x - gap0 / 2) + 0.5, yy + bh + 4); g.stroke();
        g.restore();
        // x(it) + o·W_O = x′(it): the residual stream update
        const delta = A.dX[QI].map((v) => v * fill);
        const after = A.X[QI].map((v, d) => v + delta[d]);
        micro(g, phone ? 'x(it)' : 'x(it) before', x, y0 + 2, a, { size: lsz });
        bars(g, x, yy, xw, bh, A.X[QI], { max: 1.2, pos: AM.col.linenDim, alpha: a * 0.85, hl: 0 });
        x += xw; op('+', x + opW / 2); x += opW;
        micro(g, 'o·W_O', x, y0 + 2, a, { size: lsz, color: AM.dye.verdigris });
        bars(g, x, yy, xw, bh, delta, { max: 1.2, pos: AM.dye.verdigris, alpha: a, hl: 0 });
        x += xw; op('=', x + opW / 2); x += opW;
        micro(g, phone ? 'x′(it)' : 'x(it) after', x, y0 + 2, a, { size: lsz, color: AM.dye.weld });
        bars(g, x, yy, xw, bh, after, { max: 1.2, pos: AM.dye.weld, alpha: a, hl: 0 });
        if (!phone) AM.draw.text(g, `creature ${fmt(A.X[QI][0])} → ${fmt(after[0])}`, x, yy + bh + 11, { size: 9, role: 'mono', color: AM.dye.weld, alpha: a, baseline: 'middle' });
      }

      // ------------------------------------------------ loom (matrix) scene: steps 6–7
      const PHASES = [
        { key: 'qk', label: 'Q Kᵀ: every query dotted with every key, all at once' },
        { key: 'scale', label: '÷ √4 = 2: the keys have 4 numbers, so every score is halved' },
        { key: 'mask', label: '+ M: −∞ above the diagonal hides the future' },
        { key: 'soft', label: 'softmax, row by row: positive weights that sum to 1' },
        { key: 'v', label: '· V: each row of weights blends the values into one output' },
      ];
      const PH_DUR = 2.9;

      function loomGeom(w, h) {
        const phone = w < 520;
        return gridGeom(cv.g, w, h, n, {
          top: phone ? 78 : 96, left: phone ? 46 : 66, right: phone ? 44 : 70, bottom: phone ? 44 : 50, pad: phone ? 2 : 10,
        });
      }

      function cellFor(mode, i, j) {
        const masked = j > i;
        if (mode === 'qk' || mode === 'scale') {
          const v = mode === 'qk' ? A.dots[i][j] : A.S[i][j];
          const tcol = AM.color.div(v, rawMax);
          return { color: tcol, strength: Math.min(1, Math.abs(v) / rawMax) * 0.6, masked: false, text: fmt(v, 1), glow: v >= 0 ? AM.dye.woad : AM.dye.madder };
        }
        if (mode === 'mask') {
          if (masked) return { masked: true, text: '−∞', textColor: AM.dye.madder, role: 'body' };
          const v = A.S[i][j];
          return { color: AM.color.div(v, rawMax), strength: Math.min(1, Math.abs(v) / rawMax) * 0.6, masked: false, text: fmt(v, 1), glow: AM.dye.woad };
        }
        // weights
        if (masked) return { masked: true };
        const wv = A.W[i][j];
        return { color: AM.color.heat(0.08 + 0.92 * Math.pow(wv, 0.7)), strength: wv, masked: false, text: wv < 0.005 ? '0' : wv.toFixed(2).replace(/^0/, ''), textColor: wv > 0.45 ? AM.col.ink : AM.col.linen };
      }

      function drawLoomScene(g, w, h, a, clock, R) {
        const phone = w < 520;
        const G = loomGeom(w, h);
        const inFormula = st.step === 6;
        const fA = st.tw.formula;
        // which phase of the formula walk-through
        const fAge = now() - st.tFormula; // phases advance slowly even with reduced motion
        const phase = inFormula ? Math.floor(fAge / PH_DUR) % PHASES.length : 3;
        const pp = inFormula ? (fAge % PH_DUR) / PH_DUR : 1;
        const prevPhase = (phase + PHASES.length - 1) % PHASES.length;
        const cross = inFormula && fAge > PH_DUR ? M.smoothstep(0, 0.16, pp) : 1;

        // shuttle reveal for step 6: row i is woven left→right, up to the diagonal
        const wAge = age(st.tWeave);
        const rowDur = 0.42, rowGap = 0.17;
        const reveal = (i, j) => {
          if (inFormula || AM.reducedMotion) return 1;
          const p = (wAge - 0.35 - i * rowGap) / rowDur;
          return clamp(p * (i + 1) - j + 0.5);
        };

        // tokens morph: tiles from the row fly to their column-label positions
        if (a < 0.999) {
          const e = a;
          g.save();
          R.slots.forEach((sl, j) => {
            const tx = G.gx + j * G.cell + G.cell / 2, ty = G.gy + G.gw + 14;
            const x = M.lerp(sl.cx, tx, e), y = M.lerp(R.rowY, ty, e);
            D.token(g, TOKS[j], x, y, { size: R.fs * (1 - 0.25 * e), alpha: (1 - e) * 0.9, padX: 2, h: R.tokH * (1 - 0.3 * e), selected: j === QI });
          });
          g.restore();
        }

        g.save();
        g.globalAlpha = a;

        const mode = (ph) => (ph === 3 || ph === 4 ? 'soft' : PHASES[ph].key);
        const numbersAll = !phone && G.cell >= 36;
        // small cells (phones): numbers on the focus row only, in every phase
        const showText = (i) => (numbersAll ? (inFormula || i === QI) : i === QI);
        const hatch = inFormula ? (phase === 2 ? 0.55 : phase > 2 ? 0.45 : 0) : 0.35;
        if (inFormula && cross < 1) {
          drawWeave(g, G, (i, j) => cellFor(mode(prevPhase), i, j), { alpha: 1 - cross, focus: QI, showText, hatchMask: hatch });
        }
        drawWeave(g, G, (i, j) => cellFor(inFormula ? mode(phase) : 'soft', i, j), {
          alpha: inFormula ? cross : 1, reveal, focus: QI, showText, hatchMask: hatch, clock,
        });

        // mask hint in phases before +M: show where −∞ will go
        if (inFormula && phase === 2) {
          g.save();
          const pulse = 0.5 + 0.5 * Math.sin(clock * 4);
          g.strokeStyle = AM.rgba(AM.dye.madder, 0.55 + 0.35 * pulse);
          g.lineWidth = 1.5;
          g.beginPath();
          g.moveTo(G.gx + G.cell, G.gy);
          for (let i = 0; i < n; i++) { g.lineTo(G.gx + (i + 1) * G.cell, G.gy + (i + 1) * G.cell); if (i < n - 1) g.lineTo(G.gx + (i + 2) * G.cell, G.gy + (i + 1) * G.cell); }
          g.stroke();
          g.restore();
          if (!numbersAll) haloText(g, '−∞', G.gx + G.gw * 0.74, G.gy + G.gw * 0.24, { size: phone ? 15 : 20, role: 'body', weight: 600, color: AM.dye.madder, alpha: 0.95 });
        }

        // shuttle
        if (!inFormula && !AM.reducedMotion) {
          for (let i = 0; i < n; i++) {
            const p = (wAge - 0.35 - i * rowGap) / rowDur;
            if (p <= 0 || p >= 1.15) continue;
            const x = G.gx + clamp(p) * (i + 1) * G.cell;
            const y = G.gy + i * G.cell + G.cell / 2;
            D.glowDot(g, x, y, phone ? 3 : 4, AM.dye.weld, 1 - clamp((p - 1) / 0.15));
          }
        }

        // labels: rows = queries (Q), columns = keys (Kᵀ)
        const qGlow = inFormula && phase === 0 ? 1 : 0;
        drawGridLabels(g, G, TOKS, {
          phone, focus: QI,
          rowColor: (i) => (i === QI ? AM.dye.weld : qGlow ? AM.mix(AM.col.linenDim, AM.dye.weld, 0.85) : AM.col.linenDim),
          colColor: () => (qGlow ? AM.dye.woad : AM.col.linenDim),
        });
        micro(g, phone ? 'QUERIES' : 'QUERIES ↓', G.gx - 9, G.gy - 13, 0.9, { align: 'right', size: phone ? 7.5 : 8.5 });
        micro(g, 'KEYS →', G.gx, G.gy - 13, 0.9, { size: phone ? 7.5 : 8.5 });

        // right column: Σ per row (weights) or the output vectors (· V)
        const rx = G.gx + G.gw + (phone ? 8 : 14);
        const showSum = !inFormula ? clamp((wAge - 0.35 - n * rowGap) / 0.4) : phase === 3 ? 1 : 0;
        const showOut = inFormula && phase === 4 ? 1 : 0;
        if (showSum > 0) {
          micro(g, 'Σ', rx + (phone ? 12 : 18), G.gy - 13, showSum, { align: 'center', color: AM.dye.weld });
          for (let i = 0; i < n; i++) {
            const sum = A.W[i].reduce((s2, x) => s2 + x, 0);
            AM.draw.text(g, sum.toFixed(2), rx, G.gy + i * G.cell + G.cell / 2 + 0.5, { size: phone ? 8.5 : 10, role: 'mono', color: AM.dye.weld, alpha: showSum * (i === QI ? 1 : 0.75), baseline: 'middle' });
          }
        }
        if (showOut > 0) {
          micro(g, 'o', rx + (phone ? 14 : 22), G.gy - 13, showOut, { align: 'center', color: AM.dye.verdigris });
          const bw = phone ? 30 : 48, bh = Math.max(6, G.cell * 0.55);
          const grow = easeOut(clamp(pp * 2.2));
          for (let i = 0; i < n; i++) {
            bars(g, rx, G.gy + i * G.cell + (G.cell - bh) / 2, bw, bh, A.O[i], { max: vMax, pos: AM.dye.verdigris, alpha: showOut, grow, frame: true });
          }
          // a bead travelling along the focus row into its output
          const bx = G.gx + ((pp * 1.6) % 1) * (G.gw + 10);
          D.glowDot(g, bx, G.gy + QI * G.cell + G.cell / 2, phone ? 2.5 : 3.2, AM.dye.verdigris, 0.9);
        }

        g.restore();

        // top band: title (step 6) or the formula (step 7)
        drawLoomTop(g, w, G, a, fA, phase, pp, phone);
      }

      function drawLoomTop(g, w, G, a, fA, phase, pp, phone) {
        const titleA = a * (1 - fA);
        if (titleA > 0.01) {
          AM.draw.text(g, 'ATTENTION WEIGHTS', 10, phone ? 18 : 22, { size: phone ? 9 : 10, role: 'mono', color: AM.col.mist, alpha: titleA, baseline: 'middle', letterSpacing: '0.14em' });
          AM.draw.text(g, 'every query at once · rows sum to 1 · the future stays unwoven', 10, phone ? 38 : 46, { size: phone ? 11 : 13.5, italic: true, weight: 400, color: AM.col.linenDim, alpha: titleA, baseline: 'middle', maxWidth: w - 20 });
        }
        if (fA < 0.01) return;
        // formula: softmax( Q Kᵀ / √d_k + M ) · V — each symbol lights up with its phase
        const parts = [
          { s: 'softmax(', ph: 3, role: 'mono' },
          { s: 'Q\u2009', ph: 0, role: 'display', it: true, c: AM.dye.weld },
          { s: 'K', ph: 0, role: 'display', it: true, c: AM.dye.woad },
          { s: 'ᵀ', ph: 0, role: 'display', c: AM.dye.woad },
          { s: ' / √d', ph: 1, role: 'mono' },
          { s: 'k', ph: 1, role: 'mono', sub: true },
          { s: ' + ', ph: 2, role: 'mono' },
          { s: 'M', ph: 2, role: 'display', it: true, c: AM.dye.madder },
          { s: ')', ph: 3, role: 'mono' },
          { s: ' · ', ph: 4, role: 'mono' },
          { s: 'V', ph: 4, role: 'display', it: true, c: AM.dye.verdigris },
        ];
        const big = phone ? 17 : 25;
        const fontFor = (p) => AM.font(p.sub ? big * 0.55 : p.role === 'display' ? big * 1.1 : big * 0.72, p.role, p.role === 'display' ? 700 : 400, !!p.it);
        g.save();
        g.globalAlpha = a * fA;
        const widths = parts.map((p) => { g.font = fontFor(p); return g.measureText(p.s).width; });
        const total = widths.reduce((s2, x) => s2 + x, 0);
        let x = (w - total) / 2;
        const y = phone ? 24 : 31;
        parts.forEach((p, i) => {
          const on = p.ph === phase || (phase === 3 && (p.ph === 3));
          g.font = fontFor(p);
          g.textBaseline = 'middle';
          const col = on ? (p.c || AM.dye.weld) : p.c ? AM.mix(AM.col.ink3, p.c, 0.8) : AM.col.mist;
          if (on) { g.shadowColor = AM.rgba(p.c || AM.dye.weld, 0.8); g.shadowBlur = 14; }
          g.fillStyle = col;
          g.fillText(p.s, x, y + (p.sub ? big * 0.32 : 0));
          g.shadowBlur = 0;
          if (on) {
            g.fillStyle = AM.rgba(p.c || AM.dye.weld, 0.8);
            g.fillRect(x, y + big * 0.62, widths[i], 1.5);
          }
          x += widths[i];
        });
        // phase dots + caption
        const lines = wrapLines(g, PHASES[phase].label, w - 24, AM.font(phone ? 11 : 13, 'body', 500));
        lines.slice(0, 2).forEach((ln, k) => AM.draw.text(g, ln, w / 2, (phone ? 48 : 62) + k * (phone ? 14 : 16), { size: phone ? 11 : 13, color: AM.col.linen, align: 'center', baseline: 'middle' }));
        const dotsY = 6;
        for (let k = 0; k < PHASES.length; k++) {
          g.fillStyle = k === phase ? AM.dye.weld : AM.rgba(AM.col.linen, 0.2);
          g.beginPath(); g.arc(w / 2 + (k - 2) * 10, dotsY, k === phase ? 2.6 : 2, 0, Math.PI * 2); g.fill();
        }
        g.restore();
      }

      cv.onResize(() => drawStage());
      const stageSeen = inView(cv.canvas);
      ctx.loop(() => { if (stageSeen.on) drawStage(); });
      drawStage();

      // ------------------------------------------------ free-play workbench
      buildWorkbench(body, ctx);
      // ------------------------------------------------ why √d_k
      buildWhyScale(body, ctx);
      // ------------------------------------------------ RoPE: position inside attention
      buildRope(body, ctx);

      body.appendChild(el('div', { class: 'callout att-key' },
        el('div', { class: 'callout-label' }, 'Key idea'),
        el('p', { html: 'Attention is a soft lookup. Each token sends a <span class="dye-weld">query</span>, matches it against the <span class="dye-woad">keys</span> of itself and every earlier token, and collects a blend of their <span class="dye-verdigris">values</span>. The blend weights are <span class="math">softmax(q·k/√d<sub>k</sub>)</span> under a causal mask, so they are positive, sum to 1 and never look ahead. The result is added back into the residual stream.' }),
        el('p', { html: 'A real model runs several of these heads side by side, each with its own W<sub>Q</sub>, W<sub>K</sub>, W<sub>V</sub>. That is the next chapter.' }),
      ));
    },
  });

  /** One sentence describing the strongest look-back for a few rows, from the computed weights. */
  function rowStory(A, toks) {
    const pick = [2, 3, 5, 9];
    const parts = pick.map((i) => {
      const j = M.argmax(A.W[i]);
      return `${tokPhrase(toks, i)} looks most at ${tokPhrase(toks, j)} (${pct(A.W[i][j])})`;
    });
    return parts.slice(0, -1).join(', ') + ' and ' + parts[parts.length - 1] + '.';
  }

  // ====================================================================
  // 6. Workbench: free play
  // ====================================================================
  function buildWorkbench(body, ctx) {
    const el = ctx.el, ui = AM.ui, D = AM.draw;
    const state = { end: 'tired', qi: 7, mask: true, scale: true, mult: 1, hover: null };
    let toks = sentence(state.end);
    let A = attend(embed(toks), state);
    const recompute = () => { toks = sentence(state.end); A = attend(embed(toks), state); };

    const fig = el('figure', { class: 'fig ch-wide att-bench' });
    body.appendChild(fig);
    fig.appendChild(el('div', { class: 'fig-top' }, el('span', { class: 'fig-title' }, 'The attention workbench · free play'), ui.badge('toy')));
    fig.appendChild(el('div', { class: 'prose att-bench-intro' },
      el('p', { html: '<strong>Pick any word to make it the query.</strong> Switch the mask off to let tokens cheat and look ahead. Turn off the √d<sub>k</sub> division or drag the multiplier to see softmax go from a gentle blend to a hard choice.' })));

    // controls
    const sent = ui.segmented({
      id: 'att-sent', label: 'Sentence ends with', value: 'tired',
      options: [{ value: 'tired', label: '…it was tired' }, { value: 'soft', label: '…it was soft' }],
      onChange: (v) => { state.end = v; recompute(); rebuildTokens(); rebuildTables(); refresh(); },
    });
    const maskT = ui.toggle({ id: 'att-mask', label: 'Causal mask', checked: true, onChange: (b) => { state.mask = b; recompute(); refresh(); } });
    const scaleT = ui.toggle({ id: 'att-scale', label: 'Divide by √d_k', checked: true, onChange: (b) => { state.scale = b; recompute(); refresh(); } });
    scaleT.el.lastChild.innerHTML = 'Divide by <span class="att-nocase">√d<sub>k</sub></span>';
    const mult = ui.slider({
      id: 'att-mult', label: 'Score multiplier', min: 0, max: 4, step: 0.05, value: 1,
      format: (v) => '×' + v.toFixed(2), onInput: (v) => { state.mult = v; recompute(); refresh(); },
    });
    const ctlRow = el('div', { class: 'controls att-controls' }, sent.el, el('div', { class: 'att-toggles' }, maskT.el, scaleT.el), mult.el);
    fig.appendChild(ctlRow);

    const tokRow = el('div', { class: 'att-tokrow' });
    fig.appendChild(tokRow);
    let chips;
    function rebuildTokens() {
      tokRow.innerHTML = '';
      chips = ui.tokens(toks, {
        selected: state.qi, label: 'Choose the query token',
        colors: () => AM.dye.weld,
        onSelect: (i) => { state.qi = i; refresh(); },
      });
      tokRow.appendChild(el('span', { class: 'ctl-label att-tok-label' }, 'Query'));
      tokRow.appendChild(chips.el);
    }
    rebuildTokens();

    // the two canvases: threads over the sentence, and the woven matrix
    const grid = el('div', { class: 'grid-2 att-bench-grid' });
    fig.appendChild(grid);
    const leftBox = el('div', { class: 'att-cv-box' });
    const rightBox = el('div', { class: 'att-cv-box' });
    grid.append(leftBox, rightBox);
    leftBox.appendChild(el('div', { class: 'att-cv-title' }, 'Threads from the query'));
    rightBox.appendChild(el('div', { class: 'att-cv-title' }, 'The whole weave · hover or tap a cell'));
    const cvA = ctx.canvas(leftBox, {
      label: 'Attention threads from the chosen query token to every token it can see. Thread width is the attention weight. Tap a token to choose it as the query.',
      height: (w) => (w < 420 ? 250 : Math.round(Math.min(360, Math.max(280, w * 0.64)))),
    });
    const cvB = ctx.canvas(rightBox, {
      label: 'The attention matrix as woven cloth: rows are queries, columns are keys; brighter weft means more attention. Hover or tap a cell for its numbers.',
      height: (w) => (w < 420 ? Math.round(Math.max(300, w * 0.9)) : Math.round(Math.min(360, Math.max(280, w * 0.64)))),
    });
    cvA.canvas.style.cursor = 'pointer';
    cvB.canvas.style.cursor = 'crosshair';

    // readout panel
    const panel = el('div', { class: 'panel att-readout' });
    fig.appendChild(panel);
    const head = el('div', { class: 'att-rq' });
    const stats = el('div', { class: 'att-stats' });
    panel.append(el('div', { class: 'att-rq-row' }, head, stats));
    const tableBox = el('div', { class: 'att-table' });
    panel.appendChild(tableBox);
    const outBox = el('div', { class: 'att-out' });
    panel.appendChild(outBox);

    const mats = el('div', { class: 'grid-3 att-mats' });
    fig.appendChild(mats);

    const weights = el('details', { class: 'att-weights' });
    fig.appendChild(weights);

    fig.appendChild(el('figcaption', {
      html: 'Toy numbers. The embeddings use 7 named features plus one position number (a simplification of chapter 3), and W<sub>Q</sub>, W<sub>K</sub>, W<sub>V</sub>, W<sub>O</sub> were written by hand so the patterns are easy to read. Every value in this workbench is computed from them as you watch. The question shown for each word is our reading of its hand-built query. In the real tiny model (later chapters) these matrices are <em>learned</em> from data, and the patterns are messier.',
    }));

    let scoreMx, qMx, kMx, vMx, outVec, tableNarrow = null;
    const STAGES = ['q·k', 'score', '+ mask', 'weight'];
    // Wide panels: one row per stage of the formula, one column per key.
    // Narrow panels (phones): transposed, one row per key, so it fits without sideways scrolling.
    const isNarrow = () => (panel.clientWidth || 1000) < 640;
    function buildScoreTable() {
      tableNarrow = isNarrow();
      tableBox.innerHTML = '';
      const zeros = (r, c) => Array.from({ length: r }, () => new Array(c).fill(0));
      const onHover = (r, c) => { state.hover = r >= 0 ? { i: state.qi, j: tableNarrow ? r : c } : null; paintHover(); drawB(); };
      scoreMx = tableNarrow
        ? ui.matrix({ data: zeros(toks.length, 4), rowLabels: toks, colLabels: STAGES, corner: 'key', format: (v) => fmt(v), onHover })
        : ui.matrix({ data: zeros(4, toks.length), rowLabels: STAGES, colLabels: toks, format: (v) => fmt(v), onHover });
      tableBox.classList.toggle('is-narrow', tableNarrow);
      tableBox.appendChild(scoreMx.el);
    }
    function rebuildTables() {
      buildScoreTable();
      mats.innerHTML = '';
      const mk = (title, cls, M_) => {
        const m = ui.matrix({ data: M_, rowLabels: toks, colLabels: ['1', '2', '3', '4'], format: (v) => fmt(v) });
        mats.appendChild(el('div', { class: 'att-mat ' + cls }, el('div', { class: 'fig-title', html: title }), m.el));
        return m;
      };
      qMx = mk('Q = X·W<sub>Q</sub> · queries', 'is-q', A.Q);
      kMx = mk('K = X·W<sub>K</sub> · keys', 'is-k', A.K);
      vMx = mk('V = X·W<sub>V</sub> · values', 'is-v', A.V);
      // the hand-built weights, for the curious
      weights.innerHTML = '';
      weights.appendChild(el('summary', {}, 'Show the hand-built embeddings and weight matrices'));
      const wrap = el('div', { class: 'att-wgrid' });
      const X = embed(toks);
      const sm = (title, data, rl, cl) => wrap.appendChild(el('div', { class: 'att-mat' }, el('div', { class: 'fig-title', html: title }),
        ui.matrix({ data, rowLabels: rl, colLabels: cl, format: (v) => (v === 0 ? '·' : fmt(v, v === Math.round(v * 10) / 10 ? 1 : 2)), color: (v) => AM.color.div(v, 3), textColor: (v) => textOn(AM.color.div(v, 3)) }).el));
      sm('X · embeddings <span class="att-nocase">(d<sub>model</sub> = 8)</span>', X, toks, FEAT_SHORT);
      sm('W<sub>Q</sub>', W_Q, FEAT_SHORT, ['1', '2', '3', '4']);
      sm('W<sub>K</sub>', W_K, FEAT_SHORT, ['1', '2', '3', '4']);
      sm('W<sub>V</sub>', W_V, FEAT_SHORT, ['1', '2', '3', '4']);
      sm('W<sub>O</sub>', W_O, VDIMS.map((s) => s.slice(0, 5)), FEAT_SHORT);
      weights.appendChild(wrap);
      weights.appendChild(el('p', { class: 'caption', html: 'Features: ' + FEATS.map((f, i) => `<b>${FEAT_SHORT[i]}</b> ${f}`).join(' · ') + '. Value dimensions: ' + VDIMS.join(', ') + '.' }));
    }
    rebuildTables();

    function paintTables() {
      const i = state.qi;
      const data = [A.dots[i], A.S[i], A.Sm[i], A.W[i]];
      scoreMx.update(tableNarrow ? toks.map((_, j) => data.map((r) => r[j])) : data);
      const smax = Math.max(1e-6, ...A.S[i].map(Math.abs));
      const dmax = Math.max(1e-6, ...A.dots[i].map(Math.abs));
      for (let ri = 0; ri < 4; ri++) for (let j = 0; j < toks.length; j++) {
        const c = tableNarrow ? scoreMx.cells[j][ri] : scoreMx.cells[ri][j];
        const v = data[ri][j];
        if (ri === 3) { const bg = AM.color.heat(0.1 + 0.9 * Math.sqrt(v)); c.style.background = bg; c.style.color = textOn(bg); }
        else if (v === -Infinity) { c.style.background = AM.rgba(AM.dye.madder, 0.22); c.style.color = MASK_TEXT(); }
        else { const bg = AM.color.div(v, ri === 0 ? dmax : smax); c.style.background = bg; c.style.color = textOn(bg); }
      }
      // stage labels reflect the current scaling
      const lab = scoreMx.el.querySelectorAll(tableNarrow ? '[role=columnheader]' : '.mx-head.row');
      const f = A.factor;
      // one wrapping span: the header cell is a grid, so bare text + <sub> would become two grid items
      const dk = '÷ √d<sub>k</sub>';
      if (lab[1]) lab[1].innerHTML = `<span class="att-nocase">${state.scale ? (state.mult === 1 ? dk : `${dk} × ${state.mult.toFixed(2)}`) : `× ${state.mult.toFixed(2)}`}</span>`;
      if (lab[1]) lab[1].title = `score = q·k × ${f.toFixed(3)}`;
      qMx.update(A.Q); kMx.update(A.K); vMx.update(A.V);
      paintHover();
    }
    function paintHover() {
      const i = state.qi;
      const hj = state.hover ? state.hover.j : -1;
      qMx.highlight(i, null);
      kMx.highlight(hj, null);
      vMx.highlight(hj, null);
      scoreMx.cells.forEach((r, ri) => r.forEach((c, cj) => c.classList.toggle('is-hot', hj >= 0 && (tableNarrow ? ri : cj) === hj)));
    }

    function paintReadout() {
      const i = state.qi;
      const w = A.W[i];
      // effective number of tokens attended = exp(entropy)
      let H = 0; for (const p of w) if (p > 0) H -= p * Math.log(p);
      const eff = Math.exp(H);
      const jb = M.argmax(w);
      head.innerHTML = '';
      head.append(
        el('span', { class: 'ctl-label' }, 'Query'),
        el('span', { class: 'att-qword' }, toks[i]),
        el('span', { class: 'att-qq' }, '“' + QUESTION[toks[i]] + '”'),
      );
      stats.innerHTML = '';
      stats.append(
        el('span', {}, 'strongest ', el('b', {}, `${tokShort(toks, jb)} ${pct(w[jb])}`)),
        el('span', {}, 'Σ weights ', el('b', {}, w.reduce((a, b) => a + b, 0).toFixed(3))),
        // how many tokens the weight is spread over: exp(entropy), 1 when it all sits on one token
        el('span', { title: 'exp(entropy) of the weights: 1 = all on one token' }, 'spread ', el('b', {}, `≈ ${eff.toFixed(1)} tokens`), el('span', { class: 'att-dim' }, ' (1 = all on one)')),
      );
      outBox.innerHTML = '';
      const o = A.O[i];
      if (!outVec) {
        const c = el('canvas', { role: 'img', 'aria-label': 'output vector o as bars', class: 'att-outbars' });
        const dpr = Math.min(2, window.devicePixelRatio || 1);
        c.width = 120 * dpr; c.height = 34 * dpr;
        outVec = { el: c, update: (v) => { const g = c.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, 120, 34); bars(g, 2, 2, 116, 30, v, { max: 1.2, pos: AM.dye.verdigris }); } };
      }
      outVec.update(o);
      outBox.append(
        el('div', { class: 'att-out-eq' },
          el('span', { class: 'ctl-label' }, 'Output'),
          el('span', { class: 'math' }, `o = Σ w·v = [${o.map((v) => fmt(v)).join(', ')}]`)),
        outVec.el,
        el('div', { class: 'att-out-dims' }, VDIMS.map((d, k) => el('span', {}, d, ' ', el('b', {}, fmt(o[k]))))),
      );
    }

    function refresh() {
      chips.select(state.qi);
      paintTables();
      paintReadout();
      drawA();
      drawB();
    }

    // ---- canvas A: threads over the sentence
    const beadsA = makeBeads(150);
    let lastA = now();
    let geomA = null;
    function drawA() {
      const { g, w, h } = cvA;
      if (!w) return;
      const t = now();
      const dt = Math.min(0.1, t - lastA); lastA = t;
      cvA.clear();
      const phone = w < 420;
      const pad = phone ? 4 : 10;
      const { slots, fs } = layoutTokens(g, toks, pad, w - pad, phone ? 11.5 : 14, phone ? 3 : 6);
      const tokH = Math.round(fs + (phone ? 11 : 14));
      const rowY = h - tokH / 2 - (phone ? 34 : 40);
      const anchorY = rowY - tokH / 2 - (phone ? 19 : 23);
      const labelY = rowY - tokH / 2 - (phone ? 9 : 11);
      const top = 30;
      const maxLift = (anchorY - top - 6) / 0.75;
      geomA = { slots, rowY, tokH };
      const i = state.qi;
      const W = A.W[i];
      D.weave(g, 0, 0, w, h, { step: 7, alpha: 0.02 });
      drawWarp(g, slots, top, h - 4, t);
      if (state.mask) drawCurtain(g, slots, i, top - 4, rowY + tokH / 2 + 4, w, 1, w > 360, phone);
      const paths = threadPaths(slots, i, anchorY, maxLift, phone ? 9 : 12);
      const order = W.map((x, j) => j).sort((a, b) => W[a] - W[b]);
      for (const j of order) {
        if (W[j] < 0.002) continue;
        silk(g, paths[j], { width: 0.8 + W[j] * (phone ? 7 : 10), alpha: 0.22 + 0.78 * Math.sqrt(W[j]) });
      }
      beadsA.step(dt, W, 12, !AM.reducedMotion);
      g.save();
      g.globalCompositeOperation = 'lighter';
      for (const b of beadsA.list) {
        const p = beadPos(b, paths, slots, labelY, anchorY);
        D.glowDot(g, p.x, p.y, b.r * (phone ? 1.1 : 1.4), AM.dye.verdigris, 0.95);
      }
      g.restore();
      slots.forEach((sl, j) => {
        const masked = A.Sm[i][j] === -Infinity;
        // weights row: a masked (future) key gets weight exactly 0
        haloText(g, masked ? '0' : pct(W[j]), sl.cx, labelY, { size: phone ? 9 : 10.5, color: masked ? AM.rgba(AM.dye.madder, 0.85) : W[j] > 0.25 ? AM.dye.weld : AM.col.linenDim, weight: W[j] > 0.25 ? 600 : 400 });
        D.token(g, toks[j], sl.cx, rowY, { size: fs, w: sl.w, h: tokH, selected: j === i, alpha: masked ? 0.4 : 1, padX: 2, underline: j === i ? AM.dye.weld : null, stroke: state.hover && state.hover.j === j ? AM.col.linen : undefined });
      });
      // question
      AM.draw.text(g, '“' + QUESTION[toks[i]] + '”', pad + 2, 16, { size: phone ? 12 : 14, italic: true, weight: 400, color: AM.col.linenDim, baseline: 'middle', maxWidth: w - 2 * pad });
      // output vector under the query (o = Σ w v)
      const o = A.O[i];
      const bw = phone ? 40 : 56, bh = phone ? 16 : 20;
      const ox = clamp(slots[i].cx - bw / 2, pad + 2, w - pad - bw - 2);
      const oy = rowY + tokH / 2 + (phone ? 10 : 12);
      bars(g, ox, oy, bw, bh, o, { max: 1.2, pos: AM.dye.verdigris });
      const lab = 'o =';
      const lx = ox - 6;
      if (lx - D.measure(g, lab, 10, 'mono') > 0) AM.draw.text(g, lab, lx, oy + bh / 2 + 0.5, { size: phone ? 9 : 10, role: 'mono', color: AM.dye.verdigris, baseline: 'middle', align: 'right' });
      else AM.draw.text(g, lab.replace(' =', ''), ox + bw + 6, oy + bh / 2 + 0.5, { size: phone ? 9 : 10, role: 'mono', color: AM.dye.verdigris, baseline: 'middle' });
    }
    cvA.canvas.addEventListener('pointerdown', (ev) => {
      if (!geomA) return;
      const p = cvA.pointer(ev);
      let best = -1, bd = 1e9;
      geomA.slots.forEach((sl, j) => { const d = Math.abs(sl.cx - p.x); if (d < bd) { bd = d; best = j; } });
      if (best >= 0 && bd < geomA.slots[best].w / 2 + 6 && p.y > geomA.rowY - geomA.tokH * 1.5) { state.qi = best; refresh(); }
    });

    // ---- canvas B: the woven matrix
    let geomB = null;
    function drawB() {
      const { g, w, h } = cvB;
      if (!w) return;
      cvB.clear();
      const phone = w < 420;
      const G = gridGeom(g, w, h, toks.length, { top: 22, left: phone ? 50 : 64, right: phone ? 34 : 48, bottom: phone ? 48 : 52, pad: 4 });
      geomB = G;
      const i0 = state.qi;
      const smax = Math.max(1e-6, ...A.W.flat());
      drawWeave(g, G, (i, j) => {
        if (A.Sm[i][j] === -Infinity) return { masked: true };
        const wv = A.W[i][j];
        return { color: AM.color.heat(0.08 + 0.92 * Math.pow(wv / smax, 0.7)), strength: wv, text: wv < 0.005 ? '0' : wv.toFixed(2).replace(/^0/, ''), textColor: wv > 0.45 ? AM.col.ink : AM.col.linen };
      }, { focus: i0, hover: state.hover, showText: (i) => i === i0 && G.cell >= 31, hatchMask: 0.35 });
      drawGridLabels(g, G, toks, { phone, focus: i0, colColor: (j) => (state.hover && state.hover.j === j ? AM.col.linen : AM.col.linenDim) });
      micro2(g, 'QUERIES ↓  KEYS →', G.gx, 10);
      // Σ column
      const rx = G.gx + G.gw + 8;
      micro2(g, 'Σ', rx + 10, 10, AM.dye.weld);
      A.W.forEach((r, i) => AM.draw.text(g, r.reduce((a, b) => a + b, 0).toFixed(2), rx, G.gy + i * G.cell + G.cell / 2 + 0.5, { size: phone ? 8 : 9.5, role: 'mono', color: AM.dye.weld, alpha: i === i0 ? 1 : 0.6, baseline: 'middle' }));
      // hover readout
      if (state.hover && state.hover.i >= 0) {
        const { i, j } = state.hover;
        const masked = A.Sm[i][j] === -Infinity;
        const txt = masked
          ? `${toks[i]} → ${toks[j]}: masked (future), weight 0`
          : `${toks[i]} → ${toks[j]}: q·k = ${fmt(A.dots[i][j])}, score ${fmt(A.S[i][j])}, weight ${A.W[i][j].toFixed(3)}`;
        haloText(g, txt, w / 2, h - 8, { size: phone ? 9 : 10.5, color: masked ? AM.dye.madder : AM.col.linen, maxWidth: w - 8 });
      }
    }
    function micro2(g, str, x, y, color) { AM.draw.text(g, str, x, y, { size: 8.5, role: 'mono', color: color || AM.col.mist, baseline: 'middle', letterSpacing: '0.1em' }); }
    const cellAt = (ev) => {
      if (!geomB) return null;
      const p = cvB.pointer(ev);
      const { gx, gy, cell, n } = geomB;
      const j = Math.floor((p.x - gx) / cell), i = Math.floor((p.y - gy) / cell);
      if (i < 0 || j < 0 || i >= n || j >= n) return null;
      return { i, j };
    };
    cvB.canvas.addEventListener('pointermove', (ev) => { state.hover = cellAt(ev); paintHover(); drawB(); drawA(); });
    cvB.canvas.addEventListener('pointerleave', () => { state.hover = null; paintHover(); drawB(); drawA(); });
    cvB.canvas.addEventListener('pointerdown', (ev) => {
      const c = cellAt(ev);
      if (!c) return;
      state.hover = c;
      state.qi = c.i;
      refresh();
    });

    cvA.onResize(() => drawA());
    cvB.onResize(() => drawB());
    if (typeof ResizeObserver !== 'undefined') {
      new ResizeObserver(() => { if (isNarrow() !== tableNarrow) { buildScoreTable(); paintTables(); } }).observe(panel);
    }
    const aSeen = inView(cvA.canvas);
    ctx.loop(() => { if (aSeen.on) drawA(); });
    refresh();
  }

  // ====================================================================
  // 7. Why √d_k: random queries and keys
  // ====================================================================
  function buildWhyScale(body, ctx) {
    const el = ctx.el, ui = AM.ui, D = AM.draw;
    const wrap = el('div', { class: 'grid-2 att-why' });
    body.appendChild(wrap);
    wrap.appendChild(el('div', { class: 'prose' },
      el('h3', { html: 'Why divide by √d<sub>k</sub>?' }),
      el('p', { html: 'Suppose the d<sub>k</sub> numbers in a query and a key are independent and random, with mean 0 and variance 1. Each product q<sub>n</sub>k<sub>n</sub> then has variance 1. The dot product adds up d<sub>k</sub> of them, so its variance is d<sub>k</sub> and its spread (standard deviation) is <span class="math">√d<sub>k</sub></span>.' }),
      el('p', { html: 'Real heads often use d<sub>k</sub> = 64 or 128, a spread of 8 to 11. Unscaled, softmax would put nearly all the weight on one key, and learning would stall because a saturated softmax passes back almost no gradient.' }),
      el('p', { html: 'Dividing by √d<sub>k</sub> brings the spread back to about 1 at any size. Drag the slider and watch the two curves.' }),
    ));
    const fig = el('figure', { class: 'fig' });
    wrap.appendChild(fig);
    fig.appendChild(el('div', { class: 'fig-top' }, el('span', { class: 'fig-title' }, 'Random queries & keys'), ui.badge('toy', 'Toy numbers · random')));
    const cvs = ctx.canvas(fig, {
      label: 'Histogram of dot products of random query and key vectors, unscaled and scaled by one over square root of d_k, plus the softmax of eight random scores.',
      height: (w) => Math.round(Math.min(360, Math.max(300, w * 0.7))),
    });
    const S = { e: 6, seed: 7, data: null };
    const dkOf = (e) => Math.pow(2, e);
    const slider = ui.slider({ id: 'att-dk', label: 'd_k', min: 1, max: 9, step: 1, value: S.e, format: (e) => String(dkOf(e)), onInput: (e) => { S.e = e; sample(); draw(); } });
    slider.el.querySelector('.ctl-label > span').innerHTML = '<span class="att-nocase">d<sub>k</sub></span> · key size';
    const again = ui.button({ id: 'att-dk-resample', label: 'New random draw', onClick: () => { S.seed++; sample(); draw(); } });
    fig.appendChild(el('div', { class: 'controls' }, slider.el, again));
    fig.appendChild(el('figcaption', { html: 'Each draw: 600 random query–key pairs with standard-normal entries. Madder: raw q·k. Weld: q·k/√d<sub>k</sub>. Below: softmax over the same 8 random keys for one query, without and with the scaling.' }));

    function sample() {
      const d = dkOf(S.e);
      const rnd = M.rng(S.seed * 1000 + S.e);
      const N = 600, raw = new Float64Array(N);
      for (let s = 0; s < N; s++) {
        let dot = 0;
        for (let k = 0; k < d; k++) dot += M.randn(rnd) * M.randn(rnd);
        raw[s] = dot;
      }
      // eight keys for one query
      const q = Array.from({ length: d }, () => M.randn(rnd));
      const keys = Array.from({ length: 8 }, () => Array.from({ length: d }, () => M.randn(rnd)));
      const sc = keys.map((k) => M.dot(q, k));
      const std = (arr) => { let m = 0; for (const x of arr) m += x; m /= arr.length; let v = 0; for (const x of arr) v += (x - m) * (x - m); return Math.sqrt(v / arr.length); };
      S.data = {
        d, raw, scaled: raw.map((x) => x / Math.sqrt(d)),
        sdRaw: std(raw), sdScaled: std(raw) / Math.sqrt(d),
        wRaw: M.softmax(sc), wScaled: M.softmax(sc.map((x) => x / Math.sqrt(d))),
      };
    }

    function hist(g, arr, x0, x1, y0, y1, L, bins, color, fillA) {
      const cnt = new Array(bins).fill(0);
      for (const v of arr) { const b = Math.floor(((v + L) / (2 * L)) * bins); if (b >= 0 && b < bins) cnt[b]++; }
      const mx = Math.max(...cnt);
      const bw = (x1 - x0) / bins;
      g.save();
      g.beginPath();
      g.moveTo(x0, y1);
      cnt.forEach((c, b) => { const hh = (c / mx) * (y1 - y0); g.lineTo(x0 + b * bw, y1 - hh); g.lineTo(x0 + (b + 1) * bw, y1 - hh); });
      g.lineTo(x1, y1);
      g.closePath();
      g.fillStyle = AM.rgba(color, fillA);
      g.fill();
      g.strokeStyle = color;
      g.lineWidth = 1.4;
      g.stroke();
      g.restore();
    }

    function draw() {
      const { g, w, h } = cvs;
      if (!w || !S.data) return;
      cvs.clear();
      const Dd = S.data;
      const phone = w < 420;
      const pad = 10;
      const hy0 = 30, hy1 = Math.round(h * 0.5);
      const L = 48;
      // axis
      g.strokeStyle = AM.rgba(AM.col.linen, 0.18);
      g.beginPath(); g.moveTo(pad, hy1 + 0.5); g.lineTo(w - pad, hy1 + 0.5); g.stroke();
      [-40, -20, 0, 20, 40].forEach((t) => {
        const x = pad + ((t + L) / (2 * L)) * (w - 2 * pad);
        g.fillStyle = AM.rgba(AM.col.linen, 0.25); g.fillRect(x, hy1, 1, 4);
        AM.draw.text(g, t === 0 ? '0' : fmt(t, 0), x, hy1 + 13, { size: 9, role: 'mono', color: AM.col.mist, align: 'center', baseline: 'middle' });
      });
      hist(g, Dd.raw, pad, w - pad, hy0 + 8, hy1, L, 96, AM.dye.madder, 0.16);
      hist(g, Dd.scaled, pad, w - pad, hy0 + 8, hy1, L, 96, AM.dye.weld, 0.3);
      textDk(g, `q·k   spread ${Dd.sdRaw.toFixed(2)}  (√d_k = ${Math.sqrt(Dd.d).toFixed(2)})`, pad, 10, { size: phone ? 9.5 : 10.5, role: 'mono', color: AM.dye.madder, baseline: 'middle' });
      textDk(g, `q·k / √d_k   spread ${Dd.sdScaled.toFixed(2)}`, pad, 25, { size: phone ? 9.5 : 10.5, role: 'mono', color: AM.dye.weld, baseline: 'middle' });
      // softmax rows
      const rows = [['softmax(q·k)', Dd.wRaw, AM.dye.madder], ['softmax(q·k / √d_k)', Dd.wScaled, AM.dye.weld]];
      const ry0 = hy1 + 30, rh = (h - ry0 - 8) / 2;
      rows.forEach(([lab, wv, c], r) => {
        const y = ry0 + r * rh;
        const maxW = Math.max(...wv);
        textDk(g, lab, pad, y + 8, { size: phone ? 9.5 : 10.5, role: 'mono', color: c, baseline: 'middle' });
        AM.draw.text(g, `max ${pct(maxW)}`, w - pad, y + 8, { size: phone ? 9.5 : 10.5, role: 'mono', color: AM.col.linenDim, baseline: 'middle', align: 'right' });
        const by0 = y + 18, by1 = y + rh - 6, bw = (w - 2 * pad) / 8;
        wv.forEach((p, k) => {
          const hh = p * (by1 - by0);
          g.fillStyle = AM.rgba(c, 0.15);
          g.fillRect(pad + k * bw + 3, by0, bw - 6, by1 - by0);
          g.fillStyle = c;
          g.fillRect(pad + k * bw + 3, by1 - hh, bw - 6, Math.max(1, hh));
        });
      });
    }
    sample();
    cvs.onResize(() => draw());
    draw();
  }

  // ====================================================================
  // 8. RoPE: position applied to queries and keys (moved here from chapter 3,
  //    where q and k were not yet known). Toy q and k (d_k = 8), rotated
  //    exactly with θ_i = 10000^(−2i/d_k).
  // ====================================================================
  const ROPE_BASE = 10000;
  const TAU = Math.PI * 2;
  const DR = AM.draw;
  const dotv = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
  /** Frame-rate independent smoothing toward a target (rate = speed · dt). */
  const approach = (x, target, rate) => x + (target - x) * (1 - Math.exp(-rate));
  /** Canvas text with a hand-drawn subscript (the mono face has no subscript digits); subscripts stay ≥ 8.5px. Returns the right edge. */
  function subText(g, base, subscript, x, y, o = {}) {
    const size = o.size || 10, role = o.role || 'mono', ss = Math.max(8.5, size * 0.72);
    const bw = DR.measure(g, base, size, role, o.weight);
    const sw = DR.measure(g, subscript, ss, role, o.weight);
    const x0 = o.align === 'center' ? x - (bw + sw) / 2 : o.align === 'right' ? x - bw - sw : x;
    DR.text(g, base, x0, y, { ...o, size, role, align: 'left' });
    DR.text(g, subscript, x0 + bw + 0.5, y + size * 0.3, { ...o, size: ss, role, align: 'left' });
    return x0 + bw + 0.5 + sw;
  }
  /** Dial face: soft disc, rim, optional ticks and crosshair (the clock faces of chapter 3). */
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
  /** A glowing clock hand at math angle `ang` (counter-clockwise from 3 o'clock). */
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

  function mountRope(ctx) {
    const el = ctx.el;
    const D = 8, NP = D / 2, MAXP = 30;
    const theta = (i) => Math.pow(ROPE_BASE, (-2 * i) / D); // RoPE speeds: the same clock speeds as chapter 3's sinusoids
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
    /** The attention logit before the 1/√d_k scale: RoPE(q, m) · RoPE(k, n). */
    const score = (m, n) => dotv(rope(Q, m), rope(K, n));
    const pairScore = (m, n, i) => { const q = rope(Q, m), k = rope(K, n); return q[2 * i] * k[2 * i] + q[2 * i + 1] * k[2 * i + 1]; };
    const pairNorm = (v, i) => Math.hypot(v[2 * i], v[2 * i + 1]);
    const normMax = Math.max(...[0, 1, 2, 3].flatMap((i) => [pairNorm(Q, i), pairNorm(K, i)]));
    // the score as a function of the offset alone: S(Δ) = RoPE(q, Δ)·RoPE(k, 0)
    const curve = Array.from({ length: 2 * MAXP + 1 }, (_, i) => score(i - MAXP, 0));
    const cMin = Math.min(...curve), cMax = Math.max(...curve);
    const thLabel = (t) => String(+t.toPrecision(2)); // 1, 0.1, 0.01, 0.001

    const st = { m: 7, n: 3, am: 7, an: 3, pair: 0, msg: '', dirty: true, flash: 0 };

    function geom(w) {
      const wide = w >= 440;
      const G = { w, wide };
      if (wide) {
        G.R = Math.min(w * 0.24, 132);
        G.cx = G.R + 18; G.cy = 40 + G.R; // room for the q / k labels beyond the rim
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
    const readout = el('p', { class: 'att-rope-readout' });

    const mS = AM.ui.slider({ id: 'pos-rope-m', label: 'query position m', min: 0, max: MAXP, step: 1, value: st.m, format: String, onInput: (v) => set(v, st.n, 'm') });
    const nS = AM.ui.slider({ id: 'pos-rope-n', label: 'key position n', min: 0, max: MAXP, step: 1, value: st.n, format: String, onInput: (v) => set(st.m, v, 'n') });
    mS.el.classList.add('att-lc'); nS.el.classList.add('att-lc');
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
      readout.innerHTML = `q at m = ${st.m}, k at n = ${st.n} · offset m − n = <b>${off}</b><br>q·k = ${parts} = <b>${fmt(s, 4)}</b><br><span class="att-dim">${line2}</span>`;
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
      const hx = subText(g, 'PAIR ' + i + ' · θ', String(i), 0, 11, { size: 9.5, color: AM.col.mist });
      DR.text(g, `= ${thLabel(th)} rad per position` + (G.wide ? ' · dashed: before rotation' : ''), hx + 4, 11, { size: 9.5, role: 'mono', color: AM.col.mist });
      if (!G.wide) DR.text(g, 'dashed: before rotation', 0, 26, { size: 9.5, role: 'mono', color: AM.col.mist });

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
        subText(g, 'θ', String(p), cx - 4, cy + r + 15, { size: 10.5, color: sel ? AM.dye.weld : AM.col.mist, align: 'right' });
        DR.text(g, '=' + thLabel(tp), cx - 3, cy + r + 15, { size: 10, role: 'mono', color: sel ? AM.dye.weld : AM.col.mist });
        DR.text(g, (ps >= 0 ? '+' : '') + fmt(ps, 3), cx, cy + r + 30, { size: 10.5, role: 'mono', color: AM.col.linen, align: 'center' });
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

    const controls = el('div', { class: 'controls att-rope-ctrls' }, mS.el, nS.el, el('div', { class: 'att-rope-btns' }, back, fwd), pairSeg.el);
    const fig = AM.ui.figure({
      title: 'RoPE · one dial per pair', badge: AM.ui.badge('toy', 'Exact formula'), cls: 'att-rope-fig',
      caption: 'Toy query and key (d<sub>k</sub> = 8, four pairs); the rotation is the exact RoPE formula with θ<sub>i</sub> = 10000<sup>−2i/8</sup>. The score is the raw dot product q·k, before the 1/√d<sub>k</sub> scale and softmax. The purple curve is computed from the offset alone, as RoPE(q, Δ)·RoPE(k, 0); the gold bead is computed from both positions and always lands on it.',
    }, cv.wrap, controls, readout);

    cv.onResize(() => { st.dirty = true; draw(); });
    const seen = inView(cv.wrap);
    ctx.loop((t, dt) => {
      if (!seen.on) return;
      const tm = approach(st.am, st.m, dt * 7), tn = approach(st.an, st.n, dt * 7);
      const moving = Math.abs(tm - st.m) > 1e-3 || Math.abs(tn - st.n) > 1e-3;
      st.am = moving ? tm : st.m; st.an = moving ? tn : st.n;
      if (st.flash > 0) st.flash = Math.max(0, st.flash - dt * 1.4);
      if (moving || st.dirty || st.flash > 0) { draw(); st.dirty = false; }
    });
    updateReadout();
    return { el: fig };
  }


  /** The RoPE section: prose beside the dial lab. */
  function buildRope(body, ctx) {
    const el = ctx.el;
    const rope = mountRope(ctx);
    const P = (html) => el('p', { html });
    const prose = el('div', { class: 'prose' },
      el('h3', {}, 'Position, revisited: rotate the query and key'),
      P('Chapter 3 added a position vector to every token before attention. Most open models today, including Llama, Mistral and Qwen, do something else, called <span class="term">rotary position embeddings</span> (RoPE; Su et al., 2021). Nothing is added to the token vectors. Inside every head, each query and key is split into pairs of numbers, and pair <span class="math">i</span> is turned by its position times <span class="math">θ<sub>i</sub> = 10000<sup>−2i/d<sub>k</sub></sup></span>, the clock speeds of chapter 3.'),
      P('A rotation keeps lengths. Turning the query by <span class="math">m·θ<sub>i</sub></span> and the key by <span class="math">n·θ<sub>i</sub></span> changes the angle between them by <span class="math">(m − n)·θ<sub>i</sub></span>. So the score <span class="math">q·k</span> depends on the two contents and on the offset <span class="math">m − n</span>, never on <span class="math">m</span> or <span class="math">n</span> alone. Slide both together and the score holds still.'),
      P('Position now reaches attention only as the distance between two tokens, so a head can learn a rule like “look three words back” that works anywhere in the text.'));
    body.appendChild(el('div', { class: 'grid-2 att-rope' }, prose, rope.el));
  }

  // ====================================================================
  // 9. Chapter CSS (every selector scoped to #ch-attention)
  // ====================================================================
  const CSS = `
    #ch-attention .att-intro { margin-bottom: calc(-1 * var(--space-5)); }
    #ch-attention .att-stagefig .stage-canvas canvas { border-radius: var(--radius); background: radial-gradient(120% 90% at 50% 40%, color-mix(in srgb, var(--ink-2) 92%, var(--weld)) 0%, var(--ink-2) 55%, var(--ink) 100%); border: 1px solid var(--rule); }
    #ch-attention .att-stage-cap { max-width: 60ch; }
    @media (max-width: 900px) {
      #ch-attention .att-stage-cap { display: none; }
      #ch-attention .att-stagefig { gap: 6px; }
    }
    #ch-attention .step .math.block { font-size: 0.9em; padding: 8px 10px; }
    #ch-attention .step em { color: var(--linen); }
    #ch-attention .att-nw { white-space: nowrap; }

    #ch-attention .att-bench { gap: var(--space-4); }
    #ch-attention .att-bench-intro { max-width: 70ch; }
    #ch-attention .att-controls { align-items: end; gap: var(--space-4) var(--space-6); }
    #ch-attention .att-toggles { display: flex; flex-wrap: wrap; gap: var(--space-3) var(--space-5); padding-bottom: 6px; }
    #ch-attention .att-tokrow { display: flex; align-items: center; flex-wrap: wrap; gap: var(--space-2) var(--space-4); }
    #ch-attention .att-tok-label { min-width: 3.5em; }
    #ch-attention .att-bench-grid { align-items: start; }
    #ch-attention .att-cv-box { display: grid; gap: 6px; min-width: 0; }
    #ch-attention .att-cv-box canvas { border-radius: var(--radius); background: var(--ink-2); border: 1px solid var(--rule); }
    #ch-attention .att-cv-title { font-family: var(--font-mono); font-size: var(--fs-micro); letter-spacing: 0.1em; text-transform: uppercase; color: var(--mist); }

    #ch-attention .att-readout { display: grid; gap: var(--space-4); }
    #ch-attention .att-readout > *, #ch-attention .att-bench > * { min-width: 0; }
    #ch-attention .att-table { min-width: 0; max-width: 100%; }
    #ch-attention .att-rq-row { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: baseline; gap: var(--space-3) var(--space-5); }
    #ch-attention .att-rq { display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 12px; }
    #ch-attention .att-qword { font-family: var(--font-display); font-style: italic; font-size: 1.9rem; line-height: 1; color: var(--weld); }
    #ch-attention .att-qq { font-style: italic; color: var(--linen-dim); }
    #ch-attention .att-stats { display: flex; flex-wrap: wrap; gap: 4px 18px; font-family: var(--font-mono); font-size: var(--fs-micro); letter-spacing: 0.06em; text-transform: uppercase; color: var(--mist); }
    #ch-attention .att-stats b { color: var(--weld); font-weight: 500; text-transform: none; letter-spacing: 0; }
    #ch-attention .att-table .matrix .mx-cell { min-width: 46px; }
    #ch-attention .att-table.is-narrow .matrix .mx-cell { min-width: 52px; height: 24px; }
    #ch-attention .att-table.is-narrow .matrix .mx-head { height: 24px; }
    #ch-attention .att-out { display: flex; flex-wrap: wrap; align-items: center; gap: var(--space-3) var(--space-5); }
    #ch-attention .att-out-eq { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 12px; }
    #ch-attention .att-out-eq { min-width: 0; }
    #ch-attention .att-out-eq .math { white-space: normal; overflow-wrap: anywhere; min-width: 0; font-size: 0.8rem; color: var(--verdigris); }
    #ch-attention .matrix { width: max-content; }
    #ch-attention .att-out-dims { display: flex; flex-wrap: wrap; gap: 4px 14px; font-family: var(--font-mono); font-size: 10.5px; color: var(--mist); }
    #ch-attention .att-out-dims b { color: var(--verdigris); font-weight: 500; }

    #ch-attention .att-mats { gap: var(--space-4); }
    #ch-attention .att-mat { display: grid; gap: 6px; min-width: 0; align-content: start; }
    #ch-attention .att-mat .matrix .mx-cell { min-width: 42px; height: 22px; }
    #ch-attention .att-mat .matrix .mx-head { height: 22px; }
    #ch-attention .att-mats .is-q .fig-title { color: var(--weld); }
    #ch-attention .att-mats .is-k .fig-title { color: var(--woad); }
    #ch-attention .att-mats .is-v .fig-title { color: var(--verdigris); }
    #ch-attention .att-mats .is-q .mx-cell.is-hot { outline-color: var(--weld); background: color-mix(in srgb, var(--weld) 22%, var(--ink-3)) !important; }
    #ch-attention .att-mats .is-k .mx-cell.is-hot { outline-color: var(--woad); background: color-mix(in srgb, var(--woad) 22%, var(--ink-3)) !important; }
    #ch-attention .att-mats .is-v .mx-cell.is-hot { outline-color: var(--verdigris); background: color-mix(in srgb, var(--verdigris) 22%, var(--ink-3)) !important; }

    #ch-attention .att-weights { border: 1px solid var(--rule); border-radius: var(--radius); padding: var(--space-3) var(--space-4); background: color-mix(in srgb, var(--ink-2) 60%, transparent); }
    #ch-attention .att-weights summary { cursor: pointer; font-family: var(--font-mono); font-size: var(--fs-micro); letter-spacing: 0.1em; text-transform: uppercase; color: var(--linen-dim); }
    #ch-attention .att-weights summary:hover { color: var(--weld); }
    #ch-attention .att-weights[open] summary { margin-bottom: var(--space-4); }
    #ch-attention .att-wgrid { display: flex; flex-wrap: wrap; gap: var(--space-5); }
    #ch-attention .att-wgrid .matrix .mx-cell { min-width: 34px; height: 22px; }
    #ch-attention .att-wgrid .matrix .mx-head { height: 22px; min-width: 30px; }
    #ch-attention .att-weights .caption { margin-top: var(--space-3); }

    #ch-attention .att-why { align-items: center; }
    #ch-attention .att-nocase { text-transform: none; letter-spacing: 0.02em; }
    #ch-attention .fig-title sub, #ch-attention .toggle sub, #ch-attention .ctl-label sub, #ch-attention .mx-head sub { font-size: 0.8em; line-height: 0; }
    #ch-attention .att-outbars { width: 120px; height: 34px; }
    #ch-attention .att-why canvas { border-radius: var(--radius); background: var(--ink-2); border: 1px solid var(--rule); }

    #ch-attention .att-rope { align-items: start; }
    #ch-attention .att-rope .prose { padding-top: var(--space-2); }
    #ch-attention .att-rope .prose > * { min-width: 0; }
    #ch-attention .att-rope-readout { margin: 0; font-family: var(--font-mono); font-size: 11.5px; line-height: 1.7; color: var(--linen-dim); min-height: 3.4em; font-variant-numeric: tabular-nums; }
    #ch-attention .att-rope-readout b { color: var(--weld); font-weight: 500; }
    #ch-attention .att-dim { color: var(--mist); }
    #ch-attention .att-rope-ctrls { gap: var(--space-4) var(--space-5); align-items: end; }
    #ch-attention .att-rope-ctrls .ctl-range { width: min(220px, 100%); }
    #ch-attention .att-rope-btns { display: flex; gap: var(--space-2); flex-wrap: wrap; }
    #ch-attention .att-lc .ctl-label { text-transform: none; letter-spacing: 0.06em; }
    @media (max-width: 520px) {
      #ch-attention .att-rope-ctrls { gap: var(--space-4); }
      #ch-attention .att-rope-ctrls .ctl-range { width: 100%; }
      #ch-attention .att-rope-readout { font-size: 10.5px; }
    }
  `;
})();
