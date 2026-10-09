/* Chapter 05 — Multi-head attention: "The Prism".

   Two figures, both driven by the real tiny transformer (tinyworld) that ships
   with the page:
   1. The prism (scrollytelling stage). One token's residual thread enters a
      glass prism and splits into four dyed beams, one per head of layer 0.
      Each beam fans out into that head's real attention threads, the values
      flow back, the four 16-number outputs are concatenated and pass back
      through the prism (W_O) into the residual stream. The last step shows the
      real W_Q of layer 0 as cloth, cut into four bands.
   2. The head atlas (free play). All 3 × 4 heads as small woven heatmaps on a
      chosen sentence; click one to lay its threads on the loom. Heads can be
      muted (their output zeroed before W_O); the model is then rerun by a
      small forward pass in this file that reproduces AM.model to ~1e-5.

   Head names follow docs/model-notes.md (window.AM_NOTES when it is loaded,
   else the same names kept here). Every number on screen is computed from the
   model; the "test set" figures are means over tools/fixtures/tinyworld-test.json
   (4,000 sequences), from tools/analyze-tinyworld.mjs and our own runs of the
   same set, all re-checked against the shipped weights. */
(() => {
  const ID = 'heads';
  const M = AM.math;
  const D = AM.draw;
  const clamp = M.clamp;
  const ease = M.ease;
  const DYE_KEYS = ['weld', 'woad', 'madder', 'verdigris', 'cochineal', 'saffron', 'lichen']; // order of AM.dyeList
  const hc = (h) => AM.headColor(h);
  const hdye = (h) => DYE_KEYS[h % DYE_KEYS.length];
  const now = () => performance.now() / 1000;
  const SUP = '⁰¹²³⁴⁵⁶⁷⁸⁹';
  const sup = (n) => String(n).replace(/\d/g, (c) => SUP[+c]);

  // ====================================================================
  // 1. Small helpers
  // ====================================================================
  /** Weight with two decimals; 0.995+ prints as 1.00. */
  const f2 = (w) => (w >= 0.995 ? '1.00' : w < 0.005 ? '0.00' : w.toFixed(2));
  const pct = (p) => (p >= 0.995 ? '100%' : p < 0.005 ? (p < 0.0005 ? '0%' : '<1%') : Math.round(p * 100) + '%');
  const q = (s) => `“${s}”`;
  const ORD = ['1st', '2nd', '3rd', '4th', '5th', '6th'];
  const SUBD = '₀₁₂₃₄₅₆₇₈₉';
  /** Compact form for tiles: “the₂” for the second “the”. */
  const tokTiny = (toks, j) => {
    if (toks.filter((t) => t === toks[j]).length < 2) return toks[j];
    const k = toks.slice(0, j + 1).filter((t) => t === toks[j]).length;
    return toks[j] + String(k).replace(/\d/g, (c) => SUBD[+c]);
  };
  /** A word, or “the (2nd)” when the word repeats in the sentence. */
  const tokName = (toks, j) => {
    if (toks.filter((t) => t === toks[j]).length < 2) return toks[j];
    const k = toks.slice(0, j + 1).filter((t) => t === toks[j]).length;
    return `${toks[j]} (${ORD[k - 1] || k + 'th'})`;
  };
  /** Speaker with a cross: the phone-sized mute control. */
  const MUTE_ICON = '<svg viewBox="0 0 16 16" width="13" height="13"><path d="M2 6h2.6L8.4 3v10L4.6 10H2z" fill="currentColor"/><path d="M10.6 6.1l3.8 3.8M14.4 6.1l-3.8 3.8" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" fill="none"/></svg>';
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

  /** Cubic control points for a thread arching upward from x1 to x2 at height y; a loop when x1 ≈ x2. */
  function arcPts(x1, x2, y, lift) {
    if (Math.abs(x2 - x1) < 0.5) {
      const r = lift;
      return [x1 - 1, y, x1 - r * 1.1, y - r * 1.8, x1 + r * 1.1, y - r * 1.8, x1 + 1, y];
    }
    const dx = x2 - x1;
    return [x1, y, x1 + dx * 0.08, y - lift, x1 + dx * 0.92, y - lift, x2, y];
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
  const reverseBez = (P) => [P[6], P[7], P[4], P[5], P[2], P[3], P[0], P[1]];
  /** S-curve leaving and arriving horizontally. */
  const sCurve = (x0, y0, x3, y3, k = 0.5) => { const dx = x3 - x0; return [x0, y0, x0 + dx * k, y0, x3 - dx * k, y3, x3, y3]; };

  /** A silk thread: soft glow, dyed core and a thin bright sheen. */
  function silk(g, P, { color = AM.dye.weld, width = 2, alpha = 1, f = 1, sheen = true, glow = 1 } = {}) {
    if (alpha <= 0.004 || f <= 0.001) return;
    const Q = bezPart(P, f);
    g.save();
    g.lineCap = 'round';
    g.beginPath(); g.moveTo(Q[0], Q[1]); g.bezierCurveTo(Q[2], Q[3], Q[4], Q[5], Q[6], Q[7]);
    if (glow > 0) {
      g.strokeStyle = AM.rgba(color, 0.12 * alpha * glow);
      g.lineWidth = width * 4 + 2;
      g.stroke();
    }
    g.strokeStyle = AM.rgba(color, alpha);
    g.lineWidth = width;
    g.stroke();
    if (sheen && width > 1.5) {
      g.strokeStyle = AM.rgba('#fff4d6', 0.33 * alpha);
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
    if ('letterSpacing' in g && o.letterSpacing) g.letterSpacing = o.letterSpacing;
    g.lineJoin = 'round';
    if (o.halo !== false) {
      g.strokeStyle = o.halo || AM.rgba(AM.col.ink, 0.92);
      g.lineWidth = o.haloW || 4;
      g.strokeText(str, x, y);
    }
    g.fillStyle = o.color || AM.col.linen;
    g.fillText(str, x, y);
    g.restore();
  }
  /** Is this element near the viewport? Lets a long chapter redraw only what can be seen. */
  function inView(node, margin = '120px') {
    const s = { on: true };
    if (typeof IntersectionObserver !== 'undefined') {
      new IntersectionObserver((es) => { for (const e of es) s.on = e.isIntersecting; }, { rootMargin: `${margin} 0px ${margin} 0px` }).observe(node);
    }
    return s;
  }
  /** Pre-rendered glow sprite per colour (particles are drawn with drawImage, which is cheap). */
  const sprites = new Map();
  function sprite(hex) {
    if (sprites.has(hex)) return sprites.get(hex);
    const S = 32, c = document.createElement('canvas');
    c.width = c.height = S;
    const g = c.getContext('2d'), r = S / 2;
    const grd = g.createRadialGradient(r, r, 0, r, r, r);
    grd.addColorStop(0, 'rgba(255,250,236,1)');
    grd.addColorStop(0.16, AM.rgba(hex, 1));
    grd.addColorStop(0.4, AM.rgba(hex, 0.35));
    grd.addColorStop(1, AM.rgba(hex, 0));
    g.fillStyle = grd;
    g.fillRect(0, 0, S, S);
    sprites.set(hex, c);
    return c;
  }
  /** A vector as a strip of bars around a midline (the chapter's "thread barcode"). */
  function strip(g, x, y, w, h, vec, { color = AM.col.linen, neg, alpha = 1, frame = true, max } = {}) {
    if (alpha <= 0.004) return;
    const n = vec.length;
    let m = max || 0;
    if (!m) for (let i = 0; i < n; i++) m = Math.max(m, Math.abs(vec[i]));
    m = m || 1;
    g.save();
    g.globalAlpha *= alpha;
    if (frame) {
      D.roundRect(g, x - 2, y - 2, w + 4, h + 4, 3);
      g.fillStyle = AM.rgba(AM.col.ink, 0.82);
      g.fill();
      g.strokeStyle = AM.rgba(color, 0.32);
      g.lineWidth = 1;
      g.stroke();
    }
    const bw = w / n, mid = y + h / 2;
    g.fillStyle = AM.rgba(AM.col.linen, 0.12);
    g.fillRect(x, mid - 0.5, w, 1);
    const base = g.globalAlpha;
    for (let i = 0; i < n; i++) {
      const t = clamp(vec[i] / m, -1, 1);
      const hh = Math.abs(t) * (h / 2 - 0.5);
      g.globalAlpha = base * (0.5 + 0.5 * Math.min(1, Math.abs(t) * 1.3));
      g.fillStyle = t >= 0 ? color : (neg || color);
      g.fillRect(x + i * bw + bw * 0.12, t >= 0 ? mid - hh : mid, Math.max(0.8, bw * 0.76), Math.max(0.7, hh));
    }
    g.restore();
  }
  let _mctx = null;
  /** A detached 2D context for measuring text before a canvas exists. */
  const measureCtx = () => (_mctx || (_mctx = document.createElement('canvas').getContext('2d')));
  function levenshtein(a, b) {
    const m = a.length, n = b.length;
    if (!m) return n; if (!n) return m;
    let prev = Array.from({ length: n + 1 }, (_, j) => j);
    for (let i = 1; i <= m; i++) {
      const cur = [i];
      for (let j = 1; j <= n; j++) cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
      prev = cur;
    }
    return prev[n];
  }

  // ====================================================================
  // 2. What we found the twelve heads doing (verified against the model)
  // ====================================================================
  // name: as in docs/model-notes.md. role: what our measurements show. heldout:
  // test-set averages (meta.analysis where it has them, else our own runs of the
  // same 4,000 sequences). Live evidence is computed on screen.
  const HEADS = {
    L0H0: { name: 'the backstitch', role: 'A selective look-back. At “is” in “the box is” it reads “box”, the noun being asked about; from “door” in “the keys near the old door” it reads “near”. Muting it breaks agreement and many colour questions.', heldout: ['“is” → the noun just before it', 0.89, '1,004 colour questions'] },
    L0H1: { name: 'the loose weave', role: 'No crisp pattern: it spreads its attention more evenly than any other head in layer 0. Yet the parrot leans on it: muted, the parrot’s right next word falls to 69% on average (400 parrot words from the test set).', heldout: null },
    L0H2: { name: 'the fact finder', role: 'Finds the key word further back: the country or city in a capital question, the animal before “says”, and the first name in a copying sentence.', heldout: ['capital question → the key word', 0.86, '469 sentences', 'capital', 0, 2] },
    L0H3: { name: 'the antecedent finder', role: 'At “because” and “with” it jumps back to the subject: the antecedent, the word a pronoun will refer to.', heldout: ['“because” → the subject', 0.95, '1,018 pronoun sentences', 'pronoun', 0, 3] },
    L1H0: { name: 'the second colour reader', role: 'At “is” in a colour question it spreads over all the colours, the right one and the wrong ones alike. Muting it changes nothing we measured.', heldout: ['“is” → every colour word (right 0.26, wrong 0.30)', 0.56, '1,004 colour questions'] },
    L1H1: { name: 'the clause hopper', role: 'Hops back to the previous clause: at “thanked” it finds the earlier verb “gave”, and at “the box is” it finds the full stop before it.', heldout: ['at “thanked” / “waved at” → the earlier verb', 0.86, '749 copying sentences'] },
    L1H2: { name: 'the head-noun tracker', role: 'Looks back to the noun a phrase is about: in “the keys near the old door” it looks from “door” back to “keys”. In copying sentences it finds the other name, though other heads carry the name too.', heldout: ['attractor noun → the head noun', 0.87, '899 agreement sentences', 'agreement', 1, 2] },
    L1H3: { name: 'the colour binder', role: 'At “is” it finds the colour that went with the asked-about noun. In parrot sentences it points at the next colour to repeat, a little like an induction head.', heldout: ['“is” → the right colour (wrong colours get 0.19)', 0.75, '1,004 colour questions', 'binding→colour', 1, 3] },
    L2H0: { name: 'the idle shuttle', role: 'Often rests on the first word, and muting it changes nothing we measured. One habit stands out: after “alice and bob … because” it looks at the plural “and”.', heldout: ['“because” → the “and” in “x and y”', 0.80, '61 “x and y … because” sentences'] },
    L2H1: { name: 'the verb echo', role: 'Echoes the earlier verb in copying sentences, as the clause hopper does one layer earlier. It also parks more attention on the first word than any other head.', heldout: ['at “thanked” / “waved at” → the earlier verb', 0.56, '749 copying sentences'] },
    L2H2: { name: 'the faint echo', role: 'Weaker copies of other heads’ jobs: the subject at “because”, the head noun in agreement.', heldout: ['“because” → the subject', 0.31, '1,018 pronoun sentences', 'pronoun', 2, 2] },
    L2H3: { name: 'the idle shuttle II', role: 'Often resting too, and just as unmissed. It does glance at the plural “and”, and at the animal before “says”.', heldout: ['“because” → the “and” in “x and y”', 0.72, '61 “x and y … because” sentences'] },
  };
  /** Head name, preferring the interpretability notes (AM_NOTES) when they are loaded. */
  function headName(l, h) {
    const key = `L${l}H${h}`;
    try {
      const N = window.AM_NOTES && window.AM_NOTES.heads;
      if (N) {
        let e = null;
        if (Array.isArray(N)) e = Array.isArray(N[l]) ? N[l][h] : N.find((x) => x && (x.id === key || x.key === key || (x.layer === l && x.head === h)));
        else e = N[key] || (N[l] && N[l][h]);
        const nm = e && (typeof e === 'string' ? e : (e.name || e.label || e.title));
        if (typeof nm === 'string' && nm.length > 1 && nm.length < 42) return nm;
      }
    } catch (_) { /* fall back to our own copy of the same names */ }
    return HEADS[key].name;
  }
  /** The same name as a title: “The backstitch”. */
  const titleName = (l, h) => { const n = headName(l, h); return n.charAt(0).toUpperCase() + n.slice(1); };
  /** Held-out figure for a head: [label, value, n]; meta.analysis overrides where it has the number. */
  function heldOut(m, l, h) {
    const H = HEADS[`L${l}H${h}`].heldout;
    if (!H) return null;
    let v = H[1];
    try {
      const A = m.meta && m.meta.analysis && m.meta.analysis.heads;
      if (H[3] && A && A[H[3]]) {
        const hit = A[H[3]].find((x) => x.layer === H[4] && x.head === H[5]);
        if (hit && typeof hit.attn === 'number') v = hit.attn;
      }
    } catch (_) { /* keep ours */ }
    return [H[0], v, H[2]];
  }

  // Curated sentences: each opens the atlas on the head with the clearest pattern for it (all
  // verified with the shipped weights; for “they” no single head is needed, see the mute test).
  const PRESETS = [
    { label: 'Pronoun · queen', text: 'the queen opened the door because', head: [0, 3], ans: 'she', short: 'the queen … because' },
    { label: 'Pronoun · king', text: 'the king opened the door because', head: [0, 3], ans: 'he', short: 'the king … because' },
    { label: 'Plural · they', text: 'alice and bob walked to the park and closed the box because', head: [2, 0], ans: 'they', short: 'alice and bob … because' },
    { label: 'Agreement · keys', text: 'the keys near the old door', head: [1, 2], ans: 'are', short: 'the keys near the old door' },
    { label: 'Agreement · key', text: 'the key near the old doors', head: [1, 2], ans: 'is', short: 'the key near the old doors' },
    { label: 'Colour · 2 things', text: 'the red ball and the blue box . the box is', head: [1, 3], ans: 'blue', short: 'red ball, blue box … box is' },
    { label: 'Colour · 3 things', text: 'the green hat , the white door and the pink cup . the hat is', head: [1, 3], ans: 'green', short: 'green hat, white door … hat is' },
    { label: 'Copy a name', text: 'alice gave bob a cup . bob thanked', head: [1, 2], ans: 'alice', short: 'alice gave bob … bob thanked' },
    { label: 'Parrot', text: 'the girl said red kite blue cup . the parrot said', head: [1, 3], ans: 'red', short: 'girl said red kite … parrot said' },
    { label: 'Capital', text: 'the capital of japan is', head: [0, 2], ans: 'tokyo', short: 'the capital of japan is' },
  ];
  const STAGE_TEXT = 'the queen opened the door because';
  const STAGE_LAYER = 0;
  const VOCAB_GROUPS = [
    ['people', 'king queen boy girl prince princess wizard witch children'],
    ['names', 'alice emma lucy rose bob tom sam jack'],
    ['animals', 'dog cat duck owl pig cow lion sheep dogs cats parrot'],
    ['things', 'key keys box boxes cup cups door doors book books hat hats ring ball kite crown'],
    ['colours', 'red blue green yellow white pink'],
    ['describing', 'cold tired happy sad hungry sleepy old small gold heavy broken shiny'],
    ['doing', 'walked loved gave thanked met waved helped said opened closed found lost dropped says'],
    ['little words', 'the a and because with of is are was were to at near under behind he she it they his her its their . ,'],
    ['places', 'park river market castle garden capital france japan italy spain egypt peru china kenya paris tokyo rome madrid cairo lima beijing nairobi'],
    ['sounds & facts', 'moo woof meow quack baa hoot oink roar sky grass snow sun apple'],
  ];

  // ====================================================================
  // 3. A forward pass with muting (zero-ablation of chosen heads)
  // ====================================================================
  // Same maths as js/model/transformer.js: pre-LN blocks, causal softmax(q·k/√d_head),
  // tanh-GELU, ε = 1e-5. A muted head still computes its pattern; its output is
  // zeroed before W_O. Verified at mount against AM.model.run (parity < 1e-3 or disabled).
  function makeEngine(m) {
    try {
      const P = m && m._net && m._net.params;
      if (!P) return null;
      const C = m.config, d = C.d_model, H = C.n_head, dh = d / H, NL = C.n_layer, V = C.vocab_size, F = C.d_ff;
      const get = (n, len) => { const p = P[n]; if (!p || !p.data || p.data.length !== len) throw new Error('weight ' + n); return p.data; };
      const wte = get('wte', V * d), wpe = get('wpe', C.n_ctx * d);
      const Ls = [];
      for (let l = 0; l < NL; l++) {
        const p = `h.${l}.`;
        Ls.push({
          g1: get(p + 'ln1.g', d), b1: get(p + 'ln1.b', d), wqkv: get(p + 'attn.wqkv', d * 3 * d), bqkv: get(p + 'attn.bqkv', 3 * d),
          wo: get(p + 'attn.wo', d * d), bo: get(p + 'attn.bo', d), g2: get(p + 'ln2.g', d), b2: get(p + 'ln2.b', d),
          wfc: get(p + 'mlp.wfc', d * F), bfc: get(p + 'mlp.bfc', F), wproj: get(p + 'mlp.wproj', F * d), bproj: get(p + 'mlp.bproj', d),
        });
      }
      const lnfg = get('lnf.g', d), lnfb = get('lnf.b', d), wout = get('wout', d * V), bout = get('bout', V);
      const GC = Math.sqrt(2 / Math.PI), scale = 1 / Math.sqrt(dh);
      const hb = new Float64Array(d), hid = new Float64Array(F);
      const ln = (X, o, gn, bn) => {
        let mu = 0; for (let j = 0; j < d; j++) mu += X[o + j]; mu /= d;
        let va = 0; for (let j = 0; j < d; j++) { const z = X[o + j] - mu; va += z * z; } va /= d;
        const r = 1 / Math.sqrt(va + 1e-5);
        for (let j = 0; j < d; j++) hb[j] = (X[o + j] - mu) * r * gn[j] + bn[j];
      };
      /** y[yo..] (+)= x·W + b for x of length n, W [n × mm] row-major. */
      const lin = (x, n, W, b, mm, Y, yo, add) => {
        if (!add) for (let j = 0; j < mm; j++) Y[yo + j] = b[j]; else for (let j = 0; j < mm; j++) Y[yo + j] += b[j];
        for (let i = 0; i < n; i++) { const xi = x[i]; if (xi === 0) continue; const o = i * mm; for (let j = 0; j < mm; j++) Y[yo + j] += xi * W[o + j]; }
      };
      function run(ids, mute) {
        const T = ids.length;
        const X = new Float64Array(T * d);
        for (let t = 0; t < T; t++) for (let j = 0; j < d; j++) X[t * d + j] = wte[ids[t] * d + j] + wpe[t * d + j];
        const qkv = new Float64Array(T * 3 * d), cat = new Float64Array(T * d), catRow = new Float64Array(d);
        const attn = [];
        for (let l = 0; l < NL; l++) {
          const Ly = Ls[l];
          for (let t = 0; t < T; t++) { ln(X, t * d, Ly.g1, Ly.b1); lin(hb, d, Ly.wqkv, Ly.bqkv, 3 * d, qkv, t * 3 * d, false); }
          cat.fill(0);
          const A = [];
          for (let h = 0; h < H; h++) {
            const muted = mute && mute.has(l * H + h);
            const rows = [];
            for (let i = 0; i < T; i++) {
              const row = new Float32Array(T);
              const qo = i * 3 * d + h * dh;
              let mx = -Infinity;
              const s = new Float64Array(i + 1);
              for (let j = 0; j <= i; j++) {
                const ko = j * 3 * d + d + h * dh;
                let v = 0; for (let c = 0; c < dh; c++) v += qkv[qo + c] * qkv[ko + c];
                v *= scale; s[j] = v; if (v > mx) mx = v;
              }
              let z = 0; for (let j = 0; j <= i; j++) { s[j] = Math.exp(s[j] - mx); z += s[j]; }
              for (let j = 0; j <= i; j++) {
                const a = s[j] / z; row[j] = a;
                if (!muted) { const vo = j * 3 * d + 2 * d + h * dh, co = i * d + h * dh; for (let c = 0; c < dh; c++) cat[co + c] += a * qkv[vo + c]; }
              }
              rows.push(row);
            }
            A.push(rows);
          }
          attn.push(A);
          for (let t = 0; t < T; t++) { for (let j = 0; j < d; j++) catRow[j] = cat[t * d + j]; lin(catRow, d, Ly.wo, Ly.bo, d, X, t * d, true); }
          for (let t = 0; t < T; t++) {
            ln(X, t * d, Ly.g2, Ly.b2);
            lin(hb, d, Ly.wfc, Ly.bfc, F, hid, 0, false);
            for (let k = 0; k < F; k++) { const v = hid[k]; hid[k] = 0.5 * v * (1 + Math.tanh(GC * (v + 0.044715 * v * v * v))); }
            lin(hid, F, Ly.wproj, Ly.bproj, d, X, t * d, true);
          }
        }
        const probs = [], lg = new Float64Array(V);
        for (let t = 0; t < T; t++) {
          ln(X, t * d, lnfg, lnfb);
          lin(hb, d, wout, bout, V, lg, 0, false);
          let mx = -Infinity; for (let k = 0; k < V; k++) if (lg[k] > mx) mx = lg[k];
          let z = 0; const p = new Float32Array(V);
          for (let k = 0; k < V; k++) { p[k] = Math.exp(lg[k] - mx); z += p[k]; }
          for (let k = 0; k < V; k++) p[k] /= z;
          probs.push(p);
        }
        return { attn, probs };
      }
      return { run, H, NL, dh, d, wo: Ls.map((x) => x.wo), wqkv: Ls.map((x) => x.wqkv) };
    } catch (e) {
      return null;
    }
  }

  /** Everything the prism stage shows, computed from one forward pass. */
  function stageData(m, eng) {
    const enc = m.encode(STAGE_TEXT);
    const r = m.run(enc.ids, { capture: true });
    const H = m.config.n_head, d = m.config.d_model, dh = d / H, l = STAGE_LAYER;
    const t = enc.ids.length - 1;
    const attn = [], qv = [], ov = [], wn = [];
    for (let h = 0; h < H; h++) {
      attn.push(Array.from(r.attn[l][h][t]));
      qv.push(Array.from(r.q[l][h][t]));
      const o = new Array(dh).fill(0);
      for (let j = 0; j <= t; j++) { const a = r.attn[l][h][t][j]; const v = r.v[l][h][j]; for (let c = 0; c < dh; c++) o[c] += a * v[c]; }
      ov.push(o);
      if (eng) {
        const W = eng.wo[l];
        const c = new Array(d).fill(0);
        for (let i = 0; i < dh; i++) { const oi = o[i], row = (h * dh + i) * d; for (let k = 0; k < d; k++) c[k] += oi * W[row + k]; }
        wn.push(Math.hypot(...c));
      }
    }
    const x = Array.from(r.resid[l][t]);
    const write = Array.from(r.residMid[l][t], (v, k) => v - r.resid[l][t][k]);
    let WQ = null;
    if (eng) {
      const src = eng.wqkv[l];
      WQ = [];
      for (let i = 0; i < d; i++) WQ.push(Array.from(src.subarray(i * 3 * d, i * 3 * d + d)));
    }
    return { toks: r.tokens, t, attn, qv, ov, wn: eng ? wn : null, x, write, WQ, H, d, dh, pred: m.topk(r.probs[t], 3), layer: l };
  }

  // ====================================================================
  // 4. Chapter CSS
  // ====================================================================
  const CSS = `
    #ch-${ID} .hd-intro { margin-bottom: calc(-1 * var(--space-5)); }
    #ch-${ID} .hd-stagefig .stage-canvas canvas {
      border-radius: var(--radius);
      border: 1px solid var(--rule);
      background:
        radial-gradient(90% 70% at 78% 46%, color-mix(in srgb, var(--woad) 9%, transparent) 0%, transparent 60%),
        radial-gradient(120% 90% at 40% 40%, color-mix(in srgb, var(--ink-2) 94%, var(--weld)) 0%, var(--ink-2) 55%, var(--ink) 100%);
    }
    #ch-${ID} .hd-stage-cap { max-width: 62ch; }
    @media (max-width: 520px) { #ch-${ID} .hd-long { display: none; } }
    @media (max-width: 900px) {
      #ch-${ID} .hd-stage-cap { display: none; }
      #ch-${ID} .hd-stagefig { gap: 6px; }
      #ch-${ID} .hd-stagefig .fig-title { font-size: 10px; }
    }
    #ch-${ID} .step .math.block { font-size: 0.86em; padding: 8px 10px; }
    #ch-${ID} .step em { color: var(--linen); }
    #ch-${ID} .hd-nw { white-space: nowrap; }
    #ch-${ID} .hd-nocase { text-transform: none; letter-spacing: 0.02em; }
    #ch-${ID} sub, #ch-${ID} sup { line-height: 0; }

    #ch-${ID} .hd-sec { display: grid; gap: var(--space-5); }
    #ch-${ID} .hd-sec > .prose { max-width: 68ch; }
    #ch-${ID} .hd-kicker { font-family: var(--font-mono); font-size: var(--fs-micro); letter-spacing: 0.16em; text-transform: uppercase; color: var(--weld); }

    /* atlas */
    #ch-${ID} .hd-atlas { gap: var(--space-4); }
    #ch-${ID} .hd-presets { display: flex; flex-wrap: wrap; gap: 6px; }
    #ch-${ID} .hd-chip {
      border: 1px solid var(--rule-strong); background: var(--ink-2); color: var(--linen-dim);
      border-radius: 999px; padding: 5px 12px; font-family: var(--font-mono); font-size: 10.5px; letter-spacing: 0.06em;
      cursor: pointer; transition: border-color 0.15s, color 0.15s, background 0.15s;
    }
    #ch-${ID} .hd-chip:hover { border-color: var(--linen-dim); color: var(--linen); }
    #ch-${ID} .hd-chip[aria-pressed='true'] { border-color: var(--weld); color: var(--weld); background: color-mix(in srgb, var(--weld) 10%, var(--ink-2)); }
    #ch-${ID} .hd-input-row { display: flex; gap: var(--space-3); align-items: stretch; flex-wrap: wrap; }
    #ch-${ID} .hd-input-row .text-input { flex: 1 1 260px; min-width: 0; }
    #ch-${ID} .hd-msg { font-family: var(--font-mono); font-size: 11px; line-height: 1.6; color: var(--mist); display: flex; flex-wrap: wrap; gap: 4px 8px; align-items: center; min-height: 1.6em; }
    #ch-${ID} .hd-msg .bad { color: var(--madder); border: 1px dashed color-mix(in srgb, var(--madder) 60%, transparent); border-radius: 5px; padding: 0 6px; }
    #ch-${ID} .hd-msg .fix { border: 1px solid var(--rule-strong); background: var(--ink-2); color: var(--linen); border-radius: 5px; padding: 1px 7px; cursor: pointer; font: inherit; }
    #ch-${ID} .hd-msg .fix:hover { border-color: var(--weld); color: var(--weld); }
    #ch-${ID} .hd-vocab { border: 1px solid var(--rule); border-radius: var(--radius-sm); padding: 6px 12px; background: color-mix(in srgb, var(--ink-2) 60%, transparent); }
    #ch-${ID} .hd-vocab summary { cursor: pointer; font-family: var(--font-mono); font-size: var(--fs-micro); letter-spacing: 0.1em; text-transform: uppercase; color: var(--linen-dim); padding: 3px 0; }
    #ch-${ID} .hd-vocab summary:hover { color: var(--weld); }
    #ch-${ID} .hd-vgroups { display: grid; gap: 8px; padding: 8px 0 6px; }
    #ch-${ID} .hd-vgroup { display: flex; flex-wrap: wrap; gap: 4px; align-items: baseline; }
    #ch-${ID} .hd-vgroup b { font-family: var(--font-mono); font-weight: 400; font-size: 10px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--mist); min-width: 7.5em; }
    #ch-${ID} .hd-vw { border: 1px solid var(--rule); background: var(--ink); color: var(--linen-dim); border-radius: 5px; padding: 1px 6px; font-size: 12.5px; cursor: pointer; }
    #ch-${ID} .hd-vw:hover { border-color: var(--weld); color: var(--linen); }

    #ch-${ID} .hd-board { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1.18fr); gap: var(--space-5); align-items: start; }
    #ch-${ID} .hd-cells { display: grid; grid-template-columns: auto repeat(4, minmax(0, 1fr)); gap: 8px 8px; align-items: center; }
    #ch-${ID} .hd-colh, #ch-${ID} .hd-rowh { font-family: var(--font-mono); font-size: 9.5px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--mist); text-align: center; }
    #ch-${ID} .hd-rowh { writing-mode: vertical-rl; transform: rotate(180deg); padding: 0 2px; }
    #ch-${ID} .hd-cell { position: relative; min-width: 0; }
    #ch-${ID} .hd-pick {
      display: grid; gap: 3px; width: 100%; padding: 5px 5px 6px; border-radius: 9px; cursor: pointer; text-align: left;
      border: 1px solid var(--rule); background: var(--ink); color: var(--linen); transition: border-color 0.15s, box-shadow 0.15s, background 0.15s;
    }
    #ch-${ID} .hd-pick:hover { border-color: color-mix(in srgb, var(--hc) 60%, var(--rule)); }
    #ch-${ID} .hd-pick[aria-pressed='true'] { border-color: var(--hc); background: color-mix(in srgb, var(--hc) 8%, var(--ink)); box-shadow: 0 0 0 1px var(--hc), 0 0 22px -8px var(--hc); }
    #ch-${ID} .hd-pick .stage-canvas canvas { border-radius: 5px; }
    #ch-${ID} .hd-pname { font-size: 11px; font-weight: 600; line-height: 1.2; color: var(--linen); overflow: hidden; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; min-height: 2.4em; }
    #ch-${ID} .hd-pev { font-family: var(--font-mono); font-size: 9.5px; color: var(--hc); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    #ch-${ID} .hd-mute {
      position: absolute; top: 9px; right: 9px; z-index: 1; border: 1px solid var(--rule-strong); background: color-mix(in srgb, var(--ink) 85%, transparent);
      color: var(--mist); border-radius: 999px; padding: 1px 7px 2px; font-family: var(--font-mono); font-size: 8.5px; letter-spacing: 0.08em; text-transform: uppercase; cursor: pointer;
    }
    #ch-${ID} .hd-mute-ic { display: none; }
    #ch-${ID} .hd-mute-ic svg { display: block; }
    #ch-${ID} .hd-mute:hover { color: var(--linen); border-color: var(--linen-dim); }
    #ch-${ID} .hd-mute[aria-pressed='true'] { color: var(--ink); background: var(--madder); border-color: var(--madder); font-weight: 600; }
    #ch-${ID} .hd-cell.is-muted .hd-pick { border-style: dashed; }
    #ch-${ID} .hd-cell.is-muted .hd-pick .stage-canvas { opacity: 0.35; filter: grayscale(0.8); }
    #ch-${ID} .hd-cell.is-muted .hd-pev { color: var(--madder); }

    #ch-${ID} .hd-loom { display: grid; gap: var(--space-3); }
    #ch-${ID} .hd-loom-top { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: baseline; gap: 6px var(--space-4); }
    #ch-${ID} .hd-hname { font-family: var(--font-display); font-style: italic; font-size: clamp(1.45rem, 1.1rem + 1vw, 2rem); line-height: 1.05; color: var(--hc); }
    #ch-${ID} .hd-hid { font-family: var(--font-mono); font-size: var(--fs-micro); letter-spacing: 0.12em; text-transform: uppercase; color: var(--mist); }
    #ch-${ID} .hd-role { font-size: var(--fs-small); line-height: 1.55; color: var(--linen-dim); }
    #ch-${ID} .hd-loom .stage-canvas canvas { border-radius: var(--radius-sm); background: radial-gradient(120% 100% at 50% 30%, var(--ink-2) 0%, var(--ink) 100%); border: 1px solid var(--rule); cursor: pointer; }
    #ch-${ID} .hd-loom .stage-canvas canvas:focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; }
    #ch-${ID} .hd-facts { display: grid; grid-template-columns: auto minmax(0, 1fr); gap: 5px 14px; font-family: var(--font-mono); font-size: 11px; line-height: 1.55; color: var(--linen-dim); align-items: baseline; }
    #ch-${ID} .hd-facts b { color: var(--hc); font-weight: 500; }
    #ch-${ID} .hd-facts .lab { letter-spacing: 0.08em; text-transform: uppercase; font-size: 9.5px; color: var(--mist); white-space: nowrap; }
    #ch-${ID} .hd-facts .dim { color: var(--mist); }
    #ch-${ID} .hd-pred { display: grid; gap: 5px; padding-top: var(--space-3); border-top: 1px solid var(--rule); }
    #ch-${ID} .hd-pred-title { font-family: var(--font-mono); font-size: var(--fs-micro); letter-spacing: 0.1em; text-transform: uppercase; color: var(--mist); }
    #ch-${ID} .hd-pred-title b { color: var(--linen); font-weight: 500; text-transform: none; letter-spacing: 0; }
    #ch-${ID} .hd-bar { display: grid; grid-template-columns: 5.5em minmax(0, 1fr) 3.4em; gap: 10px; align-items: center; font-size: 13px; }
    #ch-${ID} .hd-bar .w { font-weight: 600; color: var(--linen); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    #ch-${ID} .hd-bar .track { position: relative; height: 8px; border-radius: 4px; background: var(--ink-3); overflow: visible; }
    #ch-${ID} .hd-bar .fill { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 4px; background: linear-gradient(90deg, color-mix(in srgb, var(--weld) 55%, var(--ink-3)), var(--weld)); box-shadow: 0 0 10px -2px color-mix(in srgb, var(--weld) 60%, transparent); transition: width 0.35s cubic-bezier(.2,.8,.2,1); }
    #ch-${ID} .hd-bar .ghost { position: absolute; top: -3px; bottom: -3px; width: 2px; margin-left: -1px; background: var(--linen-dim); border-radius: 1px; opacity: 0.8; transition: left 0.35s; }
    #ch-${ID} .hd-bar .p { font-family: var(--font-mono); font-size: 11px; color: var(--linen-dim); text-align: right; }
    #ch-${ID} .hd-bar.is-top .p { color: var(--weld); }
    #ch-${ID} .hd-mutebar { display: flex; flex-wrap: wrap; gap: 6px 12px; align-items: center; font-family: var(--font-mono); font-size: 11px; color: var(--mist); min-height: 28px; }
    #ch-${ID} .hd-mutebar .tag { color: var(--madder); }
    #ch-${ID} .hd-mutebar .btn { min-height: 28px; padding: 3px 12px; }
    #ch-${ID} .hd-loom-mute[aria-pressed='true'] { border-color: var(--madder); color: var(--madder); }

    #ch-${ID} .hd-mt .stage-canvas canvas { border-radius: var(--radius); border: 1px solid var(--rule); background: radial-gradient(120% 100% at 50% 0%, var(--ink-2) 0%, var(--ink) 100%); }
    #ch-${ID} .hd-mt .stage-canvas canvas:focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; }
    #ch-${ID} .hd-mt-cap { font-size: var(--fs-small); line-height: 1.5; color: var(--mist); min-height: 3em; max-width: 80ch; }
    #ch-${ID} .hd-atlas figcaption { max-width: 80ch; }
    #ch-${ID} .hd-findings { display: grid; gap: var(--space-4); max-width: 68ch; }
    #ch-${ID} .hd-findings p { color: var(--linen-dim); }
    #ch-${ID} .hd-findings strong { color: var(--linen); font-weight: 600; }
    #ch-${ID} .hd-n { color: var(--weld); font-weight: 600; font-variant-numeric: tabular-nums; }

    @media (max-width: 860px) {
      #ch-${ID} .hd-board { grid-template-columns: minmax(0, 1fr); }
    }
    @media (max-width: 520px) {
      #ch-${ID} .hd-cells { gap: 6px 5px; }
      #ch-${ID} .hd-pick { padding: 4px 4px 5px; border-radius: 7px; }
      #ch-${ID} .hd-pname { font-size: 9.5px; }
      #ch-${ID} .hd-chip { font-size: 10px; padding: 4px 9px; letter-spacing: 0.03em; }
      #ch-${ID} .hd-pev { font-size: 8.5px; }
      #ch-${ID} .hd-mute { top: 3px; right: 3px; width: 24px; height: 24px; padding: 0; display: grid; place-items: center; border-radius: 50%; }
      #ch-${ID} .hd-mute-ic { display: block; }
      #ch-${ID} .hd-mute-tx { display: none; }
      #ch-${ID} .hd-pev-w { display: none; }
      #ch-${ID} .hd-vgroup b { min-width: 100%; }
      #ch-${ID} .hd-bar { grid-template-columns: 4.6em minmax(0, 1fr) 3.2em; font-size: 12.5px; }
    }
  `;

  // ====================================================================
  // 5. Figure 1 — the prism (scrollytelling stage)
  // ====================================================================
  function buildPrism(ctx, host, S) {
    const H = S.H, T = S.toks.length, tq = S.t;
    const cv = ctx.canvas(host, {
      label: `The prism, live model, layer 0. The token “because” at the end of “${STAGE_TEXT}” sends its residual vector into a glass prism that splits it into four coloured beams, one per attention head. Each beam fans out into that head's real attention threads over the sentence. Head 3 puts ${f2(S.attn[3][1])} of its attention on “queen”.`,
      height: (w) => {
        const vh = window.innerHeight;
        let h = w < 520
          ? Math.min(Math.max(340, w * 1.04), 410, Math.max(320, vh * 0.5))
          : Math.min(Math.max(500, w * 0.88), 600, Math.max(460, vh * 0.72));
        // stacked layout (stage pinned above the text): keep the stage under half the screen,
        // so the step card that is active is the one you can read below it
        if (window.innerWidth <= 900) h = Math.min(h, Math.max(260, vh * 0.45));
        return Math.round(h);
      },
    });
    const g = cv.g;
    const seen = inView(cv.canvas);
    const st = {
      step: 0, tStep: now(), tSplit: -99,
      a: { split: 0, arcs: 0, flow: 0, merge: 0, cost: 0, qs: 0, os: 0 },
      tgt: { split: 0, arcs: 0, flow: 0, merge: 0, cost: 0, qs: 0, os: 0 },
      last: now(), fly: 0,
    };
    const parts = [];
    let Lo = null, bg = null;
    const qMax = Math.max(...S.qv.flat().map(Math.abs));
    const oMax = Math.max(...S.ov.flat().map(Math.abs));
    const wqMax = S.WQ ? (() => { const a = S.WQ.flat().map(Math.abs).sort((x, y) => x - y); return a[Math.floor(a.length * 0.985)] || 1; })() : 1;

    function layout(w, h) {
      const phone = w < 520;
      const colW = Math.round(phone ? Math.max(116, w * 0.34) : Math.min(250, Math.max(172, w * 0.33)));
      const loomX0 = phone ? 6 : 14;
      const loomX1 = w - colW;
      const tokY = h - (phone ? 20 : 30);
      const tokSize = phone ? 11 : 14;
      const laneTop = phone ? 36 : 52;
      const laneBot = tokY - (phone ? 20 : 28);
      const laneH = (laneBot - laneTop) / H;
      const base = M.range(H).map((hh) => laneTop + (hh + 1) * laneH - (phone ? 5 : 8));
      const slots = D.layoutRow(g, S.toks, loomX0 + (phone ? 2 : 8), loomX1 - (phone ? 10 : 20), { size: tokSize, gap: phone ? 3 : 8 });
      const fs = tokSize * (slots[0] ? slots[0].scale : 1);
      const xq = slots[tq].cx;
      const span = Math.max(1, slots[tq].cx - slots[0].cx);
      const resX = w - (phone ? 9 : 16);
      const s = Math.min(colW * (phone ? 0.34 : 0.34), laneH * 1.0);
      const px = loomX1 + colW * (phone ? 0.5 : 0.5), py = laneTop + (laneBot - laneTop) * 0.5;
      const A = { x: px - s * 0.45, y: py - s * 0.85 }, B = { x: px - s * 0.45, y: py + s * 0.85 }, Cc = { x: px + s * 0.8, y: py };
      const lerp = (P1, P2, u) => ({ x: P1.x + (P2.x - P1.x) * u, y: P1.y + (P2.y - P1.y) * u });
      const pin = lerp(B, Cc, 0.42), pout = lerp(A, Cc, 0.42), core = { x: px - s * 0.02, y: py };
      // one exit per head on the left face; each sits at the middle of that head's quarter of the face,
      // which is where its 16-number output lands when the four are concatenated
      const exits = M.range(H).map((hh) => ({ x: A.x, y: py + (hh - (H - 1) / 2) * (1.7 * s / H) }));
      const readY = pin.y + s * 0.42;
      const writeY = Math.max(laneTop + (phone ? 26 : 40), A.y - s * 0.55);
      const faceSeg = (B.y - A.y) / H, faceBar = Math.round(s * (phone ? 0.36 : 0.3));
      const lift = (j) => {
        if (j === tq) return phone ? 7 : 10;
        const u = Math.abs(slots[j].cx - xq) / span;
        return (laneH - (phone ? 31 : 28)) * (0.16 + 0.84 * Math.pow(u, 0.75));
      };
      const arcs = M.range(H).map((hh) => M.range(T).map((j) => (j <= tq ? arcPts(xq, slots[j].cx, base[hh], lift(j)) : null)));
      const beams = M.range(H).map((hh) => {
        const E = exits[hh];
        const dx = E.x - xq;
        return [E.x, E.y, E.x - dx * 0.42, E.y, xq + dx * 0.5, base[hh], xq, base[hh]];
      });
      const tile = { h: fs + Math.round(fs * 0.42) * 2, w: slots[tq].w };
      const resPath = { x0: xq + tile.w / 2 + 2, y: tokY };
      const readB = [resX, readY, resX - colW * 0.12, readY, pin.x + s * 0.32, pin.y + s * 0.12, pin.x, pin.y];
      const writeB = [pout.x, pout.y, pout.x + s * 0.32, pout.y - s * 0.3, resX - colW * 0.1, writeY, resX, writeY];
      // strips: q^h / o^h sit in each lane's header, after the "HEAD h" label
      const cw16 = phone ? 2.4 : 3.4;
      const labW = phone ? 48 : 54;
      const qStrip = M.range(H).map((hh) => ({ x: loomX0 + (phone ? 6 : 10) + labW, y: laneTop + hh * laneH + (phone ? 6 : 9), w: cw16 * 16, h: phone ? 10 : 12 }));
      const xW = Math.min(resX - 10 - (loomX1 + (phone ? 22 : 36)), 64 * (phone ? 1.6 : 2.2));
      const xStrip = { x: resX - 10 - xW, y: tokY - (phone ? 30 : 40), w: xW, h: phone ? 10 : 13 };
      const oW = xW;
      const outStrip = { x: resX - 10 - oW, y: writeY - (phone ? 30 : 38), w: oW, h: phone ? 10 : 13 };
      // length of the residual path (right along the row, then up), for motes
      const resLen = (resX - resPath.x0) + (tokY - 6);
      return {
        w, h, phone, colW, loomX0, loomX1, tokY, tokSize, fs, laneTop, laneBot, laneH, base, slots, xq, resX, s, px, py, A, B, Cc,
        pin, pout, core, exits, readY, writeY, arcs, beams, tile, resPath, readB, writeB, qStrip, xStrip, outStrip, cw16, resLen, faceSeg, faceBar,
      };
    }

    // ---------------------------------------------------------------- static layer
    function drawBg() {
      const { w, h } = Lo;
      bg = document.createElement('canvas');
      bg.width = Math.round(w * cv.dpr); bg.height = Math.round(h * cv.dpr);
      const b = bg.getContext('2d');
      b.setTransform(cv.dpr, 0, 0, cv.dpr, 0, 0);
      D.weave(b, 0, 0, w, h, { step: 6, alpha: 0.022 });
      // lanes: a pool of dyed light under each head's baseline, fading at both ends
      const bandW = Math.ceil(Lo.xq + 40 - Lo.loomX0), bandH = Math.ceil(Lo.laneH);
      for (let hh = 0; hh < H; hh++) {
        const col = hc(hh);
        const y0 = Lo.laneTop + hh * Lo.laneH;
        const off = document.createElement('canvas');
        off.width = Math.max(1, Math.round(bandW * cv.dpr)); off.height = Math.max(1, Math.round(bandH * cv.dpr));
        const o = off.getContext('2d');
        o.setTransform(cv.dpr, 0, 0, cv.dpr, 0, 0);
        const by = Lo.base[hh] - y0;
        const vg = o.createLinearGradient(0, 0, 0, bandH);
        vg.addColorStop(0, AM.rgba(col, 0));
        vg.addColorStop(clamp((by - 4) / bandH * 0.55), AM.rgba(col, 0.012));
        vg.addColorStop(clamp(by / bandH), AM.rgba(col, 0.075));
        vg.addColorStop(clamp((by + 5) / bandH), AM.rgba(col, 0));
        vg.addColorStop(1, AM.rgba(col, 0));
        o.fillStyle = vg; o.fillRect(0, 0, bandW, bandH);
        o.globalCompositeOperation = 'destination-in';
        const hg = o.createLinearGradient(0, 0, bandW, 0);
        hg.addColorStop(0, 'rgba(0,0,0,0.15)'); hg.addColorStop(0.25, 'rgba(0,0,0,1)'); hg.addColorStop(0.9, 'rgba(0,0,0,1)'); hg.addColorStop(1, 'rgba(0,0,0,0)');
        o.fillStyle = hg; o.fillRect(0, 0, bandW, bandH);
        b.drawImage(off, Lo.loomX0, y0, bandW, bandH);
        // the lane's baseline thread
        const lg = b.createLinearGradient(Lo.loomX0, 0, Lo.xq, 0);
        lg.addColorStop(0, AM.rgba(col, 0)); lg.addColorStop(0.3, AM.rgba(col, 0.2)); lg.addColorStop(1, AM.rgba(col, 0.32));
        b.strokeStyle = lg; b.lineWidth = 1;
        b.beginPath(); b.moveTo(Lo.loomX0 + 4, Lo.base[hh] + 0.5); b.lineTo(Lo.xq, Lo.base[hh] + 0.5); b.stroke();
        D.text(b, `HEAD ${hh}`, Lo.loomX0 + (Lo.phone ? 6 : 10), Lo.qStrip[hh].y + Lo.qStrip[hh].h / 2 + 0.5, { size: Lo.phone ? 8.5 : 9.5, role: 'mono', color: col, alpha: 0.95, letterSpacing: '0.12em', baseline: 'middle' });
      }
      // warp threads, one per token
      Lo.slots.forEach((sl, j) => {
        const grd = b.createLinearGradient(0, Lo.laneTop, 0, Lo.tokY);
        const a = j <= tq ? 0.1 : 0.05;
        grd.addColorStop(0, AM.rgba(AM.col.linen, 0));
        grd.addColorStop(0.25, AM.rgba(AM.col.linen, a));
        grd.addColorStop(1, AM.rgba(AM.col.linen, a * 1.2));
        b.strokeStyle = grd;
        b.lineWidth = 1;
        b.beginPath(); b.moveTo(Math.round(sl.cx) + 0.5, Lo.laneTop + 4); b.lineTo(Math.round(sl.cx) + 0.5, Lo.tokY - Lo.tile.h / 2); b.stroke();
      });
      // tokens
      Lo.slots.forEach((sl, j) => {
        D.token(b, S.toks[j], sl.cx, Lo.tokY, { size: Lo.fs, selected: j === tq, w: sl.w - (Lo.phone ? 1 : 2), padX: 4 });
      });
      // the residual stream thread (static part)
      const rp = Lo.resPath;
      const drawRes = (wid, alpha) => {
        b.save(); b.lineCap = 'round'; b.lineJoin = 'round';
        b.strokeStyle = AM.rgba('#f3ead6', alpha); b.lineWidth = wid;
        b.beginPath(); b.moveTo(rp.x0, rp.y); b.lineTo(Lo.resX - 10, rp.y); b.quadraticCurveTo(Lo.resX, rp.y, Lo.resX, rp.y - 10); b.lineTo(Lo.resX, 6); b.stroke();
        b.restore();
      };
      drawRes(10, 0.05); drawRes(4, 0.08); drawRes(1.8, 0.75);
      D.text(b, 'residual stream', Lo.resX - 7, Lo.phone ? 12 : 16, { size: Lo.phone ? 8 : 9, role: 'mono', color: AM.col.mist, align: 'right', baseline: 'middle' });
      const xs = Lo.xStrip;
      strip(b, xs.x, xs.y, xs.w, xs.h, S.x, { color: AM.dye.woad, neg: AM.dye.madder });
      D.text(b, `x · ${S.d}`, xs.x + xs.w, xs.y - (Lo.phone ? 7 : 9), { size: Lo.phone ? 8 : 9, role: 'mono', color: AM.col.linenDim, align: 'right', baseline: 'middle' });
    }

    // ---------------------------------------------------------------- particles
    const MAXP = 300;
    function spawn(kind, hh, j, speed) { if (parts.length < MAXP) parts.push({ kind, h: hh, j, s: 0, v: speed * (0.85 + Math.random() * 0.3), r: 0.7 + Math.random() * 0.6 }); }
    const acc = { res: 0, read: 0, beam: new Array(H).fill(0), arc: M.range(H).map(() => new Array(T).fill(0)), write: 0 };
    function resAt(u) {
      const L1 = Lo.resX - Lo.resPath.x0, d = u * Lo.resLen;
      return d < L1 ? { x: Lo.resPath.x0 + d, y: Lo.tokY } : { x: Lo.resX, y: Lo.tokY - (d - L1) };
    }
    function partPos(p) {
      switch (p.kind) {
        case 'res': return resAt(p.s);
        case 'read': return bezAt(Lo.readB, p.s);
        case 'beam': return bezAt(Lo.beams[p.h], p.s);
        case 'out': return bezAt(Lo.arcs[p.h][p.j], p.s);
        case 'in': return bezAt(Lo.arcs[p.h][p.j], 1 - p.s);
        case 'back': return bezAt(Lo.beams[p.h], 1 - p.s);
        case 'write': return bezAt(Lo.writeB, p.s);
        case 'spark': { const e = ease.out(p.s); return { x: Lo.core.x + Math.cos(p.j) * e * Lo.s * p.r * 2.2, y: Lo.core.y + Math.sin(p.j) * e * Lo.s * p.r * 2.2 }; }
        default: return { x: 0, y: 0 };
      }
    }
    function stepParticles(dt) {
      const a = st.a;
      const rate = Lo.phone ? 0.7 : 1;
      acc.res += dt * 3.2 * rate; while (acc.res >= 1) { acc.res -= 1; spawn('res', -1, -1, 0.32); }
      acc.read += dt * (a.split > 0.05 ? 5 : 3) * rate; while (acc.read >= 1) { acc.read -= 1; spawn('read', -1, -1, 0.8); }
      for (let hh = 0; hh < H; hh++) {
        if (a.split > 0.6 && a.cost < 0.5 && a.flow < 0.5) { acc.beam[hh] += dt * 2.6 * rate; while (acc.beam[hh] >= 1) { acc.beam[hh] -= 1; spawn('beam', hh, -1, 0.9); } }
        if (a.arcs > 0.7 && a.cost < 0.5) {
          const inward = a.flow > 0.5;
          for (let j = 0; j <= tq; j++) {
            acc.arc[hh][j] += dt * (inward ? 11 : 9) * rate * S.attn[hh][j];
            while (acc.arc[hh][j] >= 1) { acc.arc[hh][j] -= 1; spawn(inward ? 'in' : 'out', hh, j, inward ? 0.6 : 0.55); }
          }
        }
      }
      if (a.merge > 0.6 && a.cost < 0.5) { acc.write += dt * 4 * rate; while (acc.write >= 1) { acc.write -= 1; spawn('write', -1, -1, 0.8); } }
      for (let k = parts.length - 1; k >= 0; k--) {
        const p = parts[k];
        p.s += dt * p.v;
        if (p.s >= 1) {
          if (p.kind === 'in' && st.a.flow > 0.5) { p.kind = 'back'; p.s = 0; p.v = 0.95; continue; }
          parts.splice(k, 1);
        }
      }
    }
    function drawParticles() {
      g.save();
      g.globalCompositeOperation = 'lighter';
      for (const p of parts) {
        const pos = partPos(p);
        const col = p.kind === 'write' ? AM.dye.weld : p.h < 0 ? '#f6ecd4' : hc(p.h);
        let a = p.kind === 'spark' ? (1 - p.s) * 0.95 : Math.sin(Math.PI * clamp(p.s, 0, 1)) * 0.9 + 0.1;
        if (p.kind === 'out' || p.kind === 'in') a *= 0.85;
        if (p.kind === 'res') a *= 0.55;
        const r = (Lo.phone ? 4.2 : 5.4) * p.r * (p.kind === 'res' ? 0.8 : 1);
        g.globalAlpha = a * (p.kind === 'res' || p.kind === 'read' ? 1 : 1 - st.a.cost);
        g.drawImage(sprite(col), pos.x - r, pos.y - r, r * 2, r * 2);
      }
      g.restore();
    }

    // ---------------------------------------------------------------- frame
    function draw() {
      if (!Lo) return;
      const t = now();
      const dt = Math.min(0.1, t - st.last);
      st.last = t;
      const k = AM.reducedMotion ? 1 : 1 - Math.exp(-dt * 3.2);
      for (const key in st.tgt) st.a[key] += (st.tgt[key] - st.a[key]) * k;
      // the strips' flight into the prism runs on its own clock, after the values have flowed back
      // (leaving step 3 it eases back instead of jumping)
      const fl = st.step === 3 ? clamp((t - st.tStep - 1.1) / 1.1) : st.fly * (1 - k);
      st.fly = AM.reducedMotion ? (st.step === 3 ? 1 : 0) : fl;
      if (AM.reducedMotion) parts.length = 0; else stepParticles(dt);
      const a = st.a;
      cv.clear();
      g.drawImage(bg, 0, 0, Lo.w, Lo.h);
      const phone = Lo.phone;
      const fadeLoom = 1 - a.cost;

      // caption (top-left)
      const CAPS = ['one vector in · 64 numbers', 'four heads · 16 numbers each', 'four patterns at once', 'four blends · concat · W_O', 'W_Q of layer 0 · cut four ways'];
      haloText(g, `LAYER ${S.layer} · AT “${S.toks[tq].toUpperCase()}”`, Lo.loomX0 + 2, phone ? 12 : 17, { size: phone ? 8.5 : 9.5, align: 'left', color: AM.col.mist, letterSpacing: '0.12em', halo: false });
      haloText(g, CAPS[st.step] || '', Lo.loomX0 + 2, phone ? 25 : 34, { size: phone ? 10.5 : 12.5, role: 'body', italic: true, align: 'left', color: AM.col.linenDim, halo: false });

      // step 1: the token's 64 numbers glow softly where they enter
      const hl = (st.step === 0 ? 1 : 0) * (0.55 + 0.45 * Math.sin(t * 2.4));
      if (hl > 0.01) {
        const xs = Lo.xStrip;
        g.save();
        g.globalCompositeOperation = 'lighter';
        g.shadowColor = AM.rgba('#f3ead6', 0.7 * hl); g.shadowBlur = 14;
        g.strokeStyle = AM.rgba('#f3ead6', 0.45 * hl); g.lineWidth = 1.2;
        D.roundRect(g, xs.x - 4, xs.y - 4, xs.w + 8, xs.h + 8, 4); g.stroke();
        g.restore();
      }
      // read branch into the prism
      silk(g, Lo.readB, { color: '#f3ead6', width: phone ? 1.4 : 1.8, alpha: 0.85, sheen: false });

      // inside the prism: white core splitting into four rays
      const P = Lo;
      g.save();
      g.lineCap = 'round';
      g.strokeStyle = AM.rgba('#fff8e6', 0.85);
      g.lineWidth = phone ? 1.5 : 2;
      g.beginPath(); g.moveTo(P.pin.x, P.pin.y); g.lineTo(P.core.x, P.core.y); g.stroke();
      if (a.split > 0.01) {
        for (let hh = 0; hh < H; hh++) {
          const E = P.exits[hh];
          const u = clamp(a.split * 1.6 - 0.1);
          g.strokeStyle = AM.rgba(hc(hh), 0.95 * u);
          g.lineWidth = phone ? 1.3 : 1.7;
          g.beginPath(); g.moveTo(P.core.x, P.core.y); g.lineTo(P.core.x + (E.x - P.core.x) * u, P.core.y + (E.y - P.core.y) * u); g.stroke();
        }
      }
      g.restore();

      // beams from the prism to each lane
      for (let hh = 0; hh < H; hh++) {
        const f = clamp((a.split - 0.15) * 1.35 - hh * 0.05);
        if (f <= 0) continue;
        silk(g, Lo.beams[hh], { color: hc(hh), width: phone ? 2.2 : 3, alpha: 0.85 * fadeLoom, f: ease.out(f), glow: 1.6 });
      }

      // attention threads per lane
      for (let hh = 0; hh < H; hh++) {
        const col = hc(hh);
        for (let j = 0; j <= tq; j++) {
          const wgt = S.attn[hh][j];
          const f = clamp(a.arcs * 1.5 - hh * 0.12 - (tq - j) * 0.04);
          if (f <= 0) continue;
          const wd = (phone ? 0.5 : 0.6) + (phone ? 4.2 : 5.6) * wgt;
          silk(g, reverseBez(bezPart(Lo.arcs[hh][j], ease.inOut(f))), { color: col, width: wd, alpha: (0.16 + 0.84 * Math.sqrt(wgt)) * fadeLoom, sheen: wgt > 0.3 });
          if (f > 0.95) D.glowDot(g, Lo.slots[j].cx, Lo.base[hh], (phone ? 1.6 : 2.1) * (0.4 + Math.sqrt(wgt)), col, (0.3 + 0.7 * wgt) * fadeLoom);
        }
        // label the strongest thread
        if (a.arcs > 0.8) {
          let jb = 0; for (let j = 1; j <= tq; j++) if (S.attn[hh][j] > S.attn[hh][jb]) jb = j;
          const apex = bezAt(Lo.arcs[hh][jb], 0.5);
          const big = S.attn[hh][jb] > 0.6;
          haloText(g, `${f2(S.attn[hh][jb])} → ${tokTiny(S.toks, jb)}`, jb === tq ? apex.x - 12 : apex.x, apex.y - (phone ? 7 : 9), {
            size: big ? (phone ? 10 : 12) : (phone ? 8.5 : 10), color: col, weight: big ? 600 : 400, alpha: clamp((a.arcs - 0.8) * 5) * fadeLoom, align: jb === tq ? 'right' : 'center',
          });
        }
      }

      // query strips (q^h) then output strips (o^h), 16 numbers each, in each lane's header
      const fly = ease.inOut(st.fly || 0);
      const colA = a.os * fadeLoom * clamp((fly - 0.85) / 0.15);
      if (colA > 0.01) {
        // the concatenated vector: one dark slot along the face holding all four outputs
        const bh = Lo.faceBar, x0 = Lo.A.x - bh - 5;
        g.save();
        g.globalAlpha = colA;
        D.roundRect(g, x0, Lo.A.y + 1, bh + 4, Lo.B.y - Lo.A.y - 2, 3);
        g.fillStyle = AM.rgba(AM.col.ink, 0.86); g.fill();
        g.strokeStyle = AM.rgba(AM.col.linen, 0.22); g.lineWidth = 1; g.stroke();
        g.restore();
        haloText(g, phone ? `${S.d}` : `concat · ${S.d}`, x0 + (bh + 4) / 2, Lo.A.y - (phone ? 7 : 9), { size: phone ? 8 : 9, color: AM.col.linenDim, alpha: colA, halo: AM.rgba(AM.col.ink, 0.85), haloW: 3 });
      }
      for (let hh = 0; hh < H; hh++) {
        const r = Lo.qStrip[hh], col = hc(hh);
        const sq = a.qs * fadeLoom * (1 - a.os);
        if (sq > 0.01) {
          strip(g, r.x, r.y, r.w, r.h, S.qv[hh], { color: col, alpha: sq, max: qMax });
          haloText(g, 'q' + sup(hh), r.x + r.w + 5, r.y + r.h / 2, { size: phone ? 9.5 : 11, role: 'body', italic: true, align: 'left', color: col, alpha: sq, halo: false });
        }
        const so = a.os * fadeLoom;
        if (so > 0.01) {
          // once the values have arrived, each o strip flies to its quarter of the prism's left face and
          // turns upright there: the four of them, end to end, are the concatenated 64 numbers
          const E = Lo.exits[hh], seg = Lo.faceSeg, bh = Lo.faceBar;
          const cx = M.lerp(r.x + r.w / 2, E.x - bh / 2 - 3, fly), cy = M.lerp(r.y + r.h / 2, E.y, fly);
          const sw = M.lerp(r.w, seg - 1.5, fly), sh = M.lerp(r.h, bh, fly);
          g.save();
          g.translate(cx, cy);
          g.rotate(-Math.PI / 2 * fly);
          strip(g, -sw / 2, -sh / 2, sw, sh, S.ov[hh], { color: col, alpha: so, max: oMax, frame: fly < 0.98 });
          g.restore();
          if (fly < 0.3) haloText(g, 'o' + sup(hh), r.x + r.w + 5, r.y + r.h / 2, { size: phone ? 9.5 : 11, role: 'body', italic: true, align: 'left', color: col, alpha: so * (1 - fly / 0.3), halo: false });
        }
      }
      if (a.qs > 0.05 && a.os < 0.5 && !phone) {
        const r0 = Lo.qStrip[0];
        haloText(g, `${S.dh} numbers each`, r0.x + r0.w + 22, r0.y + r0.h / 2, { size: 8.5, align: 'left', color: AM.col.mist, alpha: a.qs * fadeLoom * (1 - a.os), halo: false });
      }

      // merge: W_O thread out of the prism, and the add
      if (a.merge > 0.01) {
        const fw = clamp(fly * 1.4 - 0.25);
        silk(g, Lo.writeB, { color: AM.dye.weld, width: phone ? 1.8 : 2.4, alpha: 0.95 * fadeLoom, f: ease.out(fw), glow: 1.6 });
        if (fw > 0.9) {
          const cx = Lo.resX, cy = Lo.writeY, rr = phone ? 6 : 7.5;
          const al = clamp((fw - 0.9) * 10) * fadeLoom;
          g.save();
          g.globalAlpha = al;
          g.fillStyle = AM.col.ink; g.strokeStyle = AM.dye.weld; g.lineWidth = 1.4;
          g.beginPath(); g.arc(cx, cy, rr, 0, Math.PI * 2); g.fill(); g.stroke();
          g.beginPath(); g.moveTo(cx - rr * 0.55, cy); g.lineTo(cx + rr * 0.55, cy); g.moveTo(cx, cy - rr * 0.55); g.lineTo(cx, cy + rr * 0.55); g.stroke();
          g.restore();
          const os = Lo.outStrip;
          strip(g, os.x, os.y, os.w, os.h, S.write, { color: AM.dye.weld, neg: AM.dye.madder, alpha: al });
          haloText(g, `after W_O · ${S.d}`, os.x + os.w, os.y - (phone ? 7 : 9), { size: phone ? 8 : 9, align: 'right', color: AM.dye.weld, alpha: al, halo: false });
        }
      }

      // the prism itself
      drawPrism(a, fly);

      // W_Q as cloth (last step)
      if (a.cost > 0.01) drawCost(a.cost);

      drawParticles();

      // flash when the light first splits
      const fls = clamp(1 - (t - st.tSplit) / 0.9);
      if (fls > 0 && !AM.reducedMotion) {
        g.save();
        g.globalCompositeOperation = 'lighter';
        const rr = Lo.s * (1.3 + (1 - fls) * 2.4);
        const grd = g.createRadialGradient(Lo.core.x, Lo.core.y, 0, Lo.core.x, Lo.core.y, rr);
        grd.addColorStop(0, `rgba(255,248,230,${0.6 * fls})`);
        grd.addColorStop(0.35, AM.rgba(AM.dye.weld, 0.2 * fls));
        grd.addColorStop(1, 'rgba(0,0,0,0)');
        g.fillStyle = grd;
        g.beginPath(); g.arc(Lo.core.x, Lo.core.y, rr, 0, Math.PI * 2); g.fill();
        g.restore();
      }
    }

    function drawPrism(a, fly) {
      const { A, B, Cc, phone, s } = Lo;
      g.save();
      const path = () => { g.beginPath(); g.moveTo(A.x, A.y); g.lineTo(B.x, B.y); g.lineTo(Cc.x, Cc.y); g.closePath(); };
      // glass body
      const grd = g.createLinearGradient(A.x, A.y, Cc.x, B.y);
      grd.addColorStop(0, AM.rgba('#e8f0ff', 0.13));
      grd.addColorStop(0.45, AM.rgba(AM.dye.woad, 0.07));
      grd.addColorStop(1, AM.rgba('#fff4d6', 0.04));
      path(); g.fillStyle = grd; g.fill();
      // spectral glow inside once split (and a gold glow while W_O works)
      g.save(); path(); g.clip();
      g.globalCompositeOperation = 'lighter';
      if (a.split > 0.05) {
        for (let hh = 0; hh < H; hh++) {
          const E = Lo.exits[hh];
          const rg = g.createRadialGradient(E.x + s * 0.12, E.y, 0, E.x + s * 0.12, E.y, s * 0.6);
          rg.addColorStop(0, AM.rgba(hc(hh), 0.24 * a.split));
          rg.addColorStop(1, AM.rgba(hc(hh), 0));
          g.fillStyle = rg; g.fillRect(A.x - 4, A.y - 4, Cc.x - A.x + 8, B.y - A.y + 8);
        }
      }
      const gw = a.merge * Math.sin(Math.PI * clamp(fly * 1.2)) ;
      if (gw > 0.01) {
        const rg = g.createRadialGradient(Lo.core.x, Lo.core.y, 0, Lo.core.x, Lo.core.y, s);
        rg.addColorStop(0, AM.rgba(AM.dye.weld, 0.5 * gw)); rg.addColorStop(1, AM.rgba(AM.dye.weld, 0));
        g.fillStyle = rg; g.fillRect(A.x - 4, A.y - 4, Cc.x - A.x + 8, B.y - A.y + 8);
      }
      g.restore();
      // bevel: an inner triangle, then the rim
      const inset = (P0, k) => ({ x: P0.x + (Lo.px + s * 0.05 - P0.x) * k, y: P0.y + (Lo.py - P0.y) * k });
      const a2 = inset(A, 0.14), b2 = inset(B, 0.14), c2 = inset(Cc, 0.14);
      g.beginPath(); g.moveTo(a2.x, a2.y); g.lineTo(b2.x, b2.y); g.lineTo(c2.x, c2.y); g.closePath();
      g.strokeStyle = AM.rgba('#fbf4e4', 0.12); g.lineWidth = 1; g.stroke();
      path();
      g.lineJoin = 'round';
      g.strokeStyle = AM.rgba('#fbf4e4', 0.14); g.lineWidth = 6; g.stroke();
      g.strokeStyle = AM.rgba('#fbf4e4', 0.8); g.lineWidth = 1.25; g.stroke();
      // specular streak on the left face
      const sg = g.createLinearGradient(0, A.y, 0, B.y);
      sg.addColorStop(0, 'rgba(255,255,255,0)'); sg.addColorStop(0.3, 'rgba(255,255,255,0.7)'); sg.addColorStop(0.55, 'rgba(255,255,255,0.05)'); sg.addColorStop(1, 'rgba(255,255,255,0)');
      g.strokeStyle = sg; g.lineWidth = 1.5;
      g.beginPath(); g.moveTo(A.x + 3, A.y + 8); g.lineTo(A.x + 3, B.y - 8); g.stroke();
      g.restore();
      const isO = a.merge > 0.5;
      haloText(g, isO ? 'W_O' : (phone ? 'W_Q,K,V' : 'W_Q · W_K · W_V'), B.x + (phone ? 2 : 4), B.y + (phone ? 10 : 13), {
        size: phone ? 8 : 9.5, color: isO ? AM.dye.weld : AM.col.linenDim, align: 'left', halo: AM.rgba(AM.col.ink, 0.85), alpha: 1 - a.cost * 0.6,
      });
    }

    // W_Q as cloth: 64 × 64 cells that never change. They are painted once per size into
    // their own layer; each frame only draws that layer (faded, rows revealed by a clip).
    let costLayer = null, costKey = '';
    function renderCost(size, cell) {
      const key = `${size}|${cv.dpr}`;
      if (costLayer && costKey === key) return;
      costKey = key;
      costLayer = costLayer || document.createElement('canvas');
      costLayer.width = costLayer.height = Math.max(1, Math.round(size * cv.dpr));
      const o = costLayer.getContext('2d');
      o.setTransform(cv.dpr, 0, 0, cv.dpr, 0, 0);
      const n = S.d, dh = S.dh, gap = cell > 4 ? 0.8 : 0.3;
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          const hh = Math.floor(j / dh);
          const v = S.WQ ? S.WQ[i][j] : 0;
          const mm = Math.min(1, Math.abs(v) / wqMax);
          o.fillStyle = v >= 0 ? AM.rgba(hc(hh), 0.08 + 0.92 * Math.pow(mm, 0.8)) : AM.mix(AM.col.ink, hc(hh), 0.06 + 0.32 * Math.pow(mm, 0.8));
          o.fillRect(j * cell, i * cell, Math.max(1, cell - gap), Math.max(1, cell - gap));
        }
      }
    }
    function drawCost(c) {
      const { loomX0, loomX1, laneTop, laneBot, phone } = Lo;
      const n = S.d, dh = S.dh;
      // veil the lanes
      g.save();
      g.globalAlpha = clamp(c * 1.3);
      const vg = g.createLinearGradient(loomX1 + 30, 0, loomX1 + 60, 0);
      vg.addColorStop(0, AM.rgba(AM.col.ink2, 1)); vg.addColorStop(1, AM.rgba(AM.col.ink2, 0));
      g.fillStyle = vg;
      g.fillRect(0, laneTop - 4, loomX1 + 60, laneBot - laneTop + 8);
      g.restore();
      const top = laneTop + (phone ? 20 : 28);
      const size = Math.floor(Math.min(loomX1 - loomX0 - (phone ? 30 : 56), laneBot - top - (phone ? 30 : 40)) / n) * n;
      const cell = size / n;
      const gx = Math.round(loomX0 + (loomX1 - loomX0 - size) / 2 + (phone ? 8 : 12)), gy = Math.round(top + 4);
      g.save();
      g.globalAlpha = clamp(c * 1.6);
      D.roundRect(g, gx - 6, gy - 6, size + 12, size + 12, 7);
      g.fillStyle = AM.rgba(AM.col.ink, 0.95); g.fill();
      g.strokeStyle = AM.rgba(AM.col.linen, 0.1); g.lineWidth = 1; g.stroke();
      const rows = Math.floor(clamp(c * 1.15) * n + 0.0001);
      if (rows > 0 && size > 0) {
        renderCost(size, cell);
        g.save();
        g.beginPath(); g.rect(gx, gy, size, rows * cell); g.clip();
        g.drawImage(costLayer, gx, gy, size, size);
        g.restore();
      }
      for (let hh = 0; hh < S.H; hh++) {
        const x = gx + hh * dh * cell;
        if (hh > 0) { g.strokeStyle = AM.col.ink; g.lineWidth = 2; g.beginPath(); g.moveTo(x, gy - 4); g.lineTo(x, gy + size + 4); g.stroke(); }
        haloText(g, phone ? `h${hh}` : `head ${hh}`, x + dh * cell / 2, gy - (phone ? 10 : 13), { size: phone ? 8 : 9.5, color: hc(hh), halo: false });
        haloText(g, '16', x + dh * cell / 2, gy + size + (phone ? 8 : 10), { size: phone ? 8 : 8.5, color: hc(hh), halo: false, alpha: 0.85 });
      }
      g.restore();
      g.save();
      g.globalAlpha = clamp(c * 2 - 0.6);
      g.translate(gx - (phone ? 10 : 14), gy + size / 2);
      g.rotate(-Math.PI / 2);
      haloText(g, phone ? 'reads all 64 of x' : 'rows: all 64 numbers of x', 0, 0, { size: phone ? 8 : 9, color: AM.col.mist, halo: false });
      g.restore();
      haloText(g, `W_Q: 64 × 64 = 4 × (64 × 16)`, gx + size / 2, gy + size + (phone ? 22 : 28), { size: phone ? 9 : 11, color: AM.col.linen, alpha: clamp(c * 2 - 0.8), halo: false });
    }

    cv.onResize((w, h) => { Lo = layout(w, h); drawBg(); draw(); });

    const TG = [
      { split: 0, arcs: 0, flow: 0, merge: 0, cost: 0, qs: 0, os: 0 },
      { split: 1, arcs: 0, flow: 0, merge: 0, cost: 0, qs: 1, os: 0 },
      { split: 1, arcs: 1, flow: 0, merge: 0, cost: 0, qs: 1, os: 0 },
      { split: 1, arcs: 1, flow: 1, merge: 1, cost: 0, qs: 0, os: 1 },
      { split: 1, arcs: 1, flow: 0, merge: 0, cost: 1, qs: 0, os: 0 },
    ];
    function setStep(i) {
      const prev = st.step;
      st.step = i;
      st.tStep = now();
      Object.assign(st.tgt, TG[i] || TG[0]);
      if (i >= 1 && prev < 1) {
        st.tSplit = now();
        if (!AM.reducedMotion) {
          for (let hh = 0; hh < H; hh++) for (let k = 0; k < 5; k++) spawn('beam', hh, -1, 0.9 + Math.random() * 0.6);
          // a burst of dyed sparks from the heart of the prism
          for (let k = 0; k < 36; k++) { spawn('spark', k % H, Math.PI * (0.55 + Math.random() * 0.9), 1.1); const p = parts[parts.length - 1]; if (p) p.r = 0.5 + Math.random() * 0.9; }
        }
      }
      if (i === 3) parts.length = Math.min(parts.length, 60);
      if (AM.reducedMotion) Object.assign(st.a, st.tgt);
      cv.canvas.setAttribute('aria-label', ARIA[i] || ARIA[0]);
    }
    const best = (hh) => { let jb = 0; for (let j = 1; j <= tq; j++) if (S.attn[hh][j] > S.attn[hh][jb]) jb = j; return jb; };
    const ARIA = [
      `Live model, layer 0. The token “because” sends its 64-number residual vector up into a clear glass prism. Four empty lanes wait above the sentence “${STAGE_TEXT}”.`,
      'The prism splits the vector into four coloured beams, one per head. Each beam carries a 16-number query, shown as a small strip.',
      'Each head draws its own attention threads from “because” to the earlier words. ' + M.range(H).map((hh) => `Head ${hh}: strongest ${f2(S.attn[hh][best(hh)])} on “${tokName(S.toks, best(hh))}”.`).join(' '),
      'Values flow back along every thread. The four 16-number outputs are laid side by side into 64 numbers, pass through the prism again as W_O, and are added to the residual stream.',
      'The real query matrix W_Q of layer 0 drawn as cloth: 64 rows by 64 columns, cut into four coloured bands of 16 columns, one per head.',
    ];
    setStep(0);
    ctx.loop(() => { if (seen.on) draw(); });
    return { setStep };
  }

  // ====================================================================
  // 6. Figure 2 — the head atlas
  // ====================================================================
  function buildAtlas(ctx, body, m, eng) {
    const el = ctx.el;
    const ui = AM.ui;
    const { n_layer: NL, n_head: NH, n_ctx: NCTX } = m.config;
    const S = {
      text: PRESETS[0].text, preset: 0, ids: null, toks: null, res: null, base: null, sel: PRESETS[0].head.slice(), focus: -1,
      mute: new Set(), canMute: !!eng,
    };

    const fig = el('figure', { class: 'fig hd-atlas' });
    fig.appendChild(el('div', { class: 'fig-top' }, el('span', { class: 'fig-title' }, 'The head atlas · 3 layers × 4 heads'), ui.badge('live')));
    body.appendChild(fig);

    // ---- sentence picker
    const chips = PRESETS.map((p, i) => el('button', {
      type: 'button', class: 'hd-chip', id: `hd-preset-${i}`, 'aria-pressed': String(i === 0), title: p.text,
      onclick: () => { input.value = p.text; S.preset = i; S.sel = p.head.slice(); setSentence(p.text, true); },
    }, p.label));
    fig.appendChild(el('div', { class: 'hd-presets', role: 'group', 'aria-label': 'Example sentences' }, chips));
    const input = el('input', { class: 'text-input', id: 'hd-input', type: 'text', value: S.text, 'aria-label': 'Type a sentence using only words the model knows', autocomplete: 'off', spellcheck: 'false' });
    const runBtn = ui.button({ id: 'hd-run', label: 'Run', kind: 'primary', onClick: () => submit() });
    input.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); submit(); } });
    const msg = el('div', { class: 'hd-msg', 'aria-live': 'polite' });
    fig.appendChild(el('div', { class: 'hd-input-row' }, input, runBtn));
    fig.appendChild(msg);
    const vocab = el('details', { class: 'hd-vocab' },
      el('summary', {}, `The ${m.vocab.filter((v) => !v.startsWith('<') && /\w/.test(v)).length} words it knows (tap to add)`),
      el('div', { class: 'hd-vgroups' }, VOCAB_GROUPS.map(([lab, words]) => el('div', { class: 'hd-vgroup' },
        el('b', {}, lab),
        words.split(' ').filter((wd) => m.tokenId(wd) >= 0).map((wd) => el('button', { type: 'button', class: 'hd-vw', onclick: () => addWord(wd) }, wd))))));
    fig.appendChild(vocab);

    // ---- board: small multiples + the loom
    const board = el('div', { class: 'hd-board' });
    fig.appendChild(board);
    const cellsBox = el('div', { class: 'hd-cells', role: 'group', 'aria-label': 'Twelve attention heads' });
    board.appendChild(cellsBox);
    cellsBox.appendChild(el('span'));
    for (let h = 0; h < NH; h++) cellsBox.appendChild(el('span', { class: 'hd-colh', style: { color: hc(h) } }, `head ${h}`));
    const cells = [];
    for (let l = 0; l < NL; l++) {
      cellsBox.appendChild(el('span', { class: 'hd-rowh' }, `layer ${l}`));
      for (let h = 0; h < NH; h++) {
        const pick = el('button', { type: 'button', class: 'hd-pick', id: `hd-head-${l}-${h}`, 'aria-pressed': 'false', onclick: () => { S.sel = [l, h]; refresh(); } });
        const cell = el('div', { class: 'hd-cell', style: { '--hc': hc(h) } }, pick);
        pick.style.setProperty('--hc', hc(h));
        const cvs = ctx.canvas(pick, { aspect: 1 });
        const name = el('span', { class: 'hd-pname' }, titleName(l, h));
        const ev = el('span', { class: 'hd-pev' }, '');
        pick.append(name, ev);
        let muteBtn = null, muteTx = null;
        if (S.canMute) {
          // a word on wide screens; on phones a 24px speaker-off icon (the label stays for screen readers)
          muteTx = el('span', { class: 'hd-mute-tx' }, 'mute');
          muteBtn = el('button', { type: 'button', class: 'hd-mute', id: `hd-mute-${l}-${h}`, 'aria-pressed': 'false', title: `Mute layer ${l} head ${h}`, 'aria-label': `Mute layer ${l} head ${h}`, onclick: () => toggleMute(l, h) },
            el('span', { class: 'hd-mute-ic', 'aria-hidden': 'true', html: MUTE_ICON }), muteTx);
          cell.appendChild(muteBtn);
        }
        cellsBox.appendChild(cell);
        const c = { l, h, pick, cell, cvs, name, ev, muteBtn, muteTx };
        cvs.onResize(() => drawCell(c));
        cells.push(c);
      }
    }

    const loom = el('div', { class: 'panel hd-loom' });
    board.appendChild(loom);
    fig.appendChild(el('figcaption', { html: 'Live model: every pattern and prediction is computed on your sentence as you change it. The test-set lines are averages we measured once over the model’s 4,000-sequence test set. Muting zeroes a head’s output before W<sub>O</sub> and reruns the whole model; the muted head’s own pattern is still drawn, faded.' }));
    const hName = el('div', { class: 'hd-hname' });
    const hId = el('div', { class: 'hd-hid' });
    const loomMute = S.canMute ? el('button', { type: 'button', class: 'btn hd-loom-mute', id: 'hd-loom-mute', 'aria-pressed': 'false', onclick: () => toggleMute(S.sel[0], S.sel[1]) }, 'Mute this head') : null;
    loom.appendChild(el('div', { class: 'hd-loom-top' }, el('div', {}, hName, hId), loomMute));
    const role = el('p', { class: 'hd-role' });
    loom.appendChild(role);
    const lcv = ctx.canvas(loom, {
      label: 'Attention threads of the selected head from the selected word.',
      height: (w) => loomHeight(w),
    });
    lcv.canvas.setAttribute('tabindex', '0');
    const facts = el('div', { class: 'hd-facts' });
    loom.appendChild(facts);
    const muteBar = el('div', { class: 'hd-mutebar', 'aria-live': 'polite' });
    loom.appendChild(muteBar);
    const pred = el('div', { class: 'hd-pred' });
    loom.appendChild(pred);

    // ---------------------------------------------------------------- model runs
    function compute() {
      const r = m.run(S.ids, { capture: true });
      S.base = { attn: r.attn, probs: r.probs };
      S.res = S.mute.size && eng ? eng.run(S.ids, S.mute) : S.base;
    }
    function setSentence(text, fromPreset) {
      const e = m.encode(text);
      if (!e.ids.length) { msg.replaceChildren(el('span', {}, 'Type a few words first.')); return false; }
      const special = e.tokens.filter((t) => t.startsWith('<') && t !== '<unk>');
      if (e.unknown.length || special.length) { showUnknown(e.unknown.concat(special)); return false; }
      let ids = e.ids, note = '';
      if (ids.length > NCTX) { ids = ids.slice(0, NCTX); note = `Only the first ${NCTX} tokens fit in the model’s context.`; }
      S.text = text; S.ids = ids; S.toks = m.decode(ids); S.focus = ids.length - 1;
      chips.forEach((c, i) => c.setAttribute('aria-pressed', String(fromPreset && i === S.preset)));
      msg.replaceChildren(note ? el('span', {}, note) : el('span', {}, `${ids.length} tokens · all known to the model.`));
      compute();
      lcv.resize();
      refresh();
      return true;
    }
    function submit() {
      const v = input.value.trim();
      const hit = PRESETS.findIndex((p) => p.text === v.toLowerCase());
      if (hit >= 0) { S.preset = hit; setSentence(PRESETS[hit].text, true); return; }
      setSentence(v, false);
    }
    function addWord(wd) {
      const cur = input.value.trim();
      input.value = cur ? `${cur} ${wd}` : wd;
      submit();
    }
    // closest known word; a typo that drops a letter ("qeen", "dor") is the commonest, so it wins ties
    const subseq = (a, b) => { let i = 0; for (const ch of b) if (ch === a[i]) i++; return i === a.length; };
    function suggest(wd) {
      let best = null, bd = 99, bs = 99;
      for (const v of m.vocab) {
        if (v.startsWith('<') || !/\w/.test(v)) continue;
        const dd = levenshtein(wd, v);
        const sc = dd - (subseq(wd, v) ? 0.5 : 0);
        if (sc < bs) { bs = sc; bd = dd; best = v; }
      }
      return bd <= Math.max(2, Math.floor(wd.length / 2)) ? best : null;
    }
    function showUnknown(unknown) {
      const uniq = [...new Set(unknown)];
      const parts = [el('span', {}, 'Not in the vocabulary:')];
      uniq.forEach((wd) => {
        parts.push(el('span', { class: 'bad' }, wd));
        const s = suggest(wd);
        if (s) parts.push(el('button', { type: 'button', class: 'fix', onclick: () => { input.value = input.value.replace(new RegExp(`(^|\\s)${wd.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}(?=\\s|$|[.,])`, 'gi'), `$1${s}`); submit(); } }, `→ ${s}`));
      });
      parts.push(el('span', {}, 'The model only knows the words in the list below.'));
      msg.replaceChildren(...parts);
    }
    function toggleMute(l, h) {
      if (!eng) return;
      const k = l * NH + h;
      if (S.mute.has(k)) S.mute.delete(k); else S.mute.add(k);
      S.res = S.mute.size ? eng.run(S.ids, S.mute) : S.base;
      refresh();
    }

    // ---------------------------------------------------------------- drawing: small multiples
    function drawCell(c) {
      const { cvs, l, h } = c;
      const g = cvs.g, w = cvs.w, hgt = cvs.h;
      if (!S.res || !w) return;
      cvs.clear();
      const A = S.res.attn[l][h], n = S.ids.length;
      const pad = Math.max(3, Math.round(w * 0.05));
      const cell = (w - pad * 2) / n;
      const col = hc(h);
      g.save();
      g.fillStyle = AM.col.ink2;
      D.roundRect(g, 0, 0, w, hgt, 5); g.fill();
      // warp threads
      if (cell >= 3.5) {
        g.fillStyle = AM.mix(AM.col.ink2, AM.dye.woad, 0.18);
        for (let j = 0; j < n; j++) g.fillRect(pad + j * cell + cell * 0.42, pad, Math.max(0.6, cell * 0.16), cell * n);
      }
      for (let i = 0; i < n; i++) {
        for (let j = 0; j <= i; j++) {
          const v = A[i][j];
          const a = Math.pow(v, 0.62);
          if (a < 0.02) continue;
          g.fillStyle = AM.rgba(col, 0.1 + 0.9 * a);
          const x = pad + j * cell, y = pad + i * cell;
          if (cell >= 5) { D.roundRect(g, x + 0.5, y + cell * 0.18, cell - 1, cell * 0.64, Math.min(2.5, cell * 0.2)); g.fill(); }
          else g.fillRect(x, y, Math.max(0.8, cell - 0.4), Math.max(0.8, cell - 0.4));
        }
      }
      // focus row
      const fy = pad + S.focus * cell;
      g.strokeStyle = AM.rgba(AM.col.linen, 0.75);
      g.lineWidth = 1;
      g.strokeRect(pad - 1.5, fy - 0.5, cell * (S.focus + 1) + 3, cell + 1);
      g.restore();
    }

    // ---------------------------------------------------------------- drawing: the loom
    function loomGeom(w) {
      const phone = w < 520;
      const n = S.toks ? S.toks.length : 6;
      const size = phone ? 11.5 : 13.5;
      const g = measureCtx();
      const slots = D.layoutRow(g, S.toks || ['the'], 10, w - 10, { size, gap: phone ? 4 : 7 });
      const compact = slots[0].scale < 0.82;
      let maxLab = 0;
      if (compact) for (const t of S.toks || []) maxLab = Math.max(maxLab, D.measure(g, t, phone ? 10 : 11, 'body', 600));
      const tileH = size + Math.round(size * 0.42) * 2;
      return { phone, n, size, slots, compact, labH: compact ? Math.round(maxLab * 0.88 + 16) : Math.round(tileH / 2 + 14) };
    }
    function loomHeight(w) {
      const G = loomGeom(w);
      return Math.round((G.phone ? 170 : 210) + G.labH);
    }
    function drawLoom() {
      const g = lcv.g, w = lcv.w, h = lcv.h;
      if (!S.res || !w) return;
      lcv.clear();
      const G = loomGeom(w);
      const [l, hh] = S.sel;
      const col = hc(hh);
      const A = S.res.attn[l][hh][S.focus];
      const n = S.toks.length, qf = S.focus;
      const muted = S.mute.has(l * NH + hh);
      // positions
      let xs, baseY;
      if (G.compact) {
        const x0 = 16, x1 = w - 16;
        xs = M.range(n).map((j) => (n === 1 ? (x0 + x1) / 2 : x0 + (x1 - x0) * j / (n - 1)));
        baseY = h - G.labH;
      } else {
        xs = G.slots.map((s) => s.cx);
        baseY = h - G.labH;
      }
      const tileH = G.size + Math.round(G.size * 0.42) * 2;
      const anchorY = G.compact ? baseY - 4 : baseY - tileH / 2 - 3;
      D.weave(g, 0, 0, w, h, { step: 6, alpha: 0.02 });
      // warp threads
      for (let j = 0; j < n; j++) {
        const grd = g.createLinearGradient(0, 16, 0, anchorY);
        const a = j <= qf ? 0.1 : 0.04;
        grd.addColorStop(0, AM.rgba(AM.col.linen, 0)); grd.addColorStop(1, AM.rgba(AM.col.linen, a));
        g.strokeStyle = grd; g.lineWidth = 1;
        g.beginPath(); g.moveTo(Math.round(xs[j]) + 0.5, 16); g.lineTo(Math.round(xs[j]) + 0.5, anchorY); g.stroke();
      }
      // future curtain
      if (qf < n - 1) {
        const edge = (xs[qf] + xs[qf + 1]) / 2;
        g.save();
        g.fillStyle = AM.rgba(AM.dye.madder, 0.05);
        g.fillRect(edge, 8, w - edge - 6, anchorY - 4);
        g.strokeStyle = AM.rgba(AM.dye.madder, 0.5); g.setLineDash([3, 4]);
        g.beginPath(); g.moveTo(edge + 0.5, 10); g.lineTo(edge + 0.5, anchorY); g.stroke();
        g.restore();
        haloText(g, 'future · masked', edge + 6, 18, { size: 8.5, align: 'left', color: AM.dye.madder, halo: false, alpha: 0.85 });
      }
      // threads
      const span = Math.max(1, Math.max(xs[qf] - xs[0], xs[n - 1] - xs[0]));
      const maxLift = anchorY - 30;
      let jb = 0; for (let j = 1; j <= qf; j++) if (A[j] > A[jb]) jb = j;
      const order = M.range(qf + 1).sort((a, b) => A[a] - A[b]);
      const labels = [];
      for (const j of order) {
        const wgt = A[j];
        const lift = j === qf ? 12 : maxLift * (0.14 + 0.86 * Math.pow(Math.abs(xs[j] - xs[qf]) / span, 0.7));
        const P = arcPts(xs[qf], xs[j], anchorY, lift);
        silk(g, P, { color: col, width: 0.6 + (G.phone ? 5 : 6.5) * wgt, alpha: (0.14 + 0.86 * Math.sqrt(wgt)) * (muted ? 0.45 : 1), sheen: wgt > 0.3 });
        D.glowDot(g, xs[j], anchorY, 1.6 + 2.6 * Math.sqrt(wgt), col, 0.25 + 0.75 * wgt);
        if (wgt >= 0.08 || j === jb) labels.push({ j, wgt, ap: bezAt(P, 0.5) });
      }
      // weight labels, strongest first, skipping any that would collide
      const placed = [];
      labels.sort((a, b) => b.wgt - a.wgt).forEach((lb) => {
        const big = lb.j === jb, fsz = big ? (G.phone ? 11 : 12.5) : (G.phone ? 9 : 10);
        const bw = fsz * 2.6, bh = fsz + 4, x = lb.ap.x, y = lb.ap.y - 8;
        if (placed.some((p) => Math.abs(p.x - x) < (p.w + bw) / 2 && Math.abs(p.y - y) < (p.h + bh) / 2)) return;
        placed.push({ x, y, w: bw, h: bh });
        haloText(g, f2(lb.wgt), x, y, { size: fsz, weight: big ? 600 : 400, color: col });
      });
      // tokens
      for (let j = 0; j < n; j++) {
        const fut = j > qf;
        if (G.compact) {
          g.save();
          g.translate(xs[j] + 3, baseY + 8);
          g.rotate(-Math.PI / 3);
          D.text(g, S.toks[j], 0, 0, { size: G.phone ? 10 : 11, weight: j === qf ? 700 : 500, color: j === qf ? AM.dye.weld : AM.col.linenDim, align: 'right', baseline: 'middle', alpha: fut ? 0.4 : 1 });
          g.restore();
          D.glowDot(g, xs[j], anchorY + 2, j === qf ? 2.6 : 1.4, j === qf ? AM.dye.weld : AM.col.linenDim, fut ? 0.3 : 0.8);
        } else {
          D.token(g, S.toks[j], xs[j], baseY, { size: G.slots[0].scale * G.size, selected: j === qf, w: G.slots[j].w, padX: 4, alpha: fut ? 0.4 : 1 });
        }
      }
      if (muted) haloText(g, 'MUTED · this head’s output is zeroed (its pattern is still shown)', w / 2, 14, { size: 8.5, color: AM.dye.madder, letterSpacing: '0.08em' });
      lcv.canvas.setAttribute('aria-label', `Layer ${l} head ${hh}, ${headName(l, hh)}, at “${tokName(S.toks, qf)}”: ` +
        order.slice().reverse().slice(0, 3).map((j) => `${f2(A[j])} on “${tokName(S.toks, j)}”`).join(', ') + '. Use the arrow keys to move along the sentence.');
      lcv._xs = xs; lcv._anchorY = anchorY; lcv._baseY = baseY;
    }
    lcv.onResize(() => drawLoom());
    // pick a token on the loom
    lcv.canvas.addEventListener('pointerdown', (e) => {
      if (!lcv._xs) return;
      const p = lcv.pointer(e);
      let bj = 0, bd = Infinity;
      lcv._xs.forEach((x, j) => { const dd = Math.abs(x - p.x); if (dd < bd) { bd = dd; bj = j; } });
      if (bd < 40) { S.focus = bj; refresh(); }
    });
    lcv.canvas.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
        e.preventDefault();
        S.focus = clamp(S.focus + (e.key === 'ArrowLeft' ? -1 : 1), 0, S.ids.length - 1);
        refresh();
      }
    });

    // ---------------------------------------------------------------- text panels
    function topKey(row, qf) { let jb = 0; for (let j = 1; j <= qf; j++) if (row[j] > row[jb]) jb = j; return jb; }
    function refresh() {
      const [l, hh] = S.sel;
      const key = `L${l}H${hh}`;
      const qf = S.focus;
      // cells
      for (const c of cells) {
        const on = c.l === l && c.h === hh;
        c.pick.setAttribute('aria-pressed', String(on));
        const row = S.res.attn[c.l][c.h][qf];
        const jb = topKey(row, qf);
        const isM = S.mute.has(c.l * NH + c.h);
        c.cell.classList.toggle('is-muted', isM);
        // phones show the weight alone (the word is in the panel below); wider tiles add the word
        c.ev.innerHTML = isM ? 'muted' : `${f2(row[jb])}<span class="hd-pev-w"> ${esc(tokTiny(S.toks, jb))}</span>`;
        c.pick.setAttribute('aria-label', `Layer ${c.l} head ${c.h}, ${headName(c.l, c.h)}: at “${tokName(S.toks, qf)}” it looks most at “${tokName(S.toks, jb)}” (${f2(row[jb])})${isM ? ', muted' : ''}`);
        if (c.muteBtn) { c.muteBtn.setAttribute('aria-pressed', String(isM)); c.muteTx.textContent = isM ? 'muted' : 'mute'; }
        drawCell(c);
      }
      // header
      loom.style.setProperty('--hc', hc(hh));
      hName.textContent = titleName(l, hh);
      hId.innerHTML = `Layer ${l} · head ${hh} · <span class="hd-nocase">d<sub>head</sub></span> = ${m.config.d_model / NH}`;
      role.textContent = HEADS[key].role;
      if (loomMute) {
        const isM = S.mute.has(l * NH + hh);
        loomMute.setAttribute('aria-pressed', String(isM));
        loomMute.textContent = isM ? 'Unmute this head' : 'Mute this head';
      }
      // facts
      const row = S.res.attn[l][hh][qf];
      const ranked = M.range(qf + 1).sort((a, b) => row[b] - row[a]);
      const top2 = ranked.slice(0, 2).filter((j, k) => k === 0 || row[j] >= 0.05);
      const live = top2.map((j) => `<b>${f2(row[j])}</b> → ${esc(q(tokTiny(S.toks, j)))}`).join(' · ');
      const sinkNote = ranked[0] === 0 && qf > 0 && S.toks[0] === 'the' ? ' <span class="dim">(the first word: a common resting place)</span>' : '';
      const ho = heldOut(m, l, hh);
      facts.innerHTML = `<span class="lab">at ${esc(q(tokTiny(S.toks, qf)))}</span><span>${live}${sinkNote}</span>` +
        (ho ? `<span class="lab">test set</span><span>${esc(ho[0])}: <b>${f2(ho[1])}</b> <span class="dim">(mean over ${esc(ho[2])})</span></span>` : '<span class="lab">test set</span><span class="dim">no single job stands out in our measurements</span>');
      // muting
      if (S.canMute) {
        if (S.mute.size) {
          const list = [...S.mute].sort((a, b) => a - b).map((k) => `L${Math.floor(k / NH)}H${k % NH}`).join(', ');
          muteBar.replaceChildren(el('span', {}, 'Muted: ', el('span', { class: 'tag' }, list)),
            ui.button({ id: 'hd-unmute', label: 'Unmute all', onClick: () => { S.mute.clear(); S.res = S.base; refresh(); } }));
        } else muteBar.replaceChildren(el('span', {}, 'Mute heads (top-right of each tile) to rerun the model without them.'));
      } else muteBar.replaceChildren();
      // prediction at the focus position
      const P = S.res.probs[qf], P0 = S.base.probs[qf];
      const top = m.topk(P, 4);
      const nextTok = qf + 1 < S.toks.length ? S.toks[qf + 1] : null;
      pred.replaceChildren(...[
        el('div', { class: 'hd-pred-title', html: `Model’s next word after <b>${esc(q(tokTiny(S.toks, qf)))}</b>${S.mute.size ? ' · <span style="color:var(--madder)">with muted heads</span>' : ''}${nextTok ? ` · actual: <b>${esc(nextTok)}</b>` : ''}` }),
        ...top.map((t, k) => {
          const bar = el('div', { class: 'hd-bar' + (k === 0 ? ' is-top' : '') },
            el('span', { class: 'w' }, t.token),
            el('span', { class: 'track' },
              el('span', { class: 'fill', style: { width: (t.p * 100).toFixed(1) + '%' } }),
              S.mute.size ? el('span', { class: 'ghost', title: 'with all heads', style: { left: (P0[t.id] * 100).toFixed(1) + '%' } }) : null),
            el('span', { class: 'p' }, pct(t.p) + (S.mute.size ? '' : '')));
          return bar;
        }),
        S.mute.size ? el('div', { class: 'hd-facts', html: `<span class="lab">all heads on</span><span>${m.topk(P0, 1).map((t) => `${esc(t.token)} ${pct(t.p)}`).join('')} <span class="dim">(thin marks on the bars)</span></span>` }) : null,
      ].filter(Boolean));
      drawLoom();
    }

    setSentence(S.text, true);
    /** Open one example with one head muted (used by the mute test). */
    function openCase(pi, l, h) {
      const p = PRESETS[pi];
      input.value = p.text; S.preset = pi; S.sel = [l, h];
      S.mute.clear();
      if (eng) S.mute.add(l * NH + h);
      setSentence(p.text, true);
      const r = fig.getBoundingClientRect();
      window.scrollTo({ top: window.scrollY + r.top - 16, behavior: AM.reducedMotion ? 'auto' : 'smooth' });
    }
    return { S, openCase };
  }

  // ====================================================================
  // 6b. Figure 3 — the mute test: one head off at a time, for every example
  // ====================================================================
  function buildMuteTest(ctx, host, m, eng, onOpen, onDone) {
    const el = ctx.el;
    const { n_layer: NL, n_head: NH } = m.config;
    const K = NL * NH;
    // Nothing runs at mount: every forward pass (the unmuted baseline, twelve single
    // mutes, then all of the last layer muted at once) waits in a queue that the
    // loop works through, a few passes per frame, once the grid is on screen.
    const rows = PRESETS.map((p) => {
      const ids = m.encode(p.text).ids;
      return { p, ids, ans: m.tokenId(p.ans), base: null, lastAll: null, P: new Array(K).fill(null), top: new Array(K).fill(null), born: new Array(K).fill(0) };
    });
    const queue = [];
    rows.forEach((_, r) => { queue.push([r, 'base']); for (let k = 0; k < K; k++) queue.push([r, k]); });
    rows.forEach((_, r) => queue.push([r, 'last']));
    const LAST = new Set(M.range(NH).map((h) => (NL - 1) * NH + h));
    let labMax = 0;
    const fig = el('figure', { class: 'fig hd-mt' });
    fig.appendChild(el('div', { class: 'fig-top' }, el('span', { class: 'fig-title' }, 'The mute test · one head off at a time'), AM.ui.badge('live')));
    host.appendChild(fig);
    const cvs = ctx.canvas(fig, {
      label: 'A grid with one row per example sentence and one column per attention head. Each cell shows how likely the model still is to give the right answer when that one head is muted. Red knots mark answers that break.',
      height: (w) => geom(w).h,
    });
    cvs.canvas.setAttribute('tabindex', '0');
    const cap = el('p', { class: 'hd-mt-cap', 'aria-live': 'polite' });
    fig.appendChild(cap);
    const DEFAULT_CAP = 'Each row is one example and its right answer; each column mutes one head and reruns the model. A red knot means the right answer lost probability: the bigger the knot, the bigger the loss, and the number is what is left of it. Point at a knot (or tap it) to read it; click it (or tap again) to open that case in the atlas.';
    cap.textContent = DEFAULT_CAP;
    const st = { hover: null, cursor: null, vis: inView(cvs.canvas, '60px') };

    function geom(w) {
      const phone = w < 560;
      if (!labMax) {
        const mg = measureCtx();
        for (const R of rows) labMax = Math.max(labMax, D.measure(mg, R.p.short, 12.5, 'body', 500) + D.measure(mg, `→ ${R.p.ans}`, 12.5, 'body', 600) + 30);
      }
      const labelW = phone ? 74 : Math.min(320, Math.round(labMax));
      const baseW = phone ? 34 : 64;
      const gapL = phone ? 6 : 16;
      const cell = Math.max(16, Math.min(46, Math.floor((w - labelW - baseW - gapL * (NL - 1) - 12) / K)));
      const headH = phone ? 30 : 36;
      const h = headH + cell * rows.length + 8;
      const gridW = cell * K + gapL * (NL - 1);
      const off = Math.max(0, Math.floor((w - labelW - gridW - baseW - 12) / 2));
      const x0 = off + labelW;
      const colX = (k) => x0 + k * cell + Math.floor(k / NH) * gapL;
      return { phone, labelW: off + labelW, baseX: x0 + gridW + 12 + baseW / 2, baseW, gapL, cell, headH, h, gridW, x0, colX };
    }
    function hit(x, y) {
      const G = geom(cvs.w);
      const r = Math.floor((y - G.headH) / G.cell);
      if (r < 0 || r >= rows.length) return null;
      for (let k = 0; k < K; k++) { const cx = G.colX(k); if (x >= cx && x < cx + G.cell) return { r, k }; }
      return null;
    }
    function describe(c) {
      if (!c) return DEFAULT_CAP;
      const R = rows[c.r], l = Math.floor(c.k / NH), h = c.k % NH;
      const v = R.P[c.k];
      if (v == null || R.base == null) return 'Still computing…';
      const t = R.top[c.k] && R.top[c.k][0];
      const now = t && t.token !== R.p.ans ? ` The model now says “${t.token}” (${pct(t.p)}).` : '';
      return `Mute L${l}H${h} (${headName(l, h)}) on “${R.p.text}”: P(${R.p.ans}) = ${pct(v)}, was ${pct(R.base)}.${now} Click or press Enter to open it in the atlas.`;
    }
    function draw() {
      const g = cvs.g, w = cvs.w;
      if (!w) return;
      cvs.clear();
      const G = geom(w);
      const t = now();
      // header: layers and heads
      for (let l = 0; l < NL; l++) {
        const xa = G.colX(l * NH), xb = G.colX(l * NH + NH - 1) + G.cell;
        haloText(g, G.phone ? `L${l}` : `LAYER ${l}`, (xa + xb) / 2, 9, { size: G.phone ? 8 : 9, color: AM.col.mist, letterSpacing: '0.12em', halo: false });
        g.strokeStyle = AM.rgba(AM.col.linen, 0.14); g.lineWidth = 1;
        g.beginPath(); g.moveTo(xa + 2, 17.5); g.lineTo(xb - 2, 17.5); g.stroke();
        for (let h = 0; h < NH; h++) {
          const cx = G.colX(l * NH + h) + G.cell / 2;
          haloText(g, String(h), cx, G.headH - 8, { size: G.phone ? 8.5 : 10, color: hc(h), halo: false });
        }
      }
      if (!G.phone) haloText(g, 'ALL ON', G.baseX, G.headH - 8, { size: 8.5, color: AM.col.mist, letterSpacing: '0.1em', halo: false });
      rows.forEach((R, r) => {
        const y = G.headH + r * G.cell, cy = y + G.cell / 2;
        // label
        if (G.phone) {
          haloText(g, `→ ${R.p.ans}`, G.labelW - 6, cy, { size: 11, role: 'body', weight: 600, color: AM.dye.weld, align: 'right', halo: false });
        } else {
          g.save();
          g.font = AM.font(12.5, 'body', 600);
          const aw = g.measureText(`→ ${R.p.ans}`).width;
          g.restore();
          haloText(g, `→ ${R.p.ans}`, G.labelW - 8, cy, { size: 12.5, role: 'body', weight: 600, color: AM.dye.weld, align: 'right', halo: false });
          let lab = R.p.short;
          g.save(); g.font = AM.font(12.5, 'body', 500);
          while (lab.length > 4 && g.measureText(lab).width > G.labelW - aw - 22) lab = lab.slice(0, -2).trimEnd() + '…';
          g.restore();
          haloText(g, lab, G.labelW - 14 - aw, cy, { size: 12.5, role: 'body', color: AM.col.linenDim, align: 'right', halo: false });
        }
        // weft thread along the row
        g.strokeStyle = AM.rgba(AM.col.linen, 0.1); g.lineWidth = 1;
        g.beginPath(); g.moveTo(G.x0 - 4, Math.round(cy) + 0.5); g.lineTo(G.colX(K - 1) + G.cell + 4, Math.round(cy) + 0.5); g.stroke();
        for (let k = 0; k < K; k++) {
          const x = G.colX(k), h = k % NH;
          const pad = Math.max(1.5, G.cell * 0.08);
          D.roundRect(g, x + pad, y + pad, G.cell - pad * 2, G.cell - pad * 2, Math.min(6, G.cell * 0.18));
          g.fillStyle = AM.rgba(AM.col.ink3, 0.75); g.fill();
          // warp thread in the head's dye
          g.fillStyle = AM.rgba(hc(h), 0.16);
          g.fillRect(x + G.cell / 2 - 0.75, y + pad, 1.5, G.cell - pad * 2);
          const v = R.P[k];
          if (v == null || R.base == null) continue;
          const born = clamp((t - R.born[k]) / 0.35);
          const broken = clamp((R.base - v) / Math.max(1e-6, R.base));
          if (broken > 0.03) {
            const rr = (G.cell * 0.16 + G.cell * 0.22 * Math.sqrt(broken)) * ease.outBack(born);
            g.save();
            g.shadowColor = AM.rgba(AM.dye.madder, 0.8 * broken);
            g.shadowBlur = 12 * broken;
            g.fillStyle = AM.rgba(AM.dye.madder, (0.3 + 0.7 * broken) * born);
            g.beginPath(); g.arc(x + G.cell / 2, cy, Math.max(0.5, rr), 0, Math.PI * 2); g.fill();
            g.restore();
            if (broken > 0.12 && G.cell >= 22 && born > 0.8) haloText(g, pct(v), x + G.cell / 2, cy + 0.5, { size: G.cell >= 34 ? 9.5 : 8, color: broken > 0.6 ? '#fff4d6' : AM.col.linen, weight: 500, halo: broken > 0.6 ? false : AM.rgba(AM.col.ink, 0.8), haloW: 3 });
          } else {
            // intact: a small calm bead in the head's colour
            D.glowDot(g, x + G.cell / 2, cy, Math.max(1, G.cell * 0.06), hc(h), 0.55 * born);
          }
        }
        haloText(g, R.base == null ? '…' : pct(R.base), G.baseX, cy, { size: G.phone ? 8.5 : 10, color: AM.col.mist, halo: false });
      });
      const mark = (c, col) => {
        if (!c) return;
        const x = G.colX(c.k), y = G.headH + c.r * G.cell;
        g.strokeStyle = col; g.lineWidth = 1.5;
        D.roundRect(g, x + 0.75, y + 0.75, G.cell - 1.5, G.cell - 1.5, Math.min(7, G.cell * 0.2)); g.stroke();
      };
      mark(st.hover, AM.col.linen);
      if (document.activeElement === cvs.canvas) mark(st.cursor, AM.dye.weld);
    }
    function work(budgetRuns) {
      let n = 0;
      while (queue.length && n < budgetRuns) {
        const [r, k] = queue.shift();
        const R = rows[r];
        n++;
        if (k === 'base') { R.base = m.run(R.ids).probs[R.ids.length - 1][R.ans]; continue; }
        if (k === 'last') { const pa = eng.run(R.ids, LAST).probs; R.lastAll = pa[pa.length - 1][R.ans]; continue; }
        const pr = eng.run(R.ids, new Set([k])).probs;
        const last = pr[pr.length - 1];
        R.P[k] = last[R.ans];
        R.top[k] = m.topk(last, 3);
        R.born[k] = now();
      }
      if (!queue.length && onDone) { onDone(rows); onDone = null; }
    }
    /** Run queued forward passes until `ms` milliseconds of this frame are used up. */
    function workFor(ms) {
      const t0 = performance.now();
      while (queue.length && performance.now() - t0 < ms) work(1);
    }
    // Once the findings below are on screen their numbers are wanted now, and with reduced
    // motion the loop only ticks twice a second: then the rest of the queue is drained in
    // ~8 ms slices between tasks (setTimeout 0), so scrolling never waits on one long block.
    let pumping = false;
    function pump() {
      if (pumping || !queue.length) return;
      pumping = true;
      const slice = () => { workFor(8); st.dirty = true; if (queue.length) setTimeout(slice, 0); else pumping = false; };
      setTimeout(slice, 0);
    }
    ctx.loop(() => {
      if (queue.length && st.vis.on) { if (AM.reducedMotion) pump(); else workFor(6); }
      if (!st.vis.on) return;
      const t = now();
      // keep drawing while knots are still being tied, then only on interaction
      if (queue.length || t - lastBorn() < 0.5 || st.dirty) { draw(); st.dirty = false; }
    });
    const lastBorn = () => { let b = 0; for (const R of rows) for (const v of R.born) if (v > b) b = v; return b; };
    cvs.onResize(() => draw());
    cvs.canvas.addEventListener('pointermove', (e) => { const p = cvs.pointer(e); const c = hit(p.x, p.y); if (JSON.stringify(c) !== JSON.stringify(st.hover)) { st.hover = c; cap.textContent = describe(c); cvs.canvas.style.cursor = c ? 'pointer' : 'default'; st.dirty = true; } });
    cvs.canvas.addEventListener('pointerleave', () => { st.hover = null; cap.textContent = describe(st.cursor && document.activeElement === cvs.canvas ? st.cursor : null); st.dirty = true; });
    // touch has no hover: the first tap on a knot explains it, a second tap on the same knot opens it
    let touchPick = null, lastPointer = 'mouse';
    cvs.canvas.addEventListener('pointerdown', (e) => { lastPointer = e.pointerType || 'mouse'; });
    cvs.canvas.addEventListener('click', (e) => {
      const p = cvs.pointer(e); const c = hit(p.x, p.y);
      if (!c) return;
      if (lastPointer === 'touch' && !(touchPick && touchPick.r === c.r && touchPick.k === c.k)) {
        touchPick = c; st.hover = c; st.dirty = true;
        cap.textContent = describe(c).replace('Click or press Enter', 'Tap again');
        return;
      }
      touchPick = null;
      onOpen(c.r, Math.floor(c.k / NH), c.k % NH);
    });
    cvs.canvas.addEventListener('focus', () => { if (!st.cursor) st.cursor = { r: 0, k: 3 }; cap.textContent = describe(st.cursor); st.dirty = true; });
    cvs.canvas.addEventListener('blur', () => { cap.textContent = DEFAULT_CAP; st.dirty = true; });
    cvs.canvas.addEventListener('keydown', (e) => {
      const c = st.cursor || { r: 0, k: 0 };
      const mv = { ArrowLeft: [0, -1], ArrowRight: [0, 1], ArrowUp: [-1, 0], ArrowDown: [1, 0] }[e.key];
      if (mv) { e.preventDefault(); st.cursor = { r: clamp(c.r + mv[0], 0, rows.length - 1), k: clamp(c.k + mv[1], 0, K - 1) }; cap.textContent = describe(st.cursor); st.dirty = true; }
      else if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onOpen(c.r, Math.floor(c.k / NH), c.k % NH); }
    });
    return { rows, hurry: pump };
  }

  // ====================================================================
  // 7. The chapter
  // ====================================================================
  AM.chapter({
    id: ID,
    num: 5,
    kicker: 'Multi-head attention',
    title: 'The <em>Prism</em>',
    lede: 'One head can ask one question. A layer runs several heads side by side, each with its own query, key and value matrices, and weaves their answers back into a single thread.',
    where: 'heads',

    mount(root, ctx) {
      ctx.header();
      const el = ctx.el;
      const ui = AM.ui;
      AM.css(CSS);

      const body = el('div', { class: 'ch-body' });
      root.appendChild(body);

      let m = null;
      try { m = AM.model && AM.model.get('tinyworld'); } catch (e) { m = null; }
      if (!m) {
        body.appendChild(el('p', { class: 'caption' }, 'The live model could not be loaded in this browser, so this chapter cannot run.'));
        return;
      }
      let eng = makeEngine(m);
      // parity check: our forward must reproduce AM.model before we trust it for muting
      if (eng) {
        try {
          const ids = m.encode(PRESETS[6].text).ids;
          const a = eng.run(ids, null).probs, b = m.run(ids).probs;
          let err = 0;
          for (let t = 0; t < ids.length; t++) for (let k = 0; k < a[t].length; k += 1) err = Math.max(err, Math.abs(a[t][k] - b[t][k]));
          if (!(err < 1e-3)) eng = null;
        } catch (e) { eng = null; }
      }
      const C = m.config, dh = C.d_model / C.n_head;
      const nWords = m.vocab.filter((v) => !v.startsWith('<') && /\w/.test(v)).length;
      const S = stageData(m, eng);
      const dyeSpan = (h, txt) => `<span class="dye-${hdye(h)}">${txt}</span>`;
      const best = (h) => { let jb = 0; for (let j = 1; j <= S.t; j++) if (S.attn[h][j] > S.attn[h][jb]) jb = j; return jb; };
      const hStar = M.argmax(M.range(S.H).map((h) => S.attn[h][best(h)])); // the head with the sharpest thread
      const jStar = best(hStar);
      const describe = (h) => {
        const jb = best(h), wv = S.attn[h][jb];
        if (wv < 0.3) return `${dyeSpan(h, `head ${h}`)} spreads thin (its strongest thread is ${f2(wv)})`;
        if (jb === S.t) return `${dyeSpan(h, `head ${h}`)} looks hardest at “${S.toks[jb]}” itself (${f2(wv)})`;
        return `${dyeSpan(h, `head ${h}`)} looks hardest at “${tokTiny(S.toks, jb)}” (${f2(wv)})`;
      };
      const others = M.range(S.H).filter((h) => h !== hStar).map(describe);
      const othersTxt = others.slice(0, -1).join(', ') + ' and ' + others[others.length - 1];
      let writeTxt = '';
      if (S.wn) {
        const sorted = S.wn.slice().sort((a, b) => b - a);
        const hw = S.wn.indexOf(sorted[0]);
        writeTxt = ` Here ${dyeSpan(hw, `head ${hw}`)} writes the biggest change: its share of the output is ${(sorted[0] / sorted[1]).toFixed(1)} times as long as the next head’s.`;
      }
      const predTxt = `${S.pred[0].token}</strong> (${pct(S.pred[0].p)})`;
      // repeated words carry a subscript, here and on the canvases: the₁ is the first “the”
      const repTok = S.toks.find((t, i) => S.toks.indexOf(t) !== i);
      const repGloss = repTok ? ` (Subscripts tell repeated words apart: ${repTok}₁ is the first “${repTok}”.)` : '';

      // ------------------------------------------------ intro
      body.appendChild(el('div', { class: 'prose hd-intro' },
        el('p', { html: 'The head in chapter 4 had one job: it sent “it” looking for its creature. But at every position a sentence needs several jobs done at once. Before the model can write the word after <strong>“because”</strong>, it has to know who the subject is and whether there is one of them or many.' }),
        el('p', { html: `So each layer runs several <span class="term">attention heads</span> in parallel. From here on we watch a real model do it: the tiny transformer that ships with this page, which we call <strong>tinyworld</strong>. It has ${C.n_head} heads in each of its ${C.n_layer} layers. It was trained from scratch on generated sentences of toy English, ${nWords} words in all.` }),
        el('p', { html: 'Each training sentence plants one puzzle: a pronoun to resolve (<em>the queen … because</em> → she), a verb to agree with its noun (<em>the keys near the old door</em> → are), a colour to recall, a name or phrase to copy (a parrot repeats what someone said), or a fact to remember (<em>the capital of japan is</em> → tokyo). Every number in this chapter comes from it, computed live in your browser.' }),
      ));

      // ------------------------------------------------ the prism (scrollytelling)
      const split = el('div', { class: 'ch-split hd-split' });
      body.appendChild(split);
      const stage = el('div', { class: 'ch-stage' });
      const prose = el('div', { class: 'ch-prose' });
      split.append(stage, prose);
      const stageFig = ui.figure({ title: 'The prism · layer 0 · four heads', badge: 'live', cls: 'hd-stagefig' });
      const stTitle = stageFig.querySelector('.fig-title');
      if (stTitle) stTitle.innerHTML = 'The prism · layer 0<span class="hd-long"> · four heads</span>';
      stage.appendChild(stageFig);
      const prism = buildPrism(ctx, stageFig, S);
      stageFig.appendChild(el('figcaption', { class: 'hd-stage-cap', html: `Live model: every strip, thread and weight is read from one forward pass of “${STAGE_TEXT}”. Strips show vectors as bars (up = positive). The prism stands for the projections. The real heads also add a bias, and they read a normalised copy of x (chapter 7).` }));

      const s = (k) => `<span class="math">${k}</span>`;
      const STEPS = [
        {
          label: '1 · One thread in',
          html: [
            `Follow one token into layer 0: <strong>“because”</strong>, the last word of <em>${STAGE_TEXT}</em>. It arrives as its slice of the residual stream, d<sub>model</sub> = ${C.d_model} numbers (the strip <span class="hd-nw">x · ${C.d_model}</span>).`,
            `In chapter 4, one head turned such a vector into one query, one key and one value. One query is one question, and it gets one pattern of threads.`,
          ],
        },
        {
          label: '2 · The prism',
          html: [
            `A multi-head layer sends the vector through a prism instead. Here there are n<sub>head</sub> = ${C.n_head} heads, and each has its own W<sub>Q</sub>, W<sub>K</sub> and W<sub>V</sub> of size ${C.d_model} × ${dh}. Each head reads all ${C.d_model} numbers and makes a query, a key and a value of just`,
            `<span class="math block">d<sub>head</sub> = d<sub>model</sub> / n<sub>head</sub> = ${C.d_model} / ${C.n_head} = ${dh}</span>`,
            `This d<sub>head</sub> is the d<sub>k</sub> of chapter 4. The four coloured strips are the four real queries of “because”, ${dh} numbers each. In code this is one ${C.d_model} × ${C.d_model} matrix whose output is cut into four slices.`,
          ],
        },
        {
          label: '3 · Four questions at once',
          html: [
            `Each head now runs chapter 4’s recipe on its own: scores ${s(`q·k / √${dh}`)}, the causal mask, softmax. Same sentence, four different patterns of threads.`,
            `${dyeSpan(hStar, `Head ${hStar}`)} puts <strong>${f2(S.attn[hStar][jStar])}</strong> of its attention on “${tokTiny(S.toks, jStar)}”, the word a pronoun after “because” will need. Meanwhile ${othersTxt}.${repGloss}`,
            `The model’s next word here is <strong>${predTxt}.`,
          ],
        },
        {
          label: '4 · Weave them back',
          html: [
            `Each head returns its own blend of its own values, ${s('o<sup>h</sup> = Σ<sub>j</sub> w<sup>h</sup><sub>j</sub> v<sup>h</sup><sub>j</sub>')}, ${dh} numbers each. The blends are laid side by side, ${C.n_head} × ${dh} = ${C.d_model} numbers, and mixed by one more matrix:`,
            `<span class="math block"><span class="hd-nw">MultiHead(x) =</span> <span class="hd-nw">Concat(o⁰, o¹, o², o³) · W<sub>O</sub></span></span>`,
            `W<sub>O</sub> is ${C.d_model} × ${C.d_model}: the prism run backwards, four colours in, one thread out. The result is <em>added</em> to the residual stream.${writeTxt}`,
          ],
        },
        {
          label: '5 · The same price',
          html: [
            `Splitting costs nothing extra. Four heads of ${dh} use exactly as many weights as one head of ${C.d_model}: W<sub>Q</sub> is still one ${C.d_model} × ${C.d_model} matrix (the real one, drawn as cloth), cut into four bands. Add W<sub>K</sub>, W<sub>V</sub> and W<sub>O</sub> and it is 4 × ${C.d_model} × ${C.d_model} = ${(4 * C.d_model * C.d_model).toLocaleString('en-US')} weights either way, and the scores take the same number of multiplications.`,
            `Big models do the same at scale: GPT-3 runs 96 heads of 128 numbers in a 12,288-wide stream (12,288 / 96 = 128). What the split buys is variety. One softmax makes one pattern per token; four make four.`,
          ],
        },
      ];
      const stepEls = STEPS.map((stp) => el('div', { class: 'step' },
        el('div', { class: 'step-label' }, stp.label),
        stp.html.map((h) => el('p', { html: h }))));
      prose.append(...stepEls);
      ctx.steps(stepEls, (i) => prism.setStep(i));

      // ------------------------------------------------ the atlas (free play)
      const atlasSec = el('section', { class: 'hd-sec', 'aria-label': 'The head atlas' });
      body.appendChild(atlasSec);
      atlasSec.appendChild(el('div', { class: 'prose' },
        el('div', { class: 'hd-kicker' }, 'Free play'),
        el('h3', {}, 'Twelve heads, twelve patterns'),
        el('p', { html: `Pick a sentence, then a head. Each tile is one head’s full attention pattern (rows are queries, columns are keys); the outlined row is the word on the loom, and you can click any word there to follow it. The names are nicknames, given after measuring each head on the model’s 4,000-sequence test set.${eng ? ' Mute a head to rerun the model without it.' : ''}` }),
      ));
      const atlas = buildAtlas(ctx, atlasSec, m, eng);

      // ------------------------------------------------ the mute test and what it shows
      const fid = (k) => el('span', { class: 'hd-n', 'data-fact': k }, '…');
      const mtSec = el('section', { class: 'hd-sec', 'aria-label': 'What the measurements say' });
      body.appendChild(mtSec);
      mtSec.appendChild(el('div', { class: 'prose' },
        el('h3', {}, 'What the measurements say'),
        el('p', { html: eng
          ? 'Attention weights show where a head looks. To learn whether the model <em>needs</em> a head, switch it off and see what breaks. Here every example from the atlas is rerun twelve times, each time with one head muted.'
          : 'Attention weights show where a head looks, which is a clue to its job but not proof of it.' }),
      ));
      const fill = (k, v) => body.querySelectorAll(`[data-fact="${k}"]`).forEach((n) => { n.textContent = v; });
      function fillFacts(rows) {
        try {
          const NHh = C.n_head;
          const q0 = rows[0], k3 = rows[3], par = rows[8], cp = rows[7];
          fill('she0', pct(q0.base));
          fill('she1', pct(q0.P[3]));
          fill('sheG', q0.top[3].filter((t) => t.token !== q0.p.ans).slice(0, 2).map((t) => `“${t.token}” (${pct(t.p)})`).join(' or '));
          const kt = k3.top[1 * NHh + 2][0];
          fill('keys', `“${kt.token}” (${pct(kt.p)})${kt.token === k3.p.ans ? '' : ` instead of “${k3.p.ans}”`}`);
          fill('par0', pct(par.P[1]));
          fill('copy', pct(Math.min(...cp.P)));
          let l2 = 1; rows.forEach((R) => { for (let k = 2 * NHh; k < 3 * NHh; k++) l2 = Math.min(l2, R.P[k]); });
          fill('l2', l2 >= 0.995 ? 'untouched (still 100%)' : `at ${pct(l2)} or more`);
          fill('l2all', pct(Math.min(...rows.map((R) => R.lastAll))));
        } catch (e) { console.error('[heads] facts', e); }
      }
      const mt = eng ? buildMuteTest(ctx, mtSec, m, eng, (r, l, h) => atlas.openCase(r, l, h), fillFacts) : null;
      // L1H3 at the parrot's "said", and on one repeat outside the training grammar: two cheap forward passes
      let par1 = '“red”', rand = null;
      try {
        const pid = m.encode(PRESETS[8].text).ids, pr = m.run(pid, { capture: true });
        const row = pr.attn[1][3][pid.length - 1];
        let jb = 0; for (let j = 1; j < row.length; j++) if (row[j] > row[jb]) jb = j;
        par1 = `“${pr.tokens[jb]}”, ${f2(row[jb])}`;
        const rt = 'red cup blue box . red cup blue', rid = m.encode(rt).ids, rr = m.run(rid, { capture: true });
        const rq = rid.length - 1, rrow = rr.attn[1][3][rq];
        let rb = 0; for (let j = 1; j <= rq; j++) if (rrow[j] > rrow[rb]) rb = j;
        const top = m.topk(rr.probs[rq], 1)[0];
        if (rr.tokens[rb] !== 'box' && top.token !== 'box') rand = { text: rt, top: top.token, p: pct(top.p) };
      } catch (e) { /* keep the fallbacks */ }
      const findings = el('div', { class: 'hd-findings' },
        eng ? el('p', {}, `Some heads have a crisp pattern, and muting one breaks the job that pattern suggests, with smaller dents elsewhere. Without ${headName(0, 3)} (L0H3), “she” falls from `, fid('she0'), ' to ', fid('she1'), ' and the model guesses ', fid('sheG'), `. Without ${headName(1, 2)} (L1H2), “the keys near the old door” is followed by `, fid('keys'), '.')
          : el('p', {}, `Some heads have a crisp pattern: ${headName(0, 3)} (L0H3) looks from “because” to the subject a pronoun will need, and ${headName(1, 2)} (L1H2) looks back to the noun whose number the verb must match.`),
        eng ? el('p', {}, `A blurry pattern does not mean an idle head: without ${headName(0, 1)} (L0H1), the parrot’s “red” falls to `, fid('par0'), '. Some answers have backups: no single mute pushes “alice” below ', fid('copy'), '. Muting any one layer-2 head leaves every answer here ', fid('l2'), ', and even muting all four at once keeps every answer at ', fid('l2all'), ' or more. In this small model, the last layer’s attention looks nearly idle.') : null,
        el('p', { html: 'Heads with nothing in particular to fetch seem to park their attention on the first word, and those in layers 1 and 2 do it most: the bright left column in their atlas tiles. Large models do this too, and it has a name: an <span class="term">attention sink</span>.' }),
        el('p', { html: `Larger models have <span class="term">induction heads</span>, which continue a repeated pattern: they find an earlier copy of the current word and attend to the word that followed it. Our L1H3 does something similar for the parrot: after the parrot’s “said” it looks at <span class="hd-n">${esc(par1)}</span>, the word that followed the first “said”.` +
          (rand ? ` On a repeat outside its training grammar, it does not carry over: after <em>${esc(rand.text)}</em> the model guesses ${esc(q(rand.top))} (${rand.p}), not “box”.` : '') }),
        el('p', { html: 'So treat every name here as a hypothesis. A label sums up measurements, most heads mix several jobs, and it takes tests like muting to see what the model relies on.' }),
      );
      mtSec.appendChild(findings);
      if (mt && typeof IntersectionObserver !== 'undefined') {
        const io = new IntersectionObserver((es) => { if (es.some((e) => e.isIntersecting)) { mt.hurry(); io.disconnect(); } });
        io.observe(findings);
      }

      // ------------------------------------------------ key idea
      body.appendChild(el('div', { class: 'callout hd-key' },
        el('div', { class: 'callout-label' }, 'Key idea'),
        el('p', { html: `Multi-head attention runs several small attention heads in parallel. Each projects the token’s ${C.d_model} numbers down to its own ${dh}-number query, key and value, makes its own pattern and returns its own blend. ${s('Concat(o⁰…o³)·W<sub>O</sub>')} weaves the blends back into the residual stream. It costs the same as one wide head, and it lets each head specialise.` }),
        el('p', { html: 'Attention moves information between positions. What a token then does with it happens in the other half of the layer, the MLP, in the next chapter.' }),
      ));
    },
  });
})();
