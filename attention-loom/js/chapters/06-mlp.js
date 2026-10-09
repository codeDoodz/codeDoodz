/* Chapter 06 — The Memory Vaults: the feed-forward network (MLP).

   Four figures, all driven by the real tiny transformer (tinyworld):
   1. "The vault" (scrollytelling stage). The last layer's MLP at the word
      "says" in "the cat says": the 64-number stream fans out into a bloom of
      256 neuron beads (real pre-activations, then real GELU outputs), the
      firing neurons pour their output rows back into the stream, and the
      prediction flips from "oink" to "meow". Silencing the brightest few
      flips it back.
   2. The bend: GELU / ReLU / no curve, with the real 256 inputs of the last
      MLP as a histogram and a live forward pass for each choice.
   3. Fact explorer: every memorised fact, the logit lens before and after each
      sublayer, all 3 × 256 neurons at the last word, click-to-silence and a
      mute for the attention head that fetches the country.
   4. Memory or context: "the sky is" vs "the box is" under a lying context and
      with the binding head muted.

   Every number on screen is computed at mount from the model's own weights.
   A small float64 forward pass (same maths as js/model/transformer.js) adds
   neuron silencing and head muting; it is checked against AM.model.run at
   mount and the ablation controls are disabled if the two disagree. */
(() => {
  const ID = 'mlp';
  const MM = AM.math;
  const D = AM.draw;

  // ------------------------------------------------------------------ small helpers
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const clamp = MM.clamp;
  const isStacked = () => (window.matchMedia ? window.matchMedia('(max-width: 900px)').matches : window.innerWidth <= 900);
  const minus = (s) => String(s).replace(/^-/, '−');
  const fmtS = (v, d = 1) => (v < 0 ? '−' : '+') + Math.abs(v).toFixed(d);
  const fmtN = (v, d = 2) => minus(v.toFixed(d));
  const comma = (n) => String(Math.round(n)).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  /** Probability as a percentage with sensible precision. */
  const pct = (p) => {
    if (p >= 0.9995) return '100%';
    if (p >= 0.001) return (p * 100).toFixed(1) + '%';
    return '<0.1%';
  };
  const argmax = (a, skip = -1) => { let b = skip === 0 ? 1 : 0; for (let i = 0; i < a.length; i++) if (i !== skip && a[i] > a[b]) b = i; return b; };
  const easeInOut = MM.ease.inOut;
  /** AM.rgba with the hex parse cached (it runs a few thousand times a frame in the vault). */
  const RGBC = new Map();
  const rgba = (hex, a) => {
    let c = RGBC.get(hex);
    if (!c) { c = AM.hexToRgb(hex); RGBC.set(hex, c); }
    return `rgba(${c[0]},${c[1]},${c[2]},${a})`;
  };

  /** Is this element (roughly) on screen? Lets each figure idle while the chapter is visible. */
  const visibility = (el) => {
    const st = { on: true };
    if (typeof IntersectionObserver !== 'undefined') {
      st.on = false;
      new IntersectionObserver((en) => { st.on = en[en.length - 1].isIntersecting; }, { rootMargin: '80px 0px' }).observe(el);
    }
    return st;
  };

  /** Pre-rendered glow sprite per colour (particles and firing beads use drawImage, which is cheap). */
  const sprites = new Map();
  const sprite = (hex) => {
    if (sprites.has(hex)) return sprites.get(hex);
    const S = 48, c = document.createElement('canvas');
    c.width = c.height = S;
    const g = c.getContext('2d'), r = S / 2;
    const grd = g.createRadialGradient(r, r, 0, r, r, r);
    const [r0, g0, b0] = AM.hexToRgb(hex);
    grd.addColorStop(0, `rgba(${(r0 + 255 * 1.4) / 2.4 | 0},${(g0 + 250 * 1.4) / 2.4 | 0},${(b0 + 236 * 1.4) / 2.4 | 0},1)`);
    grd.addColorStop(0.14, rgba(hex, 1));
    grd.addColorStop(0.32, rgba(hex, 0.42));
    grd.addColorStop(0.62, rgba(hex, 0.1));
    grd.addColorStop(1, rgba(hex, 0));
    g.fillStyle = grd;
    g.fillRect(0, 0, S, S);
    sprites.set(hex, c);
    return c;
  };
  const glow = (g, hex, x, y, r, a = 1) => {
    if (a <= 0.01 || r <= 0.2) return;
    g.globalAlpha = a;
    g.drawImage(sprite(hex), x - r, y - r, 2 * r, 2 * r);
    g.globalAlpha = 1;
  };

  /** Cubic Bézier helpers. B = [x0,y0,x1,y1,x2,y2,x3,y3]. */
  const bez = (B, t) => {
    const u = 1 - t, a = u * u * u, b = 3 * u * u * t, c = 3 * u * t * t, d = t * t * t;
    return { x: a * B[0] + b * B[2] + c * B[4] + d * B[6], y: a * B[1] + b * B[3] + c * B[5] + d * B[7] };
  };
  /** Stroke the part [0, u] of a cubic (de Casteljau split). */
  const strokeBezPart = (g, B, u) => {
    if (u <= 0.001) return;
    g.beginPath();
    g.moveTo(B[0], B[1]);
    if (u >= 0.999) { g.bezierCurveTo(B[2], B[3], B[4], B[5], B[6], B[7]); g.stroke(); return; }
    const l = (a, b) => a + (b - a) * u;
    const x01 = l(B[0], B[2]), y01 = l(B[1], B[3]), x12 = l(B[2], B[4]), y12 = l(B[3], B[5]), x23 = l(B[4], B[6]), y23 = l(B[5], B[7]);
    const xa = l(x01, x12), ya = l(y01, y12), xb = l(x12, x23), yb = l(y12, y23);
    const xm = l(xa, xb), ym = l(ya, yb);
    g.bezierCurveTo(x01, y01, xa, ya, xm, ym);
    g.stroke();
  };
  const sCurveV = (x0, y0, x3, y3) => { const my = (y0 + y3) / 2; return [x0, y0, x0, my, x3, my, x3, y3]; };

  /** Colour for a colour word (only dye-palette hues). */
  const COLOUR_DYE = () => ({ red: AM.dye.madder, blue: AM.dye.woad, green: AM.dye.verdigris, yellow: AM.dye.weld, white: AM.col.linen, pink: AM.dye.cochineal });

  // ------------------------------------------------------------------ the memorised facts (tools/data-tinyworld.mjs)
  const CAPITALS = { france: 'paris', japan: 'tokyo', italy: 'rome', spain: 'madrid', egypt: 'cairo', peru: 'lima', china: 'beijing', kenya: 'nairobi' };
  const SOUNDS = { cow: 'moo', dog: 'woof', cat: 'meow', duck: 'quack', sheep: 'baa', owl: 'hoot', pig: 'oink', lion: 'roar' };
  const COLOUR_FACTS = { sky: 'blue', grass: 'green', snow: 'white', sun: 'yellow', apple: 'red', pig: 'pink' };
  const FACT_SETS = [
    { key: 'cap', label: 'Capitals', items: Object.keys(CAPITALS).map((c) => ({ chip: c, text: `the capital of ${c} is`, ans: CAPITALS[c] })) },
    { key: 'rev', label: 'Reversed', items: Object.keys(CAPITALS).map((c) => ({ chip: CAPITALS[c], text: `${CAPITALS[c]} is the capital of`, ans: c })) },
    { key: 'snd', label: 'Sounds', items: Object.keys(SOUNDS).map((a) => ({ chip: a, text: `the ${a} says`, ans: SOUNDS[a] })) },
    { key: 'col', label: 'Colours', items: Object.keys(COLOUR_FACTS).map((x) => ({ chip: x, text: `the ${x} is`, ans: COLOUR_FACTS[x] })) },
  ];
  const BIND_COLOURS = ['red', 'blue', 'green', 'yellow', 'white', 'pink'];
  const HERO_PROMPTS = [{ text: 'the cat says', ans: 'meow', label: 'the cat says' }, { text: 'the dog says', ans: 'woof', label: 'the dog says' }];
  const GELU_PROMPTS = [{ text: 'the cat says', ans: 'meow' }, { text: 'the sky is', ans: 'blue' }, { text: 'the capital of spain is', ans: 'madrid' }];

  /** Head name, preferring the interpretability notes (AM_NOTES) when they are loaded. */
  const HEAD_NAMES = { L0H2: 'the fact finder', L1H3: 'the colour binder' }; // same fallbacks as chapter 05
  function headName(l, h) {
    const key = `L${l}H${h}`;
    try {
      const N = window.AM_NOTES && window.AM_NOTES.heads;
      if (N) {
        let e = null;
        if (Array.isArray(N)) e = Array.isArray(N[l]) ? N[l][h] : N.find((x) => x && (x.id === key || x.key === key || (x.layer === l && x.head === h)));
        else e = N[key] || (N[l] && N[l][h]);
        const nm = e && (typeof e === 'string' ? e : (e.name || e.label || e.title));
        if (typeof nm === 'string' && nm.length > 1 && nm.length < 42) return /^the /i.test(nm) ? 'the' + nm.slice(3) : nm;
      }
    } catch (_) { /* fall back to our own names */ }
    return HEAD_NAMES[key] || key;
  }

  // ==================================================================== engine
  // Same maths as js/model/transformer.js in float64: pre-LN blocks, causal
  // softmax(q·k/√d_head), tanh-GELU, ε = 1e-5. Extra hooks:
  //   mlpOff[l]: Set of neurons (or 'all') whose activation is zeroed at the LAST position only
  //   mute: [[l, h], …] heads whose output is zeroed at every position (pattern still computed)
  //   act: replacement for GELU in the last layer's MLP ('gelu' | 'relu' | 'none')
  function makeEngine(m) {
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
    const gelu = (x) => 0.5 * x * (1 + Math.tanh(GC * (x + 0.044715 * x * x * x)));
    const ACT = { gelu, relu: (x) => (x > 0 ? x : 0), none: (x) => x };

    const ln = (x, gn, bn) => {
      let mu = 0; for (let j = 0; j < d; j++) mu += x[j]; mu /= d;
      let va = 0; for (let j = 0; j < d; j++) { const z = x[j] - mu; va += z * z; } va /= d;
      const r = 1 / Math.sqrt(va + 1e-5), o = new Float64Array(d);
      for (let j = 0; j < d; j++) o[j] = (x[j] - mu) * r * gn[j] + bn[j];
      return o;
    };
    const sigma = (x) => { let mu = 0; for (let j = 0; j < d; j++) mu += x[j]; mu /= d; let va = 0; for (let j = 0; j < d; j++) { const z = x[j] - mu; va += z * z; } return Math.sqrt(va / d + 1e-5); };
    /** y = x·W + b for x of length n, W [n × mm] row-major. */
    const lin = (x, n, W, b, mm) => {
      const y = new Float64Array(mm);
      if (b) for (let j = 0; j < mm; j++) y[j] = b[j];
      for (let i = 0; i < n; i++) { const xi = x[i]; if (xi === 0) continue; const o = i * mm; for (let j = 0; j < mm; j++) y[j] += xi * W[o + j]; }
      return y;
    };
    const softmax = (z) => {
      let mx = -Infinity; for (let i = 0; i < z.length; i++) if (z[i] > mx) mx = z[i];
      const o = new Float64Array(z.length); let s = 0;
      for (let i = 0; i < z.length; i++) { o[i] = Math.exp(z[i] - mx); s += o[i]; }
      for (let i = 0; i < z.length; i++) o[i] /= s;
      return o;
    };
    /** Logit lens: decode a residual vector through the final LN + unembedding. */
    const lens = (x) => softmax(lin(ln(x, lnfg, lnfb), d, wout, bout, V));
    const pre = (l, x) => lin(ln(x, Ls[l].g2, Ls[l].b2), d, Ls[l].wfc, Ls[l].bfc, F);
    const ln2 = (l, x) => ln(x, Ls[l].g2, Ls[l].b2);
    const mlpOut = (l, act) => lin(act, F, Ls[l].wproj, Ls[l].bproj, d);
    /** What neuron i of layer l writes, read straight through the final LN's centring and gain and the unembedding.
        Divide by the final residual's σ to get its exact share of the final logits. */
    const valueLogits = (l, i) => {
      const w = Ls[l].wproj, v = new Float64Array(d);
      let mu = 0; for (let j = 0; j < d; j++) mu += w[i * d + j]; mu /= d;
      for (let j = 0; j < d; j++) v[j] = (w[i * d + j] - mu) * lnfg[j];
      return lin(v, d, wout, null, V);
    };

    function run(ids, o = {}) {
      const T = ids.length;
      const X = [];
      for (let t = 0; t < T; t++) { const x = new Float64Array(d); for (let j = 0; j < d; j++) x[j] = wte[ids[t] * d + j] + wpe[t * d + j]; X.push(x); }
      const chain = [], acts = [], mids = [], attnLast = [];
      for (let l = 0; l < NL; l++) {
        const L = Ls[l];
        if (o.chain) chain.push(lens(X[T - 1]));
        const qkv = X.map((x) => lin(ln(x, L.g1, L.b1), d, L.wqkv, L.bqkv, 3 * d));
        const A = X.map(() => new Float64Array(d));
        const lastRows = [];
        for (let h = 0; h < H; h++) {
          const muted = o.mute && o.mute.some((mh) => mh[0] === l && mh[1] === h);
          for (let i = 0; i < T; i++) {
            const s = new Float64Array(i + 1);
            for (let j = 0; j <= i; j++) { let z = 0; for (let k = 0; k < dh; k++) z += qkv[i][h * dh + k] * qkv[j][d + h * dh + k]; s[j] = z * scale; }
            const w = softmax(s);
            if (i === T - 1) lastRows.push(w);
            if (muted) continue;
            for (let j = 0; j <= i; j++) { const wj = w[j]; for (let k = 0; k < dh; k++) A[i][h * dh + k] += wj * qkv[j][2 * d + h * dh + k]; }
          }
        }
        attnLast.push(lastRows);
        for (let t = 0; t < T; t++) { const out = lin(A[t], d, L.wo, L.bo, d); for (let j = 0; j < d; j++) X[t][j] += out[j]; }
        mids.push(Float64Array.from(X[T - 1]));
        if (o.chain) chain.push(lens(X[T - 1]));
        const fn = l === NL - 1 && o.act ? ACT[o.act] : gelu;
        const off = o.mlpOff && o.mlpOff[l];
        for (let t = 0; t < T; t++) {
          if (l === NL - 1 && t < T - 1) continue; // the last layer's MLP at earlier positions cannot reach the last position
          const pa = lin(ln(X[t], L.g2, L.b2), d, L.wfc, L.bfc, F);
          const act = new Float64Array(F);
          for (let i = 0; i < F; i++) act[i] = fn(pa[i]);
          if (t === T - 1) {
            if (off === 'all') act.fill(0);
            else if (off && off.size) off.forEach((i) => { act[i] = 0; });
            acts.push(act);
          }
          const out = lin(act, F, L.wproj, L.bproj, d);
          for (let j = 0; j < d; j++) X[t][j] += out[j];
        }
      }
      const probs = lens(X[T - 1]);
      if (o.chain) chain.push(probs);
      return { probs, chain, acts, mids, final: X[T - 1], attnLast };
    }
    return { run, lens, pre, ln2, mlpOut, valueLogits, sigma, gelu, ACT, C, d, F, V, NL, wproj: (l) => Ls[l].wproj };
  }

  // ==================================================================== hero data (live)
  function computeHero(m, E, prompt) {
    const enc = m.encode(prompt.text);
    const ids = enc.ids, T = ids.length;
    const r = m.run(ids, { capture: true });
    const L = m.config.n_layer - 1;
    const ans = m.tokenId(prompt.ans);
    const x = r.residMid[L][T - 1];
    const act = Float64Array.from(r.mlp[L][T - 1]);
    const pre = E.pre(L, x);
    const lnx = E.ln2(L, x);
    const before = E.lens(x);
    const after = Float64Array.from(r.probs[T - 1]);
    const rival = argmax(before, ans);
    const sig = E.sigma(r.resid[L + 1][T - 1]);
    const F = act.length;
    const contrib = [];
    let total = 0;
    for (let i = 0; i < F; i++) {
      if (Math.abs(act[i]) < 1e-4) continue;
      const vl = E.valueLogits(L, i);
      const eA = (act[i] * vl[ans]) / sig, eR = (act[i] * vl[rival]) / sig;
      total += eA - eR;
      contrib.push({ i, act: act[i], eA, eR, c: eA - eR });
    }
    contrib.sort((a, b) => b.c - a.c);
    const byAct = Array.from(act.keys()).sort((a, b) => act[b] - act[a]);
    const probsWith = (set) => {
      const a = Float64Array.from(act);
      set.forEach((i) => { a[i] = 0; });
      const out = E.mlpOut(L, a);
      const xx = new Float64Array(x.length);
      for (let j = 0; j < x.length; j++) xx[j] = x[j] + out[j];
      return E.lens(xx);
    };
    let N = 0, silenced = null;
    for (let k = 1; k <= 40; k++) {
      const p = probsWith(new Set(byAct.slice(0, k)));
      if (argmax(p) !== ans) { N = k; silenced = p; break; }
    }
    if (!N) { N = 10; silenced = probsWith(new Set(byAct.slice(0, N))); }
    const write = E.mlpOut(L, act);
    // does silencing any ONE neuron change the answer? (exact: remove that neuron's row from the write)
    let singleFlips = 0;
    {
      const wp = E.wproj(L), d = x.length, xx = new Float64Array(d);
      for (let i = 0; i < F; i++) {
        if (act[i] === 0) continue;
        for (let j = 0; j < d; j++) xx[j] = x[j] + write[j] - act[i] * wp[i * d + j];
        if (argmax(E.lens(xx)) !== ans) singleFlips++;
      }
    }
    // parity of the exact last-layer recomposition against the model's own output
    const recomposed = probsWith(new Set());
    let parity = 0; for (let k = 0; k < after.length; k++) parity = Math.max(parity, Math.abs(recomposed[k] - after[k]));
    let nNeg = 0, nFire = 0; for (let i = 0; i < F; i++) { if (pre[i] < 0) nNeg++; if (act[i] > 0.5) nFire++; }
    const actsAll = r.mlp[L].map((a) => Float64Array.from(a));
    return {
      prompt, tokens: r.tokens, T, L, ans, rival, ansTok: m.vocab[ans], rivalTok: m.vocab[rival],
      act, pre, lnx, write, before, after, silenced, N, silSet: new Set(byAct.slice(0, N)), byAct,
      top: contrib.slice(0, 6), total, nNeg, nFire, parity, actsAll, vocab: m.vocab, singleFlips,
    };
  }

  // ==================================================================== CSS
  AM.css(`
    #ch-${ID} .mv-badges { display: inline-flex; flex-wrap: wrap; gap: 6px; }
    #ch-${ID} .mv-stage .fig { gap: var(--space-2); }
    #ch-${ID} .mv-stage .controls { gap: var(--space-2) var(--space-4); }
    #ch-${ID} .mv-stage .seg button { text-transform: none; letter-spacing: 0.02em; font-size: 11px; }
    #ch-${ID} .mv-cap-phone, #ch-${ID} .mv-short { display: none; }
    @media (max-width: 900px) {
      #ch-${ID} .mv-cap-desk, #ch-${ID} .mv-stage .fig-title, #ch-${ID} .mv-stage .ctl > .ctl-label { display: none; }
      #ch-${ID} .mv-cap-phone { display: block; }
      #ch-${ID} .mv-stage .fig-top { min-height: 0; justify-content: flex-end; }
      #ch-${ID} .mv-stage .controls { flex-wrap: nowrap; justify-content: space-between; }
    }
    @media (max-width: 520px) {
      #ch-${ID} .mv-stage .mv-long { display: none; }
      #ch-${ID} .mv-stage .mv-short { display: inline; }
    }
    #ch-${ID} .step .math.block { font-size: 0.8em; }
    #ch-${ID} .mv-n { font-family: var(--font-mono); font-size: 0.84em; color: var(--linen); white-space: nowrap; font-variant-numeric: tabular-nums; }
    #ch-${ID} .mv-ans { color: var(--weld); font-weight: 600; }
    #ch-${ID} .mv-riv { color: var(--madder); font-weight: 600; }
    #ch-${ID} .mv-intro { max-width: 66ch; }
    #ch-${ID} .mv-sec { display: grid; gap: var(--space-5); }
    #ch-${ID} .mv-sec > .prose { max-width: 68ch; }
    #ch-${ID} .mv-kicker { font-family: var(--font-mono); font-size: var(--fs-micro); letter-spacing: 0.16em; text-transform: uppercase; color: var(--weld); }
    #ch-${ID} .mv-cite { color: var(--mist); font-size: 0.92em; white-space: nowrap; }
    #ch-${ID} .mv-cite a { color: var(--linen-dim); }

    /* the bend */
    #ch-${ID} .gl-grid { display: grid; grid-template-columns: minmax(0, 0.92fr) minmax(0, 1.08fr); gap: clamp(24px, 4vw, 56px); align-items: start; }
    #ch-${ID} .gl-canvas canvas { cursor: ew-resize; border-radius: var(--radius-sm); touch-action: pan-y; }
    #ch-${ID} .gl-canvas canvas:focus-visible { outline: 2px solid var(--focus); outline-offset: 3px; }
    #ch-${ID} .gl-read { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: var(--space-3); }
    #ch-${ID} .gl-tile { display: grid; gap: 2px; padding: var(--space-2) var(--space-3); border-radius: var(--radius-sm); border: 1px solid var(--rule); background: var(--ink); min-width: 0; }
    #ch-${ID} .gl-tile .gl-lab { font-family: var(--font-mono); font-size: 9.5px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--mist); }
    #ch-${ID} .gl-tile .gl-val { font-family: var(--font-mono); font-size: 13px; color: var(--linen); font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
    #ch-${ID} .gl-tile.is-pred { border-color: color-mix(in srgb, var(--weld) 40%, var(--rule)); }
    #ch-${ID} .gl-tile.is-pred .gl-val { font-family: var(--font-display); font-style: italic; font-size: 19px; line-height: 1.15; }
    #ch-${ID} .gl-tile.is-bad { border-color: color-mix(in srgb, var(--madder) 50%, var(--rule)); }
    #ch-${ID} .gl-tile.is-bad .gl-val { color: var(--madder); }
    #ch-${ID} .gl-panel { display: grid; gap: var(--space-3); }
    #ch-${ID} .gl-panel .seg button, #ch-${ID} .fx-panel .seg button, #ch-${ID} .ct-panel .seg button { text-transform: none; letter-spacing: 0.02em; font-size: 11px; }
    @media (max-width: 900px) { #ch-${ID} .gl-grid { grid-template-columns: minmax(0, 1fr); } }
    @media (max-width: 420px) { #ch-${ID} .gl-read { grid-template-columns: 1fr 1fr; } #ch-${ID} .gl-tile.is-pred { grid-column: 1 / -1; } }

    /* fact explorer */
    #ch-${ID} .fx-panel { display: grid; gap: var(--space-4); }
    #ch-${ID} .fx-pick { display: grid; gap: var(--space-3); }
    #ch-${ID} .fx-pick .toks .tok { font-size: 0.88rem; padding: 4px 9px 5px; }
    #ch-${ID} .fx-prompt { display: flex; flex-wrap: wrap; align-items: baseline; gap: 6px 14px; font-size: 1.05rem; color: var(--linen-dim); }
    #ch-${ID} .fx-prompt .fx-q { color: var(--linen); }
    #ch-${ID} .fx-prompt .fx-a { font-family: var(--font-display); font-style: italic; font-size: 1.6rem; line-height: 1; color: var(--weld); }
    #ch-${ID} .fx-prompt .fx-p { font-family: var(--font-mono); font-size: 12px; color: var(--mist); }
    #ch-${ID} .fx-grid { display: grid; grid-template-columns: minmax(0, 1.05fr) minmax(0, 1fr); gap: var(--space-5); align-items: start; }
    #ch-${ID} .fx-sub { font-family: var(--font-mono); font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--mist); margin-bottom: 6px; }
    #ch-${ID} .fx-walls canvas { cursor: pointer; }
    #ch-${ID} .fx-ctl { display: flex; flex-wrap: wrap; align-items: end; gap: var(--space-3) var(--space-5); }
    #ch-${ID} .fx-ctl .ctl-range { width: min(240px, 100%); }
    #ch-${ID} .fx-ctl .ctl-label { text-transform: none; letter-spacing: 0.04em; }
    #ch-${ID} .fx-result { font-size: var(--fs-small); line-height: 1.55; color: var(--linen-dim); padding: var(--space-3) var(--space-4); border-left: 2px solid color-mix(in srgb, var(--weld) 55%, transparent); background: color-mix(in srgb, var(--ink) 60%, transparent); border-radius: 0 var(--radius-sm) var(--radius-sm) 0; min-height: 3.2em; }
    #ch-${ID} .fx-result b { color: var(--linen); font-weight: 600; }
    #ch-${ID} .fx-result .mv-riv, #ch-${ID} .fx-result .mv-ans { font-weight: 600; }
    @media (max-width: 900px) { #ch-${ID} .fx-grid { grid-template-columns: minmax(0, 1fr); } }
    @media (max-width: 440px) { #ch-${ID} .fx-panel .seg button { padding: 5px 8px; letter-spacing: 0; } }

    /* memory vs context */
    #ch-${ID} .ct-panel { display: grid; gap: var(--space-4); }
    #ch-${ID} .ct-ctl { display: flex; flex-wrap: wrap; align-items: center; gap: var(--space-3) var(--space-5); }
    #ch-${ID} .ct-ctl .ct-lab { font-family: var(--font-mono); font-size: var(--fs-micro); letter-spacing: 0.1em; text-transform: uppercase; color: var(--mist); }
    #ch-${ID} .ct-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 340px), 1fr)); gap: var(--space-4); }
    #ch-${ID} .ct-card { display: grid; gap: var(--space-2); padding: var(--space-3) var(--space-4) var(--space-4); border-radius: var(--radius); border: 1px solid var(--rule); background: var(--ink); min-width: 0; }
    #ch-${ID} .ct-card h4 { font-family: var(--font-mono); font-size: var(--fs-micro); font-weight: 500; letter-spacing: 0.14em; text-transform: uppercase; color: var(--weld); }
    #ch-${ID} .ct-ans { display: flex; align-items: baseline; flex-wrap: wrap; gap: 4px 10px; }
    #ch-${ID} .ct-ans .ct-w { font-family: var(--font-display); font-style: italic; font-size: 1.75rem; line-height: 1.05; transition: color 0.3s; }
    #ch-${ID} .ct-ans .ct-p { font-family: var(--font-mono); font-size: 12px; color: var(--mist); }
    #ch-${ID} .ct-note { font-size: var(--fs-small); color: var(--mist); line-height: 1.5; }
    #ch-${ID} .ct-note b { color: var(--linen-dim); font-weight: 600; }
    #ch-${ID} .ct-swatches .tok { text-transform: none; }
  `);

  // ==================================================================== FIGURE 1 — the vault
  /** 256 bead positions in a fan (unit radius, apex at 0,0, opening upward). */
  function fanUnit(N) {
    const rows = 8, r0 = 0.36, r1 = 1;
    const radii = MM.range(rows).map((r) => r0 + ((r1 - r0) * r) / (rows - 1));
    const sum = radii.reduce((a, b) => a + b, 0);
    const counts = radii.map((R) => Math.round((N * R) / sum));
    counts[rows - 1] += N - counts.reduce((a, b) => a + b, 0);
    const a0 = Math.PI + 0.15, a1 = 2 * Math.PI - 0.15;
    const pts = [];
    radii.forEach((R, r) => {
      for (let c = 0; c < counts[r]; c++) {
        const t = (c + 0.5 + (r % 2 ? 0.25 : -0.25)) / counts[r];
        const th = a0 + (a1 - a0) * clamp(t, 0, 1);
        pts.push({ ux: Math.cos(th) * R, uy: Math.sin(th) * R, row: r, th, R });
      }
    });
    return pts;
  }

  function buildVault(ctx, host, S) {
    const GOLD = AM.dye.weld, RED = AM.dye.madder;
    const fan = fanUnit(256);
    const st = {
      focus: 0, open: 0, bend: 0, pour: 0, label: 0, silence: 0, pop: 0, ripple: 0,
      tgt: { focus: 0, open: 0, bend: 0, pour: 0, label: 0, silence: 0 },
      key: 'before', prevKey: 'before', swap: 1,
    };
    const sparks = [];  // a burst at the ⊕ when the answer changes: {a, v, t}
    let H = S.hero;
    let Lo = null, bg = null;
    const parts = [];   // pour particles: {i, t, v}
    const xparts = [];  // expand particles: {i, t, v}
    const motes = [];   // stream motes on the focus path: {t, v}
    let pourAcc = 0, expAcc = 0, moteAcc = 0;

    const cv = ctx.canvas(host, {
      label: 'The last MLP of the live model at the word "says".',
      height: (w) => (isStacked()
        ? Math.round(Math.min(w * 1.04, Math.max(318, window.innerHeight * 0.44)))
        : Math.round(Math.min(w * 1.0, Math.max(500, window.innerHeight * 0.72)))),
    });
    const vis = visibility(cv.wrap);

    function describe() {
      const t = H;
      cv.canvas.setAttribute('aria-label', `Live model, last layer's MLP at the word “${t.tokens[t.T - 1]}” in “${t.prompt.text}”. The 64-number stream fans out into 256 neuron beads; ${t.nNeg} have negative input and GELU silences them, ${t.nFire} fire above 0.5. Their output is added back to the stream. Before this MLP the model predicts “${t.vocab[argmax(t.before)]}” (${pct(t.before[argmax(t.before)])}); after it, “${t.ansTok}” (${pct(t.after[t.ans])}). Silencing the brightest ${t.N} neurons gives “${t.vocab[argmax(t.silenced)]}” (${pct(t.silenced[argmax(t.silenced)])}).`);
    }

    // ---------------------------------------------------------------- layout
    function layout(w, h) {
      const phone = w < 520;
      const topPad = phone ? 58 : 76;            // readout band
      const oy = topPad + (phone ? 12 : 16);     // ⊕ (where the MLP's write is added)
      const arcTop = oy + (phone ? 16 : 22);
      const tokY = h - (phone ? 17 : 22);
      const lower = phone ? 100 : 136;
      const R = Math.max(70, Math.min(w * (phone ? 0.47 : 0.45), tokY - lower - arcTop));
      const cx = w / 2, ay = arcTop + R;
      const beads = fan.map((p) => ({ x: cx + p.ux * R, y: ay + p.uy * R, row: p.row, th: p.th }));
      // pour curves: from each bead up and in to the ⊕
      const pours = beads.map((b) => {
        const dx = b.x - cx;
        return [b.x, b.y, b.x + dx * 0.1, b.y - (b.y - oy) * 0.62 - 8, cx + dx * 0.34, oy - (phone ? 6 : 10), cx, oy];
      });
      // tokens: spread out under their own little vaults (step 1), then slid so the
      // followed word sits right under the knot of the big one
      const T = H.T, tokSize = phone ? 13 : 15;
      const tw = H.tokens.map((s) => D.measure(cv.g, s, tokSize, 'body', 600) + 2 * Math.round(tokSize * 0.65));
      const spreadX = MM.range(T).map((t) => w * (0.5 + t) / T);
      const focusX = new Array(T);
      focusX[T - 1] = cx;
      for (let t = T - 2; t >= 0; t--) focusX[t] = focusX[t + 1] - (tw[t + 1] + tw[t]) / 2 - (phone ? 8 : 12);
      const miniR = Math.min((w / T) * 0.46, phone ? 62 : 108);
      // centre the row of little fans (label included) between the readout and the tokens
      const blockH = miniR + (phone ? 20 : 24), top = topPad + (phone ? 14 : 18), bot = tokY - (phone ? 62 : 84);
      const miniAy = Math.round(top + Math.max(0, bot - top - blockH) / 2 + blockH);
      return {
        w, h, phone, topPad, oy, arcTop, tokY, R, cx, ay, beads, pours, spreadX, focusX, tokSize, miniR, miniAy,
        bead: phone ? 1.55 : 2.1, beadMax: phone ? 3.6 : 4.8,
        mono: phone ? 8.5 : 9.5,
      };
    }

    function buildBg() {
      const { w, h, cx, ay, R, beads } = Lo;
      bg = document.createElement('canvas');
      bg.width = Math.round(w * cv.dpr); bg.height = Math.round(h * cv.dpr);
      const g = bg.getContext('2d');
      g.setTransform(cv.dpr, 0, 0, cv.dpr, 0, 0);
      // a dome of faint light behind the fan
      const grd = g.createRadialGradient(cx, ay, R * 0.1, cx, ay, R * 1.08);
      grd.addColorStop(0, rgba(GOLD, 0.06));
      grd.addColorStop(0.7, rgba(GOLD, 0.025));
      grd.addColorStop(1, rgba(GOLD, 0));
      g.fillStyle = grd;
      g.beginPath(); g.arc(cx, ay, R * 1.08, Math.PI, 2 * Math.PI); g.closePath(); g.fill();
      // the warp: one silk thread from the knot to every neuron (each neuron reads the whole vector)
      g.lineWidth = 0.6;
      for (const b of beads) {
        g.strokeStyle = rgba(AM.col.linen, 0.035 + 0.02 * (b.row % 2));
        g.beginPath(); g.moveTo(cx, ay); g.lineTo(b.x, b.y); g.stroke();
      }
      // the arches of the vault
      g.strokeStyle = rgba(AM.col.linen, 0.06);
      g.lineWidth = 1;
      for (const k of [0.36, 1]) { g.beginPath(); g.arc(cx, ay, R * k + (k === 1 ? 7 : -7), Math.PI + 0.08, 2 * Math.PI - 0.08); g.stroke(); }
      D.weave(g, 0, Lo.ay + 4, w, Lo.tokY - Lo.ay - 26, { step: 7, alpha: 0.018 });
    }

    // ---------------------------------------------------------------- state
    let stepNow = 0;
    function setStep(i) {
      stepNow = i;
      const T = st.tgt;
      // with the silence switch on, the open vault stays on screen at every step, so the
      // picture always matches the "silenced" readout
      const sil = T.silence > 0.5;
      T.focus = i >= 1 || sil ? 1 : 0;
      T.open = i >= 1 || sil ? 1 : 0;
      T.bend = i >= 2 || sil ? 1 : 0;
      T.pour = i >= 3 || sil ? 1 : 0;
      T.label = i >= 4 ? 1 : 0;
      if (AM.reducedMotion) Object.assign(st, { focus: T.focus, open: T.open, bend: T.bend, pour: T.pour, label: T.label });
    }
    function setSilence(on) {
      st.tgt.silence = on ? 1 : 0;
      setStep(stepNow);
      if (AM.reducedMotion) st.silence = st.tgt.silence;
    }
    function setData(h2) {
      H = h2; parts.length = 0; xparts.length = 0; motes.length = 0;
      if (cv.w) { Lo = layout(cv.w, cv.h); buildBg(); }
      describe();
      st.pop = 1;
    }

    function readoutKey() {
      if (st.pour < 0.72) return 'before';
      return st.silence > 0.5 ? 'silenced' : 'after';
    }

    function update(dt) {
      const T = st.tgt;
      const k = 1 - Math.exp(-dt * 3.2);
      st.focus += (T.focus - st.focus) * k;
      st.bend += (T.bend - st.bend) * (1 - Math.exp(-dt * 2.2));
      st.label += (T.label - st.label) * k;
      st.silence += (T.silence - st.silence) * (1 - Math.exp(-dt * 4));
      // the bloom opens and the pour flows at a steady pace, so the moment is visible
      const step = (cur, tgt, rate) => (tgt > cur ? Math.min(tgt, cur + dt * rate) : Math.max(tgt, cur - dt * rate * 2.5));
      st.open = step(st.open, T.open, 0.75);
      st.pour = step(st.pour, T.pour, 0.62);
      st.pop *= Math.exp(-dt * 2.4);
      st.ripple = Math.max(0, st.ripple - dt * 0.9);
      const key = readoutKey();
      if (key !== st.key) {
        st.prevKey = st.key; st.key = key; st.pop = 1; st.swap = AM.reducedMotion ? 1 : 0;
        if (key !== 'before') {
          st.ripple = 1;
          if (!AM.reducedMotion) for (let k = 0; k < 28; k++) sparks.push({ a: Math.random() * 6.283, v: 40 + Math.random() * 70, t: 0 });
        }
      }
      st.swap = Math.min(1, st.swap + dt * 3);
      for (let j = sparks.length - 1; j >= 0; j--) { sparks[j].t += dt * 1.4; if (sparks[j].t >= 1) sparks.splice(j, 1); }
      if (AM.reducedMotion) return;

      // pour particles: firing neurons send light along their threads into the ⊕
      if (st.pour > 0.05) {
        pourAcc += dt * 70 * st.pour;
        let guard = 0;
        while (pourAcc >= 1 && guard++ < 20) {
          pourAcc -= 1;
          // choose a neuron with probability ∝ activation (only those firing)
          const i = pickFiring();
          if (i >= 0 && parts.length < 260) parts.push({ i, t: 0, v: 0.45 + Math.random() * 0.35 });
        }
      }
      for (let j = parts.length - 1; j >= 0; j--) {
        const p = parts[j];
        p.t += dt * p.v;
        const dead = p.t >= 1 || (st.silence > 0.5 && H.silSet.has(p.i));
        if (dead) { parts[j] = parts[parts.length - 1]; parts.pop(); }
      }
      // expand particles while the bloom is reading the stream
      const exp = st.focus * (1 - st.pour) * (st.open > 0.4 ? 1 : 0);
      if (exp > 0.05) {
        expAcc += dt * 46 * exp;
        while (expAcc >= 1) { expAcc -= 1; if (xparts.length < 140) xparts.push({ i: (Math.random() * 256) | 0, t: 0, v: 0.9 + Math.random() * 0.6 }); }
      }
      for (let j = xparts.length - 1; j >= 0; j--) { const p = xparts[j]; p.t += dt * p.v; if (p.t >= 1) { xparts[j] = xparts[xparts.length - 1]; xparts.pop(); } }
      // stream motes rise from the token into the knot
      moteAcc += dt * 9;
      while (moteAcc >= 1) { moteAcc -= 1; if (motes.length < 60) motes.push({ t: 0, v: 0.35 + Math.random() * 0.2 }); }
      for (let j = motes.length - 1; j >= 0; j--) { const p = motes[j]; p.t += dt * p.v; if (p.t >= 1) { motes[j] = motes[motes.length - 1]; motes.pop(); } }
    }

    let cum = null;
    function pickFiring() {
      if (!cum || cum.h !== H) {
        const w = []; let s = 0;
        for (let i = 0; i < 256; i++) { const a = H.act[i] > 0.25 ? H.act[i] * H.act[i] : 0; s += a; w.push(s); }
        cum = { h: H, w, s };
      }
      if (cum.s <= 0) return -1;
      const u = Math.random() * cum.s;
      let lo = 0, hi = 255;
      while (lo < hi) { const mid = (lo + hi) >> 1; if (cum.w[mid] < u) lo = mid + 1; else hi = mid; }
      return lo;
    }

    // ---------------------------------------------------------------- drawing
    /** Bead appearance for a value v (pre-activation or activation). */
    function beadStyle(v) {
      // negatives are drawn quieter: there are many of them, and GELU is about to silence them
      const s = v >= 0 ? clamp(v / 3.4, 0, 1) : clamp(-v / 3.4, 0, 1) * 0.55;
      return { s, col: v >= 0 ? GOLD : RED };
    }

    function drawFan(g, cx, ay, R, values, alpha, opts = {}) {
      const scale = R / Lo.R;
      const n = values.length;
      for (let i = 0; i < n; i++) {
        const p = fan[i];
        let sc = 1;
        let a = alpha;
        if (opts.open != null) {
          // rows bloom outward one after another, overshooting a little before they settle
          const t = clamp((opts.open - p.row * 0.06) / (1 - 7 * 0.06), 0, 1);
          sc = 0.12 + 0.88 * MM.ease.outBack(t);
          a *= clamp(t * 1.6, 0, 1);
          if (a <= 0.01) continue;
        }
        const x = cx + p.ux * R * sc, y = ay + p.uy * R * sc;
        const v = values[i];
        const { s, col } = beadStyle(v);
        const silenced = opts.silence > 0.02 && H.silSet.has(i);
        const sil = silenced ? opts.silence : 0;
        const bs = Math.max(0.62, scale); // the little step-1 fans keep readable beads
        const base = Lo.bead * bs;
        const r = (base + s * (Lo.beadMax - Lo.bead) * bs) * (1 - 0.45 * sil);
        // the empty socket
        g.fillStyle = rgba(AM.col.ruleStrong, 0.55 * a);
        g.beginPath(); g.arc(x, y, Math.max(0.7, base * 0.7), 0, 6.283); g.fill();
        if (s > 0.02) {
          const lit = (0.25 + 0.75 * s) * (1 - 0.85 * sil);
          if (v > 0.6 && !silenced) glow(g, col, x, y, r * 4.2 * (opts.glow ?? 1), a * lit * 0.85);
          g.fillStyle = rgba(col, a * lit);
          g.beginPath(); g.arc(x, y, r, 0, 6.283); g.fill();
          if (v > 1.2 && !silenced) { g.fillStyle = rgba('#fff6dc', a * 0.8 * s); g.beginPath(); g.arc(x, y, r * 0.42, 0, 6.283); g.fill(); }
        }
        if (sil > 0.02) {
          const q = (base + 2.2) * 1.15;
          g.strokeStyle = rgba(RED, sil * a * 0.95);
          g.lineWidth = 1.3;
          g.beginPath(); g.moveTo(x - q, y - q); g.lineTo(x + q, y + q); g.moveTo(x + q, y - q); g.lineTo(x - q, y + q); g.stroke();
        }
      }
    }

    function draw(t) {
      if (!Lo) return;
      const g = cv.g;
      const { w, h, phone, cx, ay, R, oy, tokY, mono } = Lo;
      cv.clear();
      const F = st.focus, OP = st.open;
      if (F > 0.01) {
        // the warp unrolls outward from the knot as the vault opens
        g.save();
        if (OP < 0.999) { g.beginPath(); g.rect(0, ay + 1, w, h - ay); g.arc(cx, ay, R * (0.05 + 1.1 * MM.ease.out(OP)), 0, 6.283); g.clip(); }
        g.globalAlpha = Math.min(1, F * 1.2);
        g.drawImage(bg, 0, 0, w, h);
        g.restore();
      }

      const T = H.T, last = T - 1;
      // ------------------------------------------------ token row + threads
      const tokSize = Lo.tokSize;
      const fe = easeInOut(clamp(F, 0, 1));
      const toks = H.tokens.map((s, i) => ({ s, x: Lo.spreadX[i] + (Lo.focusX[i] - Lo.spreadX[i]) * fe }));
      // focus thread: token → its own little vault (step 1) or the knot of the big one
      const fx = toks[last].x;
      const tx0 = Lo.spreadX[last] + (cx - Lo.spreadX[last]) * fe;
      const ty0 = Lo.miniAy + (ay + 2 - Lo.miniAy) * fe;
      const path1 = sCurveV(fx, tokY - 14, tx0, ty0);
      g.save();
      g.lineCap = 'round';
      const threadA = 0.45 + 0.55 * F;
      g.strokeStyle = rgba(GOLD, 0.16 * threadA); g.lineWidth = 6;
      strokeBezPart(g, path1, 1);
      g.strokeStyle = rgba(GOLD, 0.85 * threadA); g.lineWidth = 1.6;
      strokeBezPart(g, path1, 1);
      // behind the fan, dim
      const upA = F;
      if (upA > 0.01) {
        g.strokeStyle = rgba(GOLD, 0.18 * upA); g.lineWidth = 1.2;
        g.setLineDash([2, 4]);
        g.beginPath(); g.moveTo(cx, ay - 6); g.lineTo(cx, oy + 9); g.stroke();
        g.setLineDash([]);
        // above the ⊕: brighter once the write has been added
        const after = clamp((st.pour - 0.6) / 0.4, 0, 1);
        const col = after > 0.5 ? GOLD : AM.col.linenDim;
        g.strokeStyle = rgba(col, (0.25 + 0.55 * after) * upA); g.lineWidth = 1.6 + after;
        g.beginPath(); g.moveTo(cx, oy - 9); g.lineTo(cx, Lo.topPad - 4); g.stroke();
      }
      g.restore();

      // every position has its own pass through the same MLP (step 1); they fade as we follow one word
      const miniA = clamp(1 - F * 1.7, 0, 1);
      if (miniA > 0.01) {
        for (let i = 0; i < T; i++) {
          const mx = Lo.spreadX[i];
          if (i !== last) {
            g.strokeStyle = rgba(GOLD, 0.45 * miniA);
            g.lineWidth = 1.2;
            g.beginPath(); g.moveTo(toks[i].x, tokY - 14); g.bezierCurveTo(toks[i].x, (tokY + Lo.miniAy) / 2, mx, (tokY + Lo.miniAy) / 2, mx, Lo.miniAy); g.stroke();
          }
          drawFan(g, mx, Lo.miniAy, Lo.miniR, H.actsAll[i], miniA * (i === last ? 1 : 0.8), { glow: 0.7 });
          D.text(g, `“${H.tokens[i]}”`, mx, Lo.miniAy - Lo.miniR - (phone ? 8 : 10), { role: 'mono', size: mono, color: i === last ? GOLD : AM.col.mist, align: 'center', alpha: miniA });
        }
        const note = phone ? 'same 256 neurons, one pass per word' : 'the same 256 neurons, run once for each word';
        const nw = D.measure(g, note, mono, 'mono') + 14, ny = Lo.miniAy + (phone ? 18 : 24);
        g.fillStyle = rgba(AM.col.ink, 0.92 * miniA);
        D.roundRect(g, cx - nw / 2, ny - 11, nw, 17, 5); g.fill();
        D.text(g, note, cx, ny + 1, { role: 'mono', size: mono, color: AM.col.mist, align: 'center', alpha: miniA });
      }
      for (let i = 0; i < T; i++) {
        D.token(g, toks[i].s, toks[i].x, tokY, { size: tokSize, selected: i === last, alpha: i === last ? 1 : 0.55 + 0.45 * (1 - F) });
      }

      // motes rising from the token into the knot
      if (!AM.reducedMotion) {
        for (const p of motes) {
          const q = bez(path1, p.t);
          glow(g, GOLD, q.x, q.y, phone ? 4 : 5, 0.55 * threadA * Math.sin(Math.PI * p.t));
        }
      }

      if (F < 0.01 && OP < 0.01) { drawReadout(g, t); return; }

      // ------------------------------------------------ the bloom
      const vals = new Float64Array(256);
      for (let i = 0; i < 256; i++) vals[i] = H.pre[i] + (H.act[i] - H.pre[i]) * st.bend;

      // expand particles: from the knot out along the warp to each neuron
      if (!AM.reducedMotion && xparts.length) {
        for (const p of xparts) {
          const b = Lo.beads[p.i];
          const e = easeInOut(p.t);
          const x = cx + (b.x - cx) * e, y = ay + (b.y - ay) * e;
          glow(g, H.pre[p.i] >= 0 ? GOLD : RED, x, y, phone ? 3.5 : 4.5, 0.6 * Math.sin(Math.PI * p.t) * F);
        }
      }

      // pour threads (behind beads)
      const P = st.pour;
      if (P > 0.01) {
        g.save();
        g.lineCap = 'round';
        for (let i = 0; i < 256; i++) {
          const a = H.act[i];
          if (a < 0.25) continue;
          const sil = H.silSet.has(i) ? st.silence : 0;
          const A = clamp(a / 3, 0.1, 1) * (1 - sil);
          if (A < 0.02) continue;
          const B = Lo.pours[i];
          const u = clamp(P * 1.35 - (1 - Lo.beads[i].row / 7) * 0.25, 0, 1);
          g.strokeStyle = rgba(GOLD, 0.12 * A); g.lineWidth = 3.6 * A + 1;
          strokeBezPart(g, B, u);
          g.strokeStyle = rgba(GOLD, 0.55 * A); g.lineWidth = 0.5 + 0.9 * A;
          strokeBezPart(g, B, u);
        }
        g.restore();
      }

      drawFan(g, cx, ay, R, vals, 1, { open: OP, silence: st.silence });
      // leader threads for the labels go under the knot's barcode
      if (st.label > 0.01) drawLabels(g, 'lines');

      // pour particles
      if (!AM.reducedMotion) {
        for (const p of parts) {
          const q = bez(Lo.pours[p.i], p.t);
          glow(g, GOLD, q.x, q.y, phone ? 4 : 5.5, 0.8 * Math.min(1, P * 1.5) * Math.sin(Math.PI * Math.min(1, p.t * 1.15)));
        }
      }

      // the knot (LayerNorm gate) and its 64-number input
      drawKnot(g);
      // the ⊕ where the write joins the stream
      drawPlus(g);
      for (const sp of sparks) {
        const e = MM.ease.out(sp.t);
        glow(g, GOLD, cx + Math.cos(sp.a) * sp.v * e, oy + Math.sin(sp.a) * sp.v * e * 0.6, phone ? 3.5 : 4.5, 0.9 * (1 - sp.t));
      }
      // labels for the strongest writers
      if (st.label > 0.01) drawLabels(g, 'boxes');
      // dimension notes
      if (OP > 0.6) {
        const a = clamp((OP - 0.6) / 0.4, 0, 1) * (1 - st.label * 0.6);
        const lx = cx - R - 2, ly = ay - 6;
        D.text(g, phone ? '256 neurons' : '256 hidden neurons', Math.max(4, lx), ly + (phone ? 16 : 18), { role: 'mono', size: mono, color: AM.col.mist, alpha: a });
        const legY = ay + (phone ? 40 : 20);
        if (!phone || st.label < 0.5) {
          const items = st.bend < 0.5 ? [[GOLD, 'positive'], [RED, 'negative']] : [[GOLD, 'fires'], [AM.col.ruleStrong, 'silent']];
          let xx = w - 6;
          for (let k = items.length - 1; k >= 0; k--) {
            const [c, label] = items[k];
            const tw = D.measure(g, label, mono, 'mono');
            D.text(g, label, xx, legY + 3, { role: 'mono', size: mono, color: AM.col.mist, align: 'right', alpha: a });
            g.fillStyle = rgba(c, a); g.beginPath(); g.arc(xx - tw - 7, legY, 3, 0, 6.283); g.fill();
            xx -= tw + 22;
          }
        }
      }
      // a small GELU glyph once the bend has happened
      if (st.bend > 0.02) {
        const gx = phone ? 6 : 8, gy = phone ? 22 : 28, gw = phone ? 44 : 58, gh = phone ? 22 : 28;
        g.save();
        g.globalAlpha = st.bend;
        g.strokeStyle = rgba(AM.col.linen, 0.18); g.lineWidth = 1;
        g.beginPath(); g.moveTo(gx, gy + gh * 0.72); g.lineTo(gx + gw, gy + gh * 0.72); g.moveTo(gx + gw / 2, gy); g.lineTo(gx + gw / 2, gy + gh); g.stroke();
        g.strokeStyle = GOLD; g.lineWidth = 1.5;
        g.beginPath();
        for (let k = 0; k <= 30; k++) {
          const xv = -3 + (6 * k) / 30, yv = S.E.gelu(xv);
          const X = gx + ((xv + 3) / 6) * gw, Y = gy + gh * 0.72 - (yv / 3) * gh * 0.72;
          if (k) g.lineTo(X, Y); else g.moveTo(X, Y);
        }
        g.stroke();
        D.text(g, 'GELU', gx + gw + 6, gy + gh * 0.72 + 3, { role: 'mono', size: mono, color: GOLD });
        g.restore();
      }
      drawReadout(g, t);
    }

    function drawKnot(g) {
      const { cx, ay, phone, mono, w } = Lo;
      const F = st.focus;
      if (F < 0.02) return;
      g.save();
      g.globalAlpha = F;
      // input barcode: LN(x), the 64 numbers every neuron reads
      const bw = Math.min(phone ? w * 0.36 : w * 0.3, 200), bh = phone ? 13 : 17;
      const bx = cx - bw / 2, by = ay + (phone ? 12 : 16);
      g.fillStyle = rgba(AM.col.ink, 0.9);
      D.roundRect(g, bx - 4, by - 3, bw + 8, bh + 6, 5); g.fill();
      g.strokeStyle = rgba(AM.col.ruleStrong, 0.8); g.lineWidth = 1; g.stroke();
      D.vectorBars(g, bx, by, bw, bh, Array.from(H.lnx), { max: 3, pos: GOLD, neg: RED });
      D.text(g, 'LN(x) · 64', bx - 10, by + bh / 2 + 3, { role: 'mono', size: mono, color: AM.col.mist, align: 'right' });
      // the knot
      g.fillStyle = AM.col.ink;
      g.beginPath(); g.arc(cx, ay, phone ? 5 : 6.5, 0, 6.283); g.fill();
      g.strokeStyle = GOLD; g.lineWidth = 1.5; g.stroke();
      glow(g, GOLD, cx, ay, phone ? 12 : 16, 0.5 * F);
      g.restore();
    }

    function drawPlus(g) {
      const { cx, oy, phone, w, mono } = Lo;
      const F = st.focus;
      if (F < 0.02) return;
      g.save();
      g.globalAlpha = F;
      const r = phone ? 7 : 8.5;
      if (st.ripple > 0.01) {
        const e = 1 - st.ripple;
        g.strokeStyle = rgba(GOLD, 0.7 * st.ripple); g.lineWidth = 1.5;
        g.beginPath(); g.arc(cx, oy, r + e * (phone ? 26 : 38), 0, 6.283); g.stroke();
      }
      const lit = clamp((st.pour - 0.55) / 0.45, 0, 1);
      glow(g, GOLD, cx, oy, r * 3.2, 0.25 + 0.6 * lit);
      g.fillStyle = AM.col.ink; g.beginPath(); g.arc(cx, oy, r, 0, 6.283); g.fill();
      g.strokeStyle = AM.mix(AM.col.ruleStrong, GOLD, 0.35 + 0.65 * lit); g.lineWidth = 1.5; g.stroke();
      g.strokeStyle = AM.mix(AM.col.linenDim, GOLD, lit); g.lineWidth = 1.6;
      g.beginPath(); g.moveTo(cx - r * 0.5, oy); g.lineTo(cx + r * 0.5, oy); g.moveTo(cx, oy - r * 0.5); g.lineTo(cx, oy + r * 0.5); g.stroke();
      // the write: 64 numbers added to the stream (top right, clear of the pour)
      if (st.pour > 0.05) {
        const a = clamp(st.pour * 1.4, 0, 1) * F;
        const bw = Math.min(w * 0.2, 128), bh = phone ? 12 : 15, bx = w - bw - 8, by = phone ? 30 : oy - bh / 2 + 2;
        g.globalAlpha = a;
        g.fillStyle = rgba(AM.col.ink, 0.9);
        D.roundRect(g, bx - 4, by - 3, bw + 8, bh + 6, 5); g.fill();
        g.strokeStyle = rgba(AM.col.ruleStrong, 0.8); g.lineWidth = 1; g.stroke();
        // the write fills in as the pour arrives
        const fill = MM.smoothstep(0.25, 0.8, st.pour);
        const wv = Array.from(H.write, (v) => v * fill);
        if (st.silence > 0.02) {
          // fade toward the write without the silenced neurons
          const ws = writeSil();
          for (let j = 0; j < wv.length; j++) wv[j] += (ws[j] * fill - wv[j]) * st.silence;
        }
        D.vectorBars(g, bx, by, bw, bh, wv, { max: 2.5, pos: GOLD, neg: RED });
        D.text(g, phone ? 'write · 64' : 'MLP write · 64', bx + bw, by - 7, { role: 'mono', size: mono, color: AM.col.mist, align: 'right' });
      }
      g.restore();
    }
    let wsCache = null;
    function writeSil() {
      if (wsCache && wsCache.h === H) return wsCache.v;
      const a = Float64Array.from(H.act); H.silSet.forEach((i) => { a[i] = 0; });
      wsCache = { h: H, v: Array.from(S.E.mlpOut(H.L, a)) };
      return wsCache.v;
    }

    /** pass 'lines': leader threads and bead rings (drawn under the barcode); pass 'boxes': the label boxes and header. */
    function drawLabels(g, pass) {
      const { w, phone, ay, cx, mono } = Lo;
      const K = phone ? 3 : 5;
      const items = H.top.slice(0, K).map((c) => ({ ...c, b: Lo.beads[c.i] }));
      const left = items.filter((it) => it.b.x <= cx).sort((a, b) => a.b.y - b.b.y);
      const right = items.filter((it) => it.b.x > cx).sort((a, b) => a.b.y - b.b.y);
      const a = st.label;
      const y0 = ay + (phone ? 48 : 76), dy = phone ? 21 : 24;
      const place = (arr, side) => arr.forEach((it, k) => {
        const lx = side < 0 ? 6 : w - 6;
        const ly = y0 + k * dy;
        const ansTxt = `${H.ansTok} ${fmtS(it.eA)}`, rivTxt = `${H.rivalTok} ${fmtS(it.eR)}`;
        const main = Math.abs(it.eA) >= Math.abs(it.eR) ? ansTxt : rivTxt;
        const parts2 = phone ? [[`#${it.i} `, GOLD], [main, Math.abs(it.eA) >= Math.abs(it.eR) ? GOLD : RED]]
          : [[`#${it.i}  `, GOLD], [ansTxt, GOLD], ['  ·  ', AM.col.mist], [rivTxt, RED]];
        const widths = parts2.map(([s]) => D.measure(g, s, mono, 'mono'));
        const tw = widths.reduce((p, q) => p + q, 0);
        const bx = side < 0 ? lx : lx - tw - 12;
        g.save();
        g.globalAlpha = a;
        if (pass === 'lines') {
          // leader thread from the bead to the label
          const ex = side < 0 ? bx + tw + 12 : bx;
          g.strokeStyle = rgba(GOLD, 0.45); g.lineWidth = 0.9;
          g.beginPath(); g.moveTo(it.b.x, it.b.y); g.bezierCurveTo(it.b.x, it.b.y + 30, ex + side * -24, ly, ex, ly); g.stroke();
          g.strokeStyle = GOLD; g.lineWidth = 1.2;
          g.beginPath(); g.arc(it.b.x, it.b.y, Lo.beadMax + 2.5, 0, 6.283); g.stroke();
        } else {
          g.fillStyle = rgba(AM.col.ink, 0.92);
          D.roundRect(g, bx, ly - 9, tw + 12, 18, 5); g.fill();
          g.strokeStyle = rgba(GOLD, 0.35); g.lineWidth = 1; g.stroke();
          let xx = bx + 6;
          parts2.forEach(([s, c], j) => { D.text(g, s, xx, ly + 3.5, { role: 'mono', size: mono, color: c }); xx += widths[j]; });
        }
        g.restore();
      });
      place(left, -1); place(right, 1);
      if (!phone && pass !== 'lines') {
        // header, kept clear of the stream thread at the centre
        const hs = mono - 0.5, l1 = `top ${K} writers · direct effect`, l2 = 'on the final scores (logits)';
        const one = `${l1} ${l2}`;
        if (6 + D.measure(g, one, hs, 'mono') < cx - 14) D.text(g, one, 6, y0 - 18, { role: 'mono', size: hs, color: AM.col.mist, alpha: a });
        else {
          D.text(g, l1, 6, y0 - 29, { role: 'mono', size: hs, color: AM.col.mist, alpha: a });
          D.text(g, l2, 6, y0 - 17, { role: 'mono', size: hs, color: AM.col.mist, alpha: a });
        }
      }
    }

    function probsFor(key) { return key === 'before' ? H.before : key === 'after' ? H.after : H.silenced; }
    function drawReadout(g) {
      const { phone, cx, topPad } = Lo;
      const key = st.key;
      const probs = probsFor(key);
      const lab = key === 'before' ? (phone ? 'next word · before this MLP' : 'predicted next word · before this MLP')
        : key === 'after' ? (phone ? 'next word · after this MLP' : 'predicted next word · after this MLP')
          : (phone ? `brightest ${H.N} silenced` : `after this MLP · brightest ${H.N} neurons silenced`);
      D.text(g, lab.toUpperCase(), cx, phone ? 12 : 15, { role: 'mono', size: phone ? 8 : 9, color: AM.col.mist, align: 'center', letterSpacing: '0.1em' });
      const by = phone ? 40 : 52, rise = phone ? 16 : 22;
      // the word rolls: the old answer lifts away as the new one rises into place
      const e = MM.ease.out(st.swap);
      const word = (k, y, alpha, pop) => {
        const pr = probsFor(k), top = argmax(pr);
        const isAns = top === H.ans;
        const col = isAns ? GOLD : AM.col.linen;
        const size = (phone ? 27 : 36) * (1 + 0.1 * pop);
        g.save();
        g.globalAlpha = alpha;
        g.font = AM.font(size, 'display', 500, true);
        const ww = g.measureText(H.vocab[top]).width;
        const pw = D.measure(g, pct(pr[top]), phone ? 11 : 12.5, 'mono');
        const x0 = cx - (ww + 10 + pw) / 2;
        if (isAns) { g.shadowColor = rgba(col, 0.3 + 0.4 * pop); g.shadowBlur = 22; }
        g.fillStyle = col;
        g.fillText(H.vocab[top], x0, y);
        g.restore();
        D.text(g, pct(pr[top]), x0 + ww + 10, y, { role: 'mono', size: phone ? 11 : 12.5, color: isAns ? GOLD : AM.col.linenDim, alpha });
      };
      g.save();
      g.beginPath(); g.rect(0, by - (phone ? 30 : 40), Lo.w, (phone ? 30 : 40) + 10); g.clip();
      if (e < 1 && probsFor(st.prevKey) !== probs) word(st.prevKey, by - rise * e, 1 - e, 0);
      word(key, by + rise * (1 - e), e, st.pop);
      g.restore();
      // the other word of the pair, so the flip is visible both ways
      const top = argmax(probs);
      const other = top === H.ans ? H.rival : H.ans;
      D.text(g, `${H.vocab[other]} ${pct(probs[other])}`, cx, topPad - (phone ? 8 : 10), { role: 'mono', size: phone ? 8.5 : 9.5, color: other === H.ans ? rgba(GOLD, 0.85) : rgba(RED, 0.9), align: 'center' });
    }

    cv.onResize((w, h) => { Lo = layout(w, h); buildBg(); draw(0); });
    ctx.loop((t, dt) => {
      if (!vis.on) return;
      update(dt);
      draw(t);
    });
    describe();
    return { setStep, setSilence, setData };
  }

  // ==================================================================== FIGURE 2 — the bend
  function buildGelu(ctx, hosts, S) {
    const { E, m } = S;
    const GOLD = AM.dye.weld, RED = AM.dye.madder;
    const L = m.config.n_layer - 1;
    const st = { fn: 'gelu', px: 1.2, pi: 0, drag: false };
    const cache = new Map();
    function dataFor(pi) {
      if (cache.has(pi)) return cache.get(pi);
      const pr = GELU_PROMPTS[pi];
      const ids = m.encode(pr.text).ids, T = ids.length;
      const r = E.run(ids);
      const x = r.mids[L];
      const pre = E.pre(L, x);
      const preds = {};
      for (const fn of ['gelu', 'relu', 'none']) {
        const f = E.ACT[fn];
        const a = new Float64Array(pre.length); for (let i = 0; i < pre.length; i++) a[i] = f(pre[i]);
        const out = E.mlpOut(L, a), xx = new Float64Array(x.length);
        for (let j = 0; j < x.length; j++) xx[j] = x[j] + out[j];
        const p = E.lens(xx);
        preds[fn] = { probs: p, top: argmax(p) };
      }
      let nNeg = 0; for (const v of pre) if (v < 0) nNeg++;
      const d = { pr, T, pre, preds, ans: m.tokenId(pr.ans), nNeg };
      cache.set(pi, d);
      return d;
    }
    const F = (fn, x) => E.ACT[fn](x);
    const cv = ctx.canvas(hosts.canvas, {
      label: 'GELU curve explorer',
      height: (w) => Math.round(clamp(w * 0.66, 284, 344)),
    });
    cv.canvas.tabIndex = 0;
    cv.canvas.setAttribute('role', 'slider');
    cv.canvas.setAttribute('aria-valuemin', '-4');
    cv.canvas.setAttribute('aria-valuemax', '4');
    cv.canvas.id = 'mv-gl-plot';
    let Lo = null;
    function layout(w, h) {
      const phone = w < 460;
      const padL = phone ? 30 : 38, padR = 12, padT = 14;
      const histH = phone ? 52 : 62;
      const axisY = h - histH - (phone ? 34 : 36); // room for the tick labels above the histogram's own label
      // y range [-1, 4] over [padT, axisY + something]; x range [-4, 4]
      const x0 = padL, x1 = w - padR;
      const yTop = padT, yBot = axisY;
      const ymin = -1, ymax = 4;
      const sx = (x) => x0 + ((x + 4) / 8) * (x1 - x0);
      const sy = (y) => yBot - ((y - ymin) / (ymax - ymin)) * (yBot - yTop);
      return { w, h, phone, x0, x1, yTop, yBot, sx, sy, histY: h - 8, histH, mono: phone ? 8.5 : 9.5 };
    }
    function draw() {
      if (!Lo) return;
      const g = cv.g; const { x0, x1, yTop, yBot, sx, sy, mono, histY, histH, phone } = Lo;
      cv.clear();
      const dd = dataFor(st.pi);
      // grid
      g.save();
      g.beginPath(); g.rect(x0, yTop, x1 - x0, yBot - yTop + 1); g.clip();
      g.strokeStyle = rgba(AM.col.linen, 0.05); g.lineWidth = 1;
      for (let gx = -4; gx <= 4; gx++) { g.beginPath(); g.moveTo(sx(gx) + 0.5, yTop); g.lineTo(sx(gx) + 0.5, yBot); g.stroke(); }
      for (let gy = -1; gy <= 4; gy++) { g.beginPath(); g.moveTo(x0, sy(gy) + 0.5); g.lineTo(x1, sy(gy) + 0.5); g.stroke(); }
      g.strokeStyle = rgba(AM.col.linen, 0.22);
      g.beginPath(); g.moveTo(x0, sy(0) + 0.5); g.lineTo(x1, sy(0) + 0.5); g.moveTo(sx(0) + 0.5, yTop); g.lineTo(sx(0) + 0.5, yBot); g.stroke();
      // the three curves: the chosen one bright, the others as faint references
      const curve = (fn, style) => {
        g.beginPath();
        for (let k = 0; k <= 200; k++) { const x = -4 + (8 * k) / 200; const y = F(fn, x); const X = sx(x), Y = sy(y); if (k) g.lineTo(X, Y); else g.moveTo(X, Y); }
        Object.assign(g, style); g.stroke();
      };
      g.setLineDash([3, 4]);
      for (const fn of ['gelu', 'relu', 'none']) if (fn !== st.fn) curve(fn, { strokeStyle: rgba(AM.col.linen, 0.22), lineWidth: 1.2 });
      g.setLineDash([]);
      curve(st.fn, { strokeStyle: rgba(GOLD, 0.18), lineWidth: 8 });
      curve(st.fn, { strokeStyle: GOLD, lineWidth: 2.2 });
      // probe
      const px = st.px, py = F(st.fn, px);
      g.setLineDash([2, 3]);
      g.strokeStyle = rgba(AM.col.linen, 0.45); g.lineWidth = 1;
      g.beginPath(); g.moveTo(sx(px), sy(0)); g.lineTo(sx(px), sy(py)); g.lineTo(sx(0), sy(py)); g.stroke();
      g.setLineDash([]);
      g.restore();
      glow(g, GOLD, sx(px), sy(clamp(py, -1, 4)), 14, 0.9);
      g.fillStyle = '#fff6dc'; g.beginPath(); g.arc(sx(px), sy(clamp(py, -1, 4)), 3.2, 0, 6.283); g.fill();
      // axis labels
      for (const gx of [-4, -2, 0, 2, 4]) D.text(g, minus(String(gx)), sx(gx), yBot + 13, { role: 'mono', size: mono, color: AM.col.mist, align: 'center' });
      for (const gy of [-1, 0, 2, 4]) D.text(g, minus(String(gy)), x0 - 6, sy(gy) + 3, { role: 'mono', size: mono, color: AM.col.mist, align: 'right' });
      const fname = { gelu: 'GELU(x)', relu: 'ReLU(x)', none: 'x (no bend)' }[st.fn];
      D.text(g, fname, x0 + 8, yTop + 12, { role: 'mono', size: mono + 1, color: GOLD });
      // histogram of the 256 real inputs to the last MLP's neurons
      const bins = 32, cnt = new Array(bins).fill(0); let below = 0, above = 0;
      for (const v of dd.pre) { if (v < -4) below++; else if (v >= 4) above++; else cnt[Math.min(bins - 1, Math.floor(((v + 4) / 8) * bins))]++; }
      const mx = Math.max(1, ...cnt);
      const bw = (x1 - x0) / bins;
      const room = histH - 24;
      for (let b = 0; b < bins; b++) {
        if (!cnt[b]) continue;
        const xc = -4 + (b + 0.5) * (8 / bins);
        const fy = F(st.fn, xc);
        // square-root height so the few firing neurons stay visible next to the crowd near zero
        const hh = Math.max(3, Math.sqrt(cnt[b] / mx) * room);
        const col = fy >= 0.02 ? GOLD : RED;
        const a = 0.3 + 0.7 * clamp(Math.abs(fy) / 2, 0, 1);
        const bx = x0 + b * bw + 1;
        g.fillStyle = rgba(col, a * 0.18);
        g.fillRect(bx, histY - hh, bw - 2, hh);
        g.fillStyle = rgba(col, a);
        for (let yy = histY - 2; yy >= histY - hh; yy -= 3) g.fillRect(bx, yy, bw - 2, 1.4);
        if (fy >= 0.02 && bw >= 12) D.text(g, String(cnt[b]), bx + (bw - 2) / 2, histY - hh - 3, { role: 'mono', size: mono - 1.5, color: GOLD, align: 'center', alpha: 0.85 });
      }
      g.fillStyle = rgba(AM.col.linen, 0.15); g.fillRect(x0, histY, x1 - x0, 1);
      const histLab = phone ? `256 real inputs · “${dd.pr.text.split(' ').pop()}”` : `the 256 real inputs to the last MLP's neurons at “${dd.pr.text.split(' ').pop()}”`;
      D.text(g, histLab, x0, histY - histH + 4, { role: 'mono', size: mono, color: AM.col.mist });
      if (below) D.text(g, `← ${below} below −4`, x0, histY - histH + 16, { role: 'mono', size: mono, color: rgba(RED, 0.9) });
      if (above) D.text(g, `${above} above 4 →`, x1, histY - histH + 16, { role: 'mono', size: mono, color: GOLD, align: 'right' });
    }
    // readouts
    const vX = AM.el('span', { class: 'gl-val' }), vF = AM.el('span', { class: 'gl-val' }), vP = AM.el('span', { class: 'gl-val' });
    const tP = AM.el('div', { class: 'gl-tile is-pred' }, AM.el('span', { class: 'gl-lab' }, 'Model predicts'), vP);
    hosts.read.append(
      AM.el('div', { class: 'gl-tile' }, AM.el('span', { class: 'gl-lab' }, 'Input x'), vX),
      AM.el('div', { class: 'gl-tile' }, AM.el('span', { class: 'gl-lab', id: 'mv-gl-flab' }, 'GELU(x)'), vF),
      tP);
    const flab = hosts.read.querySelector('#mv-gl-flab');
    function readout() {
      const dd = dataFor(st.pi);
      vX.textContent = fmtN(st.px, 2);
      vF.textContent = fmtN(F(st.fn, st.px), 3);
      flab.textContent = { gelu: 'GELU(x)', relu: 'ReLU(x)', none: 'no bend' }[st.fn];
      const pr = dd.preds[st.fn];
      const word = m.vocab[pr.top];
      vP.innerHTML = `${esc(word)} <span class="mv-n" style="font-size:12px;font-style:normal;color:var(--mist)">${pct(pr.probs[pr.top])}</span>`;
      tP.classList.toggle('is-bad', pr.top !== dd.ans);
      cv.canvas.setAttribute('aria-valuenow', st.px.toFixed(2));
      cv.canvas.setAttribute('aria-valuetext', `x = ${fmtN(st.px, 2)}, output ${fmtN(F(st.fn, st.px), 3)}`);
      cv.canvas.setAttribute('aria-label', `Plot of ${flab.textContent} from −4 to 4 with a probe at x = ${fmtN(st.px, 2)}, output ${fmtN(F(st.fn, st.px), 3)}. Below it, a histogram of the 256 real inputs to the last MLP's neurons for “${dd.pr.text}”: ${dd.nNeg} are negative. With this curve in the last MLP the live model predicts “${word}” (${pct(pr.probs[pr.top])}).`);
    }
    const update = () => { readout(); draw(); };
    // pointer + keyboard
    const setFromPointer = (ev) => { if (!Lo) return; const p = cv.pointer(ev); st.px = clamp(-4 + ((p.x - Lo.x0) / (Lo.x1 - Lo.x0)) * 8, -4, 4); update(); };
    cv.canvas.addEventListener('pointerdown', (ev) => { st.drag = true; try { cv.canvas.setPointerCapture(ev.pointerId); } catch (_) { /* ignore */ } setFromPointer(ev); });
    cv.canvas.addEventListener('pointermove', (ev) => { if (st.drag) setFromPointer(ev); });
    const end = () => { st.drag = false; };
    cv.canvas.addEventListener('pointerup', end);
    cv.canvas.addEventListener('pointercancel', end);
    cv.canvas.addEventListener('keydown', (ev) => {
      const k = ev.key, stepv = ev.shiftKey ? 0.5 : 0.1;
      if (k === 'ArrowLeft' || k === 'ArrowDown') st.px = clamp(st.px - stepv, -4, 4);
      else if (k === 'ArrowRight' || k === 'ArrowUp') st.px = clamp(st.px + stepv, -4, 4);
      else if (k === 'Home') st.px = -4; else if (k === 'End') st.px = 4;
      else return;
      ev.preventDefault(); update();
    });
    // controls
    const seg = AM.ui.segmented({ id: 'mv-gl-fn', label: 'Curve in the last MLP', options: [{ value: 'gelu', label: 'GELU' }, { value: 'relu', label: 'ReLU' }, { value: 'none', label: 'Straight line' }], value: 'gelu', onChange: (v) => { st.fn = v; update(); } });
    const chips = AM.ui.tokens(GELU_PROMPTS.map((p) => p.text), { selected: 0, label: 'Prompt', onSelect: (i) => { st.pi = i; update(); } });
    chips.chips.forEach((c, i) => { c.id = `mv-gl-p${i}`; });
    hosts.ctl.append(seg.el, AM.el('div', { class: 'ctl' }, AM.el('span', { class: 'ctl-label' }, 'Prompt'), chips.el));
    cv.onResize((w, h) => { Lo = layout(w, h); draw(); });
    readout();
    return { dataFor };
  }

  // ==================================================================== FIGURE 3 — fact explorer
  function buildExplorer(ctx, hosts, S) {
    const { E, m } = S;
    const GOLD = AM.dye.weld, RED = AM.dye.madder, BLUE = AM.dye.woad;
    const NL = m.config.n_layer, F = m.config.d_ff;
    const st = { set: 0, item: 0, sil: MM.range(NL).map(() => new Set()), layer: 0, mute: false, hover: null };
    const baseCache = new Map();
    let cur = null, base = null, fact = null, ids = null;

    function factNow() { return FACT_SETS[st.set].items[st.item]; }
    function compute() {
      fact = factNow();
      ids = m.encode(fact.text).ids;
      if (!baseCache.has(fact.text)) baseCache.set(fact.text, E.run(ids, { chain: true }));
      base = baseCache.get(fact.text);
      const any = st.sil.some((s) => s.size) || st.mute;
      cur = any ? E.run(ids, { chain: true, mlpOff: st.sil, mute: st.mute ? [[0, 2]] : [] }) : base;
    }

    // ---------------------------------------------------------------- ladder canvas
    const STAGES = ['embed', 'attn', 'MLP', 'attn', 'MLP', 'attn', 'MLP'];
    const lad = ctx.canvas(hosts.ladder, { label: 'Logit lens ladder', height: (w) => (w < 460 ? 214 : 236) });
    let LL = null;
    function ladLayout(w, h) {
      const phone = w < 460;
      const padL = 4, padR = 4;
      const n = STAGES.length;
      const colW = (w - padL - padR) / n;
      const xs = MM.range(n).map((k) => padL + colW * (k + 0.5));
      const yBot = h - (phone ? 34 : 36), yTop = phone ? 52 : 56;
      return { w, h, phone, xs, colW, yBot, yTop, mono: phone ? 8 : 9 };
    }
    function drawLadder() {
      if (!LL || !cur) return;
      const g = lad.g; const { w, h, phone, xs, colW, yBot, yTop, mono } = LL;
      lad.clear();
      const ans = m.tokenId(fact.ans);
      const pc = cur.chain.map((p) => p[ans]);
      const pb = base.chain.map((p) => p[ans]);
      // layer bands
      for (let l = 0; l < NL; l++) {
        const xa = xs[1 + 2 * l] - colW / 2 + 2, xb = xs[2 + 2 * l] + colW / 2 - 2;
        g.fillStyle = rgba(AM.col.linen, 0.028);
        D.roundRect(g, xa, 6, xb - xa, h - 12, 6); g.fill();
        D.text(g, `LAYER ${l}`, (xa + xb) / 2, 17, { role: 'mono', size: mono, color: AM.col.mist, align: 'center', letterSpacing: '0.1em' });
      }
      const sy = (p) => yBot - p * (yBot - yTop);
      // gridline at 50%
      g.strokeStyle = rgba(AM.col.linen, 0.07); g.setLineDash([2, 4]);
      g.beginPath(); g.moveTo(4, sy(0.5)); g.lineTo(w - 4, sy(0.5)); g.stroke(); g.setLineDash([]);
      // biggest MLP jump in the current run
      let jk = -1, jv = 0.05;
      for (let k = 2; k < 7; k += 2) { const dv = pc[k] - pc[k - 1]; if (dv > jv) { jv = dv; jk = k; } }
      const bw = Math.min(colW * 0.46, 30);
      for (let k = 0; k < STAGES.length; k++) {
        const x = xs[k];
        const col = k === 0 ? AM.col.mist : k % 2 === 0 ? GOLD : BLUE;
        // ghost of the unablated value
        if (Math.abs(pb[k] - pc[k]) > 0.01) {
          g.strokeStyle = rgba(AM.col.linen, 0.35); g.setLineDash([2, 3]); g.lineWidth = 1;
          g.strokeRect(x - bw / 2 + 0.5, sy(pb[k]) + 0.5, bw - 1, Math.max(1, yBot - sy(pb[k])) - 1);
          g.setLineDash([]);
        }
        // woven bar: thin horizontal threads
        const top = sy(pc[k]);
        g.fillStyle = rgba(col, 0.12);
        g.fillRect(x - bw / 2, top, bw, yBot - top);
        g.fillStyle = rgba(col, 0.85);
        for (let yy = yBot - 2; yy >= top; yy -= 3) g.fillRect(x - bw / 2, yy, bw, 1.4);
        g.fillStyle = col; g.fillRect(x - bw / 2, top, bw, 1.6);
        if (k === jk) glow(g, GOLD, x, top, bw * 1.4, 0.8);
        // value + the lens's top word at this stage
        D.text(g, pct(pc[k]).replace('<0.1%', '0%'), x, Math.max(yTop - 5, Math.min(top, yBot) - 5), { role: 'mono', size: mono, color: k === 0 ? AM.col.mist : AM.col.linen, align: 'center' });
        const tw = m.vocab[argmax(cur.chain[k])];
        D.text(g, tw.length > 8 ? tw.slice(0, 7) + '…' : tw, x, 34 + (phone ? 0 : 1), { role: 'mono', size: mono, color: tw === fact.ans ? GOLD : AM.col.linenDim, align: 'center' });
        D.text(g, STAGES[k], x, yBot + 14, { role: 'mono', size: mono, color: col === BLUE ? AM.mix(BLUE, AM.col.linen, 0.3) : col === GOLD ? GOLD : AM.col.mist, align: 'center' });
      }
      D.text(g, phone ? 'top word ↑ · P(answer) ↓' : 'the stream’s top word at each stage ↑   ·   P(answer) ↓', w / 2, yBot + 28, { role: 'mono', size: mono - 0.5, color: AM.col.mist, align: 'center' });
      // silk thread through the bar tops
      g.save();
      g.lineCap = 'round';
      g.strokeStyle = rgba(GOLD, 0.22); g.lineWidth = 4;
      const pathTops = () => { g.beginPath(); xs.forEach((x, k) => { const y = sy(pc[k]); if (k) { const px = xs[k - 1], py = sy(pc[k - 1]); g.bezierCurveTo(px + colW * 0.5, py, x - colW * 0.5, y, x, y); } else g.moveTo(x, y); }); };
      pathTops(); g.stroke();
      g.strokeStyle = rgba(GOLD, 0.7); g.lineWidth = 1.2; pathTops(); g.stroke();
      g.restore();
      // the biggest MLP jump, labelled on the rising thread between the two bars
      if (jk > 0) {
        // kept above the low bars' value labels, which sit just above the baseline
        const lx = xs[jk] - colW / 2;
        let ly = Math.min((sy(pc[jk - 1]) + sy(pc[jk])) / 2, yBot - 26);
        const txt = `+${Math.round(jv * 100)}`, tw = D.measure(g, txt, mono, 'mono') + 8;
        // on narrow ladders the pill can reach the jumped bar's own value label: lift it above
        const vw = D.measure(g, pct(pc[jk]), mono, 'mono') / 2, valY = Math.max(yTop - 5, Math.min(sy(pc[jk]), yBot) - 5) - 4;
        if (lx + tw / 2 > xs[jk] - vw - 2 && Math.abs(ly - valY) < 14) ly = valY - 16;
        g.fillStyle = rgba(AM.col.ink, 0.94);
        D.roundRect(g, lx - tw / 2, ly - 8, tw, 15, 4); g.fill();
        g.strokeStyle = rgba(GOLD, 0.55); g.lineWidth = 1; g.stroke();
        D.text(g, txt, lx, ly + 3, { role: 'mono', size: mono, color: GOLD, align: 'center' });
      }
      lad.canvas.setAttribute('aria-label', `Logit lens for “${fact.text}”: probability of “${fact.ans}” after each stage: ${STAGES.map((s, k) => `${s}${k ? ' ' + Math.floor((k - 1) / 2) : ''} ${pct(pc[k])}`).join(', ')}.`);
    }

    // ---------------------------------------------------------------- vault walls
    const wal = ctx.canvas(hosts.walls, { label: 'MLP neurons at the last word', height: (w) => { const p = wallPitch(w); return Math.round(16 * p + 40); } });
    function wallPitch(w) { return Math.max(5, Math.min(12, Math.floor((w - 24) / 48))); }
    let WL = null;
    function wallLayout(w, h) {
      const p = wallPitch(w), side = 16 * p, gap = Math.max(8, Math.floor((w - 3 * side) / 3));
      const total = 3 * side + 2 * gap, x0 = Math.round((w - total) / 2);
      return { w, h, p, side, gap, xs: MM.range(NL).map((l) => x0 + l * (side + gap)), y0: 22, mono: w < 460 ? 8 : 9 };
    }
    function hitWall(x, y) {
      if (!WL) return null;
      for (let l = 0; l < NL; l++) {
        const xx = x - WL.xs[l], yy = y - WL.y0;
        if (xx >= 0 && yy >= 0 && xx < WL.side && yy < WL.side) return { l, i: Math.floor(yy / WL.p) * 16 + Math.floor(xx / WL.p) };
      }
      return null;
    }
    function drawWalls() {
      if (!WL || !cur) return;
      const g = wal.g; const { p, side, xs, y0, mono } = WL;
      wal.clear();
      for (let l = 0; l < NL; l++) {
        const act = cur.acts[l];
        const x0 = xs[l];
        const sel = st.layer === l;
        g.fillStyle = rgba(AM.col.ink, 1);
        D.roundRect(g, x0 - 4, y0 - 4, side + 8, side + 8, 6); g.fill();
        g.strokeStyle = sel ? rgba(GOLD, 0.7) : AM.col.rule; g.lineWidth = 1; g.stroke();
        D.text(g, `LAYER ${l} MLP`, x0 + side / 2, y0 - 9, { role: 'mono', size: mono, color: sel ? GOLD : AM.col.mist, align: 'center', letterSpacing: '0.08em' });
        for (let i = 0; i < F; i++) {
          const cx = x0 + (i % 16) * p, cy = y0 + Math.floor(i / 16) * p;
          const a = act[i];
          const silenced = st.sil[l].has(i);
          if (silenced) {
            g.fillStyle = rgba(RED, 0.18); g.fillRect(cx + 0.5, cy + 0.5, p - 1, p - 1);
            g.strokeStyle = rgba(RED, 0.9); g.lineWidth = 1;
            g.beginPath(); g.moveTo(cx + 1.5, cy + 1.5); g.lineTo(cx + p - 1.5, cy + p - 1.5); g.moveTo(cx + p - 1.5, cy + 1.5); g.lineTo(cx + 1.5, cy + p - 1.5); g.stroke();
            continue;
          }
          const t = clamp(a / 3.5, 0, 1);
          g.fillStyle = t > 0.01 ? AM.color.heat(0.08 + 0.92 * t) : AM.col.ink3;
          g.fillRect(cx + 0.5, cy + 0.5, p - 1, p - 1);
        }
        // glow on the brightest
        for (let i = 0; i < F; i++) {
          const a = act[i];
          if (a > 2.2 && !st.sil[l].has(i)) glow(g, GOLD, x0 + (i % 16) * p + p / 2, y0 + Math.floor(i / 16) * p + p / 2, p * 1.6, 0.35 * clamp((a - 2) / 2, 0, 1));
        }
        const nOn = Array.from(act).filter((v) => v > 0.5).length;
        D.text(g, `${nOn} fire > 0.5`, x0 + side / 2, y0 + side + 16, { role: 'mono', size: mono, color: AM.col.mist, align: 'center' });
      }
      if (st.hover) {
        const { l, i } = st.hover;
        const cx = WL.xs[l] + (i % 16) * p, cy = y0 + Math.floor(i / 16) * p;
        g.strokeStyle = AM.col.linen; g.lineWidth = 1.2; g.strokeRect(cx - 0.5, cy - 0.5, p + 1, p + 1);
        const txt = `#${i} · ${fmtN(cur.acts[l][i], 2)}${st.sil[l].has(i) ? ' · silenced' : ''}`;
        const tw = D.measure(g, txt, mono + 0.5, 'mono') + 12;
        const tx = clamp(cx + p / 2 - tw / 2, 2, WL.w - tw - 2), ty = cy - 22;
        g.fillStyle = rgba(AM.col.ink3, 0.96); D.roundRect(g, tx, ty, tw, 17, 4); g.fill();
        g.strokeStyle = AM.col.ruleStrong; g.stroke();
        D.text(g, txt, tx + 6, ty + 12, { role: 'mono', size: mono + 0.5, color: AM.col.linen });
      }
    }
    wal.canvas.addEventListener('pointermove', (ev) => { const pt = wal.pointer(ev); const hv = hitWall(pt.x, pt.y); const k = hv ? hv.l * 1000 + hv.i : -1; const k0 = st.hover ? st.hover.l * 1000 + st.hover.i : -1; if (k !== k0) { st.hover = hv; drawWalls(); } });
    wal.canvas.addEventListener('pointerleave', () => { st.hover = null; drawWalls(); });
    wal.canvas.addEventListener('click', (ev) => {
      const pt = wal.pointer(ev); const hv = hitWall(pt.x, pt.y);
      if (!hv) return;
      const s = st.sil[hv.l];
      if (s.has(hv.i)) s.delete(hv.i); else s.add(hv.i);
      if (st.layer !== hv.l) { st.layer = hv.l; layerSeg.set(hv.l); }
      syncSlider();
      refresh();
    });

    // ---------------------------------------------------------------- DOM controls
    const promptEl = AM.el('div', { class: 'fx-prompt', 'aria-live': 'polite' });
    hosts.prompt.appendChild(promptEl);
    let chips = null;
    const chipHost = AM.el('div');
    const catSeg = AM.ui.segmented({ id: 'mv-fx-cat', label: 'Fact type', options: FACT_SETS.map((s, i) => ({ value: i, label: s.label })), value: 0, onChange: (v) => { st.set = v; st.item = 0; renderChips(); clearAll(false); } });
    // a new fact clears the silenced neurons (they are fact-specific) but keeps the head mute; Reset clears both
    function renderChips() {
      chipHost.innerHTML = '';
      chips = AM.ui.tokens(FACT_SETS[st.set].items.map((it) => it.chip), { selected: st.item, label: 'Fact', onSelect: (i) => { st.item = i; clearAll(false); } });
      chips.chips.forEach((c, i) => { c.id = `mv-fx-c${i}`; });
      chipHost.appendChild(chips.el);
    }
    hosts.pick.append(catSeg.el, chipHost);
    renderChips();

    const layerSeg = AM.ui.segmented({ id: 'mv-fx-layer', label: 'Silence in', options: MM.range(NL).map((l) => ({ value: l, label: `layer ${l}` })), value: 0, onChange: (v) => { st.layer = v; syncSlider(); drawWalls(); } });
    const slider = AM.ui.slider({
      id: 'mv-fx-k', label: 'Brightest neurons silenced', min: 0, max: F, step: 1, value: 0,
      format: (v) => (v >= F ? 'all' : String(v)),
      onInput: (v) => {
        const order = Array.from(base.acts[st.layer].keys()).sort((a, b) => base.acts[st.layer][b] - base.acts[st.layer][a]);
        st.sil[st.layer] = new Set(order.slice(0, v));
        refresh();
      },
    });
    const muteT = AM.ui.toggle({ id: 'mv-fx-mute', label: `Mute ${headName(0, 2)} (L0H2)`, checked: false, onChange: (b) => { st.mute = b; refresh(); } });
    const resetB = AM.ui.button({ id: 'mv-fx-reset', label: 'Reset', onClick: () => clearAll(true) });
    hosts.ctl.append(layerSeg.el, slider.el, muteT.el, resetB);
    function syncSlider() { slider.set(st.sil[st.layer].size); }
    function clearAll(all) {
      st.sil = MM.range(NL).map(() => new Set());
      if (all) muteT.set(false);
      st.mute = muteT.get();
      syncSlider();
      refresh();
    }

    function refresh() {
      compute();
      const ans = m.tokenId(fact.ans);
      const pNow = cur.probs[ans], top = argmax(cur.probs);
      const pBase = base.probs[ans];
      promptEl.innerHTML = `<span class="fx-q">${esc(fact.text)}</span><span class="fx-a">${esc(fact.ans)}</span><span class="fx-p">${pct(pBase)} · live</span>`;
      const isCap = FACT_SETS[st.set].key === 'cap' || FACT_SETS[st.set].key === 'rev';
      muteT.el.style.display = isCap ? '' : 'none';
      if (!isCap && st.mute) { st.mute = false; muteT.set(false); compute(); }
      // where is the jump (unablated)?
      const pb = base.chain.map((p) => p[ans]);
      let jk = 2, jv = -1;
      for (let k = 2; k < 7; k += 2) { const dv = pb[k] - pb[k - 1]; if (dv > jv) { jv = dv; jk = k; } }
      const jl = (jk - 2) / 2;
      const silTotal = st.sil.reduce((s, x) => s + x.size, 0);
      let msg;
      if (!silTotal && !st.mute && pb[0] > 0.5) {
        msg = `This one is a default: before any layer runs, the stream already says <b>${esc(fact.ans)}</b> (<span class="mv-n">${pct(pb[0])}</span>). Silence some neurons to see what the model needs.`;
      } else if (!silTotal && !st.mute) {
        msg = `The answer jumps most across <b>layer ${jl}'s MLP</b>: P(${esc(fact.ans)}) goes from <span class="mv-n">${pct(pb[jk - 1])}</span> to <span class="mv-n">${pct(pb[jk])}</span>. Silence some neurons to see what the model needs.`;
      } else {
        const sil = [];
        st.sil.forEach((s, l) => { if (s.size) sil.push(s.size >= F ? `all of layer ${l}'s MLP` : `${s.size} neuron${s.size > 1 ? 's' : ''} in layer ${l}`); });
        const parts = [];
        const list = (a) => (a.length < 2 ? a.join('') : `${a.slice(0, -1).join(', ')} and ${a[a.length - 1]}`);
        if (sil.length) parts.push(`${list(sil)} silenced at the last word`);
        if (st.mute) parts.push(`${headName(0, 2)} muted`);
        const verdict = top === ans
          ? `still says <span class="mv-ans">${esc(fact.ans)}</span> (<span class="mv-n">${pct(pNow)}</span>)`
          : `now says <span class="mv-riv">${esc(m.vocab[top])}</span> (<span class="mv-n">${pct(cur.probs[top])}</span>), and ${esc(fact.ans)} gets <span class="mv-n">${pct(pNow)}</span>`;
        msg = `With ${parts.join(', and ')}, the model ${verdict}.`;
      }
      hosts.result.innerHTML = msg;
      drawLadder();
      drawWalls();
    }

    lad.onResize((w, h) => { LL = ladLayout(w, h); drawLadder(); });
    wal.onResize((w, h) => { WL = wallLayout(w, h); drawWalls(); });
    refresh();
  }

  // ==================================================================== FIGURE 4 — memory or context
  function buildContrast(ctx, hosts, S) {
    const { E, m } = S;
    const HC = AM.headColor(3);
    const st = { colour: 2, mute: false };
    const sentences = (c) => ({
      mem: `the red ball and the ${c} sky . the sky is`,
      ctxs: `the red ball and the ${c} box . the box is`,
    });
    const cache = new Map();
    function dataFor(text, mute) {
      const key = text + (mute ? '|m' : '');
      if (cache.has(key)) return cache.get(key);
      const ids = m.encode(text).ids;
      const r = E.run(ids, { mute: mute ? [[1, 3]] : [] });
      const d = { tokens: m.decode(ids), probs: r.probs, attn: r.attnLast[1][3] };
      cache.set(key, d);
      return d;
    }
    const cards = ['mem', 'ctxs'].map((k) => {
      const host = hosts[k];
      const cv = ctx.canvas(host.canvas, { label: k === 'mem' ? 'Memory sentence' : 'Context sentence', height: (w) => (w < 400 ? 122 : 138) });
      return { k, host, cv, Lo: null };
    });
    function drawCard(c) {
      const cv = c.cv; if (!cv.w) return;
      const g = cv.g, w = cv.w, h = cv.h, phone = w < 420;
      cv.clear();
      const text = sentences(BIND_COLOURS[st.colour])[c.k];
      const d = dataFor(text, false);
      const toks = d.tokens, T = toks.length;
      const size = phone ? 10.5 : 12;
      const row = D.layoutRow(g, toks, 4, w - 4, { size, gap: phone ? 3 : 5 });
      const ty = h - 18;
      const qx = row[T - 1].cx;
      const dyes = COLOUR_DYE();
      // attention threads of the binding head from the final "is"
      const fade = st.mute ? 0.18 : 1;
      for (let j = 0; j < T - 1; j++) {
        const wgt = d.attn[j];
        if (wgt < 0.015) continue;
        const x2 = row[j].cx;
        const lift = Math.min(h - 40, 18 + (qx - x2) * 0.32);
        const B = [qx, ty - 14, qx - (qx - x2) * 0.15, ty - 14 - lift, x2 + (qx - x2) * 0.15, ty - 14 - lift, x2, ty - 14];
        g.save();
        g.lineCap = 'round';
        if (st.mute) g.setLineDash([3, 4]);
        g.strokeStyle = rgba(HC, 0.18 * fade); g.lineWidth = 2 + 9 * wgt;
        strokeBezPart(g, B, 1);
        g.strokeStyle = rgba(HC, (0.35 + 0.65 * wgt) * fade); g.lineWidth = 0.7 + 3 * wgt;
        strokeBezPart(g, B, 1);
        g.restore();
        if (wgt > 0.25 && !st.mute) {
          const top = bez(B, 0.5);
          D.text(g, wgt.toFixed(2), top.x, top.y - 5, { role: 'mono', size: phone ? 8 : 9, color: HC, align: 'center' });
        }
      }
      if (st.mute) {
        // a snip across the head's threads
        const sx = qx - 10, sy = ty - 34;
        g.strokeStyle = AM.dye.madder; g.lineWidth = 1.6;
        g.beginPath(); g.moveTo(sx - 6, sy - 6); g.lineTo(sx + 6, sy + 6); g.moveTo(sx + 6, sy - 6); g.lineTo(sx - 6, sy + 6); g.stroke();
      }
      toks.forEach((s, i) => {
        const ts = Math.max(6, size * Math.min(1, row[i].scale + 0.08)), px = phone ? 4 : 6;
        // D.token's colour underline needs a chip wider than 14px (it can be narrower mid-resize)
        const wide = D.measure(g, s, ts, 'body', 600) + 2 * px > 18;
        D.token(g, s, row[i].cx, ty, { size: ts, padX: px, padY: phone ? 4 : 5, selected: i === T - 1, underline: wide ? dyes[s] || null : null, radius: 5 });
      });
      D.text(g, phone ? `${headName(1, 3)} at “is”` : `${headName(1, 3)} (L1H3) at “is”`, 4, 14, { role: 'mono', size: phone ? 8 : 9, color: HC });
      cv.canvas.setAttribute('aria-label', `“${text}”. Attention of ${headName(1, 3)} (layer 1, head 3) from the last word: ${toks.slice(0, -1).map((s, j) => `${s} ${d.attn[j].toFixed(2)}`).filter((_, j) => d.attn[j] > 0.05).join(', ')}.`);
    }
    function refresh() {
      const c = BIND_COLOURS[st.colour];
      const s = sentences(c);
      const dyes = COLOUR_DYE();
      for (const card of cards) {
        drawCard(card);
        const text = s[card.k];
        const d = dataFor(text, st.mute);
        const top = argmax(d.probs);
        const word = m.vocab[top];
        card.host.ans.innerHTML = `<span class="ct-w" style="color:${dyes[word] || 'var(--linen)'}">${esc(word)}</span><span class="ct-p">${pct(d.probs[top])}${st.mute ? ' · head muted' : ''}</span>`;
      }
      const dm = dataFor(s.mem, false), dc = dataFor(s.ctxs, false);
      const dmM = dataFor(s.mem, true), dcM = dataFor(s.ctxs, true);
      const colIdxC = dc.tokens.indexOf(c, 3), colIdxM = dm.tokens.indexOf(c, 3);
      const said = m.vocab[argmax(dm.probs)];
      const lead = said === c ? `You told it the sky is <b>${c}</b>, which matches what it remembers. It answers <b>${esc(said)}</b>.` : `You told it the sky is <b>${c}</b>. It answers <b>${esc(said)}</b> anyway.`;
      hosts.mem.note.innerHTML = `${lead} The head gives the word “${c}” only <span class="mv-n">${dm.attn[colIdxM].toFixed(2)}</span> of its attention. Muted: <b>${esc(m.vocab[argmax(dmM.probs)])}</b> <span class="mv-n">${pct(dmM.probs[argmax(dmM.probs)])}</span>.`;
      hosts.ctxs.note.innerHTML = `The box's colour is only in the sentence. The head puts <span class="mv-n">${dc.attn[colIdxC].toFixed(2)}</span> of its attention on “${c}”. Muted: <b>${esc(m.vocab[argmax(dcM.probs)])}</b> <span class="mv-n">${pct(dcM.probs[argmax(dcM.probs)])}</span>, and ${c} gets <span class="mv-n">${pct(dcM.probs[m.tokenId(c)])}</span>.`;
    }
    const sw = AM.ui.tokens(BIND_COLOURS, { selected: st.colour, label: 'Colour in the sentence', colors: (i) => COLOUR_DYE()[BIND_COLOURS[i]], onSelect: (i) => { st.colour = i; refresh(); } });
    sw.chips.forEach((c, i) => { c.id = `mv-ct-c${i}`; });
    sw.el.classList.add('ct-swatches');
    const muteT = AM.ui.toggle({ id: 'mv-ct-mute', label: `Mute ${headName(1, 3)} (L1H3)`, checked: false, onChange: (b) => { st.mute = b; refresh(); } });
    hosts.ctl.append(AM.el('span', { class: 'ct-lab' }, 'Tell it a colour'), sw.el, muteT.el);
    cards.forEach((c) => c.cv.onResize(() => drawCard(c)));
    refresh();
  }

  // ==================================================================== chapter
  AM.chapter({
    id: ID,
    num: 6,
    kicker: 'Feed-forward network',
    title: 'The Memory <em>Vaults</em>',
    lede: 'Between rounds of attention, each token passes alone through a wide network of neurons. Much of what the model knows seems to be stored there.',
    where: 'mlp',
    mount(root, ctx) {
      ctx.header();
      const el = ctx.el;
      const ui = AM.ui;
      const body = el('div', { class: 'ch-body' });
      root.appendChild(body);

      // ---------------------------------------------------------------- the live model
      let m = null, E = null, parityOK = false;
      try {
        m = AM.model && AM.model.get('tinyworld');
        E = makeEngine(m);
        if (E) {
          const ids = m.encode('the capital of france is').ids;
          const a = m.run(ids).probs[ids.length - 1], b = E.run(ids).probs;
          let err = 0; for (let k = 0; k < a.length; k++) err = Math.max(err, Math.abs(a[k] - b[k]));
          parityOK = err < 1e-3;
          if (!parityOK) console.warn(`[${ID}] engine parity ${err}; ablation disabled`);
        }
      } catch (e) {
        console.warn(`[${ID}] live model unavailable`, e);
      }
      if (!m || !E || !parityOK) {
        body.appendChild(el('div', { class: 'prose' },
          el('p', { html: 'The MLP takes each token\'s vector on its own, widens it four times through a layer of neurons, bends every number with a curve called GELU and projects it back down, adding the result to the token\'s vector. The live model that drives this chapter could not be loaded in this browser.' })));
        return;
      }

      const C = m.config;
      const S = { m, E };
      // parameter bookkeeping, from the weights themselves
      const P = m._net.params;
      const size = (n) => (P[n] ? P[n].data.length : 0);
      let mlpParams = 0, total = 0;
      for (const n of Object.keys(P)) total += size(n);
      for (let l = 0; l < C.n_layer; l++) for (const k of ['wfc', 'bfc', 'wproj', 'bproj']) mlpParams += size(`h.${l}.mlp.${k}`);
      // GPT-3 175B: d = 12,288, 96 layers, d_ff = 4d (Brown et al., 2020)
      const g3d = 12288, g3L = 96, g3mlp = g3L * (2 * g3d * 4 * g3d + 4 * g3d + g3d);
      const g3share = g3mlp / 175e9;

      S.heroes = HERO_PROMPTS.map((p) => computeHero(m, E, p));
      S.hero = S.heroes[0];
      let H = S.hero;

      body.appendChild(el('div', { class: 'prose mv-intro' },
        el('p', { html: 'Attention lets tokens read from each other, but it is only half of a transformer block. The other half, and the bigger one, is a small neural network called the <span class="term">MLP</span> (multilayer perceptron) or <span class="term">feed-forward network</span>.' }),
      ));

      // ---------------------------------------------------------------- 1. scrollytelling vault
      const vaultHost = el('div');
      let vault = null;
      const promptSeg = ui.segmented({
        id: 'mv-hero-prompt', label: 'Prompt',
        options: HERO_PROMPTS.map((p, i) => ({ value: i, label: el('span', {}, el('span', { class: 'mv-long' }, p.label), el('span', { class: 'mv-short' }, p.label.split(' ')[1])) })),
        value: 0,
        onChange: (v) => { S.hero = H = S.heroes[v]; vault.setData(H); renderSteps(); silN.forEach((s) => { s.textContent = String(H.N); }); },
      });
      const silN = [el('span', {}, String(H.N)), el('span', {}, String(H.N))];
      const silLabel = el('span', {}, el('span', { class: 'mv-long' }, 'Silence the brightest ', silN[0]), el('span', { class: 'mv-short' }, 'Silence top ', silN[1]));
      const silT = ui.toggle({ id: 'mv-hero-silence', label: silLabel, checked: false, onChange: (b) => vault && vault.setSilence(b) });
      const capText = 'Live model: tinyworld, the last of its three layers. Bead brightness is each neuron\'s real input (then output) for the chosen word; gold is positive, red negative. Every neuron reads all 64 numbers; one thread per neuron stands for those 64 connections. Labels give each neuron\'s exact direct effect on the final scores (output bias left out). The fan layout itself is an illustration.';
      const stage = el('div', { class: 'ch-stage mv-stage' },
        el('figure', { class: 'fig' },
          el('div', { class: 'fig-top' },
            el('span', { class: 'fig-title' }, 'One word, one MLP'),
            el('span', { class: 'mv-badges' }, ui.badge('live'))),
          vaultHost,
          el('div', { class: 'controls' }, promptSeg.el, silT.el),
          el('figcaption', { class: 'mv-cap-desk' }, capText)));

      const n = (v) => `<span class="mv-n">${v}</span>`;
      const A = (s) => `<span class="mv-ans">${esc(s)}</span>`;
      const R = (s) => `<span class="mv-riv">${esc(s)}</span>`;
      // the strongest writer also fires in unrelated contexts (measured live for the current prompt)
      const PROBES = ['the sky is', 'the red ball and the blue box . the box is', 'the capital of china is', 'the queen opened the door because'];
      const probeActs = PROBES.map((t) => E.run(m.encode(t).ids).acts[C.n_layer - 1]);
      const messy = () => {
        const i = H.top[0].i;
        const hits = PROBES.map((t, k) => ({ t, a: probeActs[k][i] })).filter((x) => x.a > 1).sort((a, b) => b.a - a.a).slice(0, 2);
        const lead = 'Real neurons are messier than the metaphor.';
        if (hits.length < 2) return `${lead} Most respond to many unrelated contexts, and large models seem to spread each feature over many neurons.`;
        return `${lead} Neuron #${i} also fires for “${esc(hits[0].t.split(' . ').pop())}” (${n(hits[0].a.toFixed(1))}) and “${esc(hits[1].t.split(' . ').pop())}” (${n(hits[1].a.toFixed(1))}). Large models seem to pack many features into each neuron.`;
      };
      const STEP_DEFS = [
        () => ({
          label: '1 · Alone',
          h3: 'One word at a time',
          ps: [
            'Attention has just mixed information between positions. Now each position goes through the MLP on its own: the same weights for every position, and no looking sideways.',
            `Here is the MLP in the last layer of our live model. Each little fan shows the same ${n(C.d_ff)} hidden neurons, lit by how strongly they fire for that word. Follow <strong>“${esc(H.tokens[H.T - 1])}”</strong> at the end of <em>${esc(H.prompt.text)}</em>.`,
          ],
        }),
        () => ({
          label: '2 · Expand',
          h3: `Wider: ${C.d_model} → ${C.d_ff}`,
          ps: [
            `The MLP takes a normalised copy of the word’s ${n(C.d_model)} numbers (LayerNorm, chapter 7) and multiplies it by a ${C.d_model} × ${C.d_ff} matrix, W<sub>in</sub>, plus a bias. Out come ${n(C.d_ff)} numbers, one per <span class="term">hidden neuron</span>. Gold beads are positive, red negative.`,
            `<span class="math block">h = LN(x) · W<sub>in</sub> + b<sub>in</sub></span>`,
            `Four times wider than the stream is the usual choice. GPT-3 widens ${n('12,288')} numbers to ${n('49,152')}. Matrices this size add up: the MLPs hold ${n(comma(mlpParams))} of our model’s ${n(comma(total))} parameters (${n(Math.round((100 * mlpParams) / total) + '%')}), and about two thirds of GPT-3’s (${n(Math.round(g3share * 100) + '%')}).`,
          ],
        }),
        () => ({
          label: '3 · Bend',
          h3: 'GELU: most neurons go quiet',
          ps: [
            'Every one of those numbers then passes through a curve called <span class="term">GELU</span>. Negative inputs come out close to zero. Large positive inputs pass almost unchanged.',
            `For “${esc(H.tokens[H.T - 1])}”, ${n(H.nNeg)} of the ${C.d_ff} inputs are negative, so those neurons fall silent. Only ${n(H.nFire)} fire above 0.5. The word has picked out a few dozen neurons.`,
          ],
        }),
        () => ({
          label: '4 · Write',
          h3: 'Back to 64, added to the stream',
          ps: [
            `Each firing neuron adds its own row of a second matrix, W<sub>out</sub> (${C.d_ff} × ${C.d_model}), scaled by how strongly it fires. The sum is ${n(C.d_model)} numbers, added to the word’s vector:`,
            '<span class="math block">x ← x + GELU(h) · W<sub>out</sub> + b<sub>out</sub></span>',
            `Before this MLP the model would say ${R(H.vocab[argmax(H.before)])} (${n(pct(H.before[argmax(H.before)]))}). After it: ${A(H.ansTok)} (${n(pct(H.after[H.ans]))}). Nothing else runs in between, so this MLP alone changed the answer.`,
          ],
        }),
        () => ({
          label: '5 · Keys and values',
          h3: 'Each neuron is a small memory',
          ps: [
            'A useful way to read one neuron: its column of W<sub>in</sub> is a <span class="term">key</span>, a pattern it looks for in the stream. Its row of W<sub>out</sub> is a <span class="term">value</span>, a direction it writes when the key matches. <span class="mv-cite">(<a href="https://arxiv.org/abs/2012.14913" target="_blank" rel="noopener">Geva et al., 2021</a>)</span>',
            `The labels show what the strongest writers add to the final scores (logits). Some raise ${A(H.ansTok)}, others push ${R(H.rivalTok)} down. Summed over all ${C.d_ff} neurons, ${esc(H.ansTok)} gains ${n(H.total.toFixed(1))} more than ${esc(H.rivalTok)}.`,
            messy(),
          ],
        }),
        () => ({
          label: '6 · Silence',
          h3: 'Take a few away',
          ps: [
            `Now zero the ${n(H.N)} brightest neurons, at this word only, and keep the rest of the MLP as it was.`,
            `The model ${argmax(H.silenced) === argmax(H.before) ? 'goes back to' : 'now says'} ${R(H.vocab[argmax(H.silenced)])} (${n(pct(H.silenced[argmax(H.silenced)]))}). ${H.singleFlips === 0 ? `No single neuron holds the fact: silencing any one of the ${C.d_ff} leaves ${A(H.ansTok)} on top. It takes several together.` : `It takes several neurons together; only ${H.singleFlips} of the ${C.d_ff} can change the answer on their own.`} The switch under the picture repeats the experiment at any step.`,
          ],
        }),
      ];
      const stepEls = STEP_DEFS.map(() => el('div', { class: 'step' }));
      function renderSteps() {
        STEP_DEFS.forEach((fn, i) => {
          const d = fn();
          const s = stepEls[i];
          s.innerHTML = '';
          s.append(el('span', { class: 'step-label' }, d.label), el('h3', { html: d.h3 }), ...d.ps.map((p) => (p.startsWith('<span class="math block"') ? el('div', { html: p }) : el('p', { html: p }))));
        });
      }
      renderSteps();
      const prose = el('div', { class: 'ch-prose' }, stepEls);
      body.appendChild(el('div', { class: 'ch-split' }, stage, prose));
      body.appendChild(el('p', { class: 'caption mv-cap-phone' }, capText));

      vault = buildVault(ctx, vaultHost, S);
      let curStep = 0;
      ctx.steps(stepEls, (i) => {
        const was = curStep;
        curStep = i;
        vault.setStep(i);
        if (i === 5 && !silT.get()) { silT.set(true); vault.setSilence(true); }
        if (i < 5 && was === 5 && silT.get()) { silT.set(false); vault.setSilence(false); }
      });

      // ---------------------------------------------------------------- 2. the bend
      const glCanvas = el('div', { class: 'gl-canvas' });
      const glRead = el('div', { class: 'gl-read', 'aria-live': 'polite' });
      const glCtl = el('div', { class: 'controls' });
      const glNone = el('span');
      body.appendChild(el('section', { class: 'ch-wide mv-sec', 'aria-labelledby': 'mv-gl-h' },
        el('div', { class: 'gl-grid' },
          el('div', { class: 'prose' },
            el('span', { class: 'mv-kicker' }, 'The bend'),
            el('h3', { id: 'mv-gl-h' }, 'Why the curve matters'),
            el('p', { html: '<span class="term">GELU</span> (Gaussian Error Linear Unit) is <span class="math">GELU(x) = x · Φ(x)</span>, where Φ(x) is the chance that a standard normal number is below x. Large positive inputs pass through, large negative ones become 0, and near zero it bends smoothly, dipping to −0.17. Our model uses the usual tanh approximation of Φ.' }),
            el('p', { html: 'Without a bend, two matrix multiplications are just one: <span class="math">(x·W<sub>in</sub>)·W<sub>out</sub> = x·(W<sub>in</sub>W<sub>out</sub>)</span>. The whole MLP would shrink to a single 64 × 64 matrix (plus a bias), and stacking more of them would add nothing new. With the bend, each neuron acts roughly like a switch that writes only when its key matches.' }),
            el('p', { html: 'Try it on the real model. <span class="term">ReLU</span>, the sharper max(0, x) of the original Transformer, gives the same answers here. A straight line does not: ' }, glNone, ' Many recent models, LLaMA among them, use a gated variant called SwiGLU.')),
          el('div', { class: 'panel gl-panel' },
            ui.figure({ title: 'Swap the curve', badge: 'live', caption: 'The curves are exact. Bars: the 256 real inputs to the last MLP’s neurons at the prompt’s last word, binned by value (heights on a square-root scale); red bars come out at or below zero with the chosen curve. The prediction is a live forward pass with the chosen curve in place of GELU in the last MLP only. A model trained with a straight line would learn other weights; this shows that these weights rely on the bend. Drag the plot or use the arrow keys.' },
              glCanvas, glRead, glCtl)))));
      const gel = buildGelu(ctx, { canvas: glCanvas, read: glRead, ctl: glCtl }, S);
      {
        // quoted from the same live forward passes the figure shows
        const d0 = gel.dataFor(0), d1 = gel.dataFor(1);
        const p0 = d0.preds, p1 = d1.preds.none;
        glNone.innerHTML = `for “${esc(d0.pr.text)}”, P(${esc(d0.pr.ans)}) falls from ${n(pct(p0.gelu.probs[d0.ans]))} to ${n(pct(p0.none.probs[d0.ans]))}, and “${esc(d1.pr.text)}” ${p1.top === d1.ans ? 'drops to' : 'becomes'} ${p1.top === d1.ans ? n(pct(p1.probs[d1.ans])) : `${R(m.vocab[p1.top])} (${n(pct(p1.probs[p1.top]))})`}.`;
      }

      // ---------------------------------------------------------------- 3. fact explorer
      const fxPick = el('div', { class: 'fx-pick' });
      const fxPrompt = el('div');
      const fxLadder = el('div');
      const fxWalls = el('div', { class: 'fx-walls' });
      const fxCtl = el('div', { class: 'fx-ctl' });
      const fxResult = el('p', { class: 'fx-result', 'aria-live': 'polite' });
      // computed facts quoted in the prose
      const quote = (() => {
        const run = (t, o) => E.run(m.encode(t).ids, o);
        const fr = run('the capital of france is', { chain: true }), id = m.tokenId('paris');
        const frOff = run('the capital of france is', { mlpOff: [new Set(MM.range(C.d_ff)), null, null] });
        const frMute = run('the capital of france is', { mute: [[0, 2]] });
        // the same silencing at "because" in a pronoun sentence: MLP 0 is groundwork for every skill (model-notes §2)
        const pron = run('the queen opened the door because', { mlpOff: [new Set(MM.range(C.d_ff)), null, null] });
        const frIds = m.encode('the capital of france is').ids;
        const nAll = FACT_SETS.reduce((s, x) => s + x.items.length, 0);
        let ok = 0; FACT_SETS.forEach((fs) => fs.items.forEach((it) => { const p = run(it.text).probs; if (argmax(p) === m.tokenId(it.ans)) ok++; }));
        return {
          nAll, ok,
          // chain: [embed, after attn 0, after MLP 0, after attn 1, …]
          frIn: fr.chain[1][id], frOut: fr.chain[2][id], frOff: frOff.probs[id],
          muteTop: m.vocab[argmax(frMute.probs)], muteP: frMute.probs[argmax(frMute.probs)],
          pronTop: m.vocab[argmax(pron.probs)],
          // attention of L0H2 from "is" to "france"
          l0h2: fr.attnLast[0][2][frIds.indexOf(m.tokenId('france'))],
        };
      })();
      body.appendChild(el('section', { class: 'ch-wide mv-sec', 'aria-labelledby': 'mv-fx-h' },
        el('div', { class: 'prose' },
          el('span', { class: 'mv-kicker' }, 'Where facts live'),
          el('h3', { id: 'mv-fx-h' }, 'Open the vaults'),
          el('p', { html: `Our model memorised ${n(quote.nAll)} facts: the capitals of 8 countries (both ways round), what 8 animals say and the colours of 6 things. It gets ${quote.ok === quote.nAll ? 'all ' : ''}${n(quote.ok)} right. None of them can be worked out from the sentence, so they must be stored in the weights.` }),
          el('p', { html: `In large models, experiments that switch off parts of the network point to MLPs as a main store of facts. MLPs in the middle layers, working at the subject’s last word, seem to recall what the model knows about it, and attention later carries that to the end of the sentence <span class="mv-cite">(<a href="https://arxiv.org/abs/2202.05262" target="_blank" rel="noopener">Meng et al., 2022</a>)</span>. Our tiny model seems to do a simpler version: attention fetches the country to the last word, and the MLPs there turn it into the answer.` }),
          el('p', { html: `In “the capital of france is”, ${headName(0, 2)} (L0H2) puts ${n(quote.l0h2.toFixed(2))} of its attention from “is” on “france”. Mute it, and the model still names a capital, just the wrong one: ${R(quote.muteTop)} (${n(pct(quote.muteP))}). Across the first MLP, P(paris) jumps from ${n(pct(quote.frIn))} to ${n(pct(quote.frOut))}, and silencing that MLP at “is” leaves ${n(pct(quote.frOff))}. That test is blunt, though. A model this small also uses its first MLP as groundwork for everything: silenced at “because”, it turns “the queen opened the door because …” from <em>she</em> into ${R(quote.pronTop)}. The cat in the vault above is a cleaner test: there only a few neurons of the last MLP are switched off, and nothing runs after them.` }),
          el('p', { html: 'Try the others. The ladder decodes the stream after every stage as if the model stopped there, a trick called the <span class="term">logit lens</span>. The vaults show every neuron of all three MLPs at the last word. Click neurons to silence them, or use the slider.' })),
        el('div', { class: 'panel fx-panel' },
          ui.figure({ title: 'Fact explorer', badge: 'live', caption: 'Live model. Silencing sets a neuron’s output to zero at the last word only; every other word runs normally. Muting a head zeroes its output at every position. The logit lens uses the final LayerNorm and unembedding: exact after the last MLP, a rough reading at earlier stages. Dashed outlines show the unsilenced values.' },
            fxPick, fxPrompt,
            el('div', { class: 'fx-grid' },
              el('div', {}, el('div', { class: 'fx-sub' }, 'P(answer) through the layers'), fxLadder),
              el('div', {}, el('div', { class: 'fx-sub' }, 'The vaults · every neuron at the last word'), fxWalls)),
            fxCtl, fxResult))));
      buildExplorer(ctx, { pick: fxPick, prompt: fxPrompt, ladder: fxLadder, walls: fxWalls, ctl: fxCtl, result: fxResult }, S);

      // ---------------------------------------------------------------- 4. memory or context
      const mk = () => ({ canvas: el('div'), ans: el('div', { class: 'ct-ans', 'aria-live': 'polite' }), note: el('p', { class: 'ct-note' }) });
      const ctMem = mk(), ctCtx = mk();
      const ctCtl = el('div', { class: 'ct-ctl' });
      body.appendChild(el('section', { class: 'ch-wide mv-sec', 'aria-labelledby': 'mv-ct-h' },
        el('div', { class: 'prose' },
          el('span', { class: 'mv-kicker' }, 'Memory or context'),
          el('h3', { id: 'mv-ct-h' }, 'Same question, two machines'),
          el('p', { html: '“The sky is …” and “the box is …” look alike, and our model answers both with a colour. The box’s colour can only come from the sentence. The sky’s comes from memory. Tell the model a colour and watch which answer follows it.' })),
        el('div', { class: 'panel ct-panel' },
          ui.figure({ title: 'Memory vs context', badge: 'live', caption: `Live model. Threads show the attention of ${headName(1, 3)} (layer 1, head 3) from the final “is”, with its weights as numbers. In binding questions this head lands on the matching colour. Muting zeroes its output at every position.` },
            ctCtl,
            el('div', { class: 'ct-cards' },
              el('div', { class: 'ct-card' }, el('h4', {}, 'From memory'), ctMem.canvas, ctMem.ans, ctMem.note),
              el('div', { class: 'ct-card' }, el('h4', {}, 'From context'), ctCtx.canvas, ctCtx.ans, ctCtx.note))))));
      buildContrast(ctx, { mem: ctMem, ctxs: ctCtx, ctl: ctCtl }, S);

      // ---------------------------------------------------------------- key idea
      body.appendChild(el('div', { class: 'callout' },
        el('span', { class: 'callout-label' }, 'Key idea'),
        el('p', { html: `Attention moves information <strong>between</strong> tokens. The MLP processes it <strong>within</strong> each token: it widens the vector four times (${C.d_model} → ${C.d_ff} here), lets a few neurons fire through GELU, and adds their output rows back into the residual stream. Each neuron acts roughly like a key–value memory, and much of what a model memorises, facts included, appears to be stored in these weights.` })));
    },
  });
})();
