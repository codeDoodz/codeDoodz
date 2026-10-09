/* Chapter 08 — The Tower of Layers: depth, composition and the logit lens.

   Three figures, all driven by the real tiny transformer (tinyworld):
   1. The tower (scrollytelling stage). One sentence rises as warp threads
      through the model's three layers. Each layer band carries that layer's
      real attention threads (one dye per head) and MLP knots (size = how much
      the MLP wrote at that position). A composition step silences one layer-0
      head at one position, using a small float64 forward pass in this file
      that is checked against AM.model.run before it is trusted, and the
      layer-1 head that depended on it loses its way. Then a lens sweeps up the
      tower and every level grows a bead: the logit-lens top guess.
   2. The lens table (free play): positions × depth, top guess and probability
      in every cell, top 5 on hover/focus/tap. Curated sentences or the
      visitor's own words (unknown words are flagged, with suggestions).
   3. Depth profiles: P(correct next word) at each depth for five kinds of
      dependency, computed live on sentences generated from the model's own
      training grammar (redrawable), plus all 30 memorised facts.

   Head names follow docs/model-notes.md (window.AM_NOTES when present, else
   the same names kept here). The one number not computed in the page (the
   1,004-question ablation in step 2) is labelled as measured offline. */
(() => {
  const ID = 'stack';
  const M = AM.math;
  const D = AM.draw;
  const clamp = M.clamp;
  const lerp = M.lerp;
  const E = M.ease;

  // ================================================================== helpers
  const SUBS = '₀₁₂₃₄₅₆₇₈₉';
  const sub = (n) => String(n).replace(/\d/g, (c) => SUBS[+c]);
  const isStacked = () => (window.matchMedia ? window.matchMedia('(max-width: 900px)').matches : window.innerWidth <= 900);
  const pct = (p) => (p >= 0.9995 ? '100%' : p >= 0.99 ? (p * 100).toFixed(1) + '%' : p >= 0.01 ? Math.round(p * 100) + '%' : p > 0 ? '<1%' : '0%');
  const f2 = (x) => (x >= 0.995 ? '1.00' : x.toFixed(2));
  const q = (s) => `“${s}”`;
  /** A token as it should read inside a sentence of prose. */
  const say = (tok) => (tok === '.' ? 'a full stop' : tok === ',' ? 'a comma' : `<em>${String(tok).replace(/[&<>"]/g, '')}</em>`);
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const approach = (cur, target, dt, rate) => cur + (target - cur) * (1 - Math.exp(-rate * dt));
  const LEVEL_NAMES = ['embed', 'after L0', 'after L1', 'output'];
  const LEVEL_LONG = ['the embedding', 'after layer 0', 'after layer 1', 'the output'];

  const HEAD_FALLBACK = {
    L0H0: 'the backstitch', L0H1: 'the loose weave', L0H2: 'the fact finder', L0H3: 'the antecedent finder',
    L1H0: 'the second colour reader', L1H1: 'the clause hopper', L1H2: 'the head-noun tracker', L1H3: 'the colour binder',
    L2H0: 'the idle shuttle', L2H1: 'the verb echo', L2H2: 'the faint echo', L2H3: 'the idle shuttle II',
  };
  /** Head name, preferring the interpretability notes (AM_NOTES) when they are loaded. */
  const headName = (l, h) => {
    const k = `L${l}H${h}`;
    try {
      const n = window.AM_NOTES && window.AM_NOTES.heads && window.AM_NOTES.heads[k];
      if (n && typeof n.name === 'string' && n.name) return n.name;
    } catch (e) { /* fall through */ }
    return HEAD_FALLBACK[k];
  };

  /** Glow sprites per colour (drawImage is far cheaper than a gradient per bead). */
  const sprites = new Map();
  const sprite = (rgb) => {
    const key = rgb.join(',');
    if (sprites.has(key)) return sprites.get(key);
    const S = 48, c = document.createElement('canvas');
    c.width = c.height = S;
    const g = c.getContext('2d'), r = S / 2;
    const grd = g.createRadialGradient(r, r, 0, r, r, r);
    const [a, b, d] = rgb;
    grd.addColorStop(0, `rgba(${(a + 255 * 1.5) / 2.5 | 0},${(b + 248 * 1.5) / 2.5 | 0},${(d + 230 * 1.5) / 2.5 | 0},1)`);
    grd.addColorStop(0.16, `rgba(${a},${b},${d},1)`);
    grd.addColorStop(0.36, `rgba(${a},${b},${d},0.4)`);
    grd.addColorStop(0.66, `rgba(${a},${b},${d},0.09)`);
    grd.addColorStop(1, `rgba(${a},${b},${d},0)`);
    g.fillStyle = grd;
    g.fillRect(0, 0, S, S);
    sprites.set(key, c);
    return c;
  };
  const RGB = (hex) => AM.hexToRgb(hex);
  const mixRgb = (a, b, t) => [0, 1, 2].map((i) => Math.round(a[i] + (b[i] - a[i]) * t));

  /** Cubic Bézier helpers. B = [x0,y0,x1,y1,x2,y2,x3,y3]. */
  const bez = (B, t) => {
    const u = 1 - t, a = u * u * u, b = 3 * u * u * t, c = 3 * u * t * t, d = t * t * t;
    return { x: a * B[0] + b * B[2] + c * B[4] + d * B[6], y: a * B[1] + b * B[3] + c * B[5] + d * B[7] };
  };
  const strokeBez = (g, B) => { g.beginPath(); g.moveTo(B[0], B[1]); g.bezierCurveTo(B[2], B[3], B[4], B[5], B[6], B[7]); g.stroke(); };

  /** Is this element (roughly) on screen? Lets each figure idle while the chapter is visible. */
  const watch = (elm, onFirst) => {
    const st = { on: true, seen: false };
    if (typeof IntersectionObserver !== 'undefined') {
      st.on = false;
      new IntersectionObserver((en) => {
        const e = en[en.length - 1];
        st.on = e.isIntersecting;
        if (e.isIntersecting && e.intersectionRatio >= 0.25 && !st.seen) { st.seen = true; if (onFirst) onFirst(); }
      }, { rootMargin: '60px 0px', threshold: [0, 0.25, 0.5] }).observe(elm);
    } else if (onFirst) { st.seen = true; onFirst(); }
    return st;
  };

  // ================================================================== the model
  let MODEL = null, MODEL_TRIED = false;
  const getModel = () => {
    if (!MODEL_TRIED) {
      MODEL_TRIED = true;
      try { MODEL = AM.model && AM.model.get('tinyworld'); } catch (e) { MODEL = null; }
    }
    return MODEL;
  };

  /** A float64 re-implementation of the forward pass (same maths as transformer.js)
      that can silence one head's output at chosen positions. Also exposes the
      logit lens for any residual vector (used for the half-layer readings). */
  function makeNet(m) {
    const P = m && m._net && m._net.params;
    if (!P || !P['lnf.g'] || !P.wout) return null;
    const cfg = m.config, d = cfg.d_model, H = cfg.n_head, dh = d / H, F = cfg.d_ff, V = cfg.vocab_size, NL = cfg.n_layer;
    const A = (n) => P[n].data;
    const ln = (x, gm, bt) => {
      let mu = 0;
      for (let i = 0; i < d; i++) mu += x[i];
      mu /= d;
      let v = 0;
      for (let i = 0; i < d; i++) { const z = x[i] - mu; v += z * z; }
      const s = 1 / Math.sqrt(v / d + 1e-5);
      const o = new Float64Array(d);
      for (let i = 0; i < d; i++) o[i] = (x[i] - mu) * s * gm[i] + bt[i];
      return o;
    };
    const lin = (x, W, b, n, mo) => {
      const y = new Float64Array(mo);
      if (b) for (let j = 0; j < mo; j++) y[j] = b[j];
      for (let i = 0; i < n; i++) {
        const xi = x[i];
        if (xi === 0) continue;
        const o = i * mo;
        for (let j = 0; j < mo; j++) y[j] += xi * W[o + j];
      }
      return y;
    };
    const softmax = (z) => {
      let mx = -Infinity;
      for (let i = 0; i < z.length; i++) if (z[i] > mx) mx = z[i];
      const o = new Float32Array(z.length);
      let s = 0;
      for (let i = 0; i < z.length; i++) { const e = Math.exp(z[i] - mx); o[i] = e; s += e; }
      for (let i = 0; i < z.length; i++) o[i] /= s;
      return o;
    };
    const gelu = (x) => 0.5 * x * (1 + Math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
    const lens = (x) => softmax(lin(ln(x, A('lnf.g'), A('lnf.b')), A('wout'), A('bout'), d, V));

    /** mask(l, h, t) → true silences head h of layer l at position t (its output before W_O). */
    function forward(ids, mask) {
      const T = ids.length, wte = A('wte'), wpe = A('wpe');
      let X = ids.map((id, t) => { const v = new Float64Array(d); for (let i = 0; i < d; i++) v[i] = wte[id * d + i] + wpe[t * d + i]; return v; });
      const resid = [X], residMid = [], attn = [];
      const scale = 1 / Math.sqrt(dh);
      for (let l = 0; l < NL; l++) {
        const g1 = A(`h.${l}.ln1.g`), b1 = A(`h.${l}.ln1.b`), Wq = A(`h.${l}.attn.wqkv`), bq = A(`h.${l}.attn.bqkv`);
        const QKV = X.map((x) => lin(ln(x, g1, b1), Wq, bq, d, 3 * d));
        const O = Array.from({ length: T }, () => new Float64Array(d));
        const AL = [];
        for (let h = 0; h < H; h++) {
          const rows = [];
          for (let qi = 0; qi < T; qi++) {
            const s = new Float64Array(qi + 1);
            let mx = -Infinity;
            for (let k = 0; k <= qi; k++) {
              let dot = 0;
              for (let i = 0; i < dh; i++) dot += QKV[qi][h * dh + i] * QKV[k][d + h * dh + i];
              s[k] = dot * scale;
              if (s[k] > mx) mx = s[k];
            }
            let z = 0;
            for (let k = 0; k <= qi; k++) { s[k] = Math.exp(s[k] - mx); z += s[k]; }
            const row = new Float32Array(T);
            for (let k = 0; k <= qi; k++) row[k] = s[k] / z;
            rows.push(row);
            if (mask && mask(l, h, qi)) continue;
            for (let k = 0; k <= qi; k++) {
              const wk = row[k];
              for (let i = 0; i < dh; i++) O[qi][h * dh + i] += wk * QKV[k][2 * d + h * dh + i];
            }
          }
          AL.push(rows);
        }
        attn.push(AL);
        const Wo = A(`h.${l}.attn.wo`), bo = A(`h.${l}.attn.bo`);
        const Xm = X.map((x, t) => { const o = lin(O[t], Wo, bo, d, d); const y = new Float64Array(d); for (let i = 0; i < d; i++) y[i] = x[i] + o[i]; return y; });
        residMid.push(Xm);
        const g2 = A(`h.${l}.ln2.g`), b2 = A(`h.${l}.ln2.b`), Wf = A(`h.${l}.mlp.wfc`), bf = A(`h.${l}.mlp.bfc`), Wp = A(`h.${l}.mlp.wproj`), bp = A(`h.${l}.mlp.bproj`);
        X = Xm.map((x) => {
          const a = lin(ln(x, g2, b2), Wf, bf, d, F);
          for (let i = 0; i < F; i++) a[i] = gelu(a[i]);
          const o = lin(a, Wp, bp, F, d);
          const y = new Float64Array(d);
          for (let i = 0; i < d; i++) y[i] = x[i] + o[i];
          return y;
        });
        resid.push(X);
      }
      const lensRows = resid.map((R) => R.map(lens));
      return { attn, resid, residMid, lens: lensRows, probs: lensRows[NL] };
    }
    return { forward, lens, NL, H };
  }

  let NET = null, NET_OK = null;
  /** The local forward pass, but only if it reproduces AM.model.run on a probe sentence. */
  const getNet = () => {
    if (NET_OK !== null) return NET_OK ? NET : null;
    NET_OK = false;
    try {
      const m = getModel();
      NET = makeNet(m);
      if (NET) {
        const ids = m.encode('the red ball and the blue cup . the ball is').ids;
        const ref = m.run(ids, { capture: true }), mine = NET.forward(ids);
        let err = 0;
        for (let l = 0; l <= NET.NL; l++) for (let t = 0; t < ids.length; t++) {
          const a = ref.lens[l][t], b = mine.lens[l][t];
          for (let i = 0; i < a.length; i++) err = Math.max(err, Math.abs(a[i] - b[i]));
        }
        NET_OK = err < 2e-3;
      }
    } catch (e) { NET_OK = false; }
    return NET_OK ? NET : null;
  };

  const dist = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) { const z = a[i] - b[i]; s += z * z; } return Math.sqrt(s); };

  /** Everything a figure needs from one forward pass (r = AM.model.run(…, {capture}) or the local forward). */
  function analyse(m, ids, r) {
    const T = ids.length, NL = r.lens.length - 1;
    const tokens = m.decode(ids);
    const top = [];
    for (let l = 0; l <= NL; l++) { const row = []; for (let t = 0; t < T; t++) row.push(m.topk(r.lens[l][t], 5)); top.push(row); }
    const aW = [], mW = [];
    for (let l = 0; l < NL; l++) {
      aW.push(tokens.map((_, t) => dist(r.residMid[l][t], r.resid[l][t])));
      mW.push(tokens.map((_, t) => dist(r.resid[l + 1][t], r.residMid[l][t])));
    }
    // the depth from which the top guess already equals the final output (and stays)
    const lock = tokens.map((_, t) => {
      const fin = top[NL][t][0].id;
      let lv = NL;
      for (let l = NL; l >= 0; l--) { if (top[l][t][0].id === fin) lv = l; else break; }
      return lv;
    });
    return { ids, tokens, T, NL, top, lens: r.lens, attn: r.attn, resid: r.resid, residMid: r.residMid, aW, mW, lock };
  }
  const analyseText = (m, text) => {
    const enc = m.encode(text);
    const ids = enc.ids.slice(0, m.config.n_ctx);
    return analyse(m, ids, m.run(ids, { capture: true }));
  };
  const trace = (s, t) => s.top.map((row, l) => ({ l, tok: row[t][0].token, p: row[t][0].p, fin: row[t][0].id === s.top[s.NL][t][0].id }));
  const traceHTML = (s, t) => trace(s, t).map((e) => `<span class="st-tr${e.fin ? ' is-fin' : ''}"><i>x${sub(e.l)}</i> ${esc(e.tok)} <b>${pct(e.p)}</b></span>`).join('<span class="st-arr" aria-hidden="true">→</span>');

  // ================================================================== sentences
  const TOWER_SET = [
    { key: 'bind', label: 'Colour', text: 'the red ball and the blue cup . the ball is', focus: 10 },
    { key: 'pron', label: 'Pronoun', text: 'the queen opened the door because', focus: 5 },
    { key: 'agree', label: 'Agreement', text: 'the key near the old doors', focus: 5 },
    { key: 'copy', label: 'Copy a name', text: 'alice gave bob a cup . bob thanked', focus: 7 },
    { key: 'sound', label: 'Animal sound', text: 'the dog says', focus: 2 },
  ];
  const TABLE_SET = [
    { label: 'Colour', text: 'the red ball and the blue cup . the ball is' },
    { label: 'Pronoun', text: 'the queen opened the door because' },
    { label: 'Agreement', text: 'the key near the old doors' },
    { label: 'Copy', text: 'alice gave bob a cup . bob thanked' },
    { label: 'Sound', text: 'the dog says' },
    { label: 'Possessive', text: 'the dog walked to the river with' },
    { label: 'Capital', text: 'the capital of japan is' },
    { label: 'Parrot', text: 'the girl said red kite blue cup . the parrot said' },
    { label: 'Three colours', text: 'the green hat , the white door and the pink cup . the hat is' },
    { label: 'Long', text: 'the keys near the old door are gold . the red ball and the blue box . the queen opened the box because she was happy . the box is' },
  ];
  const VOCAB_GROUPS = [
    ['function', 'the a and because with of is are was were says to at near under behind capital said parrot'],
    ['verbs', 'walked loved gave thanked met waved helped opened closed found lost dropped'],
    ['pronouns', 'he she it they his her its their'],
    ['people', 'king queen boy girl prince princess wizard witch'],
    ['names', 'alice emma lucy rose bob tom sam jack'],
    ['animals', 'dog cat duck owl pig cow lion sheep children dogs cats'],
    ['things', 'key keys box boxes cup cups door doors book books hat hats ring ball kite crown'],
    ['describing', 'cold tired happy sad hungry sleepy old small gold heavy broken shiny'],
    ['colours', 'red blue green yellow white pink'],
    ['places', 'park river market castle garden'],
    ['countries', 'france japan italy spain egypt peru china kenya'],
    ['cities', 'paris tokyo rome madrid cairo lima beijing nairobi'],
    ['sounds', 'moo woof meow quack baa hoot oink roar'],
    ['nature', 'sky grass snow sun apple'],
    ['punctuation', '. ,'],
  ];

  // The training grammar, in miniature (tools/data-tinyworld.mjs), for the depth profiles.
  const G = {
    people: { king: 'm', queen: 'f', boy: 'm', girl: 'f', prince: 'm', princess: 'f', wizard: 'm', witch: 'f' },
    names: { alice: 'f', emma: 'f', lucy: 'f', rose: 'f', bob: 'm', tom: 'm', sam: 'm', jack: 'm' },
    animals: ['dog', 'cat', 'duck', 'owl', 'pig', 'cow', 'lion'],
    plural: ['children', 'dogs', 'cats'],
    pron: { m: 'he', f: 'she', n: 'it', p: 'they' },
    verbs: { opened: ['door', 'box', 'book'], closed: ['door', 'box', 'book'], found: ['key', 'cup', 'ring', 'hat', 'ball', 'kite', 'crown', 'book', 'box'], lost: ['key', 'cup', 'ring', 'hat', 'ball', 'kite', 'crown', 'book', 'box'], dropped: ['key', 'cup', 'ring', 'hat', 'ball', 'kite', 'crown', 'book', 'box'] },
    places: ['park', 'river', 'market', 'castle', 'garden'],
    pairs: [['key', 'keys'], ['box', 'boxes'], ['cup', 'cups'], ['door', 'doors'], ['book', 'books'], ['hat', 'hats']],
    preps: ['near', 'under', 'behind'],
    npAdj: ['old', 'small'],
    colors: ['red', 'blue', 'green', 'yellow', 'white', 'pink'],
    bindObjs: ['ball', 'box', 'cup', 'hat', 'kite', 'book', 'ring', 'door'],
    possObjs: ['hat', 'ball', 'book', 'ring', 'key', 'crown', 'kite', 'cup'],
    capitals: { france: 'paris', japan: 'tokyo', italy: 'rome', spain: 'madrid', egypt: 'cairo', peru: 'lima', china: 'beijing', kenya: 'nairobi' },
    sounds: { cow: 'moo', dog: 'woof', cat: 'meow', duck: 'quack', sheep: 'baa', owl: 'hoot', pig: 'oink', lion: 'roar' },
    colorFacts: { sky: 'blue', grass: 'green', snow: 'white', sun: 'yellow', apple: 'red', pig: 'pink' },
  };
  const pick = (r, xs) => xs[Math.floor(r() * xs.length)];
  const subject = (r) => {
    const cls = pick(r, ['m', 'f', 'n', 'p']);
    if (cls === 'm' || cls === 'f') {
      const pool = r() < 0.5 ? G.names : G.people;
      const w = pick(r, Object.keys(pool).filter((k) => pool[k] === cls));
      return { w: pool === G.names ? [w] : ['the', w], cls };
    }
    if (cls === 'n') return { w: ['the', pick(r, G.animals)], cls };
    if (r() < 0.3) {
      const ns = Object.keys(G.names), a = pick(r, ns);
      let b = pick(r, ns);
      while (b === a) b = pick(r, ns);
      return { w: [a, 'and', b], cls };
    }
    return { w: ['the', pick(r, G.plural)], cls };
  };
  const GEN = {
    pronoun: (r) => {
      const s = subject(r), verb = pick(r, Object.keys(G.verbs)), w = [...s.w];
      if (r() < 0.3) w.push('walked', 'to', 'the', pick(r, G.places), 'and');
      w.push(verb, 'the', pick(r, G.verbs[verb]), 'because');
      return { words: w, target: G.pron[s.cls] };
    },
    agreement: (r) => {
      const a = Math.floor(r() * G.pairs.length);
      let b = Math.floor(r() * G.pairs.length);
      while (b === a) b = Math.floor(r() * G.pairs.length);
      const pl = r() < 0.5, apl = r() < 0.5;
      const w = ['the', G.pairs[a][pl ? 1 : 0], pick(r, G.preps), 'the'];
      if (r() < 0.5) w.push(pick(r, G.npAdj));
      w.push(G.pairs[b][apl ? 1 : 0]);
      return { words: w, target: pl ? 'are' : 'is' };
    },
    binding: (r) => {
      const n = 2 + (r() < 0.35 ? 1 : 0) + (r() < 0.15 ? 1 : 0);
      const objs = [], cols = [];
      while (objs.length < n) { const o = pick(r, G.bindObjs); if (!objs.includes(o)) objs.push(o); }
      while (cols.length < n) { const c = pick(r, G.colors); if (!cols.includes(c)) cols.push(c); }
      const w = [];
      for (let k = 0; k < n; k++) {
        if (k > 0) w.push(k === n - 1 ? 'and' : ',');
        w.push('the');
        if (r() < 0.25) w.push(pick(r, G.npAdj));
        w.push(cols[k], objs[k]);
      }
      w.push('.');
      if (r() < 0.3) w.push(...subject(r).w, 'walked', 'to', 'the', pick(r, G.places), '.');
      const qi = Math.floor(r() * n);
      w.push('the', objs[qi], 'is');
      return { words: w, target: cols[qi] };
    },
    copy: (r) => {
      const ns = Object.keys(G.names), x = pick(r, ns);
      let y = pick(r, ns);
      while (y === x) y = pick(r, ns);
      const k = Math.floor(r() * 3);
      const w = k === 0 ? [x, 'gave', y, 'a', pick(r, G.possObjs), '.', y, 'thanked']
        : k === 1 ? [x, 'met', y, 'at', 'the', pick(r, G.places), '.', y, 'waved', 'at']
          : [x, 'helped', y, '.', y, 'thanked'];
      return { words: w, target: x };
    },
  };
  const FACTS = [
    ...Object.entries(G.capitals).flatMap(([c, cap]) => [
      { words: ['the', 'capital', 'of', c, 'is'], target: cap },
      { words: [cap, 'is', 'the', 'capital', 'of'], target: c },
    ]),
    ...Object.entries(G.sounds).map(([a, s]) => ({ words: ['the', a, 'says'], target: s })),
    ...Object.entries(G.colorFacts).map(([x, c]) => ({ words: ['the', x, 'is'], target: c })),
  ];

  // ================================================================== chapter CSS
  AM.css(`
    #ch-${ID} .st-split { grid-template-columns: minmax(0, 1.55fr) minmax(0, 1fr); }
    @media (max-width: 900px) { #ch-${ID} .st-split { grid-template-columns: minmax(0, 1fr); } }
    #ch-${ID} .st-stage .fig { gap: var(--space-2); }
    #ch-${ID} .st-badges { display: inline-flex; flex-wrap: wrap; gap: 6px; }
    #ch-${ID} .st-frame {
      position: relative; border: 1px solid var(--rule); border-radius: var(--radius); overflow: hidden;
      background:
        radial-gradient(90% 70% at 42% 46%, color-mix(in srgb, var(--woad) 9%, transparent) 0%, transparent 70%),
        linear-gradient(180deg, var(--ink-2), var(--ink));
    }
    #ch-${ID} .st-frame canvas { cursor: pointer; }
    #ch-${ID} .st-frame canvas:focus-visible { outline: 2px solid var(--focus); outline-offset: -3px; border-radius: var(--radius); }
    #ch-${ID} .st-cap-phone { display: none; }
    @media (max-width: 900px) {
      #ch-${ID} .st-cap-desk { display: none; }
      #ch-${ID} .st-cap-phone { display: block; }
      #ch-${ID} .st-stage .fig-top { min-height: 0; }
    }
    #ch-${ID} .st-key { display: flex; flex-wrap: wrap; gap: 4px 14px; font-family: var(--font-mono); font-size: 10px; letter-spacing: 0.06em; color: var(--mist); }
    #ch-${ID} .st-key span { display: inline-flex; align-items: center; gap: 6px; }
    #ch-${ID} .st-key i { display: inline-block; width: 8px; height: 8px; border-radius: 50%; }
    @media (max-width: 900px) { #ch-${ID} .st-stage .st-key { display: none; } }

    #ch-${ID} .step em { color: var(--linen); font-style: italic; }
    #ch-${ID} .st-chips { display: flex; flex-wrap: wrap; gap: 6px; }
    #ch-${ID} .st-chip {
      border: 1px solid var(--rule-strong); background: var(--ink); color: var(--linen-dim);
      padding: 5px 12px; border-radius: 999px; font-family: var(--font-mono); font-size: var(--fs-micro);
      letter-spacing: 0.06em; cursor: pointer; transition: border-color 0.15s, color 0.15s, background 0.15s; min-height: 30px;
    }
    #ch-${ID} .st-chip:hover { color: var(--linen); border-color: var(--linen-dim); }
    #ch-${ID} .st-chip[aria-pressed='true'] { color: var(--weld); border-color: color-mix(in srgb, var(--weld) 70%, transparent); background: color-mix(in srgb, var(--weld) 10%, var(--ink)); }
    #ch-${ID} .st-trace { display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 6px; font-family: var(--font-mono); font-size: 11px; line-height: 1.7; color: var(--linen-dim); }
    #ch-${ID} .st-tr { white-space: nowrap; padding: 1px 7px; border-radius: 5px; background: color-mix(in srgb, var(--woad) 12%, var(--ink)); border: 1px solid color-mix(in srgb, var(--woad) 30%, transparent); }
    #ch-${ID} .st-tr.is-fin { background: color-mix(in srgb, var(--weld) 12%, var(--ink)); border-color: color-mix(in srgb, var(--weld) 40%, transparent); color: var(--linen); }
    #ch-${ID} .st-tr i { font-style: normal; color: var(--mist); }
    #ch-${ID} .st-tr b { font-weight: 500; color: var(--weld); }
    #ch-${ID} .st-tr:not(.is-fin) b { color: var(--woad); }
    #ch-${ID} .st-arr { color: var(--mist); }
    #ch-${ID} .st-mute-row { display: flex; flex-wrap: wrap; gap: 8px 14px; align-items: center; }
    #ch-${ID} .st-mute-row .btn[aria-pressed='true'] { border-color: var(--madder); color: var(--madder); background: color-mix(in srgb, var(--madder) 10%, var(--ink-2)); }
    #ch-${ID} .st-result { font-size: var(--fs-small); line-height: 1.5; color: var(--linen-dim); min-height: 3em; }
    #ch-${ID} .st-result b { color: var(--linen); }
    #ch-${ID} .st-result .bad { color: var(--madder); font-weight: 600; }
    #ch-${ID} .st-result .good { color: var(--weld); font-weight: 600; }
    #ch-${ID} .st-math { font-size: 0.78em; }
    @media (max-width: 480px) { #ch-${ID} .st-math { font-size: 0.7em; white-space: nowrap; padding-left: 8px; padding-right: 8px; } }
    #ch-${ID} .st-small { display: block; font-size: var(--fs-small); line-height: 1.5; color: var(--mist) !important; }

    #ch-${ID} .st-sec { display: grid; gap: var(--space-5); }
    #ch-${ID} .st-kicker { font-family: var(--font-mono); font-size: var(--fs-micro); letter-spacing: 0.16em; text-transform: uppercase; color: var(--weld); }

    /* --- lens table --- */
    #ch-${ID} .st-lt-ctl { display: grid; gap: var(--space-3); }
    #ch-${ID} .st-lt-row { display: flex; gap: 8px; align-items: stretch; }
    #ch-${ID} .st-lt-row .text-input { flex: 1 1 auto; min-width: 0; font-size: 0.98rem; }
    #ch-${ID} .st-unk { font-size: var(--fs-small); line-height: 1.5; color: var(--mist); min-height: 1.5em; }
    #ch-${ID} .st-unk .bad { color: var(--madder); font-weight: 600; text-decoration: underline dashed; text-underline-offset: 3px; }
    #ch-${ID} .st-unk button { border: 1px dashed color-mix(in srgb, var(--weld) 60%, transparent); background: transparent; color: var(--weld); border-radius: 999px; padding: 1px 9px; margin: 0 2px; font-family: var(--font-mono); font-size: 11px; cursor: pointer; }
    #ch-${ID} .st-unk button:hover { background: color-mix(in srgb, var(--weld) 12%, transparent); }
    #ch-${ID} .st-vocab summary { cursor: pointer; font-family: var(--font-mono); font-size: var(--fs-micro); letter-spacing: 0.1em; text-transform: uppercase; color: var(--mist); width: fit-content; }
    #ch-${ID} .st-vocab summary:hover { color: var(--linen); }
    #ch-${ID} .st-vocab[open] summary { color: var(--weld); margin-bottom: var(--space-3); }
    #ch-${ID} .st-vgrid { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 250px), 1fr)); gap: 10px 22px; }
    #ch-${ID} .st-vgroup { display: flex; flex-wrap: wrap; gap: 4px; align-items: baseline; }
    #ch-${ID} .st-vgroup > span { font-family: var(--font-mono); font-size: 9.5px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--mist); margin-right: 4px; flex-basis: 100%; }
    #ch-${ID} .st-vgroup button { border: 1px solid var(--rule); background: var(--ink); color: var(--linen-dim); border-radius: 5px; padding: 1px 7px; font-size: 12.5px; cursor: pointer; }
    #ch-${ID} .st-vgroup button:hover { border-color: var(--weld); color: var(--weld); }
    #ch-${ID} .st-lt-body { display: grid; grid-template-columns: minmax(0, 1fr) 250px; gap: var(--space-5); align-items: start; }
    @media (max-width: 900px) { #ch-${ID} .st-lt-body { grid-template-columns: minmax(0, 1fr); } #ch-${ID} .st-detail { order: -1; } }
    #ch-${ID} .st-grid-wrap { overflow-x: auto; max-width: 100%; padding-bottom: 4px; }
    #ch-${ID} .st-grid { display: grid; gap: 3px; min-width: 0; }
    #ch-${ID} .st-gh { display: grid; align-content: end; gap: 1px; padding: 0 4px 4px; min-width: 0; }
    #ch-${ID} .st-gh b { font-family: var(--font-display); font-style: italic; font-weight: 400; font-size: 17px; line-height: 1; color: var(--linen); }
    #ch-${ID} .st-gh b.out { color: var(--weld); }
    #ch-${ID} .st-gh span { font-family: var(--font-mono); font-size: 9px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--mist); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    @media (max-width: 560px) { #ch-${ID} .st-gh span { font-size: 8px; letter-spacing: 0; } }
    #ch-${ID} .st-gh.mid b { font-family: var(--font-mono); font-style: normal; font-size: 10px; letter-spacing: 0.04em; color: var(--lichen); padding-bottom: 3px; }
    #ch-${ID} .st-rh { display: flex; align-items: center; justify-content: flex-end; gap: 6px; padding-right: 6px; min-width: 0; }
    #ch-${ID} .st-rh .w { font-weight: 600; font-size: 13px; color: var(--linen); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    #ch-${ID} .st-rh .i { font-family: var(--font-mono); font-size: 9px; color: var(--mist); }
    #ch-${ID} .st-cell {
      position: relative; display: grid; align-content: center; gap: 2px; min-height: 40px; min-width: 0;
      padding: 4px 7px 7px; border-radius: 6px; border: 1px solid transparent; background: var(--ink-3);
      color: var(--linen); font: 600 12.5px/1.15 var(--font-body); text-align: left; cursor: pointer; overflow: hidden;
      transition: border-color 0.12s, transform 0.12s;
    }
    #ch-${ID} .st-cell .w { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: var(--wc, var(--linen)); opacity: var(--wa, 1); }
    #ch-${ID} .st-cell .p { font: 400 9.5px/1 var(--font-mono); color: var(--linen-dim); opacity: 0.85; }
    #ch-${ID} .st-cell::after { content: ''; position: absolute; left: 6px; bottom: 3px; height: 2px; width: calc((100% - 12px) * var(--p, 0)); background: var(--bar, var(--woad)); border-radius: 2px; box-shadow: 0 0 6px var(--bar, var(--woad)); }
    #ch-${ID} .st-cell:hover { border-color: var(--linen-dim); }
    #ch-${ID} .st-cell.is-sel { border-color: var(--weld); box-shadow: 0 0 0 1px color-mix(in srgb, var(--weld) 40%, transparent), 0 0 18px -6px var(--weld); }
    #ch-${ID} .st-cell:focus-visible { outline: 2px solid var(--focus); outline-offset: 1px; }
    #ch-${ID} .st-next { display: flex; align-items: center; gap: 6px; padding-left: 8px; font-size: 12.5px; color: var(--linen-dim); white-space: nowrap; min-width: 0; }
    #ch-${ID} .st-next .ok { color: var(--verdigris); font-family: var(--font-mono); font-size: 10px; }
    #ch-${ID} .st-next .no { color: var(--mist); font-family: var(--font-mono); font-size: 10px; }
    #ch-${ID} .st-detail { position: sticky; top: 12px; display: grid; gap: 10px; padding: var(--space-4); border-radius: var(--radius-sm); background: var(--ink); border: 1px solid var(--rule); }
    @media (max-width: 900px) {
      /* phones: the detail rides along at the top while the table scrolls under it */
      #ch-${ID} .st-detail { top: 6px; z-index: 3; gap: 6px; padding: 10px 12px; box-shadow: 0 10px 24px -12px rgba(0, 0, 0, 0.8); }
      #ch-${ID} .st-detail .st-dw { font-size: 21px; }
      #ch-${ID} .st-detail .st-drow:nth-child(n+4), #ch-${ID} .st-detail .st-dtrace, #ch-${ID} .st-detail .st-dh-trace { display: none; }
    }
    #ch-${ID} .st-grid-wrap.is-scroll { -webkit-mask-image: linear-gradient(90deg, #000 calc(100% - 28px), transparent); mask-image: linear-gradient(90deg, #000 calc(100% - 28px), transparent); }
    #ch-${ID} .st-grid-wrap.is-scroll.at-end { -webkit-mask-image: none; mask-image: none; }
    #ch-${ID} .st-dw small { font-family: var(--font-mono); font-style: normal; font-size: 11px; letter-spacing: 0.06em; color: var(--mist); margin-left: 8px; }
    #ch-${ID} .st-dh { font-family: var(--font-mono); font-size: 10px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--mist); line-height: 1.5; }
    #ch-${ID} .st-dh b { color: var(--weld); font-weight: 500; text-transform: none; letter-spacing: 0.04em; }
    #ch-${ID} .st-dw { font-family: var(--font-display); font-style: italic; font-size: 26px; line-height: 1; color: var(--linen); }
    #ch-${ID} .st-drows { display: grid; gap: 5px; }
    #ch-${ID} .st-drow { display: grid; grid-template-columns: minmax(0, 6.5em) minmax(0, 1fr) 3.6em; gap: 8px; align-items: center; font-size: 12.5px; }
    #ch-${ID} .st-drow .w { font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    #ch-${ID} .st-drow .bar { height: 6px; border-radius: 3px; background: var(--ink-3); overflow: hidden; }
    #ch-${ID} .st-drow .bar i { display: block; height: 100%; border-radius: 3px; }
    #ch-${ID} .st-drow .p { font-family: var(--font-mono); font-size: 10.5px; text-align: right; color: var(--linen-dim); }
    #ch-${ID} .st-dtrace { font-size: 11px; }
    #ch-${ID} .st-legend { display: flex; flex-wrap: wrap; gap: 6px 16px; align-items: center; font-family: var(--font-mono); font-size: 10px; letter-spacing: 0.06em; color: var(--mist); }
    #ch-${ID} .st-legend span { display: inline-flex; align-items: center; gap: 6px; }
    #ch-${ID} .st-legend i { width: 18px; height: 10px; border-radius: 3px; display: inline-block; }
    #ch-${ID} .st-msg { font-size: var(--fs-small); color: var(--mist); padding: var(--space-4) 0; }

    /* --- depth profiles --- */
    #ch-${ID} .st-dp { display: grid; grid-template-columns: minmax(0, 1.25fr) minmax(0, 1fr); gap: clamp(24px, 4vw, 56px); align-items: start; }
    @media (max-width: 900px) { #ch-${ID} .st-dp { grid-template-columns: minmax(0, 1fr); } #ch-${ID} .st-dp > .prose { order: -1; } }
    #ch-${ID} .st-types { display: flex; flex-wrap: wrap; gap: 6px; }
    #ch-${ID} .st-type {
      display: inline-flex; align-items: center; gap: 7px; border: 1px solid var(--rule-strong); background: var(--ink);
      color: var(--linen-dim); padding: 4px 11px 4px 9px; border-radius: 999px; font-family: var(--font-mono);
      font-size: 10.5px; letter-spacing: 0.04em; cursor: pointer; min-height: 30px;
    }
    #ch-${ID} .st-type i { width: 14px; height: 3px; border-radius: 2px; background: var(--c); box-shadow: 0 0 6px var(--c); }
    #ch-${ID} .st-type em { font-style: normal; color: var(--mist); }
    #ch-${ID} .st-type:hover { color: var(--linen); border-color: var(--c); }
    #ch-${ID} .st-type[aria-pressed='true'] { color: var(--linen); border-color: var(--c); background: color-mix(in srgb, var(--c) 12%, var(--ink)); }
    #ch-${ID} .st-type[aria-pressed='true'] em { color: var(--c); }
    #ch-${ID} .st-dp-read { font-size: var(--fs-small); line-height: 1.55; color: var(--linen-dim); min-height: 4.7em; }
    #ch-${ID} .st-dp-read b { color: var(--linen); font-weight: 600; }
    #ch-${ID} .st-dp-read .s { font-family: var(--font-mono); font-size: 11px; color: var(--linen); }
    #ch-${ID} .st-dp-ctl { display: flex; flex-wrap: wrap; gap: 10px 16px; align-items: center; }
    #ch-${ID} .st-dp-ctl .st-status { font-family: var(--font-mono); font-size: 10.5px; color: var(--mist); }
    #ch-${ID} .st-cards { display: grid; gap: var(--space-5); max-width: calc(var(--prose) + 2 * var(--space-5)); }
    #ch-${ID} .st-card { display: grid; gap: var(--space-3); align-content: start; padding: var(--space-5); border-radius: var(--radius); border: 1px solid var(--rule); background: var(--ink-2); }
    #ch-${ID} .st-card h4 { font-family: var(--font-body); font-size: 1.05rem; font-weight: 600; color: var(--linen); }
    #ch-${ID} .st-card p { color: var(--linen-dim); font-size: 0.98rem; }
  `);

  // ================================================================== 1. the tower
  const towerHeight = (w) => {
    const ih = window.innerHeight || 800;
    if (isStacked()) return Math.round(clamp(w * 0.92, 300, Math.max(300, Math.min(420, ih * 0.47))));
    return Math.round(clamp(w * 0.82, 470, Math.max(470, ih * 0.72)));
  };
  const MODES = {
    stack: { arcs: 0.7, comp: 0, beads: 0, readout: 0, gauge: 1, out: 0, dim: 0 },
    compose: { arcs: 0.13, comp: 1, beads: 0, readout: 0, gauge: 0, out: 1, dim: 0 },
    lens: { arcs: 0.2, comp: 0, beads: 1, readout: 1, gauge: 0, out: 0, dim: 0 },
    focus: { arcs: 0.12, comp: 0, beads: 1, readout: 1, gauge: 0, out: 0, dim: 1 },
    depths: { arcs: 0.2, comp: 0, beads: 1, readout: 1, gauge: 0, out: 0, dim: 0 },
  };

  function buildTower(ctx, host) {
    const frame = AM.el('div', { class: 'st-frame' });
    host.appendChild(frame);
    const cv = ctx.canvas(frame, { height: towerHeight, label: 'The tower of layers' });
    const { canvas, g } = cv;
    canvas.tabIndex = 0;
    const srLive = AM.el('p', { class: 'sr-only', 'aria-live': 'polite' });
    host.appendChild(srLive);

    const WELD = RGB(AM.dye.weld), WOAD = RGB(AM.dye.woad), LINEN = RGB(AM.col.linen);
    const S = {
      slot: null, muted: false,
      cur: null, prev: null, tr: 1,
      focus: 0, hover: -1, mode: 'stack',
      v: Object.assign({}, MODES.stack), tg: Object.assign({}, MODES.stack),
      sweep: 1, sweeping: false, bloomed: null,
      build: 1, building: false,
      blooms: [], flow: [], arcParts: [], arcSpawn: 0,
      w: 0, h: 0, geo: null, bg: null, time: 0,
    };

    // ------------------------------------------------------------ geometry
    function geo() {
      const w = S.w, h = S.h, phone = w < 560;
      const Gw = phone ? 30 : 70;
      const R = phone ? 92 : clamp(Math.round(w * 0.27), 156, 204);
      const xa = Gw + (phone ? 6 : 14), xb = w - R - (phone ? 12 : 22);
      const botPad = phone ? 50 : 44;
      const yTop = phone ? 50 : 66;
      const yBot = h - botPad - (phone ? 12 : 18);
      const sp = (yBot - yTop) / 3;
      const ys = [0, 1, 2, 3].map((l) => yBot - l * sp);
      const gapB = phone ? 12 : 22, gapT = phone ? 8 : 14;
      const bands = [0, 1, 2].map((l) => {
        const yb = ys[l] - gapB, yt = ys[l + 1] + gapT;
        return { yb, yt, knot: yt + (phone ? 8 : 15), base: yb - 3, arcMax: (yb - yt) * 0.62 };
      });
      return { w, h, phone, Gw, R, xa, xb, ys, sp, bands, botPad, rx: w - R + (phone ? 4 : 8) };
    }
    function arcBez(Gm, L, l, h, k, qq, boost = 1) {
      const b = Gm.bands[l];
      const x0 = L.xs[k], x1 = L.xs[qq], dx = x1 - x0;
      const hh = Math.min(b.arcMax * boost, (Gm.phone ? 5 : 8) + dx * 0.42) * (1 + h * 0.04);
      const y = b.base - h * 1.1;
      const c = hh * 1.333;
      return [x0, y, x0 + dx * 0.08, y - c, x1 - dx * 0.08, y - c, x1, y];
    }
    function lay(sent) {
      const Gm = S.geo;
      if (sent._lay && sent._lay.geo === Gm) return sent._lay;
      const T = sent.T, colW = (Gm.xb - Gm.xa) / T;
      const xs = sent.tokens.map((_, t) => Gm.xa + colW * (t + 0.5));
      const fs = Gm.phone ? 10.5 : 13;
      g.font = AM.font(fs, 'body', 600);
      const tw = sent.tokens.map((s) => g.measureText(s).width);
      const rotate = Math.max(...tw) > colW - 4;
      const bfs = Gm.phone ? 9.5 : 10.5;
      g.font = AM.font(bfs, 'body', 600);
      const bw = sent.top.map((row) => row.map((tp) => g.measureText(tp[0].token).width));
      const stagger = bw.map((row) => row.some((wd, t) => t < T - 1 && (wd + row[t + 1]) / 2 + 6 > colW));
      const L = { geo: Gm, T, colW, xs, fs, rotate, bfs, bw, stagger };
      // attention threads: weight ≥ 0.3, leaving out self-attention and the habitual resting weight on the first word
      const arcs = [];
      for (let l = 0; l < sent.NL; l++) for (let h = 0; h < 4; h++) for (let qq = 1; qq < T; qq++) {
        const row = sent.attn[l][h][qq];
        for (let k = 0; k < qq; k++) {
          const wgt = row[k];
          if (wgt < 0.3) continue;
          if (k === 0 && l > 0 && wgt < 0.5) continue;
          arcs.push({ l, h, q: qq, k, w: wgt, B: arcBez(Gm, L, l, h, k, qq) });
        }
      }
      L.arcs = arcs;
      L.arcW = arcs.reduce((s, a) => s + a.w, 0);
      L.cache = null;
      let mx = 1e-6;
      for (const row of sent.mW) for (const v of row) mx = Math.max(mx, v);
      L.mMax = mx;
      sent._lay = L;
      return L;
    }
    function arcCache(L) {
      if (L.cache) return L.cache;
      const c = document.createElement('canvas');
      c.width = Math.max(1, Math.round(S.w * cv.dpr));
      c.height = Math.max(1, Math.round(S.h * cv.dpr));
      const x = c.getContext('2d');
      x.setTransform(cv.dpr, 0, 0, cv.dpr, 0, 0);
      x.lineCap = 'round';
      for (const a of L.arcs) {
        const col = AM.headColor(a.h);
        x.strokeStyle = AM.rgba(col, 0.07 + 0.12 * a.w);
        x.lineWidth = 2.5 + 5 * a.w;
        strokeBez(x, a.B);
        x.strokeStyle = AM.rgba(col, 0.35 + 0.6 * a.w);
        x.lineWidth = 0.7 + 1.5 * a.w;
        strokeBez(x, a.B);
      }
      L.cache = c;
      return c;
    }

    // ------------------------------------------------------------ static layer: bands + gutter
    function drawBands(x, Gm, eFn) {
      for (let l = 0; l < 3; l++) {
        const e = eFn(l);
        if (e <= 0.001) continue;
        const b = Gm.bands[l], off = (1 - e) * -30;
        const x0 = Gm.xa - (Gm.phone ? 7 : 14), x1 = Gm.xb + (Gm.phone ? 7 : 14);
        const yt = b.yt + off, yb = b.yb + off, rr = Gm.phone ? 6 : 9;
        x.save();
        x.globalAlpha *= e;
        D.roundRect(x, x0, yt, x1 - x0, yb - yt, rr);
        const grd = x.createLinearGradient(0, yt, 0, yb);
        grd.addColorStop(0, AM.rgba(AM.col.ink3, 0.8));
        grd.addColorStop(1, AM.rgba(AM.col.ink2, 0.45));
        x.fillStyle = grd;
        x.fill();
        x.save();
        x.clip();
        x.lineWidth = 1;
        x.strokeStyle = AM.rgba(AM.col.linen, 0.03);
        x.beginPath();
        for (let yy = yt + 2.5; yy < yb; yy += 4) { x.moveTo(x0, yy); x.lineTo(x1, yy); }
        x.stroke();
        x.strokeStyle = AM.rgba('#000000', 0.16);
        x.beginPath();
        for (let xx = x0 + 0.5; xx < x1; xx += 4) { x.moveTo(xx, yt); x.lineTo(xx, yb); }
        x.stroke();
        // the MLP lane, at the top of the band
        const lane = b.knot + (Gm.phone ? 7 : 11);
        x.fillStyle = AM.rgba(AM.dye.lichen, 0.05);
        x.fillRect(x0, yt, x1 - x0, lane - yt);
        x.setLineDash([2, 4]);
        x.strokeStyle = AM.rgba(AM.dye.lichen, 0.25);
        x.beginPath(); x.moveTo(x0 + 6, lane); x.lineTo(x1 - 6, lane); x.stroke();
        x.setLineDash([]);
        x.restore();
        D.roundRect(x, x0, yt, x1 - x0, yb - yt, rr);
        x.strokeStyle = AM.rgba(AM.col.ruleStrong, 0.95);
        x.stroke();
        // selvedge: woven ticks down both edges
        x.fillStyle = AM.rgba(AM.dye.weld, 0.32);
        for (let yy = yt + 6; yy < yb - 4; yy += 6) { x.fillRect(x0 + 2.5, yy, 2, 3); x.fillRect(x1 - 4.5, yy + 3, 2, 3); }
        x.restore();
      }
    }
    function drawGutter(x, Gm) {
      const ph = Gm.phone;
      for (let l = 0; l <= 3; l++) {
        const y = Gm.ys[l];
        D.text(x, 'x' + sub(l), Gm.Gw - (ph ? 4 : 12), y + (ph ? 4 : 2), { size: ph ? 13 : 18, role: 'display', italic: true, weight: 400, color: l === 3 ? AM.dye.weld : AM.col.linen, align: 'right' });
        if (!ph) D.text(x, LEVEL_NAMES[l].toUpperCase(), Gm.Gw - 12, y + 15, { size: 8, role: 'mono', color: AM.col.mist, align: 'right', letterSpacing: '0.08em' });
        x.save();
        x.strokeStyle = AM.rgba(AM.col.linen, 0.07);
        x.setLineDash([1, 4]);
        x.beginPath(); x.moveTo(Gm.Gw - (ph ? 2 : 6), y + 0.5); x.lineTo(Gm.xa - (ph ? 7 : 14), y + 0.5); x.stroke();
        x.restore();
      }
      for (let l = 0; l < 3; l++) {
        const b = Gm.bands[l], yc = (b.yt + b.yb) / 2;
        if (!ph) {
          D.text(x, 'LAYER ' + l, Gm.Gw - 12, yc - 1, { size: 9, role: 'mono', color: AM.dye.weld, align: 'right', letterSpacing: '0.12em', alpha: 0.9 });
          D.text(x, 'attn + mlp', Gm.Gw - 12, yc + 12, { size: 8, role: 'mono', color: AM.col.mist, align: 'right' });
        } else D.text(x, 'L' + l, Gm.Gw - 3, yc + 3, { size: 9, role: 'mono', color: AM.dye.weld, align: 'right' });
      }
    }
    function bgCache() {
      if (S.bg) return S.bg;
      const c = document.createElement('canvas');
      c.width = Math.max(1, Math.round(S.w * cv.dpr));
      c.height = Math.max(1, Math.round(S.h * cv.dpr));
      const x = c.getContext('2d');
      x.setTransform(cv.dpr, 0, 0, cv.dpr, 0, 0);
      drawBands(x, S.geo, () => 1);
      drawGutter(x, S.geo);
      S.bg = c;
      return c;
    }
    const buildE = (l) => (S.build >= 1 ? 1 : E.outBack(clamp(S.build * 1.6 - l * 0.3)));

    // ------------------------------------------------------------ sentence layers
    const dimOf = (t) => 1 - S.v.dim * (t === S.focus ? 0 : 0.6);
    const focusOf = (sent) => clamp(sent.focus != null ? sent.focus : sent.T - 1, 0, sent.T - 1);
    function drawWarps(sent, L, a) {
      const Gm = S.geo;
      const yA = S.h - Gm.botPad + (Gm.phone ? 2 : 4), yB = Gm.ys[3] - (Gm.phone ? 14 : 24);
      g.save();
      g.lineCap = 'round';
      for (let t = 0; t < sent.T; t++) {
        const f = t === S.focus, hv = t === S.hover, x = L.xs[t];
        const aa = a * dimOf(t) * (f ? 0.95 : hv ? 0.7 : 0.34);
        if (f || hv) {
          const grd = g.createLinearGradient(0, yB, 0, yA);
          grd.addColorStop(0, AM.rgba(AM.dye.weld, 0));
          grd.addColorStop(0.5, AM.rgba(f ? AM.dye.weld : AM.col.linen, (f ? 0.11 : 0.05) * a));
          grd.addColorStop(1, AM.rgba(AM.dye.weld, 0));
          g.fillStyle = grd;
          g.fillRect(x - L.colW * 0.42, yB, L.colW * 0.84, yA - yB);
        }
        const amp = f ? 1.9 : 1.3, col = f ? AM.dye.weld : AM.col.linen;
        for (const sgn of [-1, 1]) {
          g.beginPath();
          for (let y = yA; y >= yB; y -= 5) {
            const xx = x + sgn * amp * Math.sin(y * 0.055 + S.time * 1.1 + t * 1.7);
            if (y === yA) g.moveTo(xx, y); else g.lineTo(xx, y);
          }
          g.strokeStyle = AM.rgba(col, aa * (sgn > 0 ? 1 : 0.6));
          g.lineWidth = f ? 1.3 : 1;
          g.stroke();
        }
      }
      g.restore();
    }
    function drawKnots(sent, L, a) {
      if (a < 0.01) return;
      const Gm = S.geo;
      g.save();
      for (let l = 0; l < sent.NL; l++) {
        const y = Gm.bands[l].knot;
        for (let t = 0; t < sent.T; t++) {
          const n = Math.sqrt(sent.mW[l][t] / L.mMax), x = L.xs[t];
          const r = (Gm.phone ? 1.3 : 1.8) + (Gm.phone ? 2.6 : 4.2) * n;
          const aa = a * dimOf(t);
          g.beginPath();
          g.moveTo(x, y - r); g.lineTo(x + r * 0.85, y); g.lineTo(x, y + r); g.lineTo(x - r * 0.85, y); g.closePath();
          g.fillStyle = AM.rgba(AM.dye.lichen, (0.2 + 0.5 * n) * aa);
          g.fill();
          g.strokeStyle = AM.rgba(AM.dye.lichen, (0.45 + 0.5 * n) * aa);
          g.lineWidth = 0.9;
          g.stroke();
        }
      }
      g.restore();
    }
    const beadVis = (l) => (S.v.beads <= 0.001 ? 0 : clamp((S.sweep * 1.15 - l * 0.3) / 0.2) * S.v.beads);
    const sweepY = (s) => S.geo.ys[0] - ((s * 1.15 - 0.1) / 0.3) * S.geo.sp;
    function drawBeads(sent, L, a) {
      if (S.v.beads <= 0.001) return;
      const Gm = S.geo;
      g.save();
      for (let l = 0; l <= sent.NL; l++) {
        const v = beadVis(l);
        if (v <= 0.001) continue;
        const y = Gm.ys[l], sc = AM.reducedMotion ? 1 : Math.max(0, E.outBack(clamp(v)));
        for (let t = 0; t < sent.T; t++) {
          const tp = sent.top[l][t][0], same = tp.id === sent.top[sent.NL][t][0].id;
          const dm = dimOf(t), x = L.xs[t];
          const r = ((Gm.phone ? 2 : 2.8) + (Gm.phone ? 3.2 : 4.6) * tp.p) * sc;
          const col = same ? mixRgb(WELD, LINEN, 0.08 * tp.p) : mixRgb(WOAD, LINEN, 0.1);
          g.globalAlpha = a * v * dm * (0.3 + 0.7 * tp.p);
          const R = r * 3.3;
          g.drawImage(sprite(col), x - R, y - R, R * 2, R * 2);
          g.globalAlpha = a * v * dm;
          g.fillStyle = same ? '#fff2c8' : '#dbe7ff';
          g.beginPath(); g.arc(x, y, Math.max(0.8, r * 0.42), 0, Math.PI * 2); g.fill();
          if (t === S.focus && S.v.readout > 0.05) {
            g.strokeStyle = AM.rgba(AM.dye.weld, 0.7 * a * v);
            g.lineWidth = 1;
            g.beginPath(); g.arc(x, y, r + 3.5, 0, Math.PI * 2); g.stroke();
          }
          // label: the top guess
          const show = !Gm.phone || t === S.hover;
          if (!show) continue;
          const above = !L.stagger[l] || t % 2 === 0;
          const ly = above ? y - r - 6 : y + r + 13;
          const isF = t === S.focus;
          g.shadowColor = AM.col.ink;
          g.shadowBlur = 4;
          D.text(g, tp.token, x, ly, {
            size: L.bfs, weight: isF ? 700 : 600, align: 'center',
            color: same ? (isF ? '#fff0c2' : AM.dye.weld) : (isF ? '#e4ecff' : AM.dye.woad),
            alpha: a * v * dm * (0.62 + 0.38 * tp.p),
          });
          g.shadowBlur = 0;
        }
      }
      g.restore();
    }
    function drawReadout(sent, L, a) {
      if (a < 0.01) return;
      const Gm = S.geo, t = clamp(S.focus, 0, sent.T - 1), rx = Gm.rx, rw = Gm.w - rx - (Gm.phone ? 6 : 12);
      const fin = sent.top[sent.NL][t][0].id;
      g.save();
      const hy = Gm.phone ? 21 : 29, hl = Gm.phone ? 'LENS AT' : 'LOGIT LENS AT';
      D.text(g, hl, rx, hy, { size: 8.5, role: 'mono', color: AM.col.mist, letterSpacing: '0.1em', alpha: a });
      const hw = D.measure(g, hl, 8.5, 'mono') + hl.length * 0.85 + 6;
      D.text(g, q(sent.tokens[t]), rx + hw, hy + 1, { size: Gm.phone ? 13 : 17, role: 'display', italic: true, weight: 500, color: AM.dye.weld, alpha: a, maxWidth: Math.max(20, rw - hw) });
      const rows = Gm.phone ? 2 : 3, lh = Gm.phone ? 13 : 15.5;
      for (let l = 0; l <= sent.NL; l++) {
        const y = Gm.ys[l];
        const rv = S.v.beads > 0.01 ? clamp((S.sweep * 1.15 - l * 0.3) / 0.2) : 1;
        if (rv <= 0.01) continue;
        const a0 = a;
        a = a0 * rv;
        // connector from the focus bead
        const bv = beadVis(l);
        if (bv > 0.01) {
          g.strokeStyle = AM.rgba(AM.dye.weld, 0.22 * a * bv);
          g.setLineDash([2, 4]);
          g.lineWidth = 1;
          g.beginPath(); g.moveTo(L.xs[t] + 11, y + 0.5); g.lineTo(rx - 8, y + 0.5); g.stroke();
          g.setLineDash([]);
        }
        for (let i = 0; i < rows; i++) {
          const e = sent.top[l][t][i], yy = y + (i - (rows - 1) / 2) * lh, isFin = e.id === fin;
          const col = i === 0 ? (isFin ? AM.dye.weld : '#cfdcff') : AM.col.mist;
          const fsz = i === 0 ? (Gm.phone ? 10.5 : 12) : (Gm.phone ? 9.5 : 10.5);
          D.text(g, e.token, rx, yy + 3.5, { size: fsz, weight: i === 0 ? 650 : 500, color: col, alpha: a * (i === 0 ? 1 : 0.85), maxWidth: rw - (Gm.phone ? 30 : 38) });
          D.text(g, pct(e.p), rx + rw, yy + 3.5, { size: Gm.phone ? 8.5 : 9.5, role: 'mono', color: i === 0 ? AM.col.linen : AM.col.mist, align: 'right', alpha: a });
          g.globalAlpha = a * (i === 0 ? 0.9 : 0.5);
          g.fillStyle = isFin ? AM.dye.weld : AM.dye.woad;
          g.fillRect(rx, yy + 6.5, Math.max(1, rw * e.p), i === 0 ? 1.6 : 1);
          g.globalAlpha = 1;
        }
        a = a0;
      }
      g.restore();
    }
    function drawGauge(a) {
      if (a < 0.01) return;
      const Gm = S.geo, rx = Gm.rx, rw = Gm.w - rx - (Gm.phone ? 4 : 10), cw = rw / 3;
      const top = Gm.ys[3] + (Gm.phone ? 8 : 14), bot = Gm.ys[0] - (Gm.phone ? 12 : 22);
      const items = [
        { n: 3, a: 'this', b: 'model', own: true },
        { n: 12, a: 'GPT-2', b: 'small' },
        { n: 96, a: 'GPT-3', b: '175B' },
      ];
      g.save();
      g.globalAlpha = a;
      D.text(g, 'LAYERS DEEP', rx, Gm.phone ? 21 : 29, { size: 8.5, role: 'mono', color: AM.col.mist, letterSpacing: '0.1em' });
      items.forEach((it, i) => {
        const cx = rx + cw * (i + 0.5), tw = Math.min(cw * 0.52, Gm.phone ? 20 : 30);
        if (it.own) {
          Gm.bands.forEach((b) => {
            D.roundRect(g, cx - tw / 2, b.yt, tw, b.yb - b.yt, 3);
            g.fillStyle = AM.rgba(AM.dye.weld, 0.22); g.fill();
            g.strokeStyle = AM.rgba(AM.dye.weld, 0.7); g.lineWidth = 1; g.stroke();
          });
        } else {
          const step = (bot - top) / it.n, bh = Math.max(0.7, step * 0.6);
          g.fillStyle = AM.rgba(AM.col.linenDim, it.n > 40 ? 0.5 : 0.42);
          for (let k = 0; k < it.n; k++) g.fillRect(cx - tw / 2, bot - (k + 0.5) * step - bh / 2, tw, bh);
          g.strokeStyle = AM.rgba(AM.col.ruleStrong, 0.9);
          g.strokeRect(cx - tw / 2 - 2.5, top - 2.5, tw + 5, bot - top + 5);
        }
        D.text(g, String(it.n), cx, S.h - Gm.botPad + (Gm.phone ? 20 : 22), { size: Gm.phone ? 16 : 22, role: 'display', italic: true, weight: 400, color: it.own ? AM.dye.weld : AM.col.linen, align: 'center' });
        if (Gm.phone) {
          // one short label per tower: two lines of mono do not fit 28 px columns
          D.text(g, it.own ? 'HERE' : it.a.toUpperCase(), cx, Gm.ys[3] - 5, { size: 7, role: 'mono', color: it.own ? AM.dye.weld : AM.col.linenDim, align: 'center', maxWidth: cw - 3 });
        } else {
          D.text(g, it.a.toUpperCase(), cx, Gm.ys[3] - 16, { size: 8.5, role: 'mono', color: it.own ? AM.dye.weld : AM.col.linenDim, align: 'center', letterSpacing: '0.04em', maxWidth: cw - 2 });
          D.text(g, it.b.toUpperCase(), cx, Gm.ys[3] - 5, { size: 8, role: 'mono', color: AM.col.mist, align: 'center', letterSpacing: '0.04em', maxWidth: cw - 2 });
        }
      });
      g.restore();
    }
    function drawTokens(sent, L, a) {
      const Gm = S.geo;
      g.save();
      for (let t = 0; t < sent.T; t++) {
        const f = t === S.focus, col = f ? AM.dye.weld : t === S.hover ? AM.col.linen : AM.col.linenDim;
        const aa = a * (0.55 + 0.45 * dimOf(t));
        if (!L.rotate) {
          D.text(g, sent.tokens[t], L.xs[t], S.h - Gm.botPad + 24, { size: L.fs, weight: 600, color: col, align: 'center', alpha: aa });
          if (f) { g.globalAlpha = aa; g.fillStyle = AM.dye.weld; g.fillRect(L.xs[t] - 7, S.h - Gm.botPad + 30, 14, 2); g.globalAlpha = 1; }
        } else {
          g.save();
          g.translate(L.xs[t] + 3, S.h - Gm.botPad + 8);
          g.rotate(-0.92);
          D.text(g, sent.tokens[t], 0, 3, { size: L.fs - 0.5, weight: 600, color: col, align: 'right', alpha: aa });
          g.restore();
        }
      }
      g.restore();
    }
    function pill(x, y, lines, col, a, align = 'center') {
      g.save();
      g.globalAlpha = a;
      const ph = S.geo.phone;
      const fs1 = ph ? 8.5 : 9.5, fs2 = ph ? 9 : 10.5;
      g.font = AM.font(fs1, 'mono', 500);
      let wd = g.measureText(lines[0]).width;
      if (lines[1]) { g.font = AM.font(fs2, 'body', 500, true); wd = Math.max(wd, g.measureText(lines[1]).width); }
      const pw = wd + 14, phh = lines[1] ? (ph ? 30 : 34) : (ph ? 17 : 20);
      let bx = align === 'center' ? x - pw / 2 : align === 'right' ? x - pw : x;
      bx = clamp(bx, 4, S.w - pw - 4);
      D.roundRect(g, bx, y - phh / 2, pw, phh, 6);
      g.fillStyle = AM.rgba(AM.col.ink, 0.92); g.fill();
      g.strokeStyle = AM.rgba(col, 0.8); g.lineWidth = 1; g.stroke();
      D.text(g, lines[0], bx + 7, y + (lines[1] ? -3 : 3.5), { size: fs1, role: 'mono', color: col });
      if (lines[1]) D.text(g, lines[1], bx + 7, y + (ph ? 10 : 11.5), { size: fs2, italic: true, weight: 500, color: AM.col.linenDim });
      g.restore();
    }
    function drawCompose(sent, L, a) {
      if (a < 0.01) return;
      const Gm = S.geo, t = focusOf(sent);
      if (t < 2) return;
      const a0 = sent.attn[0][0][t], a1 = sent.attn[1][3][t];
      let k0 = 0;
      for (let k = 0; k < t; k++) if (a0[k] > a0[k0]) k0 = k;
      const keys1 = [];
      for (let k = 0; k < t; k++) if (a1[k] >= 0.05) keys1.push(k);
      keys1.sort((x, y) => a1[y] - a1[x]);
      g.save();
      g.lineCap = 'round';
      const c3 = AM.headColor(3), c0 = AM.headColor(0);
      for (const k of keys1) {
        const w = a1[k], B = arcBez(Gm, L, 1, 3, k, t, 1.25);
        g.strokeStyle = AM.rgba(c3, 0.16 * a * (0.3 + w)); g.lineWidth = 4 + 9 * w; strokeBez(g, B);
        g.strokeStyle = AM.rgba(c3, a * (0.35 + 0.65 * w)); g.lineWidth = 1 + 3.2 * w; strokeBez(g, B);
      }
      const B0 = arcBez(Gm, L, 0, 0, k0, t, 1.25);
      const silenced = S.muted;
      if (silenced) {
        g.setLineDash([3, 5]);
        g.strokeStyle = AM.rgba(AM.col.mist, 0.75 * a); g.lineWidth = 1.6; strokeBez(g, B0);
        g.setLineDash([]);
        const mid = bez(B0, 0.5), s = Gm.phone ? 4 : 5.5;
        g.strokeStyle = AM.rgba(AM.dye.madder, a); g.lineWidth = 2.2;
        g.beginPath(); g.moveTo(mid.x - s, mid.y - s); g.lineTo(mid.x + s, mid.y + s); g.moveTo(mid.x + s, mid.y - s); g.lineTo(mid.x - s, mid.y + s); g.stroke();
      } else {
        const w = a0[k0];
        g.strokeStyle = AM.rgba(c0, 0.18 * a); g.lineWidth = 4 + 8 * w; strokeBez(g, B0);
        g.strokeStyle = AM.rgba(c0, a * (0.4 + 0.6 * w)); g.lineWidth = 1 + 3 * w; strokeBez(g, B0);
      }
      // labels
      const top1 = keys1[0];
      if (top1 != null) {
        const B = arcBez(Gm, L, 1, 3, top1, t, 1.25), apex = bez(B, 0.5);
        const second = keys1[1];
        const l1 = Gm.phone ? `L1H3 ${sent.tokens[top1]} ${f2(a1[top1])}` : `L1H3 → ${sent.tokens[top1]} ${f2(a1[top1])}${second != null && a1[second] >= 0.1 ? ` · ${sent.tokens[second]} ${f2(a1[second])}` : ''}`;
        pill(apex.x, apex.y - (Gm.phone ? 12 : 18), Gm.phone ? [l1] : [l1, headName(1, 3)], c3, a);
      }
      {
        const apex = bez(B0, 0.5);
        const l0 = silenced ? 'L0H0 silenced here' : (Gm.phone ? `L0H0 ${sent.tokens[k0]} ${f2(a0[k0])}` : `L0H0 → ${sent.tokens[k0]} ${f2(a0[k0])}`);
        pill(apex.x - (Gm.phone ? 8 : 14), apex.y - (Gm.phone ? 10 : 14), Gm.phone ? [l0] : [l0, headName(0, 0)], silenced ? AM.dye.madder : c0, a, 'right');
      }
      g.restore();
    }
    function drawOut(sent, L, a) {
      if (a < 0.01) return;
      const Gm = S.geo, t = focusOf(sent);
      const tp = sent.top[sent.NL][t][0], x = L.xs[t], y = Gm.ys[3];
      const col = S.muted ? AM.dye.madder : AM.dye.weld;
      g.save();
      g.globalAlpha = a;
      g.drawImage(sprite(RGB(col)), x - 26, y - 26, 52, 52);
      g.fillStyle = '#fff4d6';
      g.beginPath(); g.arc(x, y, 2.6, 0, Math.PI * 2); g.fill();
      const tx = Math.min(x + 12, Gm.rx + 40);
      D.text(g, tp.token, tx, y - (Gm.phone ? 9 : 12), { size: Gm.phone ? 17 : 24, role: 'display', italic: true, weight: 500, color: col });
      D.text(g, `${pct(tp.p)} · next word`, tx, y + (Gm.phone ? 6 : 8), { size: Gm.phone ? 8 : 9, role: 'mono', color: AM.col.mist, letterSpacing: '0.06em' });
      g.restore();
    }

    // ------------------------------------------------------------ particles
    function resetFlow(sent) {
      S.flow = [];
      if (!sent || AM.reducedMotion) return;
      const Gm = S.geo;
      const yA = S.h - Gm.botPad, yB = Gm.ys[3] - 20;
      for (let t = 0; t < sent.T; t++) for (let k = 0; k < 3; k++) S.flow.push({ t, y: lerp(yA, yB, Math.random()), sp: 22 + Math.random() * 20 });
    }
    function stepParticles(dt, sent, L) {
      if (AM.reducedMotion || !sent) { S.arcParts.length = 0; return; }
      const Gm = S.geo, yA = S.h - Gm.botPad, yB = Gm.ys[3] - 20;
      for (const p of S.flow) { p.y -= p.sp * dt; if (p.y < yB) { p.y = yA; p.sp = 22 + Math.random() * 20; } }
      // threads carry particles from key to query (the direction information moves)
      const want = S.v.comp > 0.5 ? 'comp' : (S.v.arcs > 0.5 ? 'all' : null);
      if (want && S.arcParts.length < 160) {
        S.arcSpawn += dt * (want === 'all' ? 16 : 7);
        while (S.arcSpawn >= 1) {
          S.arcSpawn -= 1;
          if (want === 'all' && L.arcs.length) {
            let u = Math.random() * L.arcW, a = L.arcs[0];
            for (const c of L.arcs) { u -= c.w; if (u <= 0) { a = c; break; } }
            S.arcParts.push({ B: a.B, u: 0, dur: 1 + Math.abs(a.q - a.k) * 0.12, col: AM.headColor(a.h), s: 0.6 + a.w });
          } else if (want === 'comp') {
            const t = focusOf(sent);
            const a1 = sent.attn[1][3][t];
            let k1 = 0;
            for (let k = 0; k < t; k++) if (a1[k] > a1[k1]) k1 = k;
            const opts = [{ l: 1, h: 3, k: k1, w: a1[k1] }];
            if (!S.muted) {
              const a0 = sent.attn[0][0][t];
              let k0 = 0;
              for (let k = 0; k < t; k++) if (a0[k] > a0[k0]) k0 = k;
              opts.push({ l: 0, h: 0, k: k0, w: a0[k0] });
            }
            const o = opts[Math.floor(Math.random() * opts.length)];
            S.arcParts.push({ B: arcBez(Gm, L, o.l, o.h, o.k, t, 1.25), u: 0, dur: 1.2, col: AM.headColor(o.h), s: 1.1 });
          }
        }
      }
      for (const p of S.arcParts) p.u += dt / p.dur;
      S.arcParts = S.arcParts.filter((p) => p.u < 1);
    }
    function drawParticles(sent, L, a) {
      if (AM.reducedMotion || !sent || a < 0.02) return;
      const ph = S.geo.phone;
      g.save();
      for (const p of S.flow) {
        if (p.t >= sent.T) continue;
        const f = p.t === S.focus, x = L.xs[p.t];
        const s = (ph ? 5 : 7) * (f ? 1.3 : 1);
        g.globalAlpha = a * dimOf(p.t) * (f ? 0.85 : 0.35);
        g.drawImage(sprite(f ? WELD : LINEN), x - s / 2, p.y - s / 2, s, s);
      }
      for (const p of S.arcParts) {
        const pt = bez(p.B, E.inOut(p.u)), s = (ph ? 6 : 8) * p.s;
        g.globalAlpha = a * Math.sin(Math.PI * p.u) * 0.9;
        g.drawImage(sprite(RGB(p.col)), pt.x - s / 2, pt.y - s / 2, s, s);
      }
      g.restore();
    }
    function drawBlooms() {
      if (!S.blooms.length) return;
      g.save();
      for (const b of S.blooms) {
        const k = (S.time - b.t0) / (b.big ? 1.1 : 0.85);
        if (k < 0 || k > 1) continue;
        const r = 4 + (b.big ? 30 : 16) * E.out(k);
        g.strokeStyle = AM.rgba(AM.dye.weld, (1 - k) * (b.big ? 0.9 : 0.55));
        g.lineWidth = b.big ? 2 : 1.2;
        g.beginPath(); g.arc(b.x, b.y, r, 0, Math.PI * 2); g.stroke();
        if (b.big) {
          for (let i = 0; i < 8; i++) {
            const ang = (i / 8) * Math.PI * 2 + 0.3, rr = 6 + 34 * E.out(k);
            g.globalAlpha = 1 - k;
            g.drawImage(sprite(WELD), b.x + Math.cos(ang) * rr - 3, b.y + Math.sin(ang) * rr - 3, 6, 6);
          }
          g.globalAlpha = 1;
        }
      }
      g.restore();
      S.blooms = S.blooms.filter((b) => S.time - b.t0 < 1.2);
    }
    function drawSweep() {
      if (!S.sweeping) return;
      const s = S.sweep, y = sweepY(s), Gm = S.geo;
      const env = Math.sin(Math.PI * clamp(s * 1.05));
      const x0 = Gm.xa - 20, x1 = Gm.w - 6, hh = Gm.phone ? 18 : 28;
      g.save();
      const grd = g.createLinearGradient(0, y - hh, 0, y + hh);
      grd.addColorStop(0, AM.rgba(AM.dye.weld, 0));
      grd.addColorStop(0.5, AM.rgba(AM.dye.weld, 0.16 * env));
      grd.addColorStop(1, AM.rgba(AM.dye.weld, 0));
      g.fillStyle = grd;
      g.fillRect(x0, y - hh, x1 - x0, hh * 2);
      // glass rims
      g.fillStyle = AM.rgba('#fff3cf', 0.12 * env);
      g.fillRect(x0 + 20, y - hh * 0.62, x1 - x0 - 40, 0.8);
      g.fillRect(x0 + 20, y + hh * 0.62, x1 - x0 - 40, 0.8);
      // a glint where the lens crosses the focused thread
      if (S.cur) {
        const L = lay(S.cur), fx = L.xs[clamp(S.focus, 0, S.cur.T - 1)], R = Gm.phone ? 18 : 28;
        g.globalAlpha = 0.85 * env;
        g.drawImage(sprite(RGB(AM.dye.weld)), fx - R, y - R, R * 2, R * 2);
        g.globalAlpha = 1;
      }
      const lg = g.createLinearGradient(x0, 0, x1, 0);
      lg.addColorStop(0, AM.rgba(AM.dye.weld, 0));
      lg.addColorStop(0.15, AM.rgba('#fff3cf', 0.75 * env));
      lg.addColorStop(0.85, AM.rgba('#fff3cf', 0.75 * env));
      lg.addColorStop(1, AM.rgba(AM.dye.weld, 0));
      g.fillStyle = lg;
      g.fillRect(x0, y - 0.6, x1 - x0, 1.2);
      g.restore();
    }

    // ------------------------------------------------------------ frame
    function drawSentence(sent, a, isCur) {
      if (!sent || a <= 0.003) return;
      const L = lay(sent);
      if (S.v.arcs > 0.01) {
        g.save();
        g.globalAlpha = a * S.v.arcs;
        g.drawImage(arcCache(L), 0, 0, S.w, S.h);
        g.restore();
      }
      drawWarps(sent, L, a);
      drawKnots(sent, L, a * clamp(0.35 + S.v.arcs * 0.8));
      drawCompose(sent, L, a * S.v.comp);
      if (isCur) drawParticles(sent, L, a);
      drawBeads(sent, L, a);
      drawReadout(sent, L, a * S.v.readout);
      drawOut(sent, L, a * S.v.out);
      drawTokens(sent, L, a);
    }
    function draw() {
      if (!S.geo) return;
      cv.clear();
      if (S.build >= 1) g.drawImage(bgCache(), 0, 0, S.w, S.h);
      else { drawBands(g, S.geo, buildE); drawGutter(g, S.geo); }
      drawGauge(S.v.gauge);
      const tr = E.inOut(clamp(S.tr));
      if (S.prev && tr < 1) drawSentence(S.prev, 1 - tr, false);
      drawSentence(S.cur, S.prev ? tr : 1, true);
      drawSweep();
      drawBlooms();
    }

    // ------------------------------------------------------------ state changes
    function startSweep() {
      if (AM.reducedMotion) { S.sweep = 1; S.sweeping = false; return; }
      S.sweep = 0;
      S.sweeping = true;
      S.bloomed = new Set();
    }
    function show(sent) {
      if (sent === S.cur) return;
      S.prev = S.cur;
      S.cur = sent;
      S.tr = S.prev && !AM.reducedMotion ? 0 : 1;
      if (S.geo) resetFlow(sent);
      S.arcParts = [];
    }
    function describe() {
      const s = S.cur;
      if (!s) return;
      const t = clamp(S.focus, 0, s.T - 1);
      const tr = trace(s, t).map((e) => `${LEVEL_LONG[e.l]} ${e.tok} ${pct(e.p)}`).join(', ');
      const label = `A tower of three transformer layers reading “${s.tokens.join(' ')}”. Each word is a vertical thread; each layer band shows its attention threads and MLP edits. Focused word “${s.tokens[t]}”: the logit lens reads ${tr}.${S.muted ? ' Head L0H0 is silenced at this word.' : ''}`;
      canvas.setAttribute('aria-label', label);
    }
    /** Redraw at once after a change: the first frame is complete before the loop ticks,
        and under reduced motion (loop at ~2 fps) clicks still answer immediately. */
    const kick = () => { if (S.geo) draw(); };
    const api = {
      setSlot(slot, opts = {}) {
        const changed = slot !== S.slot;
        S.slot = slot;
        if (changed) { S.muted = false; S.focus = slot.focus; }
        show(S.muted && slot.mutedSent ? slot.mutedSent : slot.sent);
        if (changed && S.tg.beads > 0.5 && opts.sweep !== false) startSweep();
        describe();
        kick();
      },
      setMode(mode) {
        const m = MODES[mode] || MODES.stack;
        const wasBeads = S.tg.beads > 0.5;
        S.mode = mode;
        Object.assign(S.tg, m);
        if (m.beads > 0.5 && !wasBeads) startSweep();
        if (mode === 'compose' && S.slot) S.focus = S.slot.focus;
        if (AM.reducedMotion) Object.assign(S.v, m);
        kick();
      },
      setMuted(b) {
        if (!S.slot) return;
        S.muted = !!(b && S.slot.mutedSent);
        show(S.muted ? S.slot.mutedSent : S.slot.sent);
        describe();
        kick();
      },
      setFocus(t) {
        if (!S.cur) return;
        S.focus = clamp(t, 0, S.cur.T - 1);
        describe();
        const s = S.cur, tt = S.focus;
        srLive.textContent = `“${s.tokens[tt]}”: ` + trace(s, tt).map((e) => `${LEVEL_NAMES[e.l]} ${e.tok} ${pct(e.p)}`).join(', ');
        if (api.onFocus) api.onFocus(S.focus);
        kick();
      },
      get focus() { return S.focus; },
      get slot() { return S.slot; },
      onFocus: null,
    };

    cv.onResize((w, h) => {
      S.w = w; S.h = h;
      S.geo = geo();
      S.bg = null;
      // particles and blooms carry old coordinates; drop them rather than draw them in the wrong place
      S.arcParts = [];
      S.blooms = [];
      if (S.cur) resetFlow(S.cur);
      draw();
    });

    // pointer + keyboard
    const colAt = (ev) => {
      if (!S.cur || !S.geo) return -1;
      const p = cv.pointer(ev), Gm = S.geo, L = lay(S.cur);
      if (p.x < Gm.xa - 10 || p.x > Gm.xb + 10 || p.y < Gm.ys[3] - 40) return -1;
      return clamp(Math.floor((p.x - Gm.xa) / L.colW), 0, S.cur.T - 1);
    };
    const canFocus = () => S.mode !== 'compose';
    canvas.addEventListener('pointermove', (ev) => { if (ev.pointerType === 'mouse') S.hover = canFocus() ? colAt(ev) : -1; });
    canvas.addEventListener('pointerleave', () => { S.hover = -1; });
    canvas.addEventListener('pointerdown', (ev) => {
      if (!canFocus()) return;
      const c = colAt(ev);
      if (c >= 0) api.setFocus(c);
    });
    canvas.addEventListener('keydown', (ev) => {
      if (!S.cur || !canFocus()) return;
      let t = S.focus;
      if (ev.key === 'ArrowRight') t++;
      else if (ev.key === 'ArrowLeft') t--;
      else if (ev.key === 'Home') t = 0;
      else if (ev.key === 'End') t = S.cur.T - 1;
      else return;
      ev.preventDefault();
      api.setFocus(t);
    });

    const vis = watch(frame, () => {
      if (AM.reducedMotion) return;
      S.build = 0;
      S.building = true;
    });

    ctx.loop((time, dt) => {
      if (!S.geo || !vis.on) return;
      S.time = time;
      const rate = AM.reducedMotion ? 50 : 5;
      for (const k of Object.keys(S.v)) S.v[k] = approach(S.v[k], S.tg[k], dt, rate);
      if (S.building) { S.build += dt / 1.5; if (S.build >= 1) { S.build = 1; S.building = false; } }
      if (S.tr < 1) { S.tr = Math.min(1, S.tr + dt / 0.55); if (S.tr >= 1) S.prev = null; }
      if (S.sweeping) {
        S.sweep += dt / 1.9;
        const s = S.cur;
        if (s && S.bloomed) {
          for (let t = 0; t < s.T; t++) {
            const lv = s.lock[t];
            if (!S.bloomed.has(t) && beadVis(lv) > 0.55) {
              S.bloomed.add(t);
              const L = lay(s);
              S.blooms.push({ x: L.xs[t], y: S.geo.ys[lv], t0: time, big: t === S.focus });
            }
          }
        }
        if (S.sweep >= 1) { S.sweep = 1; S.sweeping = false; }
      }
      if (S.cur) stepParticles(dt, S.cur, lay(S.cur));
      draw();
    });

    return api;
  }

  // ================================================================== 2. the lens table
  function levenshtein(a, b) {
    const m = a.length, n = b.length;
    if (!m) return n;
    if (!n) return m;
    let prev = Array.from({ length: n + 1 }, (_, j) => j);
    for (let i = 1; i <= m; i++) {
      const cur = [i];
      for (let j = 1; j <= n; j++) cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
      prev = cur;
    }
    return prev[n];
  }
  const NEAR = {
    kitten: 'cat', kitty: 'cat', puppy: 'dog', doggy: 'dog', man: 'king', woman: 'queen', lady: 'queen', mother: 'queen', mom: 'queen',
    father: 'king', dad: 'king', son: 'boy', daughter: 'girl', kid: 'boy', kids: 'children', boys: 'children', girls: 'children', people: 'children',
    house: 'castle', home: 'castle', palace: 'castle', shop: 'market', store: 'market', lake: 'river', sea: 'river', forest: 'garden', field: 'park',
    bag: 'box', mug: 'cup', glass: 'cup', purple: 'pink', orange: 'yellow', black: 'blue', brown: 'red', grey: 'white', gray: 'white',
    took: 'found', picked: 'found', saw: 'met', left: 'lost', went: 'walked', ran: 'walked', liked: 'loved', angry: 'sad', glad: 'happy', big: 'heavy',
    an: 'a', this: 'the', that: 'the', hers: 'her', him: 'he', them: 'they', we: 'they', i: 'he', you: 'she',
  };
  function suggest(word, vocab) {
    if (NEAR[word] && vocab.includes(NEAR[word])) return NEAR[word];
    let best = null, bd = Infinity;
    for (const v of vocab) {
      if (v === '<pad>' || v === '<unk>') continue;
      const d = levenshtein(word, v);
      if (d < bd) { bd = d; best = v; }
    }
    return bd <= Math.max(2, Math.floor(word.length / 3)) ? best : null;
  }

  function buildTable(ctx, panel, m) {
    const el = AM.el, ui = AM.ui;
    const net = getNet();
    const S = { data: null, split: false, sel: null, chip: 0 };

    const chips = TABLE_SET.map((p, i) => el('button', {
      type: 'button', class: 'st-chip', id: `st-lt-chip-${i}`, 'aria-pressed': String(i === 0), title: p.text,
      onclick: () => { input.value = p.text; load(p.text, i); },
    }, p.label));
    const input = el('input', {
      class: 'text-input', id: 'st-lt-input', type: 'text', autocomplete: 'off', autocapitalize: 'off', spellcheck: 'false',
      maxlength: '240', 'aria-label': 'Your own sentence, in the model’s toy English', placeholder: 'type a sentence in toy English…',
    });
    const go = ui.button({ id: 'st-lt-go', label: 'Read', kind: 'primary', onClick: () => load(input.value, -1) });
    input.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); load(input.value, -1); } });
    const unk = el('p', { class: 'st-unk', 'aria-live': 'polite' });
    const vgrid = el('div', { class: 'st-vgrid' });
    const known = new Set(m.vocab);
    for (const [name, words] of VOCAB_GROUPS) {
      const ws = words.split(' ').filter((w) => known.has(w));
      if (!ws.length) continue;
      vgrid.appendChild(el('div', { class: 'st-vgroup' }, el('span', {}, name),
        ws.map((w) => el('button', { type: 'button', title: `add “${w}”`, onclick: () => { input.value = (input.value.trim() + ' ' + w).trim(); load(input.value, -1); } }, w))));
    }
    const nWords = m.vocab.filter((w) => !['<pad>', '<unk>', '.', ','].includes(w)).length;
    const vocab = el('details', { class: 'st-vocab' }, el('summary', {}, `The ${nWords} words it knows`), vgrid);
    const split = ui.toggle({ id: 'st-lt-split', label: 'Read between attention and MLP', onChange: (b) => {
      // keep the same depth selected: main column i sits at 2i in the split view
      if (S.sel && S.sel.c >= 0) S.sel.c = b ? S.sel.c * 2 : Math.ceil(S.sel.c / 2);
      S.split = b;
      render();
    } });
    if (!net) { split.input.disabled = true; split.el.title = 'Unavailable: the local forward check did not pass'; }
    const grid = el('div', { class: 'st-grid', role: 'group', 'aria-label': 'Logit lens table: rows are positions, columns are depths. Arrow keys move between cells.' });
    const gridWrap = el('div', { class: 'st-grid-wrap' }, grid);
    const detail = el('div', { class: 'st-detail', 'aria-live': 'polite' });

    panel.appendChild(el('div', { class: 'st-lt-ctl' },
      el('div', { class: 'st-chips', role: 'group', 'aria-label': 'Example sentences' }, chips),
      el('div', { class: 'st-lt-row' }, input, go),
      unk,
      el('div', { class: 'controls' }, split.el, vocab)));
    panel.appendChild(el('div', { class: 'st-legend', 'aria-hidden': 'true' },
      el('span', {}, el('i', { style: { background: 'color-mix(in srgb, var(--weld) 30%, var(--ink-2))', boxShadow: 'inset 0 -2px 0 var(--weld)' } }), 'same word as the final output'),
      el('span', {}, el('i', { style: { background: 'color-mix(in srgb, var(--woad) 30%, var(--ink-2))', boxShadow: 'inset 0 -2px 0 var(--woad)' } }), 'a different word'),
      el('span', {}, 'stronger colour, longer bar = higher probability')));
    panel.appendChild(el('div', { class: 'st-lt-body' }, gridWrap, detail));

    const stages = () => {
      const base = [
        { lab: 'x₀', sub: 'embed', long: 'x₀ · the embedding', get: (d, t) => d.lens[0][t], main: true },
        { lab: 'x₁', sub: 'after L0', long: 'x₁ · after layer 0', get: (d, t) => d.lens[1][t], main: true },
        { lab: 'x₂', sub: 'after L1', long: 'x₂ · after layer 1', get: (d, t) => d.lens[2][t], main: true },
        { lab: 'x₃', sub: 'output', long: 'x₃ · the output', get: (d, t) => d.lens[3][t], main: true, out: true },
      ];
      if (!S.split || !net) return base;
      const mid = (l) => ({ lab: `+attn ${l}`, sub: 'no mlp yet', long: `layer ${l} attention added, before its MLP`, get: (d, t) => d.midLens(l, t), main: false });
      return [base[0], mid(0), base[1], mid(1), base[2], mid(2), base[3]];
    };

    function load(text, chipIdx) {
      S.chip = chipIdx;
      chips.forEach((c, i) => c.setAttribute('aria-pressed', String(i === chipIdx)));
      const raw = String(text || '').trim();
      if (!raw) { S.data = null; unk.innerHTML = 'Type a few words, or pick an example above.'; render(); return; }
      const enc = m.encode(raw);
      const unknown = enc.unknown;
      // unknown words are left out of the input (the model has no vector for them)
      const ids = enc.ids.filter((id, i) => enc.tokens[i] !== '<unk>');
      const truncated = ids.length > m.config.n_ctx;
      const use = ids.slice(0, m.config.n_ctx);
      const parts = [];
      if (unknown.length) {
        const uniq = [...new Set(unknown)];
        parts.push(`Not in the model’s vocabulary, so left out: ${uniq.map((w) => `<span class="bad">${esc(w)}</span>`).join(', ')}.`);
        const sugg = uniq.map((w) => [w, suggest(w, m.vocab)]).filter(([, s]) => s);
        if (sugg.length) parts.push(`Swap in: ${sugg.map(([w, s]) => `<button type="button" data-from="${esc(w)}" data-to="${esc(s)}" aria-label="replace ${esc(w)} with ${esc(s)}">${esc(w)} → ${esc(s)}</button>`).join(' ')}`);
        else parts.push('Open the word list below to see what it knows.');
      }
      if (truncated) parts.push(`Only the first ${m.config.n_ctx} tokens fit the model’s context.`);
      unk.innerHTML = parts.join(' ');
      unk.querySelectorAll('button[data-from]').forEach((b) => b.addEventListener('click', () => {
        const from = b.getAttribute('data-from'), to = b.getAttribute('data-to');
        const re = new RegExp(`(^|\\s)${from.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}(?=\\s|$|[.,])`, 'gi');
        input.value = input.value.replace(re, (mm, pre) => pre + to);
        load(input.value, -1);
      }));
      if (!use.length) { S.data = null; render(); return; }
      const r = m.run(use, { capture: true });
      const d = analyse(m, use, r);
      const memo = new Map();
      d.midLens = (l, t) => {
        const k = l * 64 + t;
        if (!memo.has(k)) memo.set(k, net.lens(r.residMid[l][t]));
        return memo.get(k);
      };
      S.data = d;
      S.sel = { t: d.T - 1, c: -1 };
      render();
    }

    function cellStyle(p, fin) {
      const k = Math.round(4 + 20 * Math.pow(p, 0.9));
      const dye = fin ? 'var(--weld)' : 'var(--woad)';
      const wc = fin ? 'color-mix(in srgb, var(--weld) 82%, var(--linen))' : 'color-mix(in srgb, var(--woad) 50%, var(--linen))';
      return `background: color-mix(in srgb, ${dye} ${k}%, var(--ink-2)); --p: ${p.toFixed(3)}; --bar: ${dye}; --wc: ${wc}; --wa: ${(0.62 + 0.38 * p).toFixed(2)};`;
    }

    function render() {
      grid.innerHTML = '';
      const d = S.data;
      if (!d) { grid.appendChild(el('p', { class: 'st-msg' }, 'Nothing to read yet.')); detail.innerHTML = ''; return; }
      const st = stages();
      const phone = isStacked() && window.innerWidth < 560;
      const showNext = !phone;
      grid.style.gridTemplateColumns = `minmax(${phone ? 52 : 74}px, auto) repeat(${st.length}, minmax(${S.split ? 66 : (phone ? 54 : 74)}px, 1fr))${showNext ? ' minmax(72px, auto)' : ''}`;
      grid.appendChild(el('div', { class: 'st-gh' }, el('span', {}, 'after word')));
      st.forEach((s) => grid.appendChild(el('div', { class: 'st-gh' + (s.main ? '' : ' mid'), 'aria-hidden': 'true' }, el('b', { class: s.out ? 'out' : '' }, s.lab), el('span', {}, s.sub))));
      if (showNext) grid.appendChild(el('div', { class: 'st-gh', 'aria-hidden': 'true' }, el('span', {}, 'next in text')));
      const fin = (t) => d.top[d.NL][t][0].id;
      for (let t = 0; t < d.T; t++) {
        grid.appendChild(el('div', { class: 'st-rh', 'aria-hidden': 'true' }, el('span', { class: 'w' }, d.tokens[t]), el('span', { class: 'i' }, String(t))));
        st.forEach((s, c) => {
          const row = s.get(d, t), tk = m.topk(row, 1)[0], isFin = tk.id === fin(t);
          const cell = el('button', {
            type: 'button', class: 'st-cell' + (isFin ? ' is-fin' : ''), id: `st-lt-c-${t}-${c}`, tabindex: '-1',
            style: cellStyle(tk.p, isFin), 'aria-label': `after “${d.tokens[t]}”, ${s.long}: ${tk.token} ${pct(tk.p)}`,
          }, el('span', { class: 'w' }, tk.token), el('span', { class: 'p' }, pct(tk.p)));
          const pick = () => select(t, c);
          cell.addEventListener('pointerenter', (e) => { if (e.pointerType === 'mouse') pick(); });
          cell.addEventListener('focus', pick);
          cell.addEventListener('click', pick);
          cell.addEventListener('keydown', (e) => {
            const nav = { ArrowRight: [0, 1], ArrowLeft: [0, -1], ArrowDown: [1, 0], ArrowUp: [-1, 0] }[e.key];
            if (!nav) return;
            e.preventDefault();
            const nt = clamp(t + nav[0], 0, d.T - 1), nc = clamp(c + nav[1], 0, st.length - 1);
            const nx = grid.querySelector(`#st-lt-c-${nt}-${nc}`);
            if (nx) nx.focus();
          });
          grid.appendChild(cell);
        });
        if (showNext) {
          const nxt = d.tokens[t + 1];
          grid.appendChild(el('div', { class: 'st-next' }, nxt != null
            ? [el('span', {}, nxt), el('span', { class: fin(t) === d.ids[t + 1] ? 'ok' : 'no', title: fin(t) === d.ids[t + 1] ? 'the final output’s top guess matches' : 'the final output guessed something else' }, fin(t) === d.ids[t + 1] ? '✓' : '·')]
            : el('span', { class: 'no' }, '?  the future')));
        }
      }
      const c0 = S.sel && S.sel.c >= 0 && S.sel.c < st.length ? S.sel.c : st.length - 1;
      select(S.sel ? Math.min(S.sel.t, d.T - 1) : d.T - 1, c0);
      scrollHint();
    }
    /** A soft fade on the right edge while more columns sit off to the side (phones, split view). */
    function scrollHint() {
      const over = gridWrap.scrollWidth > gridWrap.clientWidth + 2;
      gridWrap.classList.toggle('is-scroll', over);
      gridWrap.classList.toggle('at-end', over && gridWrap.scrollLeft + gridWrap.clientWidth >= gridWrap.scrollWidth - 2);
    }
    gridWrap.addEventListener('scroll', scrollHint, { passive: true });

    function select(t, c) {
      const d = S.data;
      if (!d) return;
      S.sel = { t, c };
      grid.querySelectorAll('.st-cell.is-sel').forEach((x) => { x.classList.remove('is-sel'); x.tabIndex = -1; });
      const cell = grid.querySelector(`#st-lt-c-${t}-${c}`);
      if (cell) { cell.classList.add('is-sel'); cell.tabIndex = 0; }
      const st = stages(), s = st[c];
      const top = m.topk(s.get(d, t), 5), finId = d.top[d.NL][t][0].id;
      detail.innerHTML = '';
      detail.appendChild(el('div', { class: 'st-dh', html: `after ${esc(q(d.tokens[t]))} · <b>${esc(s.long)}</b>` }));
      detail.appendChild(el('div', { class: 'st-dw' }, top[0].token, top[0].token === '.' ? el('small', {}, 'full stop') : top[0].token === ',' ? el('small', {}, 'comma') : null));
      detail.appendChild(el('div', { class: 'st-drows' }, top.map((e) => el('div', { class: 'st-drow' },
        el('span', { class: 'w', style: { color: e.id === finId ? 'var(--weld)' : 'var(--linen)' } }, e.token),
        el('span', { class: 'bar' }, el('i', { style: { width: (e.p * 100).toFixed(1) + '%', background: e.id === finId ? 'var(--weld)' : 'var(--woad)' } })),
        el('span', { class: 'p' }, pct(e.p))))));
      detail.appendChild(el('div', { class: 'st-dh st-dh-trace' }, 'this position, up the stack'));
      detail.appendChild(el('div', { class: 'st-trace st-dtrace', html: traceHTML(d, t) }));
    }

    load(TABLE_SET[0].text, 0);
    input.value = TABLE_SET[0].text;
    let lastPhone = isStacked() && window.innerWidth < 560;
    window.addEventListener('resize', () => {
      const ph = isStacked() && window.innerWidth < 560;
      if (ph !== lastPhone) { lastPhone = ph; render(); } else scrollHint();
    });
  }

  // ================================================================== 3. depth profiles
  const TYPES = [
    { key: 'pronoun', label: 'Pronoun', dye: 'woad', gen: GEN.pronoun, n: 10, eg: 'the queen opened the door because → she' },
    { key: 'agreement', label: 'Agreement', dye: 'madder', gen: GEN.agreement, n: 10, eg: 'the key near the old doors → is' },
    { key: 'binding', label: 'Colour binding', dye: 'weld', gen: GEN.binding, n: 10, eg: 'the red ball and the blue cup . the ball is → red' },
    { key: 'copy', label: 'Copy a name', dye: 'verdigris', gen: GEN.copy, n: 10, eg: 'alice gave bob a cup . bob thanked → alice' },
    { key: 'fact', label: 'Memorised facts', dye: 'cochineal', fixed: FACTS, n: FACTS.length, eg: 'the capital of japan is → tokyo' },
  ];

  /** Monotone cubic (Fritsch–Carlson) through points, as Bézier segments (no overshoot past 0 or 1). */
  function monoPath(g, pts) {
    const n = pts.length;
    const dx = [], dy = [], m = [];
    for (let i = 0; i < n - 1; i++) { dx.push(pts[i + 1].x - pts[i].x); dy.push((pts[i + 1].y - pts[i].y) / dx[i]); }
    m.push(dy[0]);
    for (let i = 1; i < n - 1; i++) m.push(dy[i - 1] * dy[i] <= 0 ? 0 : (dy[i - 1] + dy[i]) / 2);
    m.push(dy[n - 2]);
    for (let i = 0; i < n - 1; i++) {
      if (dy[i] === 0) { m[i] = 0; m[i + 1] = 0; continue; }
      const a = m[i] / dy[i], b = m[i + 1] / dy[i], s = a * a + b * b;
      if (s > 9) { const k = 3 / Math.sqrt(s); m[i] = k * a * dy[i]; m[i + 1] = k * b * dy[i]; }
    }
    g.moveTo(pts[0].x, pts[0].y);
    for (let i = 0; i < n - 1; i++) {
      const h = dx[i] / 3;
      g.bezierCurveTo(pts[i].x + h, pts[i].y + m[i] * h, pts[i + 1].x - h, pts[i + 1].y - m[i + 1] * h, pts[i + 1].x, pts[i + 1].y);
    }
  }

  function buildProfiles(ctx, parts, m) {
    const el = AM.el;
    const S = { seed: 8, queue: [], data: TYPES.map(() => []), hi: -1, pin: -1, hover: null, dirty: true, t: 0, done: 0, total: 0 };
    const cv = ctx.canvas(parts.canvasHost, {
      height: (w) => (w < 480 ? Math.round(w * 0.86) : Math.round(clamp(w * 0.62, 300, 400))),
      label: 'Depth profiles: probability of the correct next word at each depth, for five kinds of dependency',
    });
    const { canvas, g } = cv;
    canvas.style.touchAction = 'pan-y';
    const vis = watch(parts.canvasHost);

    const typeBtns = TYPES.map((ty, i) => {
      const b = el('button', { type: 'button', class: 'st-type', id: `st-dp-type-${i}`, 'aria-pressed': 'false' },
        el('i', { 'aria-hidden': 'true' }), ty.label, el('em', { class: 'st-settle' }, '…'));
      b.style.setProperty('--c', AM.dye[ty.dye]);
      b.addEventListener('click', () => { S.pin = S.pin === i ? -1 : i; S.hi = S.pin; sync(); });
      b.addEventListener('pointerenter', (e) => { if (e.pointerType === 'mouse') { S.hi = i; sync(); } });
      b.addEventListener('pointerleave', (e) => { if (e.pointerType === 'mouse') { S.hi = S.pin; sync(); } });
      b.addEventListener('focus', () => { S.hi = i; sync(); });
      b.addEventListener('blur', () => { S.hi = S.pin; sync(); });
      return b;
    });
    parts.typesHost.appendChild(el('div', { class: 'st-types', role: 'group', 'aria-label': 'Kinds of dependency' }, typeBtns));
    // the progress counter is visual only; one polite announcement when a batch is done
    const status = el('span', { class: 'st-status', 'aria-hidden': 'true' });
    const announce = el('span', { class: 'sr-only', role: 'status' });
    const btn = AM.ui.button({ id: 'st-dp-new', label: 'New sentences', onClick: () => { S.seed += 1; enqueue(false); } });
    parts.ctlHost.appendChild(el('div', { class: 'st-dp-ctl' }, btn, status, announce));
    const read = parts.readHost;
    // rewrite the readout only when its words change (it is a live region)
    let readHTML = '';
    const setRead = (html) => { if (html !== readHTML) { readHTML = html; read.innerHTML = html; } };

    function enqueue(all) {
      // a redraw keeps any fact jobs still waiting (pressing "New sentences" early must not drop facts)
      S.queue = all ? [] : S.queue.filter((job) => TYPES[job.i].fixed);
      TYPES.forEach((ty, i) => {
        if (ty.fixed) { if (all) { S.data[i] = []; ty.fixed.forEach((ex) => S.queue.push({ i, ex })); } return; }
        S.data[i] = [];
        const r = M.rng(S.seed * 7919 + i * 104729);
        for (let k = 0; k < ty.n; k++) S.queue.push({ i, ex: ty.gen(r) });
      });
      S.total = S.queue.length;
      S.done = 0;
      S.dirty = true;
      status.textContent = 'reading…';
      announce.textContent = '';
      read.setAttribute('aria-busy', 'true');
    }
    function compute(job) {
      const ids = m.encode(job.ex.words.join(' ')).ids;
      const tgt = m.tokenId(job.ex.target);
      const r = m.run(ids, { capture: true });
      const t = ids.length - 1;
      const ps = r.lens.map((row) => row[t][tgt]);
      S.data[job.i].push({ words: job.ex.words, target: job.ex.target, ps, born: S.t });
    }
    const means = () => S.data.map((arr) => (arr.length ? [0, 1, 2, 3].map((l) => arr.reduce((s, e) => s + e.ps[l], 0) / arr.length) : null));
    const settle = (mn) => { if (!mn) return -1; for (let l = 1; l <= 3; l++) if (mn[l] >= 0.9) return l; return -1; };

    function sync() {
      typeBtns.forEach((b, i) => b.setAttribute('aria-pressed', String(i === S.hi)));
      const mns = means();
      typeBtns.forEach((b, i) => {
        const s = settle(mns[i]);
        b.querySelector('.st-settle').textContent = mns[i] ? (s > 0 ? `· layer ${s - 1}` : '· not yet') : '…';
      });
      const i = S.hover ? S.hover.i : S.hi;
      if (i >= 0 && mns[i]) {
        const ty = TYPES[i];
        const ex = S.hover ? S.data[i][S.hover.k] : null;
        const vals = (ex ? ex.ps : mns[i]).map((v, l) => `x${sub(l)} ${f2(v)}`).join(' → ');
        const s = settle(mns[i]);
        setRead(ex
          ? `<b>${esc(ty.label)}</b>, one sentence: <span class="s">${esc(ex.words.join(' '))} → ${esc(ex.target)}</span><br>P(${esc(ex.target)}): ${vals}`
          : `<b>${esc(ty.label)}</b>, mean of ${S.data[i].length} sentences: ${vals}. ${s > 0 ? `Settled (mean ≥ 0.9) after layer ${s - 1}.` : 'Not settled below the output.'} <br>e.g. <span class="s">${esc(ty.eg)}</span>`);
      } else {
        const order = TYPES.map((ty, k) => ({ ty, s: settle(mns[k]) })).filter((x) => x.s > 0);
        setRead(order.length
          ? `Mean P(correct) settles above 0.9 after ${order.map((x) => `<b>${esc(x.ty.label.toLowerCase())}</b> layer ${x.s - 1}`).join(', ')}. Hover or tap a kind, or a single thread, for its numbers.`
          : 'Reading sentences through the model…');
      }
      S.dirty = true;
      const desc = TYPES.map((ty, k) => (mns[k] ? `${ty.label}: ${mns[k].map((v) => f2(v)).join(', ')}` : '')).filter(Boolean).join('; ');
      canvas.setAttribute('aria-label', `Depth profiles. Mean probability of the correct next word at the embedding, after layer 0, after layer 1 and at the output. ${desc}.`);
    }

    // chart geometry
    let C = null;
    cv.onResize((w, h) => {
      const phone = w < 480;
      C = { w, h, phone, l: phone ? 38 : 44, r: phone ? 12 : 22, t: phone ? 28 : 32, b: phone ? 46 : 50 };
      C.px = (l) => C.l + (l / 3) * (C.w - C.l - C.r);
      C.py = (v) => C.t + (1 - v) * (C.h - C.t - C.b);
      S.dirty = true;
      drawChart();
    });

    function drawChart() {
      if (!C) return;
      cv.clear();
      const { w, h, phone } = C;
      // layer lanes
      for (let l = 0; l < 3; l++) {
        const x0 = C.px(l), x1 = C.px(l + 1);
        g.fillStyle = AM.rgba(AM.col.ink3, l % 2 ? 0.25 : 0.5);
        g.fillRect(x0, C.t - 6, x1 - x0, C.py(0) - C.t + 6);
        D.text(g, 'LAYER ' + l, (x0 + x1) / 2, C.t - 12, { size: phone ? 8 : 9, role: 'mono', color: AM.dye.weld, align: 'center', letterSpacing: '0.12em', alpha: 0.85 });
      }
      // grid
      g.save();
      g.strokeStyle = AM.rgba(AM.col.linen, 0.08);
      g.lineWidth = 1;
      [0, 0.5, 1].forEach((v) => {
        g.beginPath(); g.moveTo(C.l - 4, C.py(v) + 0.5); g.lineTo(w - C.r, C.py(v) + 0.5); g.stroke();
        D.text(g, v === 0.5 ? '0.5' : String(v), C.l - 8, C.py(v) + 3.5, { size: phone ? 8.5 : 9, role: 'mono', color: AM.col.mist, align: 'right' });
      });
      for (let l = 0; l <= 3; l++) {
        const x = C.px(l);
        g.setLineDash([2, 4]);
        g.beginPath(); g.moveTo(x + 0.5, C.t - 6); g.lineTo(x + 0.5, C.py(0) + 6); g.stroke();
        g.setLineDash([]);
        D.text(g, 'x' + sub(l), x, C.py(0) + 22, { size: phone ? 13 : 15, role: 'display', italic: true, weight: 400, color: l === 3 ? AM.dye.weld : AM.col.linen, align: 'center' });
        D.text(g, LEVEL_NAMES[l].toUpperCase(), x, C.py(0) + 36, { size: phone ? 7 : 8, role: 'mono', color: AM.col.mist, align: l === 0 ? 'left' : l === 3 ? 'right' : 'center', letterSpacing: '0.06em' });
      }
      g.restore();
      g.save();
      g.translate(phone ? 7 : 9, C.py(0.5));
      g.rotate(-Math.PI / 2);
      D.text(g, 'P(CORRECT)', 0, 3, { size: 8, role: 'mono', color: AM.col.mist, letterSpacing: '0.08em', align: 'center' });
      g.restore();

      const mns = means();
      const focus = S.hover ? S.hover.i : S.hi;
      g.save();
      g.lineCap = 'round';
      g.lineJoin = 'round';
      // single sentences
      S.data.forEach((arr, i) => {
        const col = AM.dye[TYPES[i].dye];
        const on = focus < 0 || focus === i;
        arr.forEach((ex, k) => {
          const grow = AM.reducedMotion ? 1 : E.out(clamp((S.t - ex.born) / 0.7));
          const isH = S.hover && S.hover.i === i && S.hover.k === k;
          g.save();
          g.beginPath(); g.rect(0, 0, C.l + grow * (C.w - C.l - C.r) + 2, h); g.clip();
          g.beginPath();
          monoPath(g, ex.ps.map((v, l) => ({ x: C.px(l), y: C.py(v) })));
          g.strokeStyle = AM.rgba(col, isH ? 0.95 : on ? (focus === i ? 0.32 : 0.15) : 0.04);
          g.lineWidth = isH ? 2 : 1;
          g.stroke();
          g.restore();
        });
      });
      // means
      const order = [...mns.keys()].sort((a, b) => (a === focus) - (b === focus));
      for (const i of order) {
        const mn = mns[i];
        if (!mn) continue;
        const col = AM.dye[TYPES[i].dye];
        const on = focus < 0 || focus === i;
        const pts = mn.map((v, l) => ({ x: C.px(l), y: C.py(v) }));
        g.beginPath(); monoPath(g, pts);
        g.strokeStyle = AM.rgba(col, on ? 0.16 : 0.04); g.lineWidth = focus === i ? 12 : 8; g.stroke();
        g.beginPath(); monoPath(g, pts);
        g.strokeStyle = AM.rgba(col, on ? 1 : 0.18); g.lineWidth = focus === i ? 3 : 2.2; g.stroke();
        pts.forEach((p, l) => {
          g.globalAlpha = on ? 1 : 0.2;
          const s = focus === i ? 18 : 13;
          g.drawImage(sprite(RGB(col)), p.x - s / 2, p.y - s / 2, s, s);
          g.globalAlpha = 1;
          if (focus === i) {
            // curves mostly rise to the right: low values get their label up and to the left
            // (away from the climb), high ones below the flat top, with a dark halo either way
            const low = mn[l] < 0.85, left = l === 3 || (low && l > 0);
            g.save();
            g.shadowColor = AM.col.ink;
            g.shadowBlur = 5;
            D.text(g, f2(mn[l]), p.x + (left ? -7 : 7), p.y + (low ? -9 : 16), { size: 10, role: 'mono', color: col, align: left ? 'right' : 'left' });
            g.restore();
          }
        });
      }
      g.restore();
      if (!mns.some(Boolean)) D.text(g, 'reading sentences…', w / 2, h / 2, { size: 12, role: 'mono', color: AM.col.mist, align: 'center' });
      S.dirty = false;
    }

    // hover a thread: nearest single-sentence thread, else nearest mean
    const nearest = (p) => {
      if (!C) return null;
      const lf = clamp((p.x - C.l) / (C.w - C.l - C.r) * 3, 0, 3);
      const l0 = Math.min(2, Math.floor(lf)), u = lf - l0;
      const yAt = (ps) => C.py(lerp(ps[l0], ps[l0 + 1], u));
      let best = null, bd = 9;
      S.data.forEach((arr, i) => arr.forEach((ex, k) => { const d = Math.abs(yAt(ex.ps) - p.y); if (d < bd) { bd = d; best = { i, k }; } }));
      return best;
    };
    canvas.addEventListener('pointermove', (ev) => { const n = nearest(cv.pointer(ev)); const changed = JSON.stringify(n) !== JSON.stringify(S.hover); S.hover = n; if (changed) sync(); });
    canvas.addEventListener('pointerleave', (ev) => { if (ev.pointerType === 'mouse' && S.hover) { S.hover = null; sync(); } });
    canvas.addEventListener('pointerdown', (ev) => { S.hover = nearest(cv.pointer(ev)); sync(); });

    enqueue(true);
    sync();
    ctx.loop((time) => {
      S.t = time;
      if (S.queue.length) {
        const t0 = performance.now(), budget = AM.reducedMotion ? 80 : 7;
        while (S.queue.length && performance.now() - t0 < budget) {
          const job = S.queue.shift();
          job.ex.words.length && compute(job);
          S.done++;
        }
        const finished = !S.queue.length;
        status.textContent = finished ? `${S.data.reduce((s, a) => s + a.length, 0)} sentences, read live` : `reading… ${S.done}/${S.total}`;
        sync();
        if (finished) { read.setAttribute('aria-busy', 'false'); announce.textContent = status.textContent; }
      }
      const anim = S.data.some((arr) => arr.some((ex) => S.t - ex.born < 0.8));
      if (vis.on && (S.dirty || anim)) drawChart();
    });
  }

  // ================================================================== mount
  AM.chapter({
    id: ID,
    num: 8,
    kicker: 'Depth & the logit lens',
    title: 'The Tower of <em>Layers</em>',
    lede: 'A transformer is one block stacked again and again. Each layer refines every word’s vector, and a lens slipped between the layers lets us watch a guess sharpen into an answer.',
    where: 'stack',
    mount(root, ctx) {
      ctx.header();
      const el = ctx.el, ui = AM.ui;
      const body = el('div', { class: 'ch-body' });
      root.appendChild(body);
      const m = getModel();
      if (!m || !m.run) {
        body.appendChild(el('p', { class: 'caption' }, 'The live model did not load, so this chapter cannot run. Reload the page to try again.'));
        return;
      }

      // ---------------------------------------------------------------- live numbers for the prose
      const slots = TOWER_SET.map((p) => Object.assign({}, p, { sent: Object.assign(analyseText(m, p.text), { focus: p.focus }) }));
      const hero = slots[0];
      const hs = hero.sent, hq = hero.focus;
      const a0 = hs.attn[0][0][hq], a1 = hs.attn[1][3][hq];
      let k0 = 0, k1 = 0;
      for (let k = 0; k < hq; k++) { if (a0[k] > a0[k0]) k0 = k; if (a1[k] > a1[k1]) k1 = k; }
      const heroAns = hs.top[hs.NL][hq][0];
      const lensAt = (s, l, t) => s.top[l][t][0];
      const cupIdx = hs.tokens.indexOf('cup');

      // ---------------------------------------------------------------- 1. tower + steps
      const towerHost = el('div');
      const stage = el('div', { class: 'ch-stage st-stage' },
        ui.figure({ title: 'The tower · tinyworld · 3 layers', badge: el('span', { class: 'st-badges' }, ui.badge('live')) },
          towerHost,
          el('div', { class: 'st-key', 'aria-hidden': 'true' },
            el('span', {}, el('i', { style: { background: AM.headColor(0) } }), el('i', { style: { background: AM.headColor(1) } }), el('i', { style: { background: AM.headColor(2) } }), el('i', { style: { background: AM.headColor(3) } }), 'heads 0–3'),
            el('span', {}, el('i', { style: { background: AM.dye.lichen, borderRadius: '2px', transform: 'rotate(45deg) scale(0.8)' } }), 'MLP edit'),
            el('span', {}, el('i', { style: { background: AM.dye.weld, boxShadow: '0 0 6px var(--weld)' } }), 'lens: final word'),
            el('span', {}, el('i', { style: { background: AM.dye.woad, boxShadow: '0 0 6px var(--woad)' } }), 'lens: other word')),
          el('figcaption', { class: 'st-cap-desk' }, capText())));
      function capText() {
        return 'Live model: one forward pass of tinyworld. Arcs are attention weights of 0.3 or more in each layer, coloured by head (self-attention and the habitual resting weight on the first word are left out). Knots show how far each MLP moved the stream at that position. Beads are the logit lens at each depth. In step 1 the small towers on the right compare layer counts only. Click a thread, or focus the picture and use the arrow keys.';
      }

      const step = (label, h3, ...kids) => el('div', { class: 'step' },
        el('span', { class: 'step-label' }, label),
        el('h3', {}, h3),
        ...kids.map((k) => (typeof k === 'string' ? el(k.startsWith('<div') ? 'div' : 'p', { html: k }) : k)));

      // step 2: the silence button
      const result = el('p', { class: 'st-result', 'aria-live': 'polite' });
      let muted = false;
      const muteBtn = el('button', { type: 'button', class: 'btn', id: 'st-mute', 'aria-pressed': 'false' }, `Silence L0H0 at ${q('is')}`);
      const setResult = () => {
        const s = muted && hero.mutedSent ? hero.mutedSent : hs;
        const w1 = s.attn[1][3][hq], ans = s.top[s.NL][hq][0];
        const kr = hs.tokens.indexOf(heroAns.token);
        result.innerHTML = muted
          ? `L1H3’s weight on <em>${esc(hs.tokens[kr])}</em> falls from <b>${f2(a1[kr])}</b> to <b>${f2(w1[kr])}</b>, and the model now says <span class="bad">${esc(ans.token)}</span> (${pct(ans.p)}). L1H3 was relying on what L0H0 wrote.`
          : `Normal run: the model says <span class="good">${esc(heroAns.token)}</span> (${pct(heroAns.p)}).`;
      };
      const setMute = (b) => {
        if (b && hero.mutedSent === undefined) {
          const net = getNet();
          hero.mutedSent = net ? Object.assign(analyse(m, hs.ids, net.forward(hs.ids, (l, h, t) => l === 0 && h === 0 && t === hq)), { focus: hq }) : null;
        }
        if (b && !hero.mutedSent) { muteBtn.disabled = true; result.textContent = 'Unavailable: the local forward pass did not match the model, so it is switched off.'; return; }
        muted = !!b;
        muteBtn.setAttribute('aria-pressed', String(muted));
        muteBtn.textContent = muted ? 'Restore L0H0' : `Silence L0H0 at ${q('is')}`;
        tower.setMuted(muted);
        setResult();
      };
      muteBtn.addEventListener('click', () => { tower.setSlot(hero); tower.setMode('compose'); setMute(!muted); });
      setResult();

      // step 5: sentence chips
      let chosen = 2;
      const trace5 = el('div', { class: 'st-trace', 'aria-live': 'polite' });
      const chip5 = slots.map((s, i) => el('button', { type: 'button', class: 'st-chip', id: `st-depth-${s.key}`, 'aria-pressed': String(i === chosen), title: s.text, onclick: () => choose(i) }, s.label));
      const setTrace5 = () => {
        const s = slots[chosen];
        const t = tower.slot === s ? clamp(tower.focus, 0, s.sent.T - 1) : s.focus;
        trace5.innerHTML = `<span class="st-tr"><i>at</i> ${esc(q(s.sent.tokens[t]))}</span>` + traceHTML(s.sent, t);
      };
      const choose = (i) => {
        chosen = i;
        chip5.forEach((c, j) => c.setAttribute('aria-pressed', String(i === j)));
        tower.setSlot(slots[i]);
        if (stepsCtl && stepsCtl.current !== 4) tower.setMode('depths');
        setTrace5();
      };

      /** Step 4's second paragraph: the path at "cup" wanders (worded from the live readings). */
      function cupText() {
        const tail = 'Click any thread to read its column.';
        if (cupIdx < 0) return tail;
        const e0 = lensAt(hs, 0, cupIdx), e2 = lensAt(hs, 2, cupIdx), e3 = lensAt(hs, 3, cupIdx);
        if (e2.id === e3.id) return tail;
        const start = e0.id === e3.id ? `the bare embedding already guesses ${say(e0.token)}, ` : '';
        return `Not every position climbs so neatly. At <em>cup</em>, ${start}the middle readings expect the list to go on (${say(e2.token)}), and only the last layer settles on ${say(e3.token)}. A lens path can wander on the way up. ${tail}`;
      }
      const trace4 = el('div', { class: 'st-trace', html: traceHTML(hs, hq) });
      const steps = [
        step('1 · Repeat', 'One block, stacked',
          'The last four chapters built one transformer block. Attention lets positions share information, the MLP works on each position alone, and both add their results to the residual stream. A full model is that block repeated, each copy with its own weights.',
          'The tiny model on this page stacks <strong>3</strong>. GPT-2 small stacks <strong>12</strong>. GPT-3 stacks <strong>96</strong>.',
          'In the picture, each vertical thread is one word’s residual stream, rising from its embedding at the bottom to the output at the top. The dyed arcs are this model’s real attention, one colour per head. The violet knots are MLP edits, bigger where the MLP moved the stream further.'),
        step('2 · Compose', 'Later layers build on earlier ones',
          `Look at the last word, <em>is</em>. To continue, the model has to recall which colour went with <em>ball</em>.`,
          `In layer 0, head L0H0 (${esc(headName(0, 0))}) at <em>is</em> puts <strong>${f2(a0[k0])}</strong> of its attention on <em>${esc(hs.tokens[k0])}</em>, so what it copies from there lands in the stream at <em>is</em>. In layer 1, head L1H3 (${esc(headName(1, 3))}) at the same word puts <strong>${f2(a1[k1])}</strong> of its attention on <em>${esc(hs.tokens[k1])}</em>, the colour that went with <em>ball</em>.`,
          'Does the second head depend on the first? Silence L0H0 at this one word, leave everything else switched on, and watch.',
          el('div', { class: 'st-mute-row' }, muteBtn), result,
          '<span class="st-small">Across all 1,004 held-out colour questions, silencing L0H0 at the question word alone drops accuracy from 98.6% to 46.5%, close to picking one of the listed colours at random. Measured offline with these same weights.</span>',
          'A later head reading what an earlier head wrote is called <span class="term">composition</span>. It is one reason depth helps: every layer starts from everything the layers below have worked out.'),
        step('3 · The logit lens', 'Reading the stream partway up',
          'At the top, the model turns the stream into a guess: a final LayerNorm, the unembedding matrix W<sub>U</sub> (one score per vocabulary word; chapter 9 opens it up), then softmax. Nothing stops us applying the same three steps lower down.',
          '<div class="math block st-math">lens(x<sub>ℓ</sub>) = softmax(LN<sub>f</sub>(x<sub>ℓ</sub>) W<sub>U</sub> + b<sub>U</sub>)</div>',
          'Each bead is that guess for one word at one depth. <strong style="color:var(--weld)">Gold</strong> beads already match the final answer, <strong style="color:var(--woad)">blue</strong> ones guess a different word, and bigger means more probability. This trick is the <span class="term">logit lens</span>, introduced in a 2020 blog post about GPT-2 by the writer nostalgebraist. Chapter 6’s ladder used it on one position; here it reads the whole tower.'),
        step('4 · Sharpen', 'A guess comes into focus',
          trace4,
          `Follow <em>is</em> upward. The embedding alone makes a vague guess (<em>${esc(lensAt(hs, 0, hq).token)}</em>, ${pct(lensAt(hs, 0, hq).p)}), and after layer 0 the top guess is still <em>${esc(lensAt(hs, 1, hq).token)}</em> (${pct(lensAt(hs, 1, hq).p)}). After layer 1, home of the colour binder, the lens reads <em>${esc(lensAt(hs, 2, hq).token)}</em> (${pct(lensAt(hs, 2, hq).p)}). Layer 2 only sharpens it.`,
          cupText()),
        step('5 · Depth', 'Each skill has its own height',
          'Pick a sentence and the tower re-weaves it. A pronoun is settled after layer 0. Agreement starts at <em>are</em> and flips to <em>is</em> in layer 1, where a head reads the singular noun <em>key</em>. A copied name sharpens layer by layer. For the dog, the sound stays <em>oink</em>, the model’s default, until the last layer’s MLP writes <em>woof</em>. Only the cat waits that long; the other animals get their sound in layer 0 or 1.',
          el('div', { class: 'st-chips', role: 'group', 'aria-label': 'Sentences for the tower' }, chip5),
          trace5),
      ];
      const prose = el('div', { class: 'ch-prose' }, steps);
      body.appendChild(el('div', { class: 'ch-split st-split' }, stage, prose));
      body.appendChild(el('p', { class: 'caption st-cap-phone' }, capText()));

      const tower = buildTower(ctx, towerHost);
      tower.setSlot(hero, { sweep: false });
      tower.setMode('stack');
      tower.onFocus = () => { if (stepsCtl && stepsCtl.current === 4) setTrace5(); };
      let stepsCtl = null;
      stepsCtl = ctx.steps(steps, (i) => {
        if (i !== 1 && muted) setMute(false);
        const mode = ['stack', 'compose', 'lens', 'focus', 'depths'][i];
        if (i < 4) tower.setSlot(hero);
        else tower.setSlot(slots[chosen]);
        tower.setMode(mode);
        if (i === 4) setTrace5();
        if (i === 3 && tower.focus !== hq) tower.setFocus(hq);
      });
      setTrace5();

      // ---------------------------------------------------------------- 2. the lens table
      const tablePanel = el('div', { class: 'panel' });
      body.appendChild(el('section', { class: 'ch-wide st-sec', 'aria-labelledby': 'st-lt-h' },
        el('div', { class: 'prose' },
          el('span', { class: 'st-kicker' }, 'Free play'),
          el('h3', { id: 'st-lt-h' }, 'The lens, everywhere at once'),
          el('p', { html: 'The tower shows one sentence at a time. This table shows all of it: every position down the side, every depth across, and in each cell the word the lens reads there with its probability. Each row is a guess about the word that comes <em>after</em> it. Hover, tap or tab into a cell for the top five.' }),
          el('p', { html: `Pick an example or write your own. The model knows only ${m.vocab.filter((w) => !['<pad>', '<unk>', '.', ','].includes(w)).length} words of toy English (plus the full stop and comma), so stick to its words (the list is below) and its kinds of sentence: <em>the queen walked to the park because</em>, <em>the boxes near the old key</em>, <em>tom met lucy at the river . lucy waved at</em>.` })),
        ui.figure({ title: 'Logit lens table', badge: 'live', caption: 'Live model. Each cell decodes the residual stream at that position and depth with the final LayerNorm and unembedding. x₃ is the model’s real output; the other columns are lens readings of a stream that was never trained to be read that way. With “read between attention and MLP” on, extra columns read the stream after each layer’s attention, before its MLP. Unknown words are dropped before the model runs.' }, tablePanel)));
      buildTable(ctx, tablePanel, m);

      // ---------------------------------------------------------------- 3. depth profiles
      const dpCanvas = el('div');
      const dpTypes = el('div');
      const dpCtl = el('div');
      const dpRead = el('p', { class: 'st-dp-read', 'aria-live': 'polite' });
      body.appendChild(el('section', { class: 'ch-wide st-dp', 'aria-labelledby': 'st-dp-h' },
        el('div', { class: 'panel' },
          ui.figure({ title: 'Depth profiles', badge: 'live', caption: `Live model. Sentences are generated here from the model’s own training grammar, ten per kind (press “New sentences” to draw others); facts are all ${FACTS.length} that it memorised (capitals both ways, animal sounds, colours of things). The lens reads P(correct next word) at the last position at each depth. Thin threads: single sentences. Thick threads: means.` },
            dpTypes, dpCanvas, dpRead, dpCtl)),
        el('div', { class: 'prose' },
          el('span', { class: 'st-kicker' }, 'Many sentences'),
          el('h3', { id: 'st-dp-h' }, 'Different skills, different depths'),
          el('p', { html: 'One sentence could be a fluke, so here are many. For each kind of dependency the page writes fresh sentences, runs them, and asks the lens how much probability sits on the right answer at each depth. Taken together, the memorised facts climb differently from the dog in the tower: they make more than half their climb in layer 0, where the first MLP recalls them (chapter 6), and reach certainty only at the top.' }),
          el('p', { html: 'Agreement parks near one half after layer 0 for a neat reason: at that depth the stream says <em>are</em> for every sentence (we checked all 899 held-out agreement questions), which is right for plural subjects and wrong for singular ones. Layer 1 reads the head noun and fixes the singulars.' }),
          el('p', { html: 'Part of the order follows from the wiring. A layer can only use what the layers below it have already written, so a skill built on another skill has to sit higher in the tower. Colour binding is one: L1H3 in layer 1 leans on what L0H0 wrote in layer 0, as the silence button in the tower showed. Why copied names, and the last stretch of the facts, need the final layer is harder to read from these curves alone.' }))));
      buildProfiles(ctx, { canvasHost: dpCanvas, typesHost: dpTypes, ctlHost: dpCtl, readHost: dpRead }, m);

      // ---------------------------------------------------------------- caveat
      body.appendChild(el('div', { class: 'st-cards' },
        el('div', { class: 'st-card' },
          el('h4', {}, 'Read the lens with care'),
          el('p', { html: 'Only the top of the stream is trained to be decoded. The lens assumes the lower layers already speak the same language, and they need not. Even here the lowest readings are often odd: for the colour sentence, the embedding column of the table guesses <em>are</em> after <em>the</em> and <em>of</em> after <em>is</em>.' }),
          el('p', { html: 'In bigger models the early layers often decode to nonsense, and the plain lens works worse in some model families than in others. Researchers now train a small translator for each layer, the <span class="term">tuned lens</span> (Belrose et al., 2023), to read them better. A lens reading is a hint about what the stream contains. Only the reading at the very top is what the model actually says.' }))));

      body.appendChild(el('div', { class: 'callout' },
        el('span', { class: 'callout-label' }, 'Key idea'),
        el('p', { html: 'A transformer is one block, repeated. Each layer reads the residual stream, adds a refinement and passes it up, so later layers can build on what earlier ones wrote. The logit lens decodes the stream partway up and shows the guess taking shape: in this tiny model, pronouns are settled after layer 0, colours after layer 1, and copied names are finished only at the top.' })));
    },
  });
})();
