/* Chapter 12 — Scale & recap: "From Toy to Titan".

   1. Powers of ten (scrollytelling stage). Every model is a square of cloth
      whose AREA is its parameter count; the camera zooms out on a log scale.
      The stripes are where each model's parameters sit (word table, position
      table, one attention + one MLP stripe per layer, unembedding), computed
      from its published shape with one formula (countParams), which also
      reproduces the published totals. Free play: drag the axis, use the
      slider, or fly to a model. Ends with every cloth scaled to one size.
   2. The whole machine (poster). The decoder-only transformer, bottom to top,
      each part linked to the chapter that explains it. Labels switch between
      this page's model, GPT-2 small, GPT-3 and Llama 3.1 405B. The sentence
      running through it is the live tinyworld model, one token per pass.
   3. What real models add, a recap, what the page left out, the key idea and
      a finished cloth woven from the live model's mean attention. */
(() => {
  const ID = 'epilogue';
  const M = AM.math;
  const D = AM.draw;
  const clamp = M.clamp;
  const LG = Math.log10;
  const now = () => performance.now() / 1000;

  // ================================================================== helpers
  const fmtInt = (n) => Math.round(n).toLocaleString('en-US');
  const trim0 = (s) => (s.indexOf('.') >= 0 ? s.replace(/0+$/, '').replace(/\.$/, '') : s);
  const fix = (x, d) => trim0(x.toFixed(d));
  /** 124.4M · 1.56B · 174.6B · 3T */
  function fmtShort(n) {
    const a = Math.abs(n);
    if (a >= 1e12) return fix(n / 1e12, n / 1e12 < 10 ? 1 : 0) + 'T';
    if (a >= 1e9) return fix(n / 1e9, n / 1e9 < 10 ? 2 : 1) + 'B';
    if (a >= 1e6) return fix(n / 1e6, n / 1e6 < 10 ? 2 : 1) + 'M';
    if (a >= 1e3) return fix(n / 1e3, n / 1e3 < 10 ? 1 : 0) + 'K';
    return String(Math.round(n));
  }
  /** 174.6 billion */
  function fmtWords(n) {
    if (n >= 1e12) return fix(n / 1e12, 1) + ' trillion';
    if (n >= 1e9) return fix(n / 1e9, 1) + ' billion';
    if (n >= 1e6) return fix(n / 1e6, 1) + ' million';
    return fmtInt(n);
  }
  const fmtP = (n) => (n < 1e6 ? fmtInt(n) : fmtShort(n));
  /** A length in millimetres, in the unit a person would use. */
  function fmtLen(mm) {
    if (mm < 1000) return Math.round(mm / 10) + ' cm';
    if (mm < 1e6) { const m = mm / 1000; return (m < 10 ? fix(m, 1) : String(Math.round(m))) + ' m'; }
    return fix(mm / 1e6, 1) + ' km';
  }
  const pct = (x, d = 1) => fix(x * 100, d) + '%';
  const probPct = (p) => (p >= 0.995 ? Math.round(p * 100) + '%' : p >= 0.0995 ? Math.round(p * 100) + '%' : p >= 0.001 ? (p * 100).toFixed(1) + '%' : '<0.1%');
  const POW_WORDS = ['1', '10', '100', '1,000', '10,000', '100,000', '1 million', '10 million', '100 million', '1 billion', '10 billion', '100 billion', '1 trillion', '10 trillion'];
  const powWords = (n) => POW_WORDS[n] || '10^' + n;

  /** Is a node near the viewport? Lets each figure idle while the chapter is visible. */
  function inView(node, margin = '120px') {
    const s = { on: true };
    if (typeof IntersectionObserver !== 'undefined') {
      new IntersectionObserver((es) => { for (const e of es) s.on = e.isIntersecting; }, { rootMargin: `${margin} 0px ${margin} 0px` }).observe(node);
    }
    return s;
  }

  /** Pre-rendered glow sprites (cheap beads). */
  const spriteCache = new Map();
  function sprite(hex) {
    if (spriteCache.has(hex)) return spriteCache.get(hex);
    const S = 48, c = document.createElement('canvas');
    c.width = c.height = S;
    const g = c.getContext('2d'), r = S / 2;
    const grd = g.createRadialGradient(r, r, 0, r, r, r);
    grd.addColorStop(0, 'rgba(255,248,230,1)');
    grd.addColorStop(0.16, AM.rgba(hex, 1));
    grd.addColorStop(0.36, AM.rgba(hex, 0.42));
    grd.addColorStop(0.66, AM.rgba(hex, 0.09));
    grd.addColorStop(1, AM.rgba(hex, 0));
    g.fillStyle = grd;
    g.fillRect(0, 0, S, S);
    spriteCache.set(hex, c);
    return c;
  }
  function bead(g, x, y, r, hex, a = 1) {
    if (a <= 0.01) return;
    g.globalAlpha = a;
    g.drawImage(sprite(hex), x - r * 4, y - r * 4, r * 8, r * 8);
    g.globalAlpha = 1;
  }

  /** Text with an ink halo, so it reads over cloth and threads. */
  function halo(g, str, x, y, o = {}) {
    g.save();
    g.font = AM.font(o.size || 11, o.role || 'mono', o.weight, o.italic);
    g.textAlign = o.align || 'left';
    g.textBaseline = o.baseline || 'middle';
    if ('letterSpacing' in g) g.letterSpacing = o.ls || '0px';
    g.globalAlpha *= o.alpha ?? 1;
    g.lineJoin = 'round';
    g.strokeStyle = o.halo || AM.rgba(AM.col.ink, 0.88);
    g.lineWidth = o.haloW || 4;
    g.strokeText(str, x, y);
    g.fillStyle = o.color || AM.col.linen;
    g.fillText(str, x, y);
    g.restore();
  }

  /** Point on a cubic given as [x0,y0,x1,y1,x2,y2,x3,y3]. */
  function cub(P, t) {
    const u = 1 - t;
    return {
      x: u * u * u * P[0] + 3 * u * u * t * P[2] + 3 * u * t * t * P[4] + t * t * t * P[6],
      y: u * u * u * P[1] + 3 * u * u * t * P[3] + 3 * u * t * t * P[5] + t * t * t * P[7],
    };
  }
  /** A silk thread along a cubic: soft glow, dyed core, thin sheen. */
  function silk(g, P, { color = AM.dye.weld, width = 1.6, alpha = 1, dash = null, sheen = true } = {}) {
    if (alpha <= 0.01) return;
    g.save();
    g.lineCap = 'round';
    if (dash) g.setLineDash(dash);
    g.beginPath(); g.moveTo(P[0], P[1]); g.bezierCurveTo(P[2], P[3], P[4], P[5], P[6], P[7]);
    if (!dash) {
      g.strokeStyle = AM.rgba(color, 0.14 * alpha);
      g.lineWidth = width * 4 + 2;
      g.stroke();
    }
    g.strokeStyle = AM.rgba(color, alpha);
    g.lineWidth = width;
    g.stroke();
    if (sheen && !dash && width >= 1.6) {
      g.strokeStyle = AM.rgba('#fff4d6', 0.32 * alpha);
      g.lineWidth = Math.max(0.6, width * 0.25);
      g.stroke();
    }
    g.restore();
  }
  const line = (x0, y0, x1, y1) => [x0, y0, M.lerp(x0, x1, 1 / 3), M.lerp(y0, y1, 1 / 3), M.lerp(x0, x1, 2 / 3), M.lerp(y0, y1, 2 / 3), x1, y1];
  /** S-curve from (x0,y0) to (x1,y1) that leaves and arrives vertically. */
  const sv = (x0, y0, x1, y1) => { const m = (y0 + y1) / 2; return [x0, y0, x0, m, x1, m, x1, y1]; };

  // ================================================================== the models
  // Colours for the parts of the machine (used by both big figures).
  const COMP = {
    embed: AM.dye.saffron, pos: AM.dye.lichen, attn: AM.dye.woad, mlp: AM.dye.verdigris,
    unembed: AM.dye.cochineal, stream: AM.dye.weld, loss: AM.dye.madder,
  };

  // Published shapes. Every parameter count on this page is computed from these
  // by countParams(); the published totals are listed only to compare against.
  const SHAPES = {
    ours: { key: 'ours', name: 'This page', L: 3, d: 64, H: 4, kvH: 4, F: 256, V: 138, T: 32, tied: false, pos: 'learned', norm: 'ln', mlp: 'gelu', bias: true },
    gpt2s: { key: 'gpt2s', name: 'GPT-2 small', year: 2019, L: 12, d: 768, H: 12, kvH: 12, F: 3072, V: 50257, T: 1024, tied: true, pos: 'learned', norm: 'ln', mlp: 'gelu', bias: true, pub: '124M' },
    gpt2xl: { key: 'gpt2xl', name: 'GPT-2 XL', year: 2019, L: 48, d: 1600, H: 25, kvH: 25, F: 6400, V: 50257, T: 1024, tied: true, pos: 'learned', norm: 'ln', mlp: 'gelu', bias: true, pub: '1.5B' },
    gpt3: { key: 'gpt3', name: 'GPT-3', year: 2020, L: 96, d: 12288, H: 96, kvH: 96, F: 49152, V: 50257, T: 2048, tied: true, pos: 'learned', norm: 'ln', mlp: 'gelu', bias: true, pub: '175B', pubN: 175e9 },
    llama: { key: 'llama', name: 'Llama 3.1 405B', year: 2024, L: 126, d: 16384, H: 128, kvH: 8, F: 53248, V: 128256, T: 131072, tied: false, pos: 'rope', norm: 'rms', mlp: 'swiglu', bias: false, pub: '405B' },
  };
  // DeepSeek-V3 (Dec 2024, open weights) publishes totals, not a simple per-part shape.
  const DSV3 = { P: 671e9, active: 37e9 };
  // Public guesses for undisclosed frontier models: "hundreds of billions to trillions".
  const FRONT_LO = 3e11, FRONT_HI = 3e12;
  // tinyworld training: 12,000 steps × 128 sequences × 19.3 tokens on average
  // (counted from tools/data-tinyworld.mjs with the training seed) ≈ 29.6M tokens.
  const OURS_TOKENS = 29.6e6;
  const GPT3_TOKENS = 300e9;

  /**
   * Parameters of a decoder-only transformer, part by part.
   *   attention per layer: W_Q (d·d) + W_K, W_V (d·kv each) + W_O (d·d) [+ biases] + its norm
   *   MLP per layer:       2·d·F (GELU) or 3·d·F (SwiGLU) [+ biases] + its norm
   *   plus the word table V·d, a learned position table T·d, a final norm and the
   *   unembedding d·V (+V bias) unless it is tied to the word table.
   */
  function countParams(s) {
    const dh = s.d / s.H, kv = s.kvH * dh;
    const norm = s.norm === 'ln' ? 2 * s.d : s.d;
    const attn = s.d * s.d + 2 * s.d * kv + s.d * s.d + (s.bias ? s.d + 2 * kv + s.d : 0) + norm;
    const mlp = (s.mlp === 'swiglu' ? 3 : 2) * s.d * s.F + (s.bias ? s.F + s.d : 0) + norm;
    const embed = s.V * s.d;
    const pos = s.pos === 'learned' ? s.T * s.d : 0;
    const final = norm;
    const unembed = s.tied ? 0 : s.d * s.V + (s.bias ? s.V : 0);
    const total = embed + pos + s.L * (attn + mlp) + final + unembed;
    return { dh, kv, attn, mlp, embed, pos, final, unembed, total };
  }

  /** Stripe layout of a model's cloth, in parameter order (fractions of the width). */
  function stripesOf(s, c) {
    const out = [];
    let acc = 0;
    const push = (n, col) => { if (n > 0) { out.push({ f0: acc / c.total, f1: (acc + n) / c.total, col }); acc += n; } };
    push(c.embed, COMP.embed);
    push(c.pos, COMP.pos);
    for (let l = 0; l < s.L; l++) { push(c.attn, COMP.attn); push(c.mlp + (s.tied && l === s.L - 1 ? c.final : 0), COMP.mlp); }
    if (!s.tied) push(c.final + c.unembed, COMP.unembed);
    return out;
  }

  function getModel() {
    try { return window.AM && AM.model && typeof AM.model.get === 'function' ? AM.model.get('tinyworld') : null; } catch (e) { return null; }
  }

  function buildData() {
    const lm = getModel();
    const ours = Object.assign({}, SHAPES.ours);
    let shipped = null;
    if (lm && lm.config) {
      const c = lm.config;
      Object.assign(ours, { L: c.n_layer, d: c.d_model, H: c.n_head, kvH: c.n_head, F: c.d_ff, V: c.vocab_size, T: c.n_ctx });
      if (lm.meta && lm.meta.params) shipped = lm.meta.params;
    }
    const shapes = Object.assign({}, SHAPES, { ours });
    const counts = {};
    for (const k in shapes) counts[k] = countParams(shapes[k]);
    const P_ours = shipped || counts.ours.total;
    const fab = (key, label, P, sub, extra) => Object.assign({ key, label, P, sub, shape: shapes[key] || null, stripes: shapes[key] ? stripesOf(shapes[key], counts[key]) : null }, extra || {});
    const fabrics = [
      fab('ours', 'This page', P_ours, fmtInt(P_ours) + ' parameters', { lineup: 0 }),
      fab('gpt2s', 'GPT-2 small', counts.gpt2s.total, '124M · 2019', { lineup: 1 }),
      fab('gpt2xl', 'GPT-2 XL', counts.gpt2xl.total, '1.5B · 2019', { lineup: 2 }),
      fab('gpt3', 'GPT-3', SHAPES.gpt3.pubN, '175B · 2020', { lineup: 3 }),
      fab('dsv3', 'DeepSeek-V3', DSV3.P, '671B · 37B active', { active: DSV3.active, subShort: '671B' }),
    ];
    fabrics.forEach((f) => {
      if (f.stripes) {
        const k = (col) => (col === COMP.attn ? 0.82 : col === COMP.mlp ? 0.5 : 0.64);
        f.hi = f.stripes.map((s) => AM.mix(AM.col.ink2, s.col, k(s.col)));
        f.lo = f.stripes.map((s) => AM.mix(AM.col.ink2, s.col, k(s.col) * 0.6));
        let pair = 0;
        for (const s of f.stripes) if (s.col === COMP.attn || s.col === COMP.mlp) pair += s.f1 - s.f0;
        f.pairFrac = pair / (f.shape ? f.shape.L : 1); // width of one layer (attention + MLP) as a fraction of the cloth
      }
    });
    return { lm, shapes, counts, shipped, P_ours, fabrics };
  }

  /** Chapter kicker by id (falls back to a fixed name), for the links back. */
  const FALLBACK_NAMES = { tokens: 'Tokenization', embed: 'Embeddings', position: 'Positional encoding', attention: 'Self-attention', heads: 'Multi-head attention', mlp: 'Feed-forward network', residual: 'Residual stream', stack: 'Depth', predict: 'Unembedding & sampling', train: 'Training', lab: 'Live training lab' };
  const NUMS = { tokens: 1, embed: 2, position: 3, attention: 4, heads: 5, mlp: 6, residual: 7, stack: 8, predict: 9, train: 10, lab: 11 };
  function chName(id) {
    const d = (AM.chapters || []).find((c) => c.id === id);
    const raw = (d && d.kicker) || FALLBACK_NAMES[id] || id;
    return raw.replace(/&amp;/g, '&');
  }
  const chLink = (id) => `<a class="ep-ch" href="#ch-${id}">${String(NUMS[id]).padStart(2, '0')} · ${chName(id).replace(/&/g, '&amp;')}</a>`;

  // ================================================================== CSS
  const CSS = `
    #ch-${ID} .ep-intro { margin-bottom: calc(-1 * var(--space-5)); }
    #ch-${ID} .step em { color: var(--linen); }
    #ch-${ID} .ep-zfig { gap: 10px; }
    #ch-${ID} .ep-zhold { position: relative; }
    #ch-${ID} .ep-zhold canvas { touch-action: pan-y; }
    #ch-${ID} .ep-zfig .stage-canvas canvas { border-radius: var(--radius); border: 1px solid var(--rule); background: radial-gradient(110% 90% at 18% 85%, color-mix(in srgb, var(--ink-2) 88%, var(--weld)) 0%, var(--ink-2) 46%, var(--ink) 100%); }
    #ch-${ID} .ep-zcard { position: absolute; display: grid; gap: 7px; align-content: start; padding: 14px 16px 15px; border-radius: 10px; background: color-mix(in srgb, var(--ink) 84%, transparent); border: 1px solid var(--rule); transition: opacity 0.35s; pointer-events: none; }
    #ch-${ID} .ep-zcard[hidden] { display: none; }
    #ch-${ID} .ep-zc-name { font-family: var(--font-display); font-style: italic; font-size: 1.45rem; line-height: 1.05; color: var(--linen); }
    #ch-${ID} .ep-zc-name span { display: block; margin-top: 5px; font-family: var(--font-mono); font-style: normal; font-size: 9.5px; letter-spacing: 0.14em; text-transform: uppercase; color: var(--mist); }
    #ch-${ID} .ep-zc-big { font-family: var(--font-display); font-size: 2.05rem; line-height: 1; color: var(--weld); font-variant-numeric: lining-nums; margin-top: 2px; }
    #ch-${ID} .ep-zc-big.is-words { font-size: 1.35rem; font-style: italic; }
    #ch-${ID} .ep-zc-unit { font-family: var(--font-mono); font-size: 9.5px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--mist); margin-top: -3px; }
    #ch-${ID} .ep-zc-rows { display: grid; gap: 4px; margin-top: 4px; }
    #ch-${ID} .ep-zc-row { display: flex; justify-content: space-between; align-items: baseline; gap: 10px; font-family: var(--font-mono); font-size: 10px; color: var(--mist); border-top: 1px dotted var(--rule); padding-top: 4px; }
    #ch-${ID} .ep-zc-row b { color: var(--linen); font-weight: 400; text-align: right; }
    #ch-${ID} .ep-zc-note { font-size: 12px; line-height: 1.45; color: var(--linen-dim); border-top: 1px solid var(--rule); padding-top: 8px; margin-top: 2px; }
    #ch-${ID} .ep-zc-note b { color: var(--linen); font-weight: 600; }
    #ch-${ID} .ep-zbar { display: flex; flex-wrap: wrap; align-items: end; justify-content: space-between; gap: var(--space-3) var(--space-5); }
    #ch-${ID} .ep-zslider { position: absolute; left: 0; bottom: 0; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); clip-path: inset(50%); white-space: nowrap; }
    #ch-${ID} .ep-zcap { max-width: 64ch; }
    @media (max-width: 900px) {
      #ch-${ID} .ep-zcap { display: none; }
      #ch-${ID} .ep-zfig { gap: 6px; }
      #ch-${ID} .ep-zfig .fig-top { min-height: 0; }
    }
    #ch-${ID} .ep-sm { display: none; }
    @media (max-width: 560px) {
      #ch-${ID} .ep-lg { display: none; }
      #ch-${ID} .ep-sm { display: inline; }
      #ch-${ID} .ep-zbar .seg button, #ch-${ID} .ep-pctl .seg button { padding: 5px 9px; }
      #ch-${ID} .ep-zlegend { display: none; }
    }
    #ch-${ID} .step .ep-num { font-family: var(--font-mono); font-size: 0.86em; color: var(--weld); white-space: nowrap; }

    /* ---- poster ---- */
    #ch-${ID} .ep-poster-sec { display: grid; gap: var(--space-5); }
    #ch-${ID} .ep-poster { gap: var(--space-4); }
    #ch-${ID} .ep-badges { display: inline-flex; flex-wrap: wrap; gap: 6px; }
    #ch-${ID} .ep-pctl { display: flex; flex-wrap: wrap; align-items: end; justify-content: space-between; gap: var(--space-3) var(--space-5); }
    #ch-${ID} .ep-total { font-family: var(--font-mono); font-size: var(--fs-micro); letter-spacing: 0.04em; color: var(--mist); line-height: 1.6; }
    #ch-${ID} .ep-total b { color: var(--weld); font-weight: 500; font-size: 1.2em; }
    #ch-${ID} .ep-total .ep-ok { color: var(--verdigris); }
    #ch-${ID} .ep-pbody { position: relative; padding-block: 6px 10px; }
    #ch-${ID} .ep-pcv { position: absolute; inset: 0; z-index: 0; pointer-events: none; }
    #ch-${ID} .ep-flow { position: relative; z-index: 1; list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column-reverse; gap: 20px; }
    #ch-${ID} .ep-row { display: grid; grid-template-columns: minmax(0, 440px) minmax(0, 1fr); column-gap: 56px; align-items: center; transition: opacity 0.25s; }
    #ch-${ID} .ep-row.is-gap-lg { margin-top: 6px; }
    #ch-${ID} .ep-flow.has-hot .ep-row:not(.is-hot) { opacity: 0.45; }
    #ch-${ID} .ep-node { padding-left: 66px; min-width: 0; }
    #ch-${ID} .ep-box { --c: var(--linen-dim); position: relative; display: grid; gap: 5px; padding: 9px 14px 10px 17px; border-radius: 10px; border: 1px solid var(--rule-strong); background: var(--ink-2); transition: border-color 0.3s, box-shadow 0.3s, background 0.3s; min-width: 0; }
    #ch-${ID} .ep-box::before { content: ''; position: absolute; left: -1px; top: 9px; bottom: 9px; width: 3px; border-radius: 2px; background: var(--c); box-shadow: 0 0 8px var(--c); }
    #ch-${ID} .ep-row.is-hot .ep-box, #ch-${ID} .ep-row.is-lit .ep-box { border-color: color-mix(in srgb, var(--c) 70%, var(--rule)); box-shadow: 0 0 24px -8px var(--c); }
    #ch-${ID} .ep-row.is-lit .ep-box { background: color-mix(in srgb, var(--c) 10%, var(--ink-2)); }
    #ch-${ID} .ep-box-head { display: flex; flex-wrap: wrap; align-items: center; gap: 4px 9px; }
    #ch-${ID} .ep-box-title { font-weight: 600; color: var(--linen); font-size: 0.98rem; line-height: 1.25; }
    #ch-${ID} .ep-tag { font-family: var(--font-mono); font-size: 9px; letter-spacing: 0.08em; text-transform: uppercase; padding: 1px 6px 2px; border-radius: 5px; border: 1px solid var(--rule-strong); color: var(--mist); white-space: nowrap; }
    #ch-${ID} .ep-tag.is-new { color: var(--cochineal); border-color: color-mix(in srgb, var(--cochineal) 55%, var(--rule)); }
    #ch-${ID} .ep-box-meta { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 2px 12px; font-family: var(--font-mono); font-size: 10.5px; color: var(--mist); }
    #ch-${ID} .ep-box-meta b { color: var(--linen-dim); font-weight: 400; }
    #ch-${ID} .ep-box-meta .ep-pp { color: var(--c); }
    #ch-${ID} .ep-flash { animation: ep12-flash 0.9s ease-out; }
    @keyframes ep12-flash { 0% { color: var(--weld); text-shadow: 0 0 10px var(--weld); } 100% { text-shadow: none; } }
    #ch-${ID} .ep-pair { display: grid; grid-template-columns: minmax(0, 1.25fr) minmax(0, 1fr); gap: 8px; }
    #ch-${ID} .ep-pair .ep-box.is-dashed { border-style: dashed; background: transparent; }
    #ch-${ID} .ep-blocktag { display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 12px; padding: 2px 0 2px 2px; }
    #ch-${ID} .ep-blocktag .ep-xn { font-family: var(--font-display); font-style: italic; font-size: 1.6rem; line-height: 1; color: var(--weld); }
    #ch-${ID} .ep-blocktag .ep-layer { font-family: var(--font-mono); font-size: 10px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--mist); }
    #ch-${ID} .ep-blocktag .ep-layer b { color: var(--weld); font-weight: 500; }
    #ch-${ID} .ep-box--block, #ch-${ID} .ep-box--bare { padding: 0; border: 0; background: transparent; }
    #ch-${ID} .ep-box--block::before, #ch-${ID} .ep-box--bare::before { display: none; }
    #ch-${ID} .ep-row.is-hot .ep-box--block, #ch-${ID} .ep-row.is-lit .ep-box--block, #ch-${ID} .ep-row.is-hot .ep-box--bare, #ch-${ID} .ep-row.is-lit .ep-box--bare { box-shadow: none; background: transparent; }
    #ch-${ID} .ep-note { display: grid; gap: 3px; min-width: 0; }
    #ch-${ID} .ep-links { display: flex; flex-wrap: wrap; gap: 2px 14px; }
    #ch-${ID} .ep-ch { font-family: var(--font-mono); font-size: var(--fs-micro); letter-spacing: 0.1em; text-transform: uppercase; color: var(--weld); text-decoration: none; }
    #ch-${ID} .ep-ch:hover { color: var(--linen); text-decoration: underline; }
    #ch-${ID} .ep-note p { color: var(--linen-dim); font-size: 0.95rem; line-height: 1.5; }
    #ch-${ID} .ep-note .math { font-size: 0.8em; }
    #ch-${ID} .ep-chips { display: flex; flex-wrap: wrap; gap: 4px; min-height: 26px; }
    #ch-${ID} .ep-chip { font-weight: 600; font-size: 0.84rem; line-height: 1.2; padding: 3px 7px 4px; border-radius: 6px; border: 1px solid var(--rule-strong); background: var(--ink); color: var(--linen); white-space: pre; }
    #ch-${ID} .ep-chip.is-new { border-color: var(--weld); color: var(--weld); animation: ep12-chip 0.7s ease-out; }
    @keyframes ep12-chip { 0% { transform: translateY(-6px); opacity: 0; box-shadow: 0 0 18px var(--weld); } 100% { transform: none; opacity: 1; box-shadow: none; } }
    #ch-${ID} .ep-live { font-family: var(--font-mono); font-size: 9.5px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--verdigris); }
    #ch-${ID} .ep-samp { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 6px 16px; align-items: center; }
    #ch-${ID} .ep-bars { display: grid; gap: 3px; min-width: 0; }
    #ch-${ID} .ep-bar { display: grid; grid-template-columns: 5.2em minmax(0, 1fr) 3.4em; gap: 8px; align-items: center; font-family: var(--font-mono); font-size: 10.5px; color: var(--mist); }
    #ch-${ID} .ep-bar > span:first-child { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    #ch-${ID} .ep-bar > span:last-child { text-align: right; }
    #ch-${ID} .ep-bar-track { height: 5px; border-radius: 3px; background: var(--ink-3); overflow: hidden; }
    #ch-${ID} .ep-bar-fill { display: block; height: 100%; width: 0; border-radius: 3px; background: var(--rule-strong); transition: width 0.45s ease-out, background 0.3s; }
    #ch-${ID} .ep-bar.is-pick { color: var(--linen); }
    #ch-${ID} .ep-bar.is-pick .ep-bar-fill { background: var(--weld); box-shadow: 0 0 8px var(--weld); }
    #ch-${ID} .ep-next { display: grid; justify-items: center; gap: 3px; min-width: 5.4em; }
    #ch-${ID} .ep-next-word { font-family: var(--font-display); font-style: italic; font-size: 1.75rem; line-height: 1; color: var(--weld); text-shadow: 0 0 18px color-mix(in srgb, var(--weld) 45%, transparent); }
    #ch-${ID} .ep-next-lab { font-family: var(--font-mono); font-size: 9px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--mist); }
    #ch-${ID} .ep-fly { position: absolute; left: 0; top: 0; z-index: 3; pointer-events: none; font-weight: 600; font-size: 0.84rem; padding: 3px 7px 4px; border-radius: 6px; border: 1px solid var(--weld); background: var(--ink); color: var(--weld); box-shadow: 0 0 18px -2px var(--weld); white-space: pre; will-change: transform; }
    #ch-${ID} .ep-fly[hidden] { display: none; }
    #ch-${ID} .ep-pcap { max-width: 80ch; }
    @media (max-width: 760px) {
      #ch-${ID} .ep-row { grid-template-columns: minmax(0, 1fr); row-gap: 8px; }
      #ch-${ID} .ep-node, #ch-${ID} .ep-note { padding-left: 46px; }
      #ch-${ID} .ep-more { display: none; }
      #ch-${ID} .ep-flow { gap: 24px; }
      #ch-${ID} .ep-pair { grid-template-columns: minmax(0, 1fr); }
    }

    /* ---- refinements, recap, left out ---- */
    #ch-${ID} .ep-sechead { display: grid; gap: var(--space-3); max-width: var(--prose); }
    #ch-${ID} .ep-sechead p { color: var(--linen-dim); }
    #ch-${ID} .ep-refine { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 330px), 1fr)); column-gap: var(--space-7); margin-top: var(--space-4); }
    #ch-${ID} .ep-ref { --c: var(--weld); display: grid; grid-template-columns: 16px minmax(0, 1fr); align-content: start; column-gap: 12px; row-gap: 4px; padding: 14px 0 15px; border-top: 1px solid var(--rule); }
    #ch-${ID} .ep-ref-dot { grid-row: span 2; width: 10px; height: 10px; margin-top: 4px; border-radius: 50%; background: var(--c); box-shadow: 0 0 0 3px color-mix(in srgb, var(--c) 18%, transparent), 0 0 12px var(--c); }
    #ch-${ID} .ep-ref-name { font-family: var(--font-mono); font-size: var(--fs-micro); letter-spacing: 0.12em; text-transform: uppercase; color: var(--linen); }
    #ch-${ID} .ep-ref p { font-size: 0.95rem; line-height: 1.5; color: var(--linen-dim); }
    #ch-${ID} .ep-ref .ep-ch { font-size: 9.5px; margin-left: 4px; white-space: nowrap; }
    #ch-${ID} .ep-lists { align-items: start; }
    #ch-${ID} .ep-lists h3 { font-family: var(--font-body); font-size: var(--fs-h3); font-weight: 600; color: var(--linen); margin-bottom: var(--space-4); }
    #ch-${ID} .ep-threads { list-style: none; margin: 0; padding: 0 0 0 26px; position: relative; display: grid; gap: 13px; }
    #ch-${ID} .ep-threads::before { content: ''; position: absolute; left: 7px; top: 9px; bottom: 9px; width: 2px; border-radius: 2px; background: linear-gradient(var(--weld), color-mix(in srgb, var(--weld) 35%, transparent)); box-shadow: 0 0 8px color-mix(in srgb, var(--weld) 50%, transparent); }
    #ch-${ID} .ep-threads li { --c: var(--weld); position: relative; color: var(--linen-dim); line-height: 1.5; }
    #ch-${ID} .ep-threads li::before { content: ''; position: absolute; left: -24px; top: 0.42em; width: 10px; height: 10px; border-radius: 50%; background: var(--c); box-shadow: 0 0 0 3px var(--ink), 0 0 12px var(--c); }
    #ch-${ID} .ep-threads .ep-ch { margin-right: 8px; font-size: 10px; }
    #ch-${ID} .ep-left { list-style: none; margin: 0; padding: 0; display: grid; gap: 11px; }
    #ch-${ID} .ep-left li { position: relative; padding-left: 26px; color: var(--linen-dim); line-height: 1.5; }
    #ch-${ID} .ep-left li::before { content: ''; position: absolute; left: 1px; top: 0.72em; width: 14px; height: 2px; border-radius: 2px; background: linear-gradient(90deg, var(--mist), transparent); }
    #ch-${ID} .ep-left li strong, #ch-${ID} .ep-threads li strong { color: var(--linen); font-weight: 600; }

    /* ---- key idea + coda ---- */
    #ch-${ID} .ep-chain { display: flex; flex-wrap: wrap; align-items: center; gap: 7px 7px; font-family: var(--font-mono); font-size: 11px; color: var(--mist); margin-block: 4px; }
    #ch-${ID} .ep-chain span { padding: 3px 9px 4px; border-radius: 6px; border: 1px solid var(--rule-strong); background: var(--ink); color: var(--linen); white-space: nowrap; }
    #ch-${ID} .ep-chain span.ep-blk { border-color: color-mix(in srgb, var(--weld) 55%, var(--rule)); color: var(--weld); white-space: normal; }
    #ch-${ID} .ep-coda { gap: var(--space-3); width: 100%; max-width: 860px; margin-inline: auto; }
    #ch-${ID} .ep-coda .stage-canvas canvas { border-radius: var(--radius); }
    #ch-${ID} .ep-closing { font-family: var(--font-display); font-style: italic; font-weight: 400; font-size: clamp(1.35rem, 1.05rem + 1.25vw, 2.15rem); line-height: 1.28; text-align: center; max-width: 36ch; margin: var(--space-6) auto 0; color: var(--linen); text-wrap: balance; }
    #ch-${ID} .ep-closing em { color: var(--weld); }
    #ch-${ID} .ep-codacap { text-align: center; margin-inline: auto; }
    @media (prefers-reduced-motion: reduce) {
      #ch-${ID} .ep-chip.is-new, #ch-${ID} .ep-flash { animation: none; }
    }
  `;

  // ================================================================== figure 1: powers of ten
  function buildZoom(body, ctx, DATA) {
    const el = ctx.el, ui = AM.ui;
    const FABS = DATA.fabrics;
    const byKey = Object.fromEntries(FABS.map((f) => [f.key, f]));
    const DESC = FABS.slice().sort((a, b) => b.P - a.P);
    const LINEUP = FABS.filter((f) => f.lineup != null).sort((a, b) => a.lineup - b.lineup);
    const ours = DATA.shapes.ours;
    const TARGET = { ours: LG(byKey.ours.P), gpt2: LG(byKey.gpt2xl.P), gpt3: LG(byKey.gpt3.P), frontier: LG(FRONT_HI) - 0.02 };
    const Z_MIN = 4.9, Z_MAX = 12.8, AX_MIN = 5, AX_MAX = 13;
    const FIT = 0.86; // a focused cloth fills 86% of the frame

    // ---- numbers quoted in the steps (all computed)
    const Pg2s = byKey.gpt2s.P, Pg2xl = byKey.gpt2xl.P, Pg3 = byKey.gpt3.P, Po = byKey.ours.P;
    const c = DATA.counts;
    const stitch = (P) => fmtLen(Math.sqrt(P));
    const ratioG2 = Math.round(Pg2s / Po);
    const ratioG3 = Pg3 / Po;
    const ratioTok = GPT3_TOKENS / OURS_TOKENS;
    const wordShare = (k) => c[k].embed / c[k].total;
    const mlpRatio = (k) => c[k].mlp / c[k].attn;

    const split = el('div', { class: 'ch-split ep-split' });
    body.appendChild(split);
    const stage = el('div', { class: 'ch-stage' });
    const prose = el('div', { class: 'ch-prose' });
    split.append(stage, prose);

    const fig = ui.figure({ title: 'Parameters as cloth', badge: 'illustration', cls: 'ep-zfig' });
    stage.appendChild(fig);
    const hold = el('div', { class: 'ep-zhold' });
    fig.appendChild(hold);
    const cv = ctx.canvas(hold, {
      label: `Illustration: squares of woven cloth whose areas are proportional to parameter counts. This page's model (${fmtInt(Po)} parameters) sits in the corner of GPT-2 (124 million to 1.5 billion), GPT-3 (175 billion), DeepSeek-V3 (671 billion) and a haze marking estimates for undisclosed frontier models. The view zooms out on a logarithmic scale. Stripes show where each model's parameters sit.`,
      height: (w) => (w < 560
        ? Math.round(Math.min(w * 0.96, Math.max(300, window.innerHeight * 0.42)))
        : Math.round(Math.min(Math.max(460, w * 0.82), 590, Math.max(430, window.innerHeight * 0.64)))),
    });
    const card = el('div', { class: 'ep-zcard', hidden: true, 'aria-hidden': 'true' });
    hold.appendChild(card);

    const legend = el('div', { class: 'ep-zlegend' }, ui.legend([
      { color: COMP.embed, label: 'word table' }, { color: COMP.pos, label: 'positions' },
      { color: COMP.attn, label: 'attention' }, { color: COMP.mlp, label: 'MLP' }, { color: COMP.unembed, label: 'unembedding' },
    ]));
    fig.appendChild(legend);

    const two = (long, short) => el('span', {}, el('span', { class: 'ep-lg' }, long), el('span', { class: 'ep-sm' }, short));
    const seg = ui.segmented({
      id: 'ep-zoom-to', value: 'ours',
      options: [{ value: 'ours', label: two('This page', 'Ours') }, { value: 'gpt2', label: 'GPT-2' }, { value: 'gpt3', label: 'GPT-3' }, { value: 'frontier', label: two('Frontier', 'Today') }, { value: 'lineup', label: two('Line up', 'Line up') }],
      onChange: (v) => go(v),
    });
    const slider = ui.slider({
      id: 'ep-zoom', label: 'Frame holds', min: Z_MIN, max: Z_MAX, step: 0.01, value: TARGET.ours,
      format: (v) => '≈ ' + fmtShort(Math.pow(10, v)),
      onInput: (v) => manual(v),
    });
    seg.el.setAttribute('aria-label', 'Fly to a model');
    slider.el.classList.add('ep-zslider');
    hold.appendChild(slider.el);
    fig.appendChild(el('div', { class: 'ep-zbar' }, seg.el));
    // Arrow keys move a quarter of a decade (a native step of 0.01 would take ~800 presses),
    // Page Up/Down a whole power of ten, Home/End the ends of the axis.
    slider.input.addEventListener('keydown', (ev) => {
      const d = { ArrowRight: 0.25, ArrowUp: 0.25, ArrowLeft: -0.25, ArrowDown: -0.25, PageUp: 1, PageDown: -1 }[ev.key];
      const base = st.tw ? st.tw.z1 : st.zf;
      const v = d != null ? base + d : ev.key === 'Home' ? Z_MIN : ev.key === 'End' ? Z_MAX : null;
      if (v == null) return;
      ev.preventDefault();
      nudge(v);
    });
    slider.input.addEventListener('focus', () => { st.axisFocus = true; });
    slider.input.addEventListener('blur', () => { st.axisFocus = false; });
    fig.appendChild(el('figcaption', { class: 'ep-zcap', html: 'A square’s <strong>area</strong> is proportional to its parameter count; the zoom is logarithmic. Stripe widths come from each model’s published shape through one formula, which reproduces the published totals (LayerNorm is counted with the sublayer it feeds). The frontier haze is a range of public estimates.' }));

    // ---- steps
    const STEPS = [
      {
        key: 'ours', label: '1 · This page’s loom',
        html: [
          `Most of this page’s live figures came from one small transformer: <strong>${fmtInt(Po)} parameters</strong> in ${ours.L} layers, with d<sub>model</sub> = ${ours.d}, a ${ours.V}-word vocabulary and a ${ours.T}-token context. It trained on about 30 million tokens of toy English (a 12-million-token corpus, read about two and a half times), in 25 minutes on four CPU cores.`,
          `On the stage it is a square of cloth whose area is its parameter count. The stripes are where those parameters sit, in order: the <span class="dye-saffron">word table</span>, the <span class="dye-lichen">position table</span>, one <span class="dye-woad">attention</span> and one <span class="dye-verdigris">MLP</span> stripe per layer, then the <span class="dye-cochineal">unembedding</span>.`,
          `If every parameter were a 1 mm stitch, this cloth would be ${stitch(Po)} across. About the size of a cushion cover.`,
        ],
      },
      {
        key: 'gpt2', label: '2 · GPT-2, 2019',
        html: [
          `GPT-2 came in four sizes, from <strong>124 million</strong> to <strong>1.5 billion</strong> parameters: 12 to 48 layers, d<sub>model</sub> from 768 to 1,600, a vocabulary of 50,257 subword tokens and a 1,024-token context.`,
          `The smallest is already ${fmtInt(ratioG2)} times our model. As 1 mm stitches the two cloths here are ${stitch(Pg2s)} and ${stitch(Pg2xl)} across. Our loom is the gold speck in the corner.`,
          `The wide <span class="dye-saffron">saffron</span> stripe is GPT-2 small’s word table, ${pct(wordShare('gpt2s'), 0)} of the model. GPT-2 reuses that table to unembed, so it has no separate unembedding stripe.`,
        ],
      },
      {
        key: 'gpt3', label: '3 · GPT-3, 2020',
        html: [
          `GPT-3 has <strong>175 billion</strong> parameters in 96 layers, with d<sub>model</sub> = 12,288, 96 heads per layer and a 2,048-token context. It was trained on about 300 billion tokens.`,
          `That is about ${fmtWords(Math.round(ratioG3 / 1e5) * 1e5).replace('.0', '')} times our parameters and ${fmtInt(Math.round(ratioTok / 1000) * 1000)} times our training text. Its 96 layers are the fine blue-green pinstripes. As stitches it would be a square ${stitch(Pg3)} across, with our whole model a cushion cover in one corner.`,
        ],
      },
      {
        key: 'frontier', label: '4 · Today’s frontier',
        html: [
          `Most of today’s largest models do not publish their sizes. Public estimates run from hundreds of billions to several trillion parameters, with contexts of hundreds of thousands to millions of tokens, trained on trillions of tokens. The haze marks that range of guesses.`,
          `Some open-weight models do publish. DeepSeek-V3 (December 2024) has <strong>671 billion</strong> parameters and is a <span class="term">mixture of experts</span>: a router sends each token through a few of many expert MLPs, so only <strong>37 billion</strong> parameters work on any one token. The glowing patches add up to 37/671 of its cloth; which experts light up changes from token to token.`,
        ],
      },
      {
        key: 'lineup', label: '5 · The same weave',
        html: [
          `Now shrink every cloth to the same size. All four start with the word and position tables, then add an attention stripe and an MLP stripe for each layer. The number of pairs grows: ${ours.L}, 12, 48, 96.`,
          `In each one the <span class="dye-verdigris">MLP</span> stripe is twice the <span class="dye-woad">attention</span> stripe (${['ours', 'gpt2s', 'gpt2xl', 'gpt3'].map((k) => mlpRatio(k).toFixed(2) + '×').join(', ')}). An MLP holds about 8·d<sup>2</sup> weights (d → 4d → d); attention holds 4·d<sup>2</sup> (W<sub>Q</sub>, W<sub>K</sub>, W<sub>V</sub>, W<sub>O</sub>).`,
          `What scale changes most is the word table’s share: ${pct(wordShare('gpt2s'), 0)} of GPT-2 small, ${pct(wordShare('gpt3'), 2)} of GPT-3. Big models are mostly blocks.`,
        ],
      },
    ];
    const stepEls = STEPS.map((s) => el('div', { class: 'step' }, el('div', { class: 'step-label' }, s.label), s.html.map((h) => el('p', { html: h }))));
    prose.append(...stepEls);

    // ---- state
    const st = { zf: TARGET.ours, tw: null, lineup: 0, lineupT: 0, key: 'ours', last: now(), sparks: [], patch: new Float32Array(144), patchOn: [], patchClock: 9, drag: false, cardKey: '', L: null };
    const fst = {};
    FABS.forEach((f) => { fst[f.key] = { a: f.key === 'ours' ? 1 : 0, rev: f.key === 'ours' ? 1 : 0 }; });
    const frontSt = { a: 0 };

    function go(key) {
      st.key = key;
      seg.set(key);
      if (key === 'lineup') { st.lineupT = 1; st.tw = null; return; }
      st.lineupT = 0;
      const z1 = TARGET[key];
      if (AM.reducedMotion) { st.zf = z1; st.tw = null; slider.set(z1); return; }
      const dz = Math.abs(z1 - st.zf);
      st.tw = { z0: st.zf, z1, t0: now(), dur: Math.min(2.9, 0.7 + 0.42 * dz) };
    }
    function nearestKey(z) { return z < 6.6 ? 'ours' : z < 10.2 ? 'gpt2' : z < 11.55 ? 'gpt3' : 'frontier'; }
    function manual(v) {
      st.tw = null; st.lineupT = 0; st.zf = clamp(v, Z_MIN, Z_MAX);
      st.key = nearestKey(st.zf); seg.set(st.key);
    }
    /** A short glide to v (keyboard). */
    function nudge(v) {
      const z1 = clamp(v, Z_MIN, Z_MAX);
      if (AM.reducedMotion) { manual(z1); slider.set(z1); return; }
      st.lineupT = 0;
      st.tw = { z0: st.zf, z1, t0: now(), dur: 0.35 };
      st.key = nearestKey(z1); seg.set(st.key);
    }

    ctx.steps(stepEls, (i) => go(STEPS[i].key));

    // ---- layout
    function layout(w, h) {
      const phone = w < 560;
      const padL = phone ? 12 : 20, padR = phone ? 12 : 20;
      const top = phone ? 30 : 46, axisH = phone ? 42 : 56;
      const ax = padL, ay = h - axisH;
      const R = Math.max(80, phone ? Math.min(w - padL - padR, ay - top) : Math.min(ay - top, (w - padL - padR - 26 - 214) / FIT));
      const cardX = ax + R * FIT + 26;
      const cardW = w - padR - cardX;
      return { w, h, phone, padL, padR, top, ax, ay, R, cardX, cardW, showCard: !phone && cardW >= 180, axY: h - (phone ? 25 : 31) };
    }

    // ---- woven texture (a 2×2 plain-weave tile, scaled with the zoom)
    const TILE = 12; // tile drawn at 12 px per cell, scaled down to the on-screen pitch
    const tile = document.createElement('canvas');
    tile.width = tile.height = TILE * 2;
    (function paintTile() {
      const g = tile.getContext('2d');
      g.fillStyle = 'rgba(0,0,0,0.3)';
      g.fillRect(0, 0, TILE * 2, TILE * 2);
      const cell = (cx, cy, horiz) => {
        const x = cx * TILE, y = cy * TILE, i = 1.2;
        if (horiz) {
          g.clearRect(x, y + i, TILE, TILE - 2 * i);
          const grd = g.createLinearGradient(0, y + i, 0, y + TILE - i);
          grd.addColorStop(0, 'rgba(255,248,230,0.26)');
          grd.addColorStop(0.35, 'rgba(255,248,230,0.06)');
          grd.addColorStop(0.7, 'rgba(0,0,0,0.04)');
          grd.addColorStop(1, 'rgba(0,0,0,0.26)');
          g.fillStyle = grd; g.fillRect(x, y + i, TILE, TILE - 2 * i);
        } else {
          g.clearRect(x + i, y, TILE - 2 * i, TILE);
          const grd = g.createLinearGradient(x + i, 0, x + TILE - i, 0);
          grd.addColorStop(0, 'rgba(255,248,230,0.2)');
          grd.addColorStop(0.4, 'rgba(255,248,230,0.04)');
          grd.addColorStop(1, 'rgba(0,0,0,0.28)');
          g.fillStyle = grd; g.fillRect(x + i, y, TILE - 2 * i, TILE);
        }
      };
      cell(0, 0, true); cell(1, 0, false); cell(0, 1, false); cell(1, 1, true);
    })();
    let pat = null;
    const canPat = typeof DOMMatrix !== 'undefined';
    function weavePattern(g) {
      if (!pat) { pat = g.createPattern(tile, 'repeat'); if (pat && !pat.setTransform) pat = null; }
      return pat;
    }
    /** Weave scale: one cell holds 10^n parameters; pitch on screen stays in [3.6, 11.4) px. */
    function weaveLevels(k) {
      const lp0 = LG(3.6), q = LG(k);
      const n = Math.ceil(2 * (lp0 - q));
      const u = (n / 2 + q - lp0) / 0.5;
      return { n, u, pitch: Math.pow(10, n / 2) * k };
    }
    function overlay(g, x, y, w, h, lv, alpha, anchorX, anchorY) {
      if (!canPat || w <= 0 || h <= 0) return;
      const p = weavePattern(g);
      if (!p) return;
      const a1 = M.smoothstep(0, 0.45, lv.u);
      const fillAt = (pitch, a) => {
        if (a <= 0.01) return;
        const s = pitch / TILE;
        p.setTransform(new DOMMatrix([s, 0, 0, s, anchorX, anchorY]));
        g.globalAlpha = alpha * a;
        g.fillStyle = p;
        g.fillRect(x, y, w, h);
      };
      fillAt(lv.pitch, a1 * 0.62);
      fillAt(lv.pitch * Math.sqrt(10), (1 - a1) * 0.62);
      g.globalAlpha = 1;
    }

    // ---- MoE patches: a 12×12 grid, 8 lit at once, each 37/671/8 of the cloth's area
    const PN = 12, PON = 8, PSIDE = Math.sqrt(DSV3.active / DSV3.P / PON);
    // GPT-3's cloth covers the bottom-left √(175/671) ≈ 0.51 of DeepSeek-V3's side, so experts
    // light up only in cells that stay visible (otherwise fewer than 8 would show).
    const PCOVER = Math.ceil(Math.sqrt(byKey.gpt3.P / DSV3.P) * PN);
    const PFREE = [];
    for (let i = 0; i < PN * PN; i++) if (!((i % PN) < PCOVER && Math.floor(i / PN) >= PN - PCOVER)) PFREE.push(i);
    function pickPatches() {
      const s = new Set();
      while (s.size < PON) s.add(PFREE[Math.floor(Math.random() * PFREE.length)]);
      st.patchOn = Array.from(s);
    }

    // ---- per-frame update
    function update(dt, t) {
      if (st.tw) {
        const u = clamp((t - st.tw.t0) / st.tw.dur);
        st.zf = M.lerp(st.tw.z0, st.tw.z1, M.ease.inOut(u));
        if (u >= 1) st.tw = null;
        slider.set(st.zf);
      }
      const vt = `frame holds about ${fmtWords(Math.pow(10, st.zf))} parameters`;
      if (vt !== st.vt) { st.vt = vt; slider.input.setAttribute('aria-valuetext', vt); }
      const k = 1 - Math.exp(-dt * (AM.reducedMotion ? 60 : 2.4));
      st.lineup += (st.lineupT - st.lineup) * k;
      if (Math.abs(st.lineupT - st.lineup) < 0.002) st.lineup = st.lineupT;
      const kA = 1 - Math.exp(-dt * (AM.reducedMotion ? 60 : 5));
      // a cloth appears as soon as the camera is heading for its scale, so a big one
      // weaves itself upward past the frame while the camera pulls back to show it whole
      const zSeen = Math.max(st.zf, st.tw ? st.tw.z1 : st.zf);
      for (const f of FABS) {
        const s = fst[f.key];
        let vis;
        if (st.lineupT > 0.5) vis = f.lineup != null;
        else vis = f.key === 'ours' || zSeen >= LG(f.P) - (s.a > 0.5 ? 0.5 : 0.35);
        if (vis && s.rev === 0) s.dur = st.tw ? clamp(st.tw.dur * 0.9 + 0.2, 1.3, 2.8) : 1.4;
        s.a += ((vis ? 1 : 0) - s.a) * kA;
        if (vis && s.rev < 1) s.rev = AM.reducedMotion ? 1 : Math.min(1, s.rev + dt / (s.dur || 1.4));
        if (!vis && s.a < 0.02) s.rev = 0;
      }
      const fv = st.lineupT < 0.5 && zSeen >= LG(FRONT_LO) - 0.15;
      frontSt.a += ((fv ? 1 : 0) - frontSt.a) * kA;
      // experts: re-pick every 0.8 s
      st.nFlash = Math.max(0, (st.nFlash || 0) - dt * 1.6);
      st.patchClock += dt;
      if (st.patchClock >= 0.8) { st.patchClock = 0; pickPatches(); }
      const kp = 1 - Math.exp(-dt * (AM.reducedMotion ? 60 : 10));
      for (let i = 0; i < st.patch.length; i++) st.patch[i] += ((st.patchOn.includes(i) ? 1 : 0) - st.patch[i]) * kp;
    }

    // ---- drawing
    function isFocus(f) {
      if (st.lineup > 0.5) return true;
      const k = nearestKey(st.zf);
      return (k === 'ours' && f.key === 'ours') || (k === 'gpt2' && (f.key === 'gpt2s' || f.key === 'gpt2xl')) || (k === 'gpt3' && f.key === 'gpt3') || (k === 'frontier' && f.key === 'dsv3');
    }

    function drawFabric(g, L, f, x, y, S, alpha, lv, anchorX, anchorY, t, focus) {
      if (S < 0.35 || alpha < 0.01) return null;
      const s = fst[f.key];
      const revE = M.ease.inOut(s.rev);
      const yF = y + S * (1 - revE); // weaving front: the cloth is woven from the bottom up
      const vx0 = Math.max(x, -2), vx1 = Math.min(x + S, L.w + 2);
      const vy0 = Math.max(yF, -2), vy1 = Math.min(y + S, L.h + 2);
      if (vx1 <= vx0 || vy1 <= vy0) return null;
      g.save();
      g.globalAlpha = alpha;
      if (S > 8) { g.fillStyle = 'rgba(3,5,12,0.5)'; g.fillRect(vx0 + 4, vy0 + 5, vx1 - vx0, Math.max(0, Math.min(vy1 + 5, L.h) - vy0 - 5)); }
      g.beginPath(); g.rect(vx0, vy0, vx1 - vx0, vy1 - vy0); g.clip();
      g.fillStyle = AM.col.ink2;
      g.fillRect(vx0, vy0, vx1 - vx0, vy1 - vy0);
      if (f.stripes) {
        const cols = focus ? f.hi : f.lo;
        const dpr = cv.dpr || 1;
        const snap = f.pairFrac * S * dpr >= 2.6 ? (v) => Math.round(v * dpr) / dpr : (v) => v;
        for (let i = 0; i < f.stripes.length; i++) {
          const sp = f.stripes[i];
          const sx0 = snap(x + sp.f0 * S), sx1 = snap(x + sp.f1 * S);
          if (sx1 < vx0 || sx0 > vx1 || sx1 <= sx0) continue;
          g.fillStyle = cols[i];
          g.fillRect(Math.max(sx0, vx0), vy0, Math.min(sx1, vx1) - Math.max(sx0, vx0), vy1 - vy0);
        }
      } else {
        g.fillStyle = AM.mix(AM.col.ink2, AM.col.linen, focus ? 0.17 : 0.1);
        g.fillRect(vx0, vy0, vx1 - vx0, vy1 - vy0);
      }
      if (f.active) {
        const cs = S / PN, ps = S * PSIDE;
        for (let i = 0; i < st.patch.length; i++) {
          const v = st.patch[i];
          if (v < 0.02) continue;
          const cx = x + ((i % PN) + 0.5) * cs, cy = y + (Math.floor(i / PN) + 0.5) * cs;
          g.globalAlpha = alpha * v * 0.22;
          g.fillStyle = AM.dye.weld;
          g.fillRect(cx - ps / 2 - 3, cy - ps / 2 - 3, ps + 6, ps + 6);
          g.globalAlpha = alpha * v;
          g.fillStyle = AM.mix(AM.col.ink2, AM.dye.weld, 0.78);
          g.fillRect(cx - ps / 2, cy - ps / 2, ps, ps);
        }
        g.globalAlpha = alpha;
      }
      overlay(g, vx0, vy0, vx1 - vx0, vy1 - vy0, lv, alpha, anchorX, anchorY);
      if (focus && S > 40 && !AM.reducedMotion) {
        // light catching the silk: a slow diagonal sheen
        const ph = ((t * 0.1 + (f.lineup || 0) * 0.23) % 1.5) - 0.25;
        const sx = x + ph * S * 1.6 - S * 0.3;
        const sh = g.createLinearGradient(sx, y, sx + S * 0.32, y + S * 0.32);
        sh.addColorStop(0, 'rgba(255,244,214,0)');
        sh.addColorStop(0.5, 'rgba(255,244,214,0.075)');
        sh.addColorStop(1, 'rgba(255,244,214,0)');
        g.globalAlpha = alpha;
        g.globalCompositeOperation = 'lighter';
        g.fillStyle = sh;
        g.fillRect(vx0, vy0, vx1 - vx0, vy1 - vy0);
        g.globalCompositeOperation = 'source-over';
      }
      g.restore();
      // selvedge
      g.save();
      g.globalAlpha = alpha;
      if (focus && S > 30) {
        g.strokeStyle = AM.rgba(AM.dye.weld, 0.16);
        g.lineWidth = 5;
        g.strokeRect(x - 1.5, yF - 1.5, S + 3, y + S - yF + 3);
      }
      g.strokeStyle = AM.rgba(AM.col.linen, focus ? 0.7 : 0.32);
      g.lineWidth = 1;
      g.strokeRect(x + 0.5, yF + 0.5, Math.max(0, S - 1), Math.max(0, y + S - yF - 1));
      g.restore();
      if (s.rev > 0 && s.rev < 1) drawFront(g, L, vx0, vx1, yF, t, alpha);
      return { x, y, S };
    }

    function drawFront(g, L, x0, x1, y, t, alpha) {
      if (y < -4 || y > L.h + 4) return;
      g.save();
      g.globalAlpha = alpha;
      const grd = g.createLinearGradient(0, y - 7, 0, y + 7);
      grd.addColorStop(0, AM.rgba(AM.dye.weld, 0));
      grd.addColorStop(0.5, AM.rgba(AM.dye.weld, 0.28));
      grd.addColorStop(1, AM.rgba(AM.dye.weld, 0));
      g.fillStyle = grd;
      g.fillRect(x0, y - 7, x1 - x0, 14);
      g.strokeStyle = AM.rgba('#fff4d6', 0.85);
      g.lineWidth = 1.1;
      g.beginPath(); g.moveTo(x0, y + 0.5); g.lineTo(x1, y + 0.5); g.stroke();
      const ph = (t * 1.5) % 2, u = ph < 1 ? ph : 2 - ph;
      const sx = x0 + (x1 - x0) * M.ease.inOut(u);
      g.restore();
      bead(g, sx, y, 2.8, AM.dye.weld, alpha);
      if (!AM.reducedMotion && st.sparks.length < 90) {
        for (let i = 0; i < 2; i++) st.sparks.push({ x: sx, y, vx: (Math.random() - 0.5) * 70, vy: -20 - Math.random() * 60, life: 0, max: 0.4 + Math.random() * 0.5 });
      }
    }

    function drawFrontier(g, L, side, a, t) {
      if (a < 0.01) return;
      const lo = side(FRONT_LO), hi = side(FRONT_HI);
      const ax = L.ax, ay = L.ay;
      g.save();
      g.globalAlpha = a;
      // the band between the two estimates, fading on past the upper one
      const out = hi * 1.45;
      const grd = g.createRadialGradient(ax, ay, lo * 0.9, ax, ay, out);
      grd.addColorStop(0, AM.rgba(AM.dye.lichen, 0.2));
      grd.addColorStop(0.5, AM.rgba(AM.dye.lichen, 0.11));
      grd.addColorStop(0.8, AM.rgba(AM.dye.lichen, 0.04));
      grd.addColorStop(1, AM.rgba(AM.dye.lichen, 0));
      g.beginPath();
      g.rect(ax, ay - out, out, out);
      g.rect(ax, ay - lo, lo, lo);
      g.fillStyle = grd;
      g.fill('evenodd');
      // drifting motes give the haze some life
      for (let i = 0; i < 26; i++) {
        const r1 = M.rng(i + 7)(), r2 = M.rng(i * 31 + 3)();
        const s = M.lerp(lo, hi * 1.25, r1);
        const along = (r2 + t * 0.012 * (0.5 + r1)) % 1;
        const px = along < 0.5 ? ax + s * along * 2 : ax + s;
        const py = along < 0.5 ? ay - s : ay - s + s * (along - 0.5) * 2;
        bead(g, px, py, 0.9 + r1, AM.dye.lichen, 0.35 * a);
      }
      g.restore();
    }
    /** Labels for the estimate band, drawn after the cloths so nothing covers them. */
    function drawFrontierLabels(g, L, side, a) {
      if (a < 0.02) return;
      const lo = side(FRONT_LO), hi = side(FRONT_HI);
      const ax = L.ax, ay = L.ay, fs = L.phone ? 9 : 10, col = AM.mix(AM.dye.lichen, AM.col.linen, 0.25);
      // the two estimate outlines go over the cloths, so the 300-billion line stays visible on DeepSeek-V3
      g.save();
      g.globalAlpha = a;
      g.setLineDash([4, 4]);
      g.lineWidth = 1.2;
      g.strokeStyle = 'rgba(214,200,250,0.85)'; // lichen, lightened so it reads over the cloths
      g.strokeRect(ax + 0.5, ay - lo + 0.5, lo, lo);
      g.lineWidth = 1;
      g.setLineDash([3, 5]);
      g.strokeStyle = AM.rgba(AM.dye.lichen, 0.5);
      g.strokeRect(ax + 0.5, ay - hi + 0.5, hi, hi);
      g.restore();
      if (hi > L.w * 2.2 || lo < 50) return;
      if (ax + hi < L.w - 8 && ay - hi > 4) {
        halo(g, L.phone ? 'frontier models' : 'frontier models · sizes undisclosed', ax + 10, ay - hi + 13, { size: fs, color: col, alpha: a, ls: '0.04em' });
        if (ay - hi > L.top - 6) halo(g, '≈ 3 trillion', ax + hi, ay - hi - 9, { size: fs, color: AM.dye.lichen, align: 'right', alpha: a });
      }
      // on a phone the corner of the 300-billion square sits under DeepSeek-V3's label, so the
      // number goes just inside the square's top edge instead
      if (ay - lo > 24) {
        if (L.phone) halo(g, '≈ 300 billion', ax + 6, ay - lo + 10, { size: fs, color: 'rgb(214,200,250)', alpha: a });
        else halo(g, '≈ 300 billion', ax + lo + 6, ay - lo + 2, { size: fs, color: AM.dye.lichen, alpha: a });
      }
    }

    function drawLabels(g, L, rects, fade) {
      if (fade < 0.02) return;
      for (const f of FABS) {
        const r = rects[f.key];
        if (!r) continue;
        const s = fst[f.key];
        const a = fade * s.a * M.smoothstep(0.7, 1, s.rev);
        if (a < 0.02) continue;
        const { x, y, S } = r;
        const fsN = L.phone ? 11.5 : 13, fsS = L.phone ? 9 : 10;
        if (S >= (L.phone ? 96 : 118) && x + S <= L.w + 1 && y >= -1) {
          halo(g, f.label, x + S - 10, y + 17, { size: fsN, role: 'body', weight: 600, align: 'right', alpha: a, color: f.key === 'ours' ? AM.dye.weld : AM.col.linen });
          g.font = AM.font(fsS, 'mono');
          const sub = f.subShort && g.measureText(f.sub).width > S - 20 ? f.subShort : f.sub;
          halo(g, sub, x + S - 10, y + 33, { size: fsS, align: 'right', alpha: a * 0.9, color: AM.col.linenDim });
        } else if (S >= 18 && S < 118 && f.key !== 'ours') {
          const tx = x + S + 7, ty = y + 7;
          if (tx + 70 < L.w) {
            halo(g, f.label, tx, ty, { size: fsN - 1, role: 'body', weight: 600, alpha: a });
            halo(g, f.sub, tx, ty + 14, { size: fsS, alpha: a * 0.9, color: AM.col.linenDim });
          }
        }
      }
    }

    /** "This page" marker: once our cloth is a speck, a pulsing ring keeps it findable. */
    function drawMarker(g, L, r, fade, t) {
      if (!r || fade < 0.02 || r.S >= 26) return;
      const a = fade * clamp((26 - r.S) / 10);
      const cx = L.ax + r.S / 2, cy = L.ay - r.S / 2;
      const ph = (t * 0.7) % 1;
      g.save();
      g.globalAlpha = a * (1 - ph);
      g.strokeStyle = AM.dye.weld;
      g.lineWidth = 1.2;
      g.beginPath(); g.arc(cx, cy, 5 + ph * 16, 0, Math.PI * 2); g.stroke();
      g.globalAlpha = a;
      g.beginPath(); g.arc(cx, cy, 5.5, 0, Math.PI * 2); g.stroke();
      g.restore();
      bead(g, cx, cy, 1.8, AM.dye.weld, a);
      g.save();
      g.globalAlpha = a;
      g.strokeStyle = AM.rgba(AM.dye.weld, 0.7);
      g.beginPath(); g.moveTo(cx + 4, cy + 4); g.lineTo(cx + 12, L.ay + 12); g.lineTo(cx + 20, L.ay + 12); g.stroke();
      g.restore();
      halo(g, `this page · ${fmtInt(byKey.ours.P)}`, cx + 24, L.ay + 12, { size: L.phone ? 9 : 10, color: AM.dye.weld, alpha: a });
    }

    function drawWeaveLabel(g, L, lv, fade) {
      if (fade < 0.02) return;
      const n = lv.u > 0.22 ? lv.n : lv.n + 1;
      const y = L.phone ? 17 : 22;
      g.save();
      g.globalAlpha = fade;
      const sq = L.phone ? 8 : 9;
      g.fillStyle = AM.mix(AM.col.ink2, AM.col.linen, 0.25);
      g.fillRect(L.ax, y - sq / 2, sq, sq);
      g.strokeStyle = AM.rgba(AM.col.linen, 0.6);
      g.strokeRect(L.ax + 0.5, y - sq / 2 + 0.5, sq - 1, sq - 1);
      g.restore();
      if (st.lastN != null && n !== st.lastN && !AM.reducedMotion) st.nFlash = 1;
      st.lastN = n;
      const fs = L.phone ? 9.5 : 10.5;
      const a = '1 woven cell ≈ ', b = powWords(Math.max(0, n)), c = ' parameters';
      g.save();
      g.font = AM.font(fs, 'mono');
      const wa = g.measureText(a).width, wb = g.measureText(b).width;
      g.restore();
      const x0 = L.ax + sq + 8;
      if (st.nFlash > 0.01) {
        g.save();
        g.globalAlpha = fade * st.nFlash * 0.5;
        g.fillStyle = AM.dye.weld;
        g.shadowColor = AM.dye.weld;
        g.shadowBlur = 14;
        D.roundRect(g, x0 + wa - 4, y - fs * 0.75, wb + 8, fs * 1.5, 4);
        g.fill();
        g.restore();
      }
      halo(g, a, x0, y, { size: fs, color: AM.col.linenDim, alpha: fade });
      halo(g, b, x0 + wa, y, { size: fs, color: AM.dye.weld, alpha: fade });
      halo(g, c, x0 + wa + wb, y, { size: fs, color: AM.col.linenDim, alpha: fade });
    }

    function axisX(L, lg) { const x0 = L.ax + 4, x1 = L.w - L.padR - 4; return x0 + (lg - AX_MIN) / (AX_MAX - AX_MIN) * (x1 - x0); }
    function drawAxis(g, L, fade) {
      if (fade < 0.02) return;
      const y = L.axY, x0 = L.ax + 4, x1 = L.w - L.padR - 4;
      g.save();
      g.globalAlpha = fade;
      // frontier estimate range
      const fx0 = axisX(L, LG(FRONT_LO)), fx1 = axisX(L, LG(FRONT_HI));
      const grd = g.createLinearGradient(fx0, 0, fx1 + 26, 0);
      grd.addColorStop(0, AM.rgba(AM.dye.lichen, 0.55));
      grd.addColorStop(0.75, AM.rgba(AM.dye.lichen, 0.45));
      grd.addColorStop(1, AM.rgba(AM.dye.lichen, 0));
      g.fillStyle = grd;
      g.fillRect(fx0, y - 2.5, fx1 - fx0 + 26, 5);
      g.strokeStyle = AM.rgba(AM.col.linen, st.drag ? 0.5 : 0.24);
      g.lineWidth = 1;
      g.beginPath(); g.moveTo(x0, y + 0.5); g.lineTo(x1, y + 0.5); g.stroke();
      const names = ['100K', '1M', '10M', '100M', '1B', '10B', '100B', '1T', '10T'];
      for (let p = AX_MIN; p <= AX_MAX; p++) {
        const x = axisX(L, p);
        g.strokeStyle = AM.rgba(AM.col.linen, 0.3);
        g.beginPath(); g.moveTo(Math.round(x) + 0.5, y - 3); g.lineTo(Math.round(x) + 0.5, y + 3); g.stroke();
        if (!L.phone || p % 2 === 1) D.text(g, names[p - AX_MIN], x, y + 15, { size: L.phone ? 8.5 : 9, role: 'mono', color: AM.col.mist, align: 'center', baseline: 'middle' });
      }
      for (const f of FABS) {
        const x = axisX(L, LG(f.P));
        g.fillStyle = f.key === 'ours' ? AM.dye.weld : f.key === 'dsv3' ? AM.mix(AM.col.linen, AM.dye.lichen, 0.4) : AM.col.linen;
        g.beginPath(); g.arc(x, y + 0.5, 2.4, 0, Math.PI * 2); g.fill();
      }
      if (!L.phone && !st.axisFocus) D.text(g, 'parameters · log scale · drag', x1, y - 10, { size: 9, role: 'mono', color: AM.col.mist, align: 'right', baseline: 'middle', alpha: 0.85 });
      g.restore();
      const vx = axisX(L, clamp(st.zf, AX_MIN, AX_MAX));
      g.save();
      g.globalAlpha = fade;
      g.strokeStyle = AM.rgba(AM.dye.weld, 0.6);
      g.beginPath(); g.moveTo(vx + 0.5, y - 9); g.lineTo(vx + 0.5, y + 6); g.stroke();
      g.restore();
      bead(g, vx, y + 0.5, 3.4, AM.dye.weld, fade);
      if (st.axisFocus && fade > 0.5) {
        g.save();
        g.strokeStyle = AM.dye.weld;
        g.lineWidth = 1.5;
        g.beginPath(); g.arc(vx, y + 0.5, 9, 0, Math.PI * 2); g.stroke();
        g.restore();
        halo(g, '← → to zoom', vx + (vx > L.w - 110 ? -14 : 14), y - 11, { size: 9, color: AM.dye.weld, align: vx > L.w - 110 ? 'right' : 'left' });
      }
    }

    /** 2 × 2 grid of equal squares, labels underneath. */
    function lineupSlots(L) {
      const labH = L.phone ? 50 : 62, gapX = L.phone ? 16 : 28, gapY = L.phone ? 8 : 12, top = L.phone ? 30 : 42;
      const s = Math.min((L.w - L.padL - L.padR - gapX) / 2, (L.h - top - 8 - 2 * labH - gapY) / 2);
      const x0 = (L.w - (2 * s + gapX)) / 2;
      const y0 = top + Math.max(0, (L.h - top - 8 - (2 * (s + labH) + gapY)) / 2);
      return LINEUP.map((f, i) => ({ x: x0 + (i % 2) * (s + gapX), y: y0 + Math.floor(i / 2) * (s + labH + gapY), s }));
    }

    /** A magnifier on cloths whose layers are too fine to see: shows the stripe pairs × mag. */
    function drawLoupe(g, L, f, sl, a) {
      const pairW = f.pairFrac * sl.s;
      if (pairW >= 6 || a < 0.02) return;
      const rL = Math.max(L.phone ? 19 : 24, sl.s * 0.2);
      const mag = Math.max(2, Math.round(16 / pairW));
      const cx = sl.x + sl.s - rL - 6, cy = sl.y + sl.s - rL - 6;
      const fx = sl.x + sl.s * 0.55, fy = sl.y + sl.s * 0.3, win = rL / mag;
      g.save();
      g.globalAlpha = a;
      // the sampled patch, and a thread to the lens
      g.strokeStyle = AM.rgba('#fff4d6', 0.75);
      g.lineWidth = 1;
      g.strokeRect(fx - win, fy - win, win * 2, win * 2);
      g.strokeStyle = AM.rgba('#fff4d6', 0.35);
      g.beginPath(); g.moveTo(fx, fy + win); g.lineTo(cx - rL * 0.5, cy - rL * 0.82); g.stroke();
      // lens
      g.save();
      g.beginPath(); g.arc(cx, cy, rL, 0, Math.PI * 2); g.clip();
      g.fillStyle = AM.col.ink2;
      g.fillRect(cx - rL, cy - rL, rL * 2, rL * 2);
      for (let i = 0; i < f.stripes.length; i++) {
        const sp = f.stripes[i];
        const X0 = cx + (sl.x + sp.f0 * sl.s - fx) * mag, X1 = cx + (sl.x + sp.f1 * sl.s - fx) * mag;
        if (X1 < cx - rL || X0 > cx + rL) continue;
        g.fillStyle = f.hi[i];
        g.fillRect(Math.round(X0), cy - rL, Math.max(1, Math.round(X1) - Math.round(X0)), rL * 2);
      }
      overlay(g, cx - rL, cy - rL, rL * 2, rL * 2, { n: 0, u: 0.7, pitch: 5 }, 1, cx, cy);
      const sh = g.createRadialGradient(cx - rL * 0.4, cy - rL * 0.5, 0, cx, cy, rL);
      sh.addColorStop(0, 'rgba(255,248,230,0.16)');
      sh.addColorStop(0.6, 'rgba(255,248,230,0)');
      sh.addColorStop(1, 'rgba(0,0,0,0.3)');
      g.globalAlpha = a;
      g.fillStyle = sh;
      g.fillRect(cx - rL, cy - rL, rL * 2, rL * 2);
      g.restore();
      g.strokeStyle = AM.rgba(AM.dye.weld, 0.22);
      g.lineWidth = 5;
      g.beginPath(); g.arc(cx, cy, rL + 1.5, 0, Math.PI * 2); g.stroke();
      g.strokeStyle = AM.rgba('#fff4d6', 0.85);
      g.lineWidth = 1.4;
      g.beginPath(); g.arc(cx, cy, rL, 0, Math.PI * 2); g.stroke();
      g.restore();
      halo(g, `×${mag}`, cx + rL * 0.74, cy - rL * 0.92, { size: L.phone ? 8.5 : 9.5, color: '#fff4d6', alpha: a });
    }

    function drawLineupLabels(g, L, slots, e) {
      const a = M.smoothstep(0.55, 1, e);
      if (a < 0.02) return;
      halo(g, 'every cloth scaled to the same size', L.w / 2, L.phone ? 15 : 22, { size: L.phone ? 9 : 10.5, align: 'center', color: AM.col.linenDim, alpha: a, ls: '0.05em' });
      LINEUP.forEach((f, i) => {
        const sl = slots[i], s = f.shape, cc = DATA.counts[f.key];
        drawLoupe(g, L, f, sl, a * M.smoothstep(0.75, 1, fst[f.key].rev)); // not over cloth still being woven
        const x = sl.x, y = sl.y + sl.s + (L.phone ? 12 : 15);
        const fs = L.phone ? 8.5 : 10, lh = L.phone ? 12 : 15;
        halo(g, f.label, x, y, { size: L.phone ? 10.5 : 12.5, role: 'body', weight: 600, alpha: a, color: f.key === 'ours' ? AM.dye.weld : AM.col.linen });
        if (L.phone) halo(g, `${s.L} layers`, x, y + lh, { size: fs, alpha: a, color: AM.col.linenDim });
        else halo(g, `${s.L} layers`, x + sl.s, y, { size: fs, alpha: a, color: AM.col.linenDim, align: 'right' });
        const r = (cc.mlp / cc.attn).toFixed(2);
        if (L.phone) {
          halo(g, `MLP ${r}× attn`, x, y + 2 * lh, { size: fs, alpha: a, color: AM.dye.verdigris });
        } else {
          halo(g, `MLP = ${r} × attention`, x, y + lh, { size: fs, alpha: a, color: AM.dye.verdigris });
          halo(g, `word table ${pct(cc.embed / cc.total, cc.embed / cc.total < 0.01 ? 2 : 1)}`, x, y + 2 * lh, { size: fs, alpha: a, color: AM.dye.saffron });
        }
      });
    }

    function drawSparks(g, dt) {
      if (!st.sparks.length) return;
      g.save();
      g.globalCompositeOperation = 'lighter';
      for (let i = st.sparks.length - 1; i >= 0; i--) {
        const p = st.sparks[i];
        p.life += dt;
        if (p.life >= p.max) { st.sparks.splice(i, 1); continue; }
        p.x += p.vx * dt; p.y += p.vy * dt; p.vy += 90 * dt;
        bead(g, p.x, p.y, 0.9, AM.dye.weld, 0.8 * (1 - p.life / p.max));
      }
      g.restore();
    }

    function updateCard(L) {
      const key = !L.showCard || st.lineup > 0.4 ? '' : nearestKey(st.zf);
      if (key === st.cardKey) {
        if (key) { card.style.left = L.cardX + 'px'; card.style.top = L.top + 'px'; card.style.width = L.cardW + 'px'; }
        return;
      }
      st.cardKey = key;
      if (!key) { card.hidden = true; return; }
      card.hidden = false;
      card.style.left = L.cardX + 'px'; card.style.top = L.top + 'px'; card.style.width = L.cardW + 'px';
      card.innerHTML = cardHTML(key);
    }
    const crow = (k, v) => `<div class="ep-zc-row"><span>${k}</span><b>${v}</b></div>`;
    function cardHTML(key) {
      if (key === 'ours') {
        return `<div class="ep-zc-name">This page<span>tinyworld</span></div><div class="ep-zc-big">${fmtInt(Po)}</div><div class="ep-zc-unit">parameters</div><div class="ep-zc-rows">`
          + crow('layers · heads', `${ours.L} · ${ours.H}`) + crow('d_model', ours.d) + crow('vocabulary', `${ours.V} words`) + crow('context', `${ours.T} tokens`)
          + crow('data', '≈ 30M tokens') + crow('1 mm stitches', stitch(Po)) + '</div>';
      }
      if (key === 'gpt2') {
        return `<div class="ep-zc-name">GPT-2<span>OpenAI · 2019</span></div><div class="ep-zc-big">124M–1.5B</div><div class="ep-zc-unit">parameters</div><div class="ep-zc-rows">`
          + crow('layers', '12 – 48') + crow('d_model', '768 – 1,600') + crow('vocabulary', '50,257 tokens') + crow('context', '1,024 tokens')
          + crow('1 mm stitches', `${stitch(Pg2s)} – ${stitch(Pg2xl)}`.replace(' m –', ' –')) + '</div>';
      }
      if (key === 'gpt3') {
        return `<div class="ep-zc-name">GPT-3<span>OpenAI · 2020</span></div><div class="ep-zc-big">175B</div><div class="ep-zc-unit">parameters</div><div class="ep-zc-rows">`
          + crow('layers · heads', '96 · 96') + crow('d_model', '12,288') + crow('context', '2,048 tokens') + crow('data', '≈ 300B tokens')
          + crow('1 mm stitches', stitch(Pg3)) + '</div>';
      }
      return `<div class="ep-zc-name">The frontier<span>today · undisclosed</span></div><div class="ep-zc-big is-words">hundreds of billions to trillions</div><div class="ep-zc-unit">parameters, public estimates</div><div class="ep-zc-rows">`
        + crow('context', '100K – millions') + crow('data', 'trillions of tokens') + '</div>'
        + `<div class="ep-zc-note"><b>DeepSeek-V3</b> (open weights, Dec 2024): 671B parameters, 37B active per token, 14.8T training tokens, 128K context.</div>`;
    }

    function drawStage() {
      const { g, w, h } = cv;
      if (!w) return;
      const t = now();
      const dt = Math.min(0.1, Math.max(0, t - st.last));
      st.last = t;
      update(dt, t);
      const L = layout(w, h);
      st.L = L;
      cv.clear();
      D.weave(g, 0, 0, w, h, { step: 7, alpha: 0.018 });
      const e = M.ease.inOut(clamp(st.lineup));
      const zoomFade = 1 - e;
      const k = FIT * L.R / Math.pow(10, st.zf / 2); // px per √parameter
      const lv = weaveLevels(k);
      const lvFixed = { n: 0, u: 0.7, pitch: L.phone ? 4.6 : 5.4 };
      const side = (P) => Math.sqrt(P) * k;
      drawFrontier(g, L, side, frontSt.a * zoomFade, t);
      const slots = e > 0.001 ? lineupSlots(L) : null;
      const rects = {};
      for (const f of DESC) {
        const s = fst[f.key];
        const inLine = f.lineup != null;
        let S = side(f.P), x = L.ax, y = L.ay - S, anchorX = L.ax, anchorY = L.ay;
        const alpha = s.a * (inLine ? 1 : zoomFade);
        if (alpha < 0.01) continue;
        let level = lv;
        if (inLine && slots) {
          const sl = slots[f.lineup];
          const S0 = clamp(S, 0.6, L.R * 4);
          S = Math.exp(M.lerp(Math.log(S0), Math.log(sl.s), e));
          const bx = M.lerp(L.ax, sl.x, e), by = M.lerp(L.ay, sl.y + sl.s, e);
          x = bx; y = by - S; anchorX = bx; anchorY = by;
          if (e > 0.5) level = lvFixed;
        }
        const r = drawFabric(g, L, f, x, y, S, alpha, level, anchorX, anchorY, t, isFocus(f));
        if (r) rects[f.key] = r;
        else if (f.key === 'ours') rects.ours = { x, y, S };
      }
      if (!rects.ours) rects.ours = { x: L.ax, y: L.ay - side(byKey.ours.P), S: side(byKey.ours.P) };
      drawFrontierLabels(g, L, side, frontSt.a * zoomFade);
      drawLabels(g, L, rects, zoomFade);
      drawMarker(g, L, rects.ours, zoomFade, t);
      drawWeaveLabel(g, L, lv, zoomFade);
      drawAxis(g, L, zoomFade);
      if (slots) drawLineupLabels(g, L, slots, e);
      drawSparks(g, dt);
      updateCard(L);
    }

    // ---- axis drag (free play)
    const setFromX = (x) => {
      const L = st.L;
      if (!L) return;
      const x0 = L.ax + 4, x1 = L.w - L.padR - 4;
      const v = AX_MIN + ((x - x0) / (x1 - x0)) * (AX_MAX - AX_MIN);
      manual(v);
      slider.set(st.zf);
    };
    cv.canvas.addEventListener('pointerdown', (ev) => {
      const p = cv.pointer(ev);
      if (!st.L || p.y < st.L.axY - 24 || st.lineup > 0.5) return;
      st.drag = true;
      try { cv.canvas.setPointerCapture(ev.pointerId); } catch (e) { /* not capturable */ }
      setFromX(p.x);
    });
    cv.canvas.addEventListener('pointermove', (ev) => {
      const p = cv.pointer(ev);
      if (st.drag) setFromX(p.x);
      else cv.canvas.style.cursor = st.L && p.y >= st.L.axY - 24 && st.lineup < 0.5 ? 'ew-resize' : 'default';
    });
    const endDrag = () => { st.drag = false; };
    cv.canvas.addEventListener('pointerup', endDrag);
    cv.canvas.addEventListener('pointercancel', endDrag);

    cv.onResize(() => { st.cardKey = '__'; drawStage(); });
    const seen = inView(cv.wrap);
    ctx.loop(() => { if (seen.on) drawStage(); });
    drawStage();
  }

  // ================================================================== figure 2: the whole machine
  const SAMPLE_T = 0.8;
  const PROMPTS = ['the queen', 'the king lost his', 'alice gave', 'the dog walked to the river with', 'the capital of japan is', 'the witch said green cup yellow cup . the parrot said', 'the keys near the old door'];

  function buildPoster(body, ctx, DATA) {
    const el = ctx.el, ui = AM.ui;
    const lm = DATA.lm;
    const sec = el('div', { class: 'ep-poster-sec' });
    body.appendChild(sec);
    sec.appendChild(el('div', { class: 'prose ep-sechead' },
      el('h3', {}, 'The whole machine on one page'),
      el('p', { html: 'Here is everything this page built, in the order data flows through it, from the bottom up as in the 2017 paper’s figure. Each part links back to its chapter. The switch relabels the parts with the sizes of real models; the shape of the diagram never changes.' }),
      lm ? el('p', { html: 'It is also running. On every pass, the real model on this page reads the words so far and samples the next one, which loops back to the input.' }) : null,
    ));

    const fig = el('figure', { class: 'fig ep-poster' });
    sec.appendChild(fig);
    fig.appendChild(el('div', { class: 'fig-top' },
      el('span', { class: 'fig-title' }, 'Decoder-only transformer · read it bottom to top'),
      el('span', { class: 'ep-badges' }, ui.badge('illustration'), lm ? ui.badge('live', 'Live model · the sentence') : null)));
    let preset = 'ours';
    const two = (long, short) => el('span', {}, el('span', { class: 'ep-lg' }, long), el('span', { class: 'ep-sm' }, short));
    const seg = ui.segmented({
      id: 'ep-preset', label: 'Label the parts with', value: 'ours',
      options: [{ value: 'ours', label: two('This page', 'Ours') }, { value: 'gpt2s', label: 'GPT-2 small' }, { value: 'gpt3', label: 'GPT-3' }, { value: 'llama', label: two('Llama 3.1 405B', 'Llama 405B') }],
      onChange: (v) => setPreset(v),
    });
    const totalEl = el('div', { class: 'ep-total', 'aria-live': 'polite' });
    fig.appendChild(el('div', { class: 'ep-pctl' }, seg.el, totalEl));
    const pbody = el('div', { class: 'ep-pbody' });
    fig.appendChild(pbody);
    const flow = el('ol', { class: 'ep-flow', 'aria-label': 'The parts of a decoder-only transformer, in the order data flows through them' });
    pbody.appendChild(flow);
    const fly = el('div', { class: 'ep-fly', hidden: true, 'aria-hidden': 'true' });
    pbody.appendChild(fly);

    const rows = {};
    const COL = { tokens: AM.col.linenDim, embed: COMP.embed, attn: COMP.attn, mlp: COMP.mlp, block: COMP.stream, out: COMP.unembed, sample: AM.dye.weld, train: COMP.loss };
    function addRow(k, boxContent, links, opts = {}) {
      const box = el('div', { class: `ep-box ep-box--${k}` }, boxContent);
      box.style.setProperty('--c', COL[k]);
      const text = el('p', {});
      const note = el('div', { class: 'ep-note' }, el('div', { class: 'ep-links', html: links.map(chLink).join('') }), text);
      const node = el('div', { class: 'ep-node' }, box);
      const li = el('li', { class: 'ep-row' + (opts.cls ? ' ' + opts.cls : ''), 'data-k': k }, node, note);
      // mouse and pen: hover; touch: a tap toggles (a touch pointer "leaves" as soon as it lifts)
      li.addEventListener('pointerenter', (e) => { if (e.pointerType !== 'touch') setHot(k); });
      li.addEventListener('pointerleave', (e) => { if (e.pointerType !== 'touch') setHot(null); });
      li.addEventListener('pointerup', (e) => { if (e.pointerType === 'touch' && !(e.target.closest && e.target.closest('a'))) setHot(st.hot === k ? null : k); });
      li.addEventListener('focusin', () => setHot(k));
      li.addEventListener('focusout', () => setHot(null));
      flow.appendChild(li);
      rows[k] = { li, box, note, text, node };
      return rows[k];
    }
    const head = (title, tag) => el('div', { class: 'ep-box-head' }, tag || null, el('span', { class: 'ep-box-title' }, title));

    // tokens (live chips)
    const chips = el('div', { class: 'ep-chips', 'aria-live': 'off' });
    const tokMeta = el('div', { class: 'ep-box-meta' });
    addRow('tokens', [el('div', { class: 'ep-box-head' }, el('span', { class: 'ep-box-title' }, 'Tokens'), lm ? el('span', { class: 'ep-live' }, 'live input') : null), chips, tokMeta], ['tokens']);
    // embedding + position
    const embMeta = el('div', { class: 'ep-box-meta' });
    const posTitle = el('span', { class: 'ep-box-title' }, '+ Position');
    const posTag = el('span', { class: 'ep-tag is-new', hidden: true }, 'RoPE');
    const posMeta = el('div', { class: 'ep-box-meta' });
    const embBox = el('div', { class: 'ep-box' }, head('Token embedding'), embMeta);
    embBox.style.setProperty('--c', COMP.embed);
    const posBox = el('div', { class: 'ep-box' }, el('div', { class: 'ep-box-head' }, posTitle, posTag), posMeta);
    posBox.style.setProperty('--c', COMP.pos);
    const embRow = addRow('embed', el('div', { class: 'ep-pair' }, embBox, posBox), ['embed', 'position']);
    embRow.box.classList.add('ep-box--bare');
    // attention
    const attnNorm = el('span', { class: 'ep-tag' }, 'LayerNorm');
    const attnTitle = el('span', { class: 'ep-box-title' }, 'Masked multi-head attention');
    const attnTag = el('span', { class: 'ep-tag is-new', hidden: true }, 'GQA');
    const attnMeta = el('div', { class: 'ep-box-meta' });
    addRow('attn', [el('div', { class: 'ep-box-head' }, attnNorm, attnTitle, attnTag), attnMeta], ['attention', 'heads']);
    // MLP
    const mlpNorm = el('span', { class: 'ep-tag' }, 'LayerNorm');
    const mlpTitle = el('span', { class: 'ep-box-title' }, 'MLP');
    const mlpTag = el('span', { class: 'ep-tag is-new', hidden: true }, 'SwiGLU');
    const mlpMeta = el('div', { class: 'ep-box-meta' });
    addRow('mlp', [el('div', { class: 'ep-box-head' }, mlpNorm, mlpTitle, mlpTag), mlpMeta], ['mlp']);
    // block × N
    const xn = el('span', { class: 'ep-xn' }, '× 3');
    const layerLab = el('span', { class: 'ep-layer' }, 'blocks');
    addRow('block', el('div', { class: 'ep-blocktag' }, xn, layerLab), ['residual', 'stack'], { cls: 'is-gap-lg' });
    // final norm + unembedding
    const outNorm = el('span', { class: 'ep-tag' }, 'LayerNorm');
    const outMeta = el('div', { class: 'ep-box-meta' });
    addRow('out', [el('div', { class: 'ep-box-head' }, outNorm, el('span', { class: 'ep-box-title' }, 'Unembedding → logits')), outMeta], ['predict'], { cls: 'is-gap-lg' });
    // softmax + sample + next token (live)
    const bars = el('div', { class: 'ep-bars' });
    const barEls = [0, 1, 2].map(() => {
      const fill = el('span', { class: 'ep-bar-fill' });
      const tok = el('span', {}, '·'), p = el('span', {}, '');
      const row = el('div', { class: 'ep-bar' }, tok, el('span', { class: 'ep-bar-track' }, fill), p);
      bars.appendChild(row);
      return { row, tok, p, fill };
    });
    const nextWord = el('span', { class: 'ep-next-word' }, '…');
    const nextLab = el('span', { class: 'ep-next-lab' }, 'next token');
    const sampHead = el('div', { class: 'ep-box-head' }, el('span', { class: 'ep-box-title' }, 'Softmax → sample'), lm ? el('span', { class: 'ep-tag', title: 'The bars show the probabilities the next token is drawn from, after dividing the logits by 0.8' }, `temperature ${SAMPLE_T}`) : null);
    addRow('sample', [sampHead, lm
      ? el('div', { class: 'ep-samp' }, bars, el('div', { class: 'ep-next' }, nextWord, nextLab))
      : el('div', { class: 'ep-box-meta' }, el('span', {}, 'probabilities → one token drawn'))], ['predict']);
    // training
    addRow('train', [head('Training loss'), el('div', { class: 'ep-box-meta', html: '<b>−log p(the real next token)</b>' })], ['train', 'lab'], { cls: 'is-gap-lg' });

    // one live, verified head behaviour for the attention note (name from AM_NOTES when present)
    let headW = null;
    if (lm) {
      try {
        const ids = lm.encode('the queen opened the door because').ids;
        const r = lm.run(ids, { capture: true });
        if (r.tokens[1] === 'queen' && r.attn[0] && r.attn[0][3]) headW = r.attn[0][3][ids.length - 1][1];
      } catch (e) { headW = null; }
    }
    const notesHead = window.AM_NOTES && window.AM_NOTES.heads && window.AM_NOTES.heads.L0H3 ? window.AM_NOTES.heads.L0H3.name : null;
    const headLine = headW == null ? '' : ` <span class="ep-more">In “the queen opened the door because”, head L0H3${notesHead ? ' (' + String(notesHead) + ')' : ''} puts ${headW.toFixed(2)} of its attention from “because” on “queen”.</span>`;

    // ---- labels per preset
    function setPreset(key, first = false) {
      preset = key;
      const s = DATA.shapes[key], c = DATA.counts[key];
      const isOurs = key === 'ours', rope = s.pos === 'rope', gqa = s.kvH < s.H, swi = s.mlp === 'swiglu', tied = s.tied;
      const normName = s.norm === 'rms' ? 'RMSNorm' : 'LayerNorm';
      const m = (left, right) => `<span>${left}</span>${right != null ? `<span class="ep-pp">${right}</span>` : ''}`;
      const perL = (n) => `${fmtP(n)} × ${s.L}`;
      const changed = [];
      const set = (node, html) => { if (node.innerHTML !== html) { node.innerHTML = html; changed.push(node); } };
      set(tokMeta, m(isOurs ? `vocabulary ${s.V} words` : `vocabulary ${fmtInt(s.V)} tokens`, `context ${fmtInt(s.T)}`));
      set(embMeta, m(`${fmtInt(s.V)} × ${fmtInt(s.d)}`, fmtP(c.embed)));
      posTitle.textContent = rope ? 'Position' : '+ Position';
      posTag.hidden = !rope;
      posBox.classList.toggle('is-dashed', rope);
      set(posMeta, rope ? m('no table · inside attention') : m(`${fmtInt(s.T)} × ${fmtInt(s.d)}`, fmtP(c.pos)));
      attnNorm.textContent = normName; mlpNorm.textContent = normName; outNorm.textContent = normName;
      attnNorm.classList.toggle('is-new', s.norm === 'rms'); mlpNorm.classList.toggle('is-new', s.norm === 'rms'); outNorm.classList.toggle('is-new', s.norm === 'rms');
      attnTitle.textContent = gqa ? 'Grouped-query attention' : 'Masked multi-head attention';
      attnTag.hidden = !gqa;
      set(attnMeta, m(`${s.H} heads × ${c.dh}${gqa ? ` · ${s.kvH} K/V heads` : ''}`, perL(c.attn)));
      mlpTitle.textContent = 'MLP';
      mlpTag.hidden = !swi;
      set(mlpMeta, m(`${fmtInt(s.d)} → ${fmtInt(s.F)} → ${fmtInt(s.d)} · ${swi ? 'Swish gate' : 'GELU'}`, perL(c.mlp)));
      set(xn, `× ${s.L}`);
      set(outMeta, m(`${fmtInt(s.d)} → ${fmtInt(s.V)} logits`, tied ? (key === 'gpt3' ? 'tied (assumed)' : 'tied to the word table') : fmtP(c.unembed + c.final)));
      // notes
      const R = rows;
      set(R.tokens.text, isOurs
        ? `Text is cut into tokens, and each becomes an integer id. This model reads whole words from a list of ${s.V}; up to ${s.T} fit in its context.`
        : `Text is cut into tokens, and each becomes an integer id: ${fmtInt(s.V)} subword pieces, up to ${fmtInt(s.T)} at a time.`);
      set(R.embed.text, `Each id picks one row of a learned table: a vector of d<sub>model</sub> = ${fmtInt(s.d)} numbers. ` + (rope
        ? 'There is no position table. RoPE rotates each query and key by its position, inside attention.'
        : 'A learned position vector is added, so the model knows the order.'));
      set(R.attn.text, `The only step where tokens read from one another: <span class="math">softmax(QK<sup>T</sup>/√d<sub>k</sub> + M)·V</span>, in ${s.H} heads at once.` + (gqa ? ` Here the ${s.H} query heads share ${s.kvH} key/value heads.` : '') + (isOurs ? headLine : ` <span class="ep-more">Its ${normName} comes first.</span>`));
      const mr = c.mlp / c.attn;
      set(R.mlp.text, `Each token on its own: widen from ${fmtInt(s.d)} to ${fmtInt(s.F)} numbers, apply ${swi ? 'a Swish-gated product' : 'GELU'}, project back. ` + `<span class="ep-more">It holds ${Math.abs(mr - 2) < 0.06 ? 'about twice' : fix(mr, 1) + ' times'} the attention’s parameters.</span>`);
      set(R.block.text, `The gold thread is the <strong>residual stream</strong>, ${fmtInt(s.d)} numbers per token. Each sublayer reads it and adds its result back. The block repeats ${s.L} times, each layer building on the last.`);
      set(R.out.text, `A final ${normName}, then one score (logit) for every entry in the vocabulary, ${fmtInt(s.V)} of them.` + (tied ? (key === 'gpt3' ? ' We assume it reuses its word table for this, as GPT-2 does; its paper says it keeps GPT-2’s design.' : ' This model reuses its word table to do it.') : ''));
      set(R.sample.text, 'Softmax turns logits into probabilities, and one token is drawn. Then it is appended and the whole stack runs again, once per token.' + ' <span class="ep-more">Real systems keep a KV cache so earlier positions are not recomputed.</span>');
      set(R.train.text, 'Where all the numbers come from: compare the prediction with the real next token, send the error back down through every box, and nudge each parameter. Repeat for millions to trillions of tokens. <span class="ep-more">Hover or tap here to send the error down as red beads.</span>');
      // total
      let tot;
      if (isOurs) {
        const ok = DATA.shipped != null && DATA.shipped === c.total;
        tot = `<b>${fmtInt(c.total)}</b> parameters by the formula` + (DATA.shipped != null ? (ok ? ' · <span class="ep-ok">equals the shipped weights ✓</span>' : ` · shipped weights: ${fmtInt(DATA.shipped)}`) : '');
      } else {
        tot = `<b>${c.total < 1e9 ? fmtInt(c.total) : fmtWords(c.total)}</b> parameters by the formula · published: ${s.pub}`;
      }
      set(totalEl, tot);
      if (!first && !AM.reducedMotion) {
        changed.forEach((n) => { n.classList.remove('ep-flash'); void n.offsetWidth; n.classList.add('ep-flash'); });
      }
      requestAnimationFrame(() => remeasure());
    }

    // ---- geometry + static threads
    const cv = ctx.canvas(pbody, {
      label: 'Diagram threads: a gold residual stream rises from the embeddings to the unembedding; each attention and MLP box draws a copy off the stream and adds its result back at a plus sign; a dashed thread loops the sampled token back to the input.',
      height: () => Math.max(200, pbody.offsetHeight),
      maxDpr: 1.5,
    });
    cv.wrap.classList.add('ep-pcv');
    pbody.insertBefore(cv.wrap, pbody.firstChild);
    cv.canvas.removeAttribute('role');
    cv.canvas.setAttribute('aria-hidden', 'true');
    const cache = document.createElement('canvas');
    let G = null;
    const st = { hot: null, beads: [], spawn: 0, passY: null, live: null };

    function rel(r, pr) { return { x: r.left - pr.left, y: r.top - pr.top, w: r.width, h: r.height, r: r.right - pr.left, b: r.bottom - pr.top, cx: r.left - pr.left + r.width / 2, cy: r.top - pr.top + r.height / 2 }; }
    function remeasure() {
      if (!cv.w) return;
      const pr = pbody.getBoundingClientRect();
      const R = {};
      for (const k in rows) R[k] = { box: rel(rows[k].box.getBoundingClientRect(), pr), note: rel(rows[k].note.getBoundingClientRect(), pr), node: rel(rows[k].node.getBoundingClientRect(), pr) };
      R.embed.box = rel(embBox.getBoundingClientRect(), pr);
      R.embed.pos = rel(posBox.getBoundingClientRect(), pr);
      const stacked = R.tokens.note.y >= R.tokens.box.b - 2;
      const colX = R.tokens.node.x;
      G = {
        R, stacked, w: cv.w, h: cv.h, colX, colR: R.tokens.node.r,
        streamX: colX + (stacked ? 23 : 36), loopX: colX + (stacked ? 6 : 12),
      };
      G.yB = R.embed.box.y - 16; // stream starts above the embedding boxes
      G.yT = R.out.box.b + 16;   // and ends below the unembedding
      const sub = (k) => {
        const b = R[k].box;
        return {
          off: sv(G.streamX, b.b + 10, b.x + 1, b.b - 12),
          ret: sv(b.x + 1, b.y + 12, G.streamX, b.y - 10),
          plusY: b.y - 10,
        };
      };
      G.sub = { attn: sub('attn'), mlp: sub('mlp') };
      // loop path: next token → left margin → input
      const sb = R.sample.box, tb = R.tokens.box;
      G.loop = [
        { x: sb.x - 2, y: sb.cy }, { x: G.loopX + 10, y: sb.cy }, { x: G.loopX, y: sb.cy + 10 },
        { x: G.loopX, y: tb.cy - 10 }, { x: G.loopX + 10, y: tb.cy }, { x: tb.x - 2, y: tb.cy },
      ];
      renderStatic();
    }

    function renderStatic() {
      if (!G || !cv.w) return;
      const { w, h, dpr } = cv;
      cache.width = Math.round(w * dpr);
      cache.height = Math.round(h * dpr);
      const g = cache.getContext('2d');
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      g.clearRect(0, 0, w, h);
      const R = G.R, hot = st.hot;
      const dimA = (k) => (hot && hot !== k ? 0.4 : 1);
      const hotA = (k) => (hot === k ? 1 : 0);
      // block bracket
      const bt = R.block.node.y - 8, bb = R.attn.box.b + 22;
      const bx0 = G.colX + (G.stacked ? 12 : 18);
      g.save();
      g.setLineDash([3, 5]);
      g.strokeStyle = AM.rgba(AM.dye.weld, hot === 'block' ? 0.7 : 0.26);
      g.lineWidth = 1;
      if (bb - bt > 24 && G.colR - bx0 > 24) { D.roundRect(g, bx0, bt, G.colR - bx0 - 1, bb - bt, 12); g.stroke(); }
      g.restore();
      // loop path (append and repeat)
      g.save();
      g.setLineDash([2, 5]);
      g.strokeStyle = AM.rgba(AM.col.linen, hot === 'sample' ? 0.6 : 0.24);
      g.lineWidth = 1.2;
      g.beginPath();
      const lp = G.loop;
      g.moveTo(lp[0].x, lp[0].y);
      g.lineTo(lp[1].x, lp[1].y);
      g.quadraticCurveTo(G.loopX, lp[0].y, lp[2].x, lp[2].y);
      g.lineTo(lp[3].x, lp[3].y);
      g.quadraticCurveTo(G.loopX, lp[5].y, lp[4].x, lp[4].y);
      g.lineTo(lp[5].x, lp[5].y);
      g.stroke();
      g.restore();
      if (!G.stacked) {
        g.save();
        g.translate(G.loopX - 1, (R.tokens.box.cy + R.embed.box.cy) / 2);
        g.rotate(-Math.PI / 2);
        D.text(g, 'append · repeat', 0, -5, { size: 8.5, role: 'mono', color: AM.col.mist, align: 'center', baseline: 'middle', alpha: 0.9 });
        g.restore();
      }
      // vertical connectors outside the stream
      const tb = R.tokens.box, eb = R.embed.box, pb = R.embed.pos, ob = R.out.box, sb = R.sample.box, trb = R.train.box;
      if (!G.stacked) { // on phones the notes sit between the boxes, so these would cross the text
        silk(g, line(tb.x + tb.w * 0.3, tb.y, eb.cx, eb.b), { color: AM.col.linen, width: 1, alpha: 0.32 * dimA('tokens') });
        silk(g, line(tb.x + tb.w * 0.72, tb.y, pb.cx, pb.b), { color: AM.col.linen, width: 1, alpha: 0.2 * dimA('tokens') });
        silk(g, line(sb.x + 40, ob.y, sb.x + 40, sb.b), { color: COMP.unembed, width: 1.2, alpha: 0.5 * dimA('out') });
        silk(g, line(sb.x + 40, sb.y, trb.x + 40, trb.b), { color: COMP.loss, width: 1.2, alpha: 0.6 * dimA('train'), dash: [3, 4] });
      }
      // embeddings into the stream, stream into the unembedding
      const sx = G.streamX;
      silk(g, sv(eb.x + 18, eb.y, sx, G.yB), { color: COMP.stream, width: 2.2, alpha: 0.85 });
      if (pb.x > eb.r && pb.y < eb.b) silk(g, sv(pb.x + 18, pb.y, sx, G.yB), { color: COMP.pos, width: 1.3, alpha: 0.55 * dimA('embed') });
      silk(g, sv(sx, G.yT, ob.x + 18, ob.b), { color: COMP.stream, width: 2.2, alpha: 0.85 });
      // the residual stream
      const sw = hot === 'block' ? 3.2 : 2.4;
      silk(g, line(sx, G.yB, sx, G.yT), { color: COMP.stream, width: sw, alpha: hot && !['block', 'attn', 'mlp'].includes(hot) ? 0.55 : 1 });
      // sublayer branches
      for (const k of ['attn', 'mlp']) {
        const b = G.sub[k], col = COMP[k];
        const a = (0.6 + 0.4 * hotA(k)) * dimA(k);
        silk(g, b.off, { color: col, width: 1.5, alpha: a });
        silk(g, b.ret, { color: col, width: 1.5, alpha: a });
        g.save();
        g.fillStyle = AM.col.ink;
        g.strokeStyle = AM.rgba(COMP.stream, 0.95);
        g.lineWidth = 1.3;
        g.beginPath(); g.arc(sx, b.plusY, 6.5, 0, Math.PI * 2); g.fill(); g.stroke();
        g.beginPath(); g.moveTo(sx - 3.5, b.plusY); g.lineTo(sx + 3.5, b.plusY); g.moveTo(sx, b.plusY - 3.5); g.lineTo(sx, b.plusY + 3.5); g.stroke();
        g.restore();
      }
      if (!G.stacked) {
        g.save();
        g.translate(sx - 11, (G.yB + G.yT) / 2 + 40);
        g.rotate(-Math.PI / 2);
        D.text(g, 'residual stream', 0, 0, { size: 8.5, role: 'mono', color: AM.dye.weld, align: 'center', baseline: 'middle', alpha: 0.85 });
        g.restore();
      }
      // leaders from boxes to notes
      if (!G.stacked) {
        for (const k in R) {
          const b = k === 'embed' ? R.embed.pos : R[k].box, n = R[k].note;
          const x0 = (k === 'block' ? G.colR : b.r) + 4, y0 = k === 'block' ? R.block.box.cy : b.cy;
          const x1 = n.x - 8, y1 = n.y + 9;
          const a = (hot === k ? 0.9 : 0.24) * (hot && hot !== k ? 0.6 : 1);
          silk(g, [x0, y0, x0 + (x1 - x0) * 0.5, y0, x0 + (x1 - x0) * 0.5, y1, x1, y1], { color: hot === k ? COL[k] : AM.col.linen, width: 1, alpha: a, sheen: false });
          g.fillStyle = AM.rgba(COL[k], hot === k ? 1 : 0.6);
          g.beginPath(); g.arc(x1 + 2, y1, 2.2, 0, Math.PI * 2); g.fill();
        }
      }
    }

    function setHot(k) {
      if (st.hot === k) return;
      st.hot = k;
      flow.classList.toggle('has-hot', !!k);
      for (const key in rows) rows[key].li.classList.toggle('is-hot', key === k);
      renderStatic();
    }

    // ---- the live sentence
    const live = {
      on: !!lm, phase: 'start', t: 0, ids: [], pi: 0, chosen: -1, top: [], layers: lm ? lm.config.n_layer : 3,
      lit: {}, flyPts: null,
    };
    // timeline of one pass: tokens → embed → (block × layers) → out → sample
    const SEG_IN = 0.45, SEG_LAYER = 0.55, SEG_OUT = 0.45;
    const passDur = () => SEG_IN + SEG_LAYER * live.layers + SEG_OUT;
    function renderChips(newLast) {
      chips.textContent = '';
      const toks = lm.decode(live.ids);
      toks.forEach((tk, i) => chips.appendChild(el('span', { class: 'ep-chip' + (newLast && i === toks.length - 1 ? ' is-new' : '') }, tk)));
    }
    // The distribution m.sample() draws from: p^(1/T), renormalised, with <pad>/<unk>
    // dropped. This equals softmax(logits / T), so the bars show what the draw really uses.
    const BANNED = lm ? new Set(['<pad>', '<unk>'].map((tk) => lm.vocab.indexOf(tk)).filter((i) => i >= 0)) : null;
    function tempered(row, T) {
      let mx = 0;
      for (let i = 0; i < row.length; i++) if (!BANNED.has(i) && row[i] > mx) mx = row[i];
      const out = new Float64Array(row.length), lmx = Math.log(mx);
      let z = 0;
      for (let i = 0; i < row.length; i++) if (!BANNED.has(i) && row[i] > 0) { out[i] = Math.exp((Math.log(row[i]) - lmx) / T); z += out[i]; }
      for (let i = 0; i < out.length; i++) out[i] /= z;
      return out;
    }
    function compute() {
      const r = lm.run(live.ids, {});
      const row = r.probs[live.ids.length - 1];
      live.q = tempered(row, SAMPLE_T);
      live.top = lm.topk(live.q, 3);
      live.chosen = lm.sample(row, { temperature: SAMPLE_T });
    }
    function newPrompt() {
      const p = PROMPTS[live.pi++ % PROMPTS.length];
      live.ids = lm.encode(p).ids;
      renderChips(false);
      nextWord.textContent = '…';
      nextLab.textContent = 'next token';
      barEls.forEach((b) => { b.fill.style.width = '0%'; b.tok.textContent = '·'; b.p.textContent = ''; b.row.classList.remove('is-pick'); });
      beginPass();
    }
    function beginPass() { compute(); live.phase = 'pass'; live.t = 0; live.lit = {}; }
    function reveal() {
      const pick = lm.vocab[live.chosen];
      live.top.forEach((tp, i) => {
        const b = barEls[i];
        b.tok.textContent = tp.token;
        b.p.textContent = probPct(tp.p);
        b.fill.style.width = Math.max(1.5, tp.p * 100) + '%';
        b.row.classList.toggle('is-pick', tp.id === live.chosen);
      });
      nextWord.textContent = pick;
      nextLab.textContent = 'next token · ' + probPct(live.q[live.chosen]);
    }
    function startFly() {
      if (AM.reducedMotion || !G) { append(); return; }
      const pr = pbody.getBoundingClientRect();
      const a = rel(nextWord.getBoundingClientRect(), pr);
      const last = chips.lastElementChild;
      const lr = last ? rel(last.getBoundingClientRect(), pr) : rel(chips.getBoundingClientRect(), pr);
      fly.textContent = lm.vocab[live.chosen];
      fly.hidden = false;
      const fw = fly.offsetWidth, fh = fly.offsetHeight;
      let ex = (last ? lr.r + 4 : lr.x), ey = lr.cy;
      if (ex + fw > G.R.tokens.box.r - 6) { ex = G.R.tokens.box.x + 17; ey = lr.cy + fh + 4; }
      const lp = G.loop;
      live.flyPts = [
        { x: a.cx, y: a.cy }, { x: lp[1].x, y: lp[0].y }, { x: G.loopX, y: lp[2].y },
        { x: G.loopX, y: lp[3].y }, { x: lp[4].x, y: ey }, { x: ex + fw / 2, y: ey },
      ];
      live.flyLen = live.flyPts.slice(1).reduce((s, p, i) => s + Math.hypot(p.x - live.flyPts[i].x, p.y - live.flyPts[i].y), 0);
      live.flySize = { fw, fh };
      live.phase = 'fly'; live.t = 0;
    }
    function flyAt(u) {
      const P = live.flyPts;
      let d = u * live.flyLen;
      for (let i = 1; i < P.length; i++) {
        const seg = Math.hypot(P[i].x - P[i - 1].x, P[i].y - P[i - 1].y);
        if (d <= seg || i === P.length - 1) { const f = seg ? Math.min(1, d / seg) : 1; return { x: M.lerp(P[i - 1].x, P[i].x, f), y: M.lerp(P[i - 1].y, P[i].y, f) }; }
        d -= seg;
      }
      return P[P.length - 1];
    }
    function append() {
      fly.hidden = true;
      live.ids = live.ids.concat(live.chosen);
      renderChips(true);
      const tok = lm.vocab[live.chosen];
      if (tok === '.' || live.ids.length >= 18) { live.phase = 'rest'; live.t = 0; }
      else beginPass();
    }
    /** Where the pass front is (y), and which layer it is in. */
    function frontAt(t) {
      const R = G.R;
      const yTok = R.tokens.box.cy, yIn = G.yB, yBlockTop = R.mlp.box.y - 18, yOut = R.out.box.cy, ySam = R.sample.box.cy;
      if (t < SEG_IN) return { y: M.lerp(yTok, yIn, M.ease.inOut(t / SEG_IN)), layer: -1 };
      t -= SEG_IN;
      if (t < SEG_LAYER * live.layers) {
        const l = Math.floor(t / SEG_LAYER), u = (t - l * SEG_LAYER) / SEG_LAYER;
        return { y: M.lerp(yIn, yBlockTop, M.ease.inOut(u)), layer: l };
      }
      t -= SEG_LAYER * live.layers;
      return { y: M.lerp(live.layers ? yBlockTop : yIn, ySam, M.ease.inOut(clamp(t / SEG_OUT))), layer: live.layers };
    }
    function lightRows(f, t) {
      const R = G.R;
      const order = ['tokens', 'embed', 'attn', 'mlp', 'out', 'sample'];
      for (const k of order) {
        const cy = k === 'embed' ? R.embed.box.cy : R[k].box.cy;
        const key = k + ':' + (k === 'attn' || k === 'mlp' ? f.layer : 0);
        if (f.y <= cy + 2 && !live.lit[key] && (k !== 'attn' && k !== 'mlp' || (f.layer >= 0 && f.layer < live.layers))) {
          live.lit[key] = t;
          rows[k].li.classList.add('is-lit');
          clearTimeout(rows[k]._lt);
          rows[k]._lt = setTimeout(() => rows[k].li.classList.remove('is-lit'), 520);
          if ((k === 'attn' || k === 'mlp') && !AM.reducedMotion) spawnBranch(k);
        }
      }
      const lay = f.layer >= 0 && f.layer < live.layers ? f.layer : -1;
      if (lay !== live.shownLayer) {
        live.shownLayer = lay;
        if (lay >= 0) layerLab.innerHTML = preset === 'ours' ? `blocks · now in layer <b>${lay + 1}</b> of ${live.layers}` : `blocks · our live model: layer <b>${lay + 1}</b> of ${live.layers}`;
        else layerLab.textContent = 'blocks';
      }
    }
    function tickLive(dt) {
      if (!live.on || !G) return;
      live.t += dt;
      switch (live.phase) {
        case 'start': newPrompt(); break;
        case 'pass': {
          const f = frontAt(Math.min(live.t, passDur()));
          st.passY = f.y;
          lightRows(f, live.t);
          if (!AM.reducedMotion && f.layer >= 0 && f.layer < live.layers && Math.random() < dt * 14) spawnStream(f.y);
          if (live.t >= passDur()) { st.passY = null; live.shownLayer = -1; layerLab.textContent = 'blocks'; reveal(); live.phase = 'show'; live.t = 0; }
          break;
        }
        case 'show': if (live.t >= 0.85) startFly(); break;
        case 'fly': {
          const u = clamp(live.t / 0.85);
          const p = flyAt(M.ease.inOut(u));
          fly.style.transform = `translate(${(p.x - live.flySize.fw / 2).toFixed(1)}px, ${(p.y - live.flySize.fh / 2).toFixed(1)}px)`;
          if (u >= 1) append();
          break;
        }
        case 'rest': if (live.t >= 2.6) newPrompt(); break;
        default: break;
      }
    }

    // ---- beads
    function spawnStream(y0) {
      if (st.beads.length > 90) return;
      st.beads.push({ kind: 'up', y: y0 == null ? G.yB : y0 + (Math.random() - 0.5) * 10, v: 60 + Math.random() * 40, ph: Math.random() * 6.28, col: COMP.stream, r: 1.1 + Math.random() * 0.7 });
    }
    function spawnBranch(k) {
      for (let i = 0; i < 3; i++) st.beads.push({ kind: 'branch', k, t: -i * 0.12, col: COMP[k], r: 1.3 });
    }
    function spawnDown() {
      if (st.beads.length > 90) return;
      st.beads.push({ kind: 'down', y: G.yT, v: 70 + Math.random() * 40, ph: Math.random() * 6.28, col: COMP.loss, r: 1.2 + Math.random() * 0.6 });
    }

    function drawPoster(dt, t) {
      if (!G || !cv.w) return;
      const g = cv.g;
      cv.clear();
      g.drawImage(cache, 0, 0, cv.w, cv.h);
      if (AM.reducedMotion) {
        if (st.passY != null) bead(g, G.streamX, clamp(st.passY, G.yT, G.yB), 3, COMP.stream, 1);
        return;
      }
      // idle trickle up the stream, and red beads down when training is in focus
      st.spawn += dt;
      if (st.spawn > (st.hot === 'train' ? 0.09 : 0.42)) { st.spawn = 0; if (st.hot === 'train') spawnDown(); else spawnStream(null); }
      // the pass front: a bright knot travelling up the stream
      if (st.passY != null) {
        const y = st.passY;
        const inStream = y <= G.yB + 1 && y >= G.yT - 1;
        bead(g, inStream ? G.streamX : G.R.tokens.box.x + G.R.tokens.box.w * 0.3, y, 4.2, COMP.stream, 0.95);
      }
      g.save();
      g.globalCompositeOperation = 'lighter';
      for (let i = st.beads.length - 1; i >= 0; i--) {
        const b = st.beads[i];
        if (b.kind === 'up' || b.kind === 'down') {
          b.y += (b.kind === 'up' ? -1 : 1) * b.v * dt;
          if ((b.kind === 'up' && b.y < G.yT) || (b.kind === 'down' && b.y > G.yB)) { st.beads.splice(i, 1); continue; }
          const x = G.streamX + Math.sin(b.ph + t * 3 + b.y * 0.05) * 1.6;
          const edge = Math.min(1, (b.y - G.yT) / 20, (G.yB - b.y) / 20);
          bead(g, x, b.y, b.r, b.col, 0.85 * clamp(edge));
        } else {
          b.t += dt;
          if (b.t < 0) continue;
          const S = G.sub[b.k];
          const T1 = 0.32, T2 = 0.62, T3 = 0.94;
          if (b.t >= T3) { st.beads.splice(i, 1); continue; }
          let p;
          if (b.t < T1) p = cub(S.off, b.t / T1);
          else if (b.t < T2) continue; // inside the box
          else p = cub(S.ret, (b.t - T2) / (T3 - T2));
          bead(g, p.x, p.y, b.r, b.col, 0.95);
        }
      }
      g.restore();
    }

    cv.onResize(() => { remeasure(); drawPoster(0, now()); });
    const seen = inView(pbody, '60px');
    let lastT = now();
    ctx.loop(() => {
      const t = now();
      const dt = Math.min(0.1, t - lastT);
      lastT = t;
      if (!seen.on) return;
      tickLive(AM.reducedMotion ? 0.5 : dt);
      drawPoster(dt, t);
    });
    setPreset('ours', true);
    if (lm) {
      // a complete first frame: the first prompt, its real next-token distribution and a draw
      live.ids = lm.encode(PROMPTS[0]).ids;
      live.pi = 1;
      renderChips(false);
      try { compute(); reveal(); live.phase = 'show'; live.t = 0; } catch (e) { live.phase = 'start'; }
    }
    else { chips.appendChild(el('span', { class: 'ep-chip' }, 'the')); chips.appendChild(el('span', { class: 'ep-chip' }, 'queen')); }

    fig.appendChild(el('figcaption', { class: 'ep-pcap', html: 'The diagram is an illustration. The parameter counts for each preset come from the formula used above (per layer × number of layers); GPT-3 is assumed to share its word table with the unembedding, as GPT-2 does. '
      + (lm ? `The running sentence always comes from this page’s model. Its bars are the top three probabilities after the logits are divided by the temperature, ${SAMPLE_T}: the distribution the next token is really drawn from. The real forward pass takes a few milliseconds; it is slowed down here so you can follow it. ` : '')
      + 'Tags in <span class="dye-cochineal">red-violet</span> mark refinements that GPT-2 and GPT-3 do not use.' }));
  }

  // ================================================================== refinements, recap, left out
  const REFINE = [
    { name: 'RMSNorm', dye: 'woad', ch: 'residual', text: 'LayerNorm without subtracting the mean: divide each vector by its root-mean-square, then multiply by a learned gain. Cheaper, and works about as well.' },
    { name: 'Rotary position embeddings', dye: 'lichen', ch: 'position', text: 'No position table. Pairs of numbers in each query and key are rotated by angles set by the token’s position, so q·k depends on how far apart two tokens are.' },
    { name: 'SwiGLU MLPs', dye: 'verdigris', ch: 'mlp', text: 'The MLP gets a gate: Swish(x·W₁) multiplies x·V element by element before the down-projection W₂.' },
    { name: 'Grouped-query attention', dye: 'woad', ch: 'heads', text: 'Many query heads share a few key/value heads (128 and 8 in Llama 3.1 405B), which shrinks the KV cache.' },
    { name: 'Mixture of experts', dye: 'verdigris', ch: 'mlp', text: 'Many expert MLPs per layer and a router that sends each token to a few of them. Total parameters grow much faster than the work per token.' },
    { name: 'The KV cache', dye: 'weld', ch: 'predict', text: 'While generating, the keys and values of earlier tokens are kept, so each new token computes one new row of attention instead of redoing every position.' },
    { name: 'Instruction tuning and RLHF', dye: 'madder', ch: 'train', text: 'After pre-training on raw text, fine-tuning on example dialogues and on human preference comparisons turns a text continuer into an assistant.' },
    { name: 'Tool use', dye: 'saffron', ch: 'predict', text: 'The model learns to write special tokens that call a search engine, a calculator or a code runner. The results are pasted into its context, and it carries on.' },
  ];

  function buildRefinements(body, ctx) {
    const el = ctx.el;
    const sec = el('div', { class: 'ep-refsec' });
    sec.appendChild(el('div', { class: 'ep-sechead' },
      el('h3', { class: 'ep-h3' }, 'What real models add'),
      el('p', { html: 'Production models keep the loop and tune the parts. Each change fits in a line. Pick <strong>Llama 3.1 405B</strong> in the diagram above to see the first four in place.' })));
    const grid = el('div', { class: 'ep-refine' });
    REFINE.forEach((r) => {
      const item = el('div', { class: 'ep-ref' },
        el('span', { class: 'ep-ref-dot', 'aria-hidden': 'true' }),
        el('div', { class: 'ep-ref-name' }, r.name),
        el('p', { html: r.text + ' ' + chLink(r.ch) }));
      item.style.setProperty('--c', AM.dye[r.dye]);
      grid.appendChild(item);
    });
    sec.appendChild(grid);
    body.appendChild(sec);
  }

  const RECAP = [
    { chs: ['tokens'], dye: 'weld', text: 'Text becomes a list of integer ids.' },
    { chs: ['embed', 'position'], dye: 'saffron', text: 'Each id becomes a learned vector, and its position is mixed in.' },
    { chs: ['attention'], dye: 'woad', text: 'Each token gathers from itself and earlier tokens, weighted by softmax(q·k/√d<sub>k</sub>).' },
    { chs: ['heads'], dye: 'lichen', text: 'Several heads do this side by side, each asking its own question.' },
    { chs: ['mlp'], dye: 'verdigris', text: 'An MLP then transforms every token on its own.' },
    { chs: ['residual', 'stack'], dye: 'weld', text: 'Each block reads the residual stream and adds to it, layer after layer.' },
    { chs: ['predict'], dye: 'cochineal', text: 'The last position’s vector becomes a score for every token in the vocabulary; one is sampled, appended, and the loop runs again.' },
    { chs: ['train', 'lab'], dye: 'madder', text: 'Every number in it was learned by predicting the next token, over and over.' },
  ];
  const LEFT_OUT = [
    '<strong>Real tokenizers at full size.</strong> Our model reads 138 whole words; real ones read subword pieces from vocabularies of tens of thousands to a few hundred thousand.',
    '<strong>Engineering at scale.</strong> Thousands of GPUs, one model split across many chips, low-precision arithmetic, weeks to months of training.',
    '<strong>Data.</strong> Collecting, filtering and deduplicating trillions of tokens, which matters as much as the architecture.',
    '<strong>Long contexts.</strong> FlashAttention, sliding windows and other tricks that make long inputs affordable.',
    '<strong>Other inputs.</strong> Images, audio and video, which multimodal models also turn into tokens.',
    '<strong>Other shapes.</strong> Encoder models such as BERT, and the original 2017 encoder–decoder built for translation.',
    '<strong>Safety and evaluation.</strong> Measuring what a model can do, and training it to decline harmful requests.',
    '<strong>How big models think.</strong> Interpretability at this scale is still largely an open question.',
  ];

  function buildRecap(body, ctx) {
    const el = ctx.el;
    const grid = el('div', { class: 'grid-2 ep-lists' });
    const ol = el('ol', { class: 'ep-threads' });
    RECAP.forEach((r) => {
      const li = el('li', { html: r.chs.map(chLink).join(' ') + ' ' + r.text });
      li.style.setProperty('--c', AM.dye[r.dye]);
      ol.appendChild(li);
    });
    const ul = el('ul', { class: 'ep-left' });
    LEFT_OUT.forEach((t) => ul.appendChild(el('li', { html: t })));
    grid.append(el('div', {}, el('h3', {}, 'The journey in eight threads'), ol), el('div', {}, el('h3', {}, 'What this page left out'), ul));
    body.appendChild(grid);
  }

  // ================================================================== coda: the finished cloth
  const CODA_SENTENCE = 'the queen opened the door because she was cold .';

  function buildCoda(body, ctx, DATA) {
    const el = ctx.el, ui = AM.ui;
    const lm = DATA.lm;
    // the motif: mean attention over every head of every layer (live), or a toy fallback
    let A = null, toks = CODA_SENTENCE.split(' ');
    if (lm) {
      try {
        const ids = lm.encode(CODA_SENTENCE).ids;
        const r = lm.run(ids, { capture: true });
        toks = r.tokens;
        const T = ids.length, nL = r.attn.length, nH = r.attn[0].length;
        A = Array.from({ length: T }, () => new Float32Array(T));
        for (let l = 0; l < nL; l++) for (let hh = 0; hh < nH; hh++) for (let q = 0; q < T; q++) for (let k = 0; k <= q; k++) A[q][k] += r.attn[l][hh][q][k] / (nL * nH);
      } catch (e) { A = null; }
    }
    const isLive = !!A;
    if (!A) {
      const T = toks.length;
      A = Array.from({ length: T }, (_, q) => { const row = new Float32Array(T); let s = 0; for (let k = 0; k <= q; k++) { row[k] = Math.exp(-(q - k) * 0.5) + (k === 0 ? 0.6 : 0); s += row[k]; } for (let k = 0; k <= q; k++) row[k] /= s; return row; });
    }
    const N = A.length;
    let amax = 0;
    for (let q = 1; q < N; q++) for (let k = 0; k <= q; k++) amax = Math.max(amax, A[q][k]);

    const fig = el('figure', { class: 'fig ep-coda' });
    fig.appendChild(el('div', { class: 'fig-top' }, el('span', { class: 'fig-title' }, 'The finished cloth'), ui.badge(isLive ? 'live' : 'illustration', isLive ? 'Live model · the pattern' : undefined)));
    body.appendChild(fig);
    const cv = ctx.canvas(fig, {
      label: `A finished piece of woven cloth in indigo, bordered with stripes of every dye on the page. In its centre, gold weft threads form a staircase triangle: the ${isLive ? 'average attention of this page\'s model' : 'shape of causal attention'} over the sentence “${CODA_SENTENCE}”. Each row, labelled with its token, looks back at earlier tokens; nothing is woven above the diagonal.`,
      height: (w) => (w < 560 ? 250 : 360),
    });
    fig.appendChild(el('figcaption', { class: 'ep-codacap', html: isLive
      ? `The gold staircase is the mean attention of all ${lm.config.n_layer * lm.config.n_head} heads in this page’s model, reading “${CODA_SENTENCE}”. Each row is the token named on its left, looking back at the tokens before it; brighter gold means more attention. Above the diagonal there is no gold: the causal mask.`
      : `The gold staircase is the shape of causal attention: each row looks only back. (Toy weights; the live model did not load.)` }));
    body.appendChild(el('p', { class: 'ep-closing', html: 'That is the whole loom. Every reply a chatbot writes is woven this way: <em>one token at a time</em>, each new thread pulled through everything that came before it.' }));

    const st = { grid: null, done: 0, started: false, cache: document.createElement('canvas'), pointer: null, last: now() };
    const seen = inView(cv.wrap, '0px');
    const DYES = ['madder', 'saffron', 'weld', 'verdigris', 'woad', 'lichen', 'cochineal'].map((k) => AM.dye[k]);

    const rgb = (hex) => AM.hexToRgb(hex);
    const mixC = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
    const css = (c, k = 1) => `rgb(${Math.min(255, c[0] * k) | 0},${Math.min(255, c[1] * k) | 0},${Math.min(255, c[2] * k) | 0})`;
    function buildGrid(w, h) {
      const p = w < 560 ? 5 : 6, fringe = w < 560 ? 12 : 16;
      const cols = Math.floor((w - 2) / p), rows = Math.floor((h - 2 * fringe) / p);
      const ox = Math.round((w - cols * p) / 2), oy = Math.round((h - rows * p) / 2);
      const m = Math.max(1, Math.floor((rows - 12) / N));
      const mc0 = Math.floor((cols - N * m) / 2), mr0 = Math.floor((rows - N * m) / 2);
      const INK = rgb(AM.col.ink), WOAD = rgb(AM.dye.woad), WELD = rgb(AM.dye.weld);
      const groundWeft = mixC(INK, WOAD, 0.3), groundWarp = mixC(INK, WOAD, 0.15), deep = mixC(INK, WOAD, 0.38);
      const dyes = DYES.map(rgb);
      const rr = M.rng(12), rowJ = Array.from({ length: rows }, () => (rr() - 0.5) * 0.16), colJ = Array.from({ length: cols }, () => (rr() - 0.5) * 0.12);
      const cells = new Array(rows * cols);
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          let over = (r + c) % 4 < 2; // twill ground
          let weft = groundWeft, warp = groundWarp, shine = 0.1;
          const br = Math.min(r, rows - 1 - r), bc = Math.min(c, cols - 1 - c);
          // borders: a band of every dye at top and bottom, dyed warp near the selvedges
          if (br >= 2 && br <= 4) { weft = mixC(INK, dyes[Math.floor((c + (br === 3 ? 4 : 0)) / 8) % 7], br === 3 ? 0.72 : 0.42); over = br === 3 ? true : (r + c) % 2 === 0; shine = 0.18; }
          if (bc >= 2 && bc <= 5 && br > 5) { warp = mixC(INK, dyes[(bc - 2 + Math.floor(r / 9)) % 7], 0.6); over = (r + c) % 2 === 0; shine = 0.16; }
          // motif: the attention matrix; row i is token i looking back
          const i = Math.floor((r - mr0) / m), j = Math.floor((c - mc0) / m);
          if (r >= mr0 && c >= mc0 && i < N && j < N) {
            if (j <= i) {
              const v = clamp(Math.pow(A[i][j] / amax, 0.6));
              weft = mixC(deep, WELD, 0.1 + 0.9 * v);
              over = (r * 3 + c) % 7 !== 0; // gold floats, tied down now and then
              shine = 0.12 + 0.34 * v;
            } else {
              over = (r + c) % 2 === 0;
              weft = mixC(INK, WOAD, 0.17);
              warp = mixC(INK, WOAD, 0.09);
            }
          }
          const k = 1 + (over ? rowJ[r] : colJ[c]);
          cells[r * cols + c] = { over, col: css(over ? weft : warp, k), shine };
        }
      }
      return { p, cols, rows, ox, oy, cells, m, mc0, mr0 };
    }
    function drawCell(g, G, r, c) {
      const cl = G.cells[r * G.cols + c], p = G.p;
      const x = G.ox + c * p, y = G.oy + r * p;
      g.fillStyle = 'rgba(2,4,10,0.75)';
      g.fillRect(x, y, p, p);
      g.fillStyle = cl.col;
      if (cl.over) {
        g.fillRect(x, y + 0.7, p, p - 1.4);
        g.fillStyle = AM.rgba('#fff4d6', cl.shine);
        g.fillRect(x, y + 0.7, p, 1);
        g.fillStyle = 'rgba(0,0,0,0.22)';
        g.fillRect(x, y + p - 1.7, p, 1);
      } else {
        g.fillRect(x + 0.7, y, p - 1.4, p);
        g.fillStyle = AM.rgba('#fff4d6', cl.shine * 0.7);
        g.fillRect(x + 0.7, y, 1, p);
        g.fillStyle = 'rgba(0,0,0,0.22)';
        g.fillRect(x + p - 1.7, y, 1, p);
      }
    }
    function resetCache() {
      const { w, h, dpr } = cv;
      st.cache.width = Math.round(w * dpr);
      st.cache.height = Math.round(h * dpr);
      const g = st.cache.getContext('2d');
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      g.fillStyle = AM.col.ink;
      g.fillRect(0, 0, w, h);
      // bare warp, waiting for the weft; its loose ends stay as a fringe
      const G = st.grid;
      g.strokeStyle = AM.rgba(AM.col.linen, 0.09);
      g.lineWidth = 1;
      g.beginPath();
      for (let c = 0; c < G.cols; c++) { const x = G.ox + c * G.p + G.p / 2; g.moveTo(Math.round(x) + 0.5, G.oy - 2); g.lineTo(Math.round(x) + 0.5, G.oy + G.rows * G.p + 2); }
      g.stroke();
      const fr = M.rng(5);
      for (let c = 0; c < G.cols; c++) {
        const x = Math.round(G.ox + c * G.p + G.p / 2) + 0.5;
        const top = G.oy - 3 - fr() * (G.oy - 6), bot = G.oy + G.rows * G.p + 3 + fr() * (G.oy - 6);
        g.strokeStyle = AM.rgba(AM.col.linen, 0.16 + 0.12 * fr());
        g.beginPath(); g.moveTo(x, G.oy); g.lineTo(x + (fr() - 0.5) * 2, top); g.moveTo(x, G.oy + G.rows * G.p); g.lineTo(x + (fr() - 0.5) * 2, bot); g.stroke();
      }
      for (let k = 0; k < st.done; k++) { const r = G.rows - 1 - k; for (let c = 0; c < G.cols; c++) drawCell(g, G, r, c); }
    }
    cv.onResize((w, h) => {
      st.grid = buildGrid(w, h);
      if (AM.reducedMotion) st.done = st.grid.rows;
      st.done = Math.min(st.done, st.grid.rows);
      if (st.started && st.prog != null) st.done = Math.min(st.grid.rows, Math.floor(st.prog * st.grid.rows));
      resetCache();
      draw(0);
    });
    cv.canvas.addEventListener('pointermove', (e) => { st.pointer = cv.pointer(e); });
    cv.canvas.addEventListener('pointerleave', () => { st.pointer = null; });

    const WEAVE_S = 6;
    function draw(dt) {
      const G = st.grid;
      if (!G || !cv.w) return;
      const g = cv.g, w = cv.w, h = cv.h;
      if (st.started && st.done < G.rows) {
        st.prog = Math.min(1, (st.prog || 0) + dt / WEAVE_S);
        const target = AM.reducedMotion ? G.rows : Math.floor(st.prog * G.rows);
        if (target > st.done) {
          const cg = st.cache.getContext('2d');
          for (let k = st.done; k < target; k++) { const r = G.rows - 1 - k; for (let c = 0; c < G.cols; c++) drawCell(cg, G, r, c); }
          st.done = target;
        }
      }
      cv.clear();
      g.drawImage(st.cache, 0, 0, w, h);
      // the row being woven, and its shuttle
      if (st.started && st.done < G.rows) {
        const r = G.rows - 1 - st.done;
        const frac = (st.prog * G.rows) % 1;
        const ltr = st.done % 2 === 0;
        const n = Math.floor(frac * G.cols);
        for (let k = 0; k < n; k++) drawCell(g, G, r, ltr ? k : G.cols - 1 - k);
        const sx = G.ox + (ltr ? n : G.cols - n) * G.p, sy = G.oy + r * G.p + G.p / 2;
        g.save();
        g.strokeStyle = AM.rgba('#fff4d6', 0.5);
        g.lineWidth = 1;
        g.beginPath(); g.moveTo(ltr ? G.ox : sx, sy); g.lineTo(ltr ? sx : G.ox + G.cols * G.p, sy); g.stroke();
        g.restore();
        bead(g, sx, sy, 3, AM.dye.weld, 1);
      } else if (st.done >= G.rows && !AM.reducedMotion) {
        // a slow sheen across the silk, and light where the pointer rests
        const t = now();
        const ph = ((t * 0.09) % 1.6) - 0.3;
        const x = ph * (w + h) - h;
        g.save();
        g.globalCompositeOperation = 'lighter';
        const grd = g.createLinearGradient(x, 0, x + h, h);
        grd.addColorStop(0, 'rgba(255,244,214,0)');
        grd.addColorStop(0.5, 'rgba(255,244,214,0.07)');
        grd.addColorStop(1, 'rgba(255,244,214,0)');
        g.fillStyle = grd;
        g.fillRect(0, 0, w, h);
        if (st.pointer) {
          const rg = g.createRadialGradient(st.pointer.x, st.pointer.y, 0, st.pointer.x, st.pointer.y, 90);
          rg.addColorStop(0, 'rgba(233,194,74,0.16)');
          rg.addColorStop(1, 'rgba(233,194,74,0)');
          g.fillStyle = rg;
          g.fillRect(st.pointer.x - 90, st.pointer.y - 90, 180, 180);
        }
        g.restore();
      }
      // name each row of the motif once it is woven: row i is token i looking back
      const fs = w < 560 ? 8.5 : 9.5, lx = G.ox + G.mc0 * G.p - (w < 560 ? 5 : 8);
      for (let i = 0; i < N; i++) {
        const top = G.mr0 + i * G.m;
        const a = clamp((st.done - (G.rows - top - G.m)) / G.m);
        if (a <= 0.01) continue;
        halo(g, toks[i], lx, G.oy + (top + G.m / 2) * G.p, { size: fs, align: 'right', color: AM.col.linenDim, alpha: 0.9 * a, haloW: 3.5 });
      }
    }
    ctx.loop(() => {
      const t = now();
      const dt = Math.min(0.1, t - st.last);
      st.last = t;
      if (!seen.on) return;
      if (!st.started) { st.started = true; st.prog = AM.reducedMotion ? 1 : 0; }
      draw(AM.reducedMotion ? 99 : dt);
    });
  }

  // ================================================================== chapter
  AM.chapter({
    id: ID,
    num: 12,
    kicker: 'Scale & recap',
    title: 'From Toy to <em>Titan</em>',
    lede: 'The machine on this page has the same architecture as the models behind modern chatbots. What separates them is mostly scale, plus a handful of refinements.',
    where: 'all',
    mount(root, ctx) {
      ctx.header();
      AM.css(CSS);
      const el = ctx.el;
      const DATA = buildData();
      const body = el('div', { class: 'ch-body' });
      root.appendChild(body);

      body.appendChild(el('div', { class: 'prose ep-intro' },
        el('p', { html: 'You have now seen every part of the machine, running in a model small enough to live inside this page. The transformers behind modern chatbots are built from the same parts, in the same order.' }),
        el('p', { html: 'The difference is size. Scroll, and the stage zooms out by powers of ten.' })));

      buildZoom(body, ctx, DATA);
      buildPoster(body, ctx, DATA);
      buildRefinements(body, ctx);
      buildRecap(body, ctx);

      body.appendChild(el('div', { class: 'callout ep-key' },
        el('div', { class: 'callout-label' }, 'Key idea'),
        el('p', {}, 'Take away the scale and the refinements, and the model behind a modern chatbot is the machine on this page:'),
        el('div', { class: 'ep-chain', 'aria-label': 'tokens, then embeddings plus positions, then attention and MLP blocks added to the residual stream, repeated N times, then unembed, sample, repeat' },
          el('span', {}, 'tokens'), '→', el('span', {}, 'embeddings + positions'), '→',
          el('span', { class: 'ep-blk' }, '[ attention + MLP, each added to the residual stream ] × N'), '→',
          el('span', {}, 'unembed'), '→', el('span', {}, 'sample'), '→', el('span', {}, 'repeat')),
        el('p', {}, 'Scale buys more layers, wider vectors, more heads, longer contexts and far more training text. The refinements make it cheaper to run, steadier to train and more useful as an assistant. The loop is the one you have been watching all along.')));

      buildCoda(body, ctx, DATA);
    },
  });
})();
