/* Prologue: the hero loom.
   Enhances the static hero in index.html (eyebrow, title, subtitle stay as
   they are) with a full-bleed canvas. A real sentence is read by the real
   tiny transformer that ships with the page (tinyworld: 3 layers × 4 heads).
   Its words hang as warp threads; every attention weight of every head is
   woven as a silk weft arc in that head's dye, layer 0 nearest the words and
   layer 2 on top. A shuttle weaves each layer in, a focus thread is picked
   out with the model's own next-word prediction, then the cloth unweaves and
   the next sentence goes on the loom.

   Everything drawn is computed live from AM.model.get('tinyworld').run():
   arcs for weights ≥ 6%, beads for a word attending to itself. */
(() => {
  const ID = 'prologue';
  const M = AM.math;

  // Curated sentences (docs/model-training.md). q = the word whose attention we
  // spotlight, k = the earlier word it should find. The head, the weight and the
  // prediction are all looked up from the live run, never hard-coded.
  const SENTENCES = [
    { text: 'the queen opened the door because she was cold .', q: 'because', k: 'queen' },
    { text: 'alice gave bob a cup . bob thanked alice .', q: 'thanked', k: 'alice' },
    { text: 'the red ball and the blue box . the box is blue .', q: 'is', k: 'blue' },
    { text: 'the keys near the old door are gold .', q: 'door', k: 'keys' },
    { text: 'the capital of japan is tokyo .', q: 'is', k: 'japan' },
  ];
  const THRESH = 0.06;   // weft arcs are drawn for attention weights ≥ 6%
  const KNOT_MIN = 0.1;  // a word attending ≥ 10% to itself shows as a bead on its warp
  const GHOST = 0.22;    // the not-yet-woven pattern, faintly drawn on the cloth
  const DUR = { in: 0.85, band: 1.35, swoop: 0.32, hold: 1.5, focus: 5.2, rest: 1.6, out: 1.05 };
  const WEAVE = 3 * DUR.band + 2 * DUR.swoop;
  const ORDER = ['in', 'weave', 'hold', 'focus', 'rest', 'out'];

  const clamp = M.clamp;
  const smooth = M.smoothstep;
  const easeIO = M.ease.inOut;
  const easeOut = M.ease.out;
  const outBack = M.ease.outBack;
  const pct = (w) => (w >= 0.995 ? '100%' : w < 0.005 ? '<1%' : Math.round(w * 100) + '%');
  const headName = (l, h) => `L${l}·H${h}`;
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

  /** Optional nickname for a head from the interpretability notes, if they are loaded. */
  function headNick(l, h) {
    try {
      const N = window.AM_NOTES;
      if (!N || !N.heads) return '';
      const key = `L${l}H${h}`;
      let e = Array.isArray(N.heads) ? (N.heads.find && N.heads.find((x) => x && (x.id === key || x.key === key || (x.layer === l && x.head === h)))) : N.heads[key];
      if (!e && Array.isArray(N.heads) && Array.isArray(N.heads[l])) e = N.heads[l][h];
      const name = e && (typeof e === 'string' ? e : (e.short || e.name || e.label));
      return typeof name === 'string' && name.length < 40 ? name : '';
    } catch (_) { return ''; }
  }

  // ---------------------------------------------------------------- bezier helpers
  function bezAt(P, t) {
    const u = 1 - t, a = u * u * u, b = 3 * u * u * t, c = 3 * u * t * t, d = t * t * t;
    return { x: a * P[0] + b * P[2] + c * P[4] + d * P[6], y: a * P[1] + b * P[3] + c * P[5] + d * P[7] };
  }
  /** Sub-curve of a cubic between t0 and t1 (de Casteljau). */
  function bezSub(P, t0, t1) {
    const split = (Q, f) => {
      const L = (a, b) => a + (b - a) * f;
      const x01 = L(Q[0], Q[2]), y01 = L(Q[1], Q[3]), x12 = L(Q[2], Q[4]), y12 = L(Q[3], Q[5]);
      const x23 = L(Q[4], Q[6]), y23 = L(Q[5], Q[7]);
      const x012 = L(x01, x12), y012 = L(y01, y12), x123 = L(x12, x23), y123 = L(y12, y23);
      const xm = L(x012, x123), ym = L(y012, y123);
      return [[Q[0], Q[1], x01, y01, x012, y012, xm, ym], [xm, ym, x123, y123, x23, y23, Q[6], Q[7]]];
    };
    let Q = P;
    if (t1 < 1) Q = split(Q, t1)[0];
    if (t0 > 0) Q = split(Q, t0 / t1)[1];
    return Q;
  }
  /** Parameter t where the (x-monotonic) arc reaches x. */
  function tAtX(P, x) {
    let lo = 0, hi = 1;
    for (let i = 0; i < 14; i++) {
      const m = (lo + hi) / 2, u = 1 - m;
      const xm = u * u * u * P[0] + 3 * u * u * m * P[2] + 3 * u * m * m * P[4] + m * m * m * P[6];
      if (xm < x) lo = m; else hi = m;
    }
    return (lo + hi) / 2;
  }

  /** Radial glow sprite (a bead of light) in one dye, drawn once. */
  function makeSprite(color, core = '#fff8e6') {
    const S = 64, c = document.createElement('canvas');
    c.width = c.height = S;
    const g = c.getContext('2d');
    const grd = g.createRadialGradient(S / 2, S / 2, 0, S / 2, S / 2, S / 2);
    grd.addColorStop(0, AM.rgba(core, 1));
    grd.addColorStop(0.16, AM.rgba(color, 0.95));
    grd.addColorStop(0.42, AM.rgba(color, 0.28));
    grd.addColorStop(1, AM.rgba(color, 0));
    g.fillStyle = grd;
    g.fillRect(0, 0, S, S);
    return c;
  }

  const CSS = `
    #ch-${ID}.hero {
      min-height: 100vh; min-height: 100svh;
      align-content: stretch;
      grid-template-rows: auto minmax(0, 1fr);
      padding-block: clamp(52px, 9vh, 116px) 0;
    }
    #ch-${ID} .pl-back {
      position: absolute; top: 0; bottom: 0; z-index: 0;
      left: calc(-1 * var(--gutter)); right: calc(-1 * var(--gutter));
    }
    @media (min-width: 1240px) { #ch-${ID} .pl-back { left: calc(-1 * (var(--rail-w) + var(--gutter))); } }
    #ch-${ID} .pl-back .stage-canvas { position: absolute; inset: 0; }
    #ch-${ID} .pl-back canvas { touch-action: pan-y; -webkit-tap-highlight-color: transparent; }
    /* a soft ink scrim so the title and subtitle always read over the threads */
    #ch-${ID} .hero-inner::before {
      content: ''; position: absolute; z-index: -1; pointer-events: none;
      inset: -56px 0 -40px -40px;
      background: radial-gradient(64% 72% at 34% 52%, color-mix(in srgb, var(--ink) 86%, transparent) 0%, color-mix(in srgb, var(--ink) 58%, transparent) 46%, transparent 78%);
    }
    #ch-${ID} .pl-floor {
      position: relative; z-index: 2; width: 100%; max-width: var(--content); margin: var(--space-3) auto 0;
      display: grid; grid-template-rows: minmax(0, 1fr) auto; pointer-events: none;
    }
    #ch-${ID} .pl-space { position: relative; min-height: clamp(260px, 36vh, 560px); }
    #ch-${ID} .pl-ring { position: absolute; left: 0; top: 0; width: 0; height: 0; border-radius: 16px; pointer-events: none; }
    #ch-${ID} .pl-ring:focus-visible { outline: 1px solid color-mix(in srgb, var(--weld) 75%, transparent); outline-offset: 4px; }
    #ch-${ID} .pl-dock {
      pointer-events: auto; display: grid; align-items: center;
      grid-template-columns: auto minmax(0, 1fr) auto; grid-template-areas: 'badge read ctrl';
      gap: 8px 22px; padding: 12px 0 64px;
    }
    #ch-${ID} .pl-dock > .badge { grid-area: badge; justify-self: start; }
    #ch-${ID} .pl-read {
      grid-area: read; max-width: 92ch; min-height: 4.5em; font-size: var(--fs-small); line-height: 1.5; color: var(--linen-dim);
      display: flex; align-items: center;
    }
    #ch-${ID} .pl-read > span { display: block; }
    #ch-${ID} .pl-read b { color: var(--linen); font-weight: 600; }
    #ch-${ID} .pl-read .pl-w { font-family: var(--font-mono); font-size: 0.9em; color: var(--linen); white-space: nowrap; }
    #ch-${ID} .pl-read .pl-hd { font-family: var(--font-mono); font-size: 0.86em; white-space: nowrap; color: var(--c, var(--linen)); }
    #ch-${ID} .pl-read .pl-pred { font-family: var(--font-display); font-style: italic; font-size: 1.2em; line-height: 1; color: var(--weld); }
    #ch-${ID} .pl-ctrl { grid-area: ctrl; display: inline-flex; align-items: center; gap: 8px; justify-self: end; }
    #ch-${ID} .pl-pips { display: inline-flex; align-items: center; gap: 0; }
    #ch-${ID} .pl-pip {
      display: grid; place-items: center; width: 24px; height: 30px; padding: 0; border: 0; background: none; cursor: pointer; border-radius: 6px;
    }
    #ch-${ID} .pl-pip::before {
      content: ''; width: 7px; height: 7px; border-radius: 4px; background: var(--rule-strong);
      transition: width 0.3s, background 0.3s, box-shadow 0.3s;
    }
    #ch-${ID} .pl-pip:hover::before { background: var(--linen-dim); }
    #ch-${ID} .pl-pip[aria-pressed='true']::before { width: 18px; background: var(--weld); box-shadow: 0 0 10px color-mix(in srgb, var(--weld) 60%, transparent); }
    #ch-${ID} .pl-play {
      display: grid; place-items: center; width: 30px; height: 30px; padding: 0; border-radius: 50%;
      border: 1px solid var(--rule-strong); background: var(--ink); color: var(--linen-dim); cursor: pointer;
      transition: border-color 0.15s, color 0.15s;
    }
    #ch-${ID} .pl-play:hover { border-color: var(--weld); color: var(--weld); }
    #ch-${ID} .pl-play svg { width: 12px; height: 12px; display: block; }
    #ch-${ID} .pl-cue {
      position: absolute; left: 50%; bottom: 10px; transform: translateX(-50%); z-index: 3;
      display: grid; justify-items: center; gap: 6px; padding: 4px 10px;
      font-family: var(--font-mono); font-size: 10px; letter-spacing: 0.22em; text-transform: uppercase;
      color: var(--mist); text-decoration: none;
    }
    #ch-${ID} .pl-cue:hover { color: var(--linen); }
    #ch-${ID} .pl-cue i { position: relative; display: block; width: 1px; height: 26px; background: linear-gradient(var(--rule-strong), transparent); overflow: hidden; }
    #ch-${ID} .pl-cue i::after {
      content: ''; position: absolute; left: -2px; top: -8px; width: 5px; height: 8px; border-radius: 3px;
      background: var(--weld); box-shadow: 0 0 8px var(--weld); animation: pl-drop 2.4s cubic-bezier(.6,0,.4,1) infinite;
    }
    @keyframes pl-drop { 0% { transform: translateY(0); opacity: 0; } 15% { opacity: 1; } 70% { opacity: 1; } 100% { transform: translateY(34px); opacity: 0; } }
    @media (prefers-reduced-motion: reduce) { #ch-${ID} .pl-cue i::after { animation: none; top: 8px; } }
    @media (max-width: 900px) {
      #ch-${ID} .pl-dock { grid-template-columns: auto minmax(0, 1fr); grid-template-areas: 'badge ctrl' 'read read'; gap: 6px 12px; padding-bottom: 54px; }
      #ch-${ID} .pl-read { min-height: 4.5em; align-items: flex-start; }
    }
    @media (max-width: 640px) {
      #ch-${ID}.hero { padding-top: 40px; }
      #ch-${ID} .hero-sub { font-size: 1.0625rem; margin-top: var(--space-4); }
      #ch-${ID} .hero-title { margin-top: var(--space-3); }
      #ch-${ID} .hero-inner::before { inset: -40px -16px -24px -16px; }
      #ch-${ID} .pl-space { min-height: 250px; }
      #ch-${ID} .pl-read { font-size: 0.8125rem; min-height: 6em; }
      #ch-${ID} .pl-cue { bottom: 6px; gap: 4px; }
      #ch-${ID} .pl-cue i { height: 20px; }
    }
  `;

  AM.chapter({
    id: ID,
    num: null,
    kicker: 'Prologue',
    title: 'The Attention <em>Loom</em>',
    lede: 'A real sentence, read by a real tiny transformer, woven on a loom.',

    mount(root, ctx) {
      const el = ctx.el;
      AM.css(CSS);

      let model = null;
      try { model = AM.model && typeof AM.model.get === 'function' ? AM.model.get('tinyworld') : null; } catch (e) { model = null; }
      if (!model) return; // the static hero stands on its own
      const NL = model.config.n_layer, NH = model.config.n_head;
      const inner = root.querySelector('.hero-inner') || root;

      // ------------------------------------------------------------ DOM
      const back = el('div', { class: 'pl-back' });
      root.insertBefore(back, root.firstChild);
      const cv = ctx.canvas(back, {
        label: 'Live attention threads of the tiny transformer.',
        height: () => Math.max(360, root.clientHeight),
      });

      const ring = el('div', {
        class: 'pl-ring', tabindex: '0', role: 'group', id: 'pl-loom',
        'aria-label': 'The loom. Use the left and right arrow keys to inspect what each word attends to, Escape to clear.',
      });
      const space = el('div', { class: 'pl-space' }, ring);
      const read = el('p', { class: 'pl-read', id: 'pl-read' });
      const pipBtns = SENTENCES.map((s, i) => el('button', {
        type: 'button', class: 'pl-pip', id: `pl-pip-${i}`, 'aria-pressed': String(i === 0),
        'aria-label': `Sentence ${i + 1}: ${s.text}`, title: s.text,
        onclick: () => choose(i),
      }));
      const ICON_PAUSE = '<svg viewBox="0 0 12 12" aria-hidden="true"><rect x="2" y="1.5" width="2.6" height="9" rx="1" fill="currentColor"/><rect x="7.4" y="1.5" width="2.6" height="9" rx="1" fill="currentColor"/></svg>';
      const ICON_PLAY = '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M3 1.6 L10.2 6 L3 10.4 Z" fill="currentColor"/></svg>';
      const playBtn = el('button', { type: 'button', class: 'pl-play', id: 'pl-play', onclick: () => setAuto(!st.auto) });
      const dock = el('div', { class: 'pl-dock' },
        AM.ui.badge('live'),
        read,
        el('div', { class: 'pl-ctrl' }, el('div', { class: 'pl-pips', role: 'group', 'aria-label': 'Choose a sentence' }, pipBtns), playBtn));
      const floor = el('div', { class: 'pl-floor' }, space, dock);
      root.appendChild(floor);
      const next = root.nextElementSibling;
      root.appendChild(el('a', { class: 'pl-cue', href: next && next.id ? '#' + next.id : '#ch-tokens', 'aria-label': 'Scroll to chapter 1' }, el('span', {}, 'Scroll'), el('i', { 'aria-hidden': 'true' })));

      // ------------------------------------------------------------ live model data
      const cache = [];
      function analyse(si) {
        if (cache[si]) return cache[si];
        const S = SENTENCES[si];
        const enc = model.encode(S.text);
        const r = model.run(enc.ids, { capture: true });
        const T = r.tokens.length;
        const threads = [], knots = [];
        const byQ = Array.from({ length: T }, () => []);
        for (let l = 0; l < NL; l++) for (let h = 0; h < NH; h++) for (let q = 0; q < T; q++) {
          const row = r.attn[l][h][q];
          for (let k = 0; k <= q; k++) {
            const w = row[k];
            if (k === q) { if (w >= KNOT_MIN) knots.push({ l, h, i: q, w }); }
            else if (w >= THRESH) { const th = { l, h, q, k, w }; threads.push(th); byQ[q].push(th); }
          }
        }
        threads.sort((a, b) => a.w - b.w);
        const qi = r.tokens.indexOf(S.q), ki = r.tokens.indexOf(S.k);
        let focus = null;
        if (qi > 0 && ki >= 0 && ki < qi) {
          let best = null;
          for (let l = 0; l < NL; l++) for (let h = 0; h < NH; h++) {
            const w = r.attn[l][h][qi][ki];
            if (!best || w > best.w) best = { l, h, w };
          }
          const top = model.topk(r.probs[qi], 1)[0];
          focus = { q: qi, k: ki, l: best.l, h: best.h, w: best.w, pred: top.token, pp: top.p, th: threads.find((t) => t.q === qi && t.k === ki && t.l === best.l && t.h === best.h) || null };
        }
        const preds = r.tokens.map((_, i) => model.topk(r.probs[i], 1)[0]);
        return (cache[si] = { si, tokens: r.tokens, T, attn: r.attn, threads, knots, byQ, focus, preds });
      }

      // ------------------------------------------------------------ state
      const reduced0 = AM.reducedMotion;
      const st = {
        idx: 0, phase: reduced0 ? 'rest' : 'weave', pt: 0, next: null,
        auto: !reduced0, hover: -1, sel: -1, hvA: 0, hvI: -1, lastPoke: 0,
        full: true, trail: [], lastT: 0,
      };
      let data = analyse(0);
      let G = null;            // geometry for the current sentence
      let bg = null;           // cached warps + words for the current sentence
      let bands = [];          // cached woven layers
      let sprites = null;
      const particles = [];
      let pickTable = null;

      // ------------------------------------------------------------ geometry
      function measure() {
        const cr = cv.canvas.getBoundingClientRect();
        const sr = space.getBoundingClientRect();
        const ir = inner.getBoundingClientRect();
        return { cL: sr.left - cr.left, cR: sr.right - cr.left, top: sr.top - cr.top, bottom: sr.bottom - cr.top, textBottom: ir.bottom - cr.top };
      }

      function layout(d) {
        const { g, w } = cv;
        const Mz = measure();
        const phone = w < 640;
        const s = phone ? 0.8 : 1;
        const fs = phone ? 11.5 : 15;
        const labW = phone ? 20 : 38;
        let x0 = Mz.cL + labW, x1 = Mz.cR - (phone ? 2 : 10);
        const T = d.T;
        let sp = (x1 - x0) / T;
        const spMax = phone ? 62 : 124;
        if (sp > spMax) { const extra = (sp - spMax) * T; x0 += extra / 2; x1 -= extra / 2; sp = spMax; }
        const xs = d.tokens.map((_, i) => x0 + sp * (i + 0.5));
        g.save(); g.font = AM.font(fs, 'body', 600);
        const tw = d.tokens.map((t) => g.measureText(t).width);
        g.restore();
        let stagger = false;
        for (let i = 0; i < T - 1; i++) if ((tw[i] + tw[i + 1]) / 2 > sp - 7) stagger = true;
        const tokH = Math.round(fs * 1.9);
        const rowGap = stagger ? Math.round(fs * 1.45) : 0;
        const tokY0 = Mz.bottom - tokH / 2 - rowGap;
        const tokY = xs.map((_, i) => tokY0 + (stagger && i % 2 ? rowGap : 0));
        const yb0 = tokY0 - tokH / 2 - (phone ? 10 : 16);
        const predH = phone ? 44 : 58;
        const avail = yb0 - Mz.top - predH;
        const bandH = clamp(avail / 3.15, phone ? 44 : 56, phone ? 86 : 124);
        const yb = Array.from({ length: NL }, (_, l) => yb0 - l * bandH);
        const maxLift = bandH * 0.98;
        const loomTop = yb[NL - 1] - maxLift * 1.16;
        // the redrawn region is snapped to device pixels so partial redraws leave no seams
        const dp = cv.dpr || 1;
        const ry0 = Math.max(0, Math.floor((loomTop - predH - 6) * dp) / dp);
        const ry1 = Math.min(cv.h, Math.ceil((Mz.bottom + 22) * dp) / dp);
        const span = Math.max(1, xs[T - 1] - xs[0]);
        const hoff = Array.from({ length: NH }, (_, h) => (h - (NH - 1) / 2) * (phone ? 2.2 : 3));
        const geo = { phone, s, fs, labW, x0, x1, sp, xs, tw, stagger, tokH, rowGap, tokY, yb, bandH, maxLift, loomTop, predH, ry0, ry1, span, hoff, Mz, cL: Mz.cL, cR: Mz.cR };
        // arc geometry for every thread (key → query, rising above the layer's baseline)
        for (const th of d.threads) {
          const xk = xs[th.k] + hoff[th.h], xq = xs[th.q] + hoff[th.h], y = yb[th.l];
          const dd = xq - xk;
          const lift = maxLift * (0.2 + 0.8 * Math.pow(dd / span, 0.6)) * (0.86 + 0.09 * th.h);
          const c = dd * 0.1, up = lift * 4 / 3;
          th.p = [xk, y, xk + c, y - up, xq - c, y - up, xq, y];
          th.len = dd + lift * 1.6;
          th.width = s * (0.35 + 3.0 * th.w);
          th.alpha = Math.min(1, 0.08 + 0.92 * Math.pow(th.w, 0.9));
        }
        for (const kn of d.knots) {
          kn.x = xs[kn.i] + hoff[kn.h];
          kn.y = yb[kn.l] - 1;
          kn.r = s * (0.9 + 2.4 * Math.sqrt(kn.w));
        }
        return geo;
      }

      // ------------------------------------------------------------ painting primitives
      /** An offscreen layer in CSS px, reusing an old canvas element when there is one. */
      function offscreen(old, hCss) {
        const c = old || document.createElement('canvas');
        const W = Math.max(1, Math.round(cv.w * cv.dpr)), H = Math.max(1, Math.round(hCss * cv.dpr));
        if (c.width !== W || c.height !== H) { c.width = W; c.height = H; }
        const g = c.getContext('2d');
        g.setTransform(1, 0, 0, 1, 0, 0);
        g.globalAlpha = 1; g.globalCompositeOperation = 'source-over';
        g.clearRect(0, 0, W, H);
        g.setTransform(cv.dpr, 0, 0, cv.dpr, 0, 0);
        return { c, g, h: hCss };
      }
      function arcPath(g, P) { g.beginPath(); g.moveTo(P[0], P[1]); g.bezierCurveTo(P[2], P[3], P[4], P[5], P[6], P[7]); }
      /** A silk weft thread: soft glow (additive), dyed core, thin sheen on strong ones. */
      function paintThread(g, th, o = {}) {
        const P = o.P || th.p;
        const a = (o.alpha ?? 1) * th.alpha;
        if (a <= 0.004) return;
        const col = AM.headColor(th.h);
        const wd = th.width * (o.widthMul || 1);
        g.lineCap = 'round';
        if (th.w > 0.22 || o.glow) {
          g.globalCompositeOperation = 'lighter';
          arcPath(g, P);
          if (o.glow) {
            g.strokeStyle = AM.rgba(col, 0.07 * a);
            g.lineWidth = wd * 6 + 4;
            g.stroke();
          }
          g.strokeStyle = AM.rgba(col, (o.glow ? 0.2 : 0.1) * a);
          g.lineWidth = wd * (o.glow ? 2.6 : 4.2) + 1.5;
          g.stroke();
          g.globalCompositeOperation = 'source-over';
        }
        arcPath(g, P);
        g.strokeStyle = AM.rgba(col, a);
        g.lineWidth = wd;
        g.stroke();
        if (th.w > 0.4 || o.glow) {
          g.strokeStyle = AM.rgba('#fff6dc', 0.42 * a);
          g.lineWidth = Math.max(0.5, wd * 0.26);
          g.stroke();
        }
      }
      function paintKnot(g, kn, a = 1, scale = 1) {
        if (a <= 0.004 || scale <= 0.01) return;
        const r = kn.r * scale;
        const sp = sprites[kn.h];
        g.globalAlpha = a * (0.55 + 0.45 * kn.w);
        g.drawImage(sp, kn.x - r * 3, kn.y - r * 3, r * 6, r * 6);
        g.globalAlpha = 1;
      }

      // Ground light and warp gradient are defined in canvas coordinates, so the cached
      // loom region and the faint warps drawn directly above it join seamlessly.
      function groundGlow(g, Q) {
        const cx = (Q.x0 + Q.x1) / 2, cy = (Q.loomTop + Q.yb[0]) / 2;
        const rad = Math.max(cv.w * 0.55, 320);
        const glow = g.createRadialGradient(cx, cy, 0, cx, cy, rad);
        glow.addColorStop(0, AM.rgba(AM.dye.woad, 0.075));
        glow.addColorStop(0.45, AM.rgba(AM.dye.lichen, 0.03));
        glow.addColorStop(1, AM.rgba(AM.dye.woad, 0));
        return glow;
      }
      function warpGradient(g, Q) {
        const yEnd = Q.yb[0] + 10;
        const f = (y) => clamp(y / Math.max(1, yEnd));
        const grad = g.createLinearGradient(0, 0, 0, yEnd);
        grad.addColorStop(0, AM.rgba(AM.col.linen, 0));
        grad.addColorStop(f(Q.Mz.textBottom * 0.55), AM.rgba(AM.col.linen, 0.03));
        grad.addColorStop(f(Q.ry0), AM.rgba(AM.col.linen, 0.1));
        grad.addColorStop(f(Q.loomTop + Q.bandH * 0.4), AM.rgba(AM.col.linen, 0.2));
        grad.addColorStop(1, AM.rgba(AM.col.linen, 0.3));
        return grad;
      }
      function strokeWarps(g, Q, yFrom, yTo) {
        g.beginPath();
        Q.xs.forEach((x, i) => { const xx = Math.round(x) + 0.5; g.moveTo(xx, yFrom); g.lineTo(xx, yTo == null ? Q.tokY[i] - Q.fs * 0.95 : yTo); });
        g.strokeStyle = warpGradient(g, Q); g.lineWidth = 1; g.stroke();
        g.globalAlpha *= 0.35; g.lineWidth = 4; g.stroke();
      }
      /** Everything outside the cached loom region: the ground light and the warps hanging from above. */
      function drawUpper(g, Q, a) {
        if (a <= 0.004) return;
        g.save();
        g.globalAlpha = a;
        g.beginPath(); g.rect(0, 0, cv.w, Q.ry0); g.rect(0, Q.ry1, cv.w, cv.h - Q.ry1); g.clip();
        g.fillStyle = groundGlow(g, Q);
        g.fillRect(0, 0, cv.w, cv.h);
        strokeWarps(g, Q, 0, Q.ry0 + 1);
        g.restore();
      }
      /** The cached loom region [ry0, ry1]: ground light, heddle bars, warps, beads and words. */
      function buildBg(d, Q, slot) {
        const B = offscreen(slot.bg && slot.bg.c, Q.ry1 - Q.ry0);
        B.y0 = Q.ry0;
        const g = B.g;
        const phone = Q.phone;
        g.translate(0, -Q.ry0);
        g.fillStyle = groundGlow(g, Q);
        g.fillRect(0, Q.ry0, cv.w, Q.ry1 - Q.ry0);
        // heddle bars: one faint baseline per layer, labelled
        for (let l = 0; l < NL; l++) {
          const y = Math.round(Q.yb[l]) + 0.5;
          const grd = g.createLinearGradient(Q.x0 - 20, 0, Q.x1 + 10, 0);
          grd.addColorStop(0, AM.rgba(AM.col.linen, 0));
          grd.addColorStop(0.06, AM.rgba(AM.col.linen, 0.09));
          grd.addColorStop(0.94, AM.rgba(AM.col.linen, 0.09));
          grd.addColorStop(1, AM.rgba(AM.col.linen, 0));
          g.strokeStyle = grd;
          g.lineWidth = 1;
          g.setLineDash([2, 4]);
          g.beginPath(); g.moveTo(Q.x0 - 20, y); g.lineTo(Q.x1 + 10, y); g.stroke();
          g.setLineDash([]);
          g.font = AM.font(phone ? 9 : 10, 'mono', 500);
          g.fillStyle = AM.rgba(AM.col.mist, 0.95);
          g.textBaseline = 'middle';
          if ('letterSpacing' in g) g.letterSpacing = '0.08em';
          const lab = phone ? 'L' + l : 'LAYER ' + l;
          const edge = Q.xs[0] - Math.min(Q.sp * 0.5, 44) - 10;
          const near = edge - g.measureText(lab).width > Q.cL + 40;
          g.textAlign = near ? 'right' : 'left';
          g.fillText(lab, near ? edge : Q.cL, y - Q.bandH * 0.42);
          if ('letterSpacing' in g) g.letterSpacing = '0px';
        }
        // warp threads, brightest in the loom, ending in a small bead above each word
        g.save();
        strokeWarps(g, Q, Q.ry0 - 1, null);
        g.restore();
        Q.xs.forEach((x, i) => {
          const y = Q.tokY[i] - Q.fs * 0.95;
          g.drawImage(sprites.linen, x - 6, y - 6, 12, 12);
        });
        g.font = AM.font(Q.fs, 'body', 600);
        g.textAlign = 'center';
        g.textBaseline = 'middle';
        g.fillStyle = AM.col.linenDim;
        d.tokens.forEach((t, i) => g.fillText(t, Q.xs[i], Q.tokY[i] + 1));
        return (slot.bg = B);
      }

      function buildBands(d, Q, slot) {
        slot.bands = Array.from({ length: NL }, (_, l) => {
          const y0 = Math.floor(Q.yb[l] - Q.maxLift * 1.22 - 6);
          const y1 = Math.ceil(Q.yb[l] + 10);
          const old = slot.bands && slot.bands[l];
          const B = offscreen(old && old.c, y1 - y0);
          B.y0 = y0;
          const g = B.g;
          g.translate(0, -y0);
          for (const th of d.threads) if (th.l === l) paintThread(g, th);
          for (const kn of d.knots) if (kn.l === l) paintKnot(g, kn);
          return B;
        });
        return slot.bands;
      }

      function blitBand(g, B, a, squash = 1, l = 0) {
        if (a <= 0.004 || !B) return;
        g.globalAlpha = a;
        if (squash >= 0.999) g.drawImage(B.c, 0, B.y0, cv.w, B.h);
        else {
          const base = G.yb[l];
          const top = base - (base - B.y0) * squash;
          g.drawImage(B.c, 0, top, cv.w, B.h * squash);
        }
        g.globalAlpha = 1;
      }

      // Sentences are prepared (model run, geometry, cached layers) ahead of time,
      // one stage per frame while the loom rests, so switching is only a swap.
      let geoVer = 0;
      const slots = [{ bg: null, bands: null }, { bg: null, bands: null }];
      let cur = 0;               // the slot holding the installed sentence's layers
      const prep = { si: -1, ver: -1, slot: -1, stage: 0 };
      function ensureSprites() {
        if (sprites) return;
        sprites = Array.from({ length: NH }, (_, h) => makeSprite(AM.headColor(h)));
        sprites.linen = makeSprite(AM.col.linen);
        sprites.weld = makeSprite(AM.dye.weld);
      }
      function prepare(si) {
        const slot = 1 - cur, d = analyse(si), Q = layout(d);
        buildBg(d, Q, slots[slot]);
        buildBands(d, Q, slots[slot]);
        return { si, ver: geoVer, slot, stage: 3, d, G: Q };
      }
      function stepPrep(si) {
        if (prep.si !== si || prep.ver !== geoVer || prep.slot !== 1 - cur) { prep.si = si; prep.ver = geoVer; prep.slot = 1 - cur; prep.stage = 0; }
        if (prep.stage === 0) { prep.d = analyse(si); prep.stage = 1; }
        else if (prep.stage === 1) { prep.G = layout(prep.d); buildBg(prep.d, prep.G, slots[prep.slot]); prep.stage = 2; }
        else if (prep.stage === 2) { buildBands(prep.d, prep.G, slots[prep.slot]); prep.stage = 3; }
      }
      const prepReady = (i) => prep.si === i && prep.ver === geoVer && prep.slot === 1 - cur && prep.stage === 3;
      function install(P) {
        cur = P.slot;
        data = P.d; G = P.G; bg = slots[cur].bg; bands = slots[cur].bands;
        prep.si = -1;
        // the keyboard focus ring hugs the loom
        const sr = space.getBoundingClientRect(), cr = cv.canvas.getBoundingClientRect();
        const ox = cr.left - sr.left, oy = cr.top - sr.top;
        const rTop = G.ry0 + G.predH * 0.25, rBot = G.Mz.bottom + 4;
        ring.style.left = (G.x0 - G.labW + ox - 6) + 'px';
        ring.style.top = (rTop + oy) + 'px';
        ring.style.width = (G.x1 - G.x0 + G.labW + 12) + 'px';
        ring.style.height = (rBot - rTop) + 'px';
        cv.canvas.setAttribute('aria-label', `The sentence “${SENTENCES[P.si].text}” as read by the live model: each word hangs as a vertical warp thread, and coloured arcs show which earlier words each of its ${NL * NH} attention heads (${NL} layers × ${NH} heads) attends to; a bead on a word’s own thread marks attention to itself. Weights under 6% are not drawn.`);
        resetParticles();
        st.full = true;
      }
      function rebuild() {
        if (!cv.w) return;
        ensureSprites();
        geoVer++;
        install(prepare(st.idx));
      }

      // ------------------------------------------------------------ particles: information travelling key → query
      function resetParticles() {
        particles.length = 0;
        const n = G.phone ? 54 : 104;
        const cum = [];
        let acc = 0;
        for (const th of data.threads) { acc += Math.pow(th.w, 1.6); cum.push(acc); }
        pickTable = { cum, acc };
        const rnd = M.rng(17 + st.idx);
        for (let i = 0; i < n; i++) { const pa = {}; spawn(pa, rnd); pa.t = rnd(); particles.push(pa); }
      }
      function spawn(pa, rnd = Math.random) {
        const { cum, acc } = pickTable;
        const u = rnd() * acc;
        let lo = 0, hi = cum.length - 1;
        while (lo < hi) { const m = (lo + hi) >> 1; if (cum[m] < u) lo = m + 1; else hi = m; }
        pa.th = data.threads[lo];
        pa.t = 0;
        pa.v = (46 + 54 * rnd()) / Math.max(40, pa.th ? pa.th.len : 100);
      }
      function drawParticles(g, dt, aAll, act) {
        if (!data.threads.length || aAll <= 0.01) return;
        g.globalCompositeOperation = 'lighter';
        for (const pa of particles) {
          pa.t += pa.v * dt;
          if (pa.t >= 1) spawn(pa);
          const th = pa.th;
          if (!th) continue;
          const pos = bezAt(th.p, pa.t);
          let a = Math.pow(Math.sin(Math.PI * pa.t), 0.8) * (0.3 + 0.7 * th.w) * aAll;
          if (act >= 0) a *= th.q === act ? 1.6 : 0.18;
          if (a <= 0.01) continue;
          const r = G.s * (2.2 + 3.6 * th.w);
          g.globalAlpha = Math.min(1, a);
          g.drawImage(sprites[th.h], pos.x - r, pos.y - r, r * 2, r * 2);
        }
        g.globalAlpha = 1;
        g.globalCompositeOperation = 'source-over';
      }

      // ------------------------------------------------------------ overlays
      function wordGlow(g, i, color, a, strong = false) {
        if (a <= 0.01) return;
        g.save();
        g.globalAlpha = a;
        g.font = AM.font(G.fs, 'body', strong ? 700 : 600);
        g.textAlign = 'center'; g.textBaseline = 'middle';
        g.shadowColor = AM.rgba(color, 0.9);
        g.shadowBlur = strong ? 16 : 10;
        g.fillStyle = strong ? '#fff6dc' : AM.mix(AM.col.linen, color, 0.35);
        g.fillText(data.tokens[i], G.xs[i], G.tokY[i] + 1);
        g.shadowBlur = 0;
        // an underline thread in the dye
        const tw = G.tw[i];
        g.fillStyle = AM.rgba(color, 0.9);
        g.fillRect(G.xs[i] - tw / 2, G.tokY[i] + G.fs * 0.72, tw, 1.5);
        g.restore();
      }
      function warpGlow(g, i, color, a, yTop) {
        if (a <= 0.01) return;
        const x = Math.round(G.xs[i]) + 0.5;
        const y1 = G.tokY[i] - G.fs * 0.95;
        const grd = g.createLinearGradient(0, yTop, 0, y1);
        grd.addColorStop(0, AM.rgba(color, 0));
        grd.addColorStop(0.25, AM.rgba(color, 0.55 * a));
        grd.addColorStop(1, AM.rgba(color, 0.8 * a));
        g.strokeStyle = grd;
        g.lineWidth = 1.2;
        g.beginPath(); g.moveTo(x, yTop); g.lineTo(x, y1); g.stroke();
      }
      /** The model's next-word prediction, floated above a warp. */
      function drawPred(g, i, tok, p, a, color = AM.dye.weld) {
        if (a <= 0.01) return;
        const big = G.phone ? 21 : 28;
        const sub = `NEXT-WORD GUESS · ${pct(p)}`;
        g.save();
        g.font = AM.font(big, 'display', 500, true);
        const ww = g.measureText(tok).width;
        g.font = AM.font(G.phone ? 8.5 : 9.5, 'mono', 500);
        const sw = g.measureText(sub).width * 1.12;
        const half = Math.max(ww, sw) / 2 + 4;
        const x = clamp(G.xs[i], G.cL + half, G.cR - half);
        const yWord = G.loomTop - (G.phone ? 10 : 14);
        g.globalAlpha = a;
        // stem from the warp up to the word
        g.strokeStyle = AM.rgba(color, 0.55);
        g.lineWidth = 1;
        g.setLineDash([2, 3]);
        g.beginPath(); g.moveTo(G.xs[i] + 0.5, G.yb[NL - 1] - G.bandH * 0.2); g.lineTo(x + 0.5, yWord + 4); g.stroke();
        g.setLineDash([]);
        g.font = AM.font(big, 'display', 500, true);
        g.textAlign = 'center'; g.textBaseline = 'alphabetic';
        g.lineJoin = 'round';
        g.strokeStyle = AM.rgba(AM.col.ink, 0.85); g.lineWidth = 5;
        g.strokeText(tok, x, yWord);
        g.shadowColor = AM.rgba(color, 0.7); g.shadowBlur = 14;
        g.fillStyle = color;
        g.fillText(tok, x, yWord);
        g.shadowBlur = 0;
        g.font = AM.font(G.phone ? 8.5 : 9.5, 'mono', 500);
        if ('letterSpacing' in g) g.letterSpacing = '0.12em';
        g.fillStyle = AM.col.mist;
        g.strokeText(sub, x, yWord - big - 2);
        g.fillText(sub, x, yWord - big - 2);
        g.restore();
      }

      // ------------------------------------------------------------ the weave (live drawing of one layer)
      function weaveBand(g, l, u, t) {
        const dir = l % 2 === 0 ? 1 : -1;
        const xa = G.xs[0] - G.sp * 0.5, xb = G.xs[data.T - 1] + G.sp * 0.5;
        const e = easeIO(clamp(u));
        const sx = dir > 0 ? M.lerp(xa, xb, e) : M.lerp(xb, xa, e);
        const tips = [];
        for (const th of data.threads) {
          if (th.l !== l) continue;
          const xk = th.p[0], xq = th.p[6];
          if (dir > 0) {
            if (sx <= xk) continue;
            if (sx >= xq) { paintThread(g, th); continue; }
            const tt = tAtX(th.p, sx);
            paintThread(g, th, { P: bezSub(th.p, 0, tt) });
            tips.push([bezAt(th.p, tt), th]);
          } else {
            if (sx >= xq) continue;
            if (sx <= xk) { paintThread(g, th); continue; }
            const tt = tAtX(th.p, sx);
            paintThread(g, th, { P: bezSub(th.p, tt, 1) });
            tips.push([bezAt(th.p, tt), th]);
          }
        }
        for (const kn of data.knots) {
          if (kn.l !== l) continue;
          const passed = (sx - kn.x) * dir;
          if (passed > 0) paintKnot(g, kn, 1, outBack(clamp(passed / 46)));
        }
        // needle tips riding with the shuttle
        g.globalCompositeOperation = 'lighter';
        for (const [pos, th] of tips) {
          const r = G.s * (2.4 + 4 * th.w);
          g.globalAlpha = 0.5 + 0.5 * th.w;
          g.drawImage(sprites[th.h], pos.x - r, pos.y - r, r * 2, r * 2);
        }
        g.globalAlpha = 1;
        g.globalCompositeOperation = 'source-over';
        return { x: sx, y: G.yb[l] - G.bandH * 0.5, dir };
      }
      function drawShuttle(g, pos, l, reedA) {
        const { x, y } = pos;
        // the reed: a soft vertical light sweeping the layer
        if (reedA > 0.01) {
          const yT = G.yb[l] - G.maxLift * 1.18, yB = G.yb[l] + 4;
          const grd = g.createLinearGradient(0, yT, 0, yB);
          grd.addColorStop(0, AM.rgba(AM.dye.weld, 0));
          grd.addColorStop(0.5, AM.rgba(AM.dye.weld, 0.32 * reedA));
          grd.addColorStop(1, AM.rgba(AM.dye.weld, 0.05 * reedA));
          g.strokeStyle = grd;
          g.lineWidth = 1;
          g.beginPath(); g.moveTo(x, yT); g.lineTo(x, yB); g.stroke();
          g.globalCompositeOperation = 'lighter';
          g.lineWidth = 12; g.globalAlpha = 0.16; g.stroke(); g.globalAlpha = 1;
          g.globalCompositeOperation = 'source-over';
        }
        // comet trail
        const tr = st.trail;
        tr.push({ x, y });
        if (tr.length > 18) tr.shift();
        g.globalCompositeOperation = 'lighter';
        g.lineCap = 'round';
        for (let i = 1; i < tr.length; i++) {
          const a = i / tr.length;
          g.strokeStyle = AM.rgba(AM.dye.weld, 0.4 * a * a);
          g.lineWidth = 0.8 + 3.2 * a;
          g.beginPath(); g.moveTo(tr[i - 1].x, tr[i - 1].y); g.lineTo(tr[i].x, tr[i].y); g.stroke();
        }
        // the shuttle: an elongated bead of light
        const ang = tr.length > 2 ? Math.atan2(y - tr[tr.length - 3].y, x - tr[tr.length - 3].x) : 0;
        g.save();
        g.translate(x, y);
        g.rotate(ang);
        g.drawImage(sprites.weld, -26, -11, 52, 22);
        g.globalCompositeOperation = 'source-over';
        g.fillStyle = '#fff6dc';
        g.beginPath(); g.ellipse(0, 0, 9, 2.6, 0, 0, Math.PI * 2); g.fill();
        g.restore();
        g.globalCompositeOperation = 'source-over';
      }

      // ------------------------------------------------------------ readout
      let lastRead = '';
      function setRead(html) { if (html !== lastRead) { read.innerHTML = '<span>' + html + '</span>'; lastRead = html; } }
      const hoverWord = () => (window.matchMedia && window.matchMedia('(hover: hover)').matches ? 'Hover' : 'Tap');
      const DEFAULT_READ = () => `Real attention from the tiny transformer on this page (${NL} layers × ${NH} heads). Each arc is a word looking back at an earlier word, in its head’s dye: ${Array.from({ length: NH }, (_, h) => `<span class="pl-hd" style="--c:${AM.headColor(h)}">H${h}</span>`).join(' ')}. Weights under 6% are not drawn. ${hoverWord()} a word.`;
      const W = (t) => `<span class="pl-w">${esc(t)}</span>`;
      const HD = (l, h) => `<span class="pl-hd" style="--c:${AM.headColor(h)}">${headName(l, h)}</span>`;
      function readFocus() {
        const f = data.focus;
        if (!f) return DEFAULT_READ();
        const nick = headNick(f.l, f.h);
        return `At ${W('“' + data.tokens[f.q] + '”')}, head ${HD(f.l, f.h)}${nick ? ` (${esc(nick)})` : ''} puts <b>${pct(f.w)}</b> of its attention on ${W('“' + data.tokens[f.k] + '”')}. The model’s top guess for the next word: <span class="pl-pred">${esc(f.pred)}</span> <b>${pct(f.pp)}</b>.`;
      }
      function readWord(i) {
        const tok = data.tokens[i];
        const pr = data.preds[i];
        const tail = ` Top guess for the next word: <span class="pl-pred">${esc(pr.token)}</span> <b>${pct(pr.p)}</b>.`;
        if (i === 0) return `${W('“' + tok + '”')} is the first word, so each of the ${NL * NH} heads can only attend to that word itself (100%).${tail}`;
        const links = [];
        for (let l = 0; l < NL; l++) for (let h = 0; h < NH; h++) for (let k = 0; k < i; k++) links.push({ l, h, k, w: data.attn[l][h][i][k] });
        links.sort((a, b) => b.w - a.w);
        const top = links.slice(0, 3).filter((x) => x.w >= 0.05);
        if (!top.length) return `${W('“' + tok + '”')} mostly attends to itself here.${tail}`;
        return `${W('“' + tok + '”')} looks back at ` + top.map((x) => `${W(data.tokens[x.k])} <b>${pct(x.w)}</b> ${HD(x.l, x.h)}`).join(', ') + '.' + tail;
      }

      // ------------------------------------------------------------ interaction
      function pick(x, y) {
        if (!G || st.phase === 'out' || st.phase === 'in') return -1;
        if (y < G.ry0 + G.predH * 0.5 || y > G.ry1 + 4) return -1;
        if (x < G.xs[0] - G.sp * 0.6 || x > G.xs[data.T - 1] + G.sp * 0.6) return -1;
        return clamp(Math.round((x - G.xs[0]) / G.sp), 0, data.T - 1);
      }
      const poke = () => { st.lastPoke = performance.now(); if (AM.reducedMotion) frame(st.lastT, 0); };
      cv.canvas.addEventListener('pointermove', (ev) => {
        if (ev.pointerType !== 'mouse') return;
        const p = cv.pointer(ev);
        const i = pick(p.x, p.y);
        if (i !== st.hover) { st.hover = i; poke(); }
        cv.canvas.style.cursor = i >= 0 ? 'pointer' : '';
      });
      cv.canvas.addEventListener('pointerleave', () => { if (st.hover !== -1) { st.hover = -1; poke(); } });
      // a tap or click pins a word (and a second one lets go); 'click' never fires for a scroll gesture
      cv.canvas.addEventListener('click', (ev) => {
        const p = cv.pointer(ev);
        const i = pick(p.x, p.y);
        st.sel = i >= 0 && i !== st.sel ? i : -1;
        if (ev.pointerType && ev.pointerType !== 'mouse') st.hover = -1;
        poke();
      });
      ring.addEventListener('keydown', (ev) => {
        const T = data.T;
        let handled = true;
        if (ev.key === 'ArrowRight') st.sel = st.sel < 0 ? 0 : Math.min(T - 1, st.sel + 1);
        else if (ev.key === 'ArrowLeft') st.sel = st.sel < 0 ? T - 1 : Math.max(0, st.sel - 1);
        else if (ev.key === 'Home') st.sel = 0;
        else if (ev.key === 'End') st.sel = T - 1;
        else if (ev.key === 'Escape') st.sel = -1;
        else handled = false;
        if (handled) { ev.preventDefault(); poke(); }
      });
      ring.addEventListener('focus', () => read.setAttribute('aria-live', 'polite'));
      ring.addEventListener('blur', () => read.removeAttribute('aria-live'));
      pipBtns.forEach((b) => {
        b.addEventListener('focus', () => read.setAttribute('aria-live', 'polite'));
        b.addEventListener('blur', () => read.removeAttribute('aria-live'));
      });

      function setAuto(on) {
        st.auto = on;
        playBtn.innerHTML = on ? ICON_PAUSE : ICON_PLAY;
        playBtn.setAttribute('aria-label', on ? 'Pause the loom on this sentence' : 'Play: cycle through the sentences');
        playBtn.setAttribute('aria-pressed', String(!on));
        playBtn.title = on ? 'Pause' : 'Play';
      }
      setAuto(st.auto);

      function choose(i) {
        if (i === st.idx && st.phase !== 'out') return;
        st.next = i;
        st.sel = -1; st.hover = -1;
        if (AM.reducedMotion) { switchTo(i); st.phase = 'rest'; st.pt = 0; frame(st.lastT, 0); return; }
        if (st.phase !== 'out') { st.phase = 'out'; st.pt = 0; st.full = true; }
      }
      function switchTo(i) {
        st.idx = i; st.next = null;
        st.sel = -1; st.hover = -1; st.hvA = 0; st.hvI = -1;
        pipBtns.forEach((b, j) => b.setAttribute('aria-pressed', String(j === i)));
        if (!cv.w) return;
        ensureSprites();
        install(prepReady(i) ? Object.assign({}, prep) : prepare(i));
      }

      // ------------------------------------------------------------ timeline
      function advance(dt) {
        if (AM.reducedMotion) {
          if (st.phase !== 'rest') { st.phase = 'rest'; st.pt = 0; st.full = true; }
          const a = st.sel >= 0 ? st.sel : st.hover;
          st.hvI = a; st.hvA = a >= 0 ? 1 : 0;
          return;
        }
        // a pinned word on a touch screen lets go after a while so the loom moves on
        if (st.sel >= 0 && performance.now() - st.lastPoke > 12000 && document.activeElement !== ring) st.sel = -1;
        const act = st.sel >= 0 ? st.sel : st.hover;
        const k = 1 - Math.exp(-dt * 9);
        if (act >= 0) st.hvI = act;
        st.hvA += ((act >= 0 ? 1 : 0) - st.hvA) * k;
        if (st.hvA < 0.01 && act < 0) st.hvI = -1;
        const ph = st.phase;
        if ((ph === 'rest' || ph === 'out') && (st.auto || st.next != null)) {
          const ni = st.next != null ? st.next : (st.idx + 1) % SENTENCES.length;
          if (!prepReady(ni)) stepPrep(ni);
        }
        // hovering or pinning a word freezes the cloth; pausing only stops it moving on to the next sentence
        const timed = ph === 'in' || ph === 'weave' || ph === 'out' || st.next != null || (act < 0 && (ph !== 'rest' || st.auto));
        if (timed) st.pt += dt;
        const dur = ph === 'weave' ? WEAVE : DUR[ph];
        if (st.pt >= dur) {
          st.pt = 0;
          st.full = true;
          if (ph === 'out') { switchTo(st.next != null ? st.next : (st.idx + 1) % SENTENCES.length); st.phase = 'in'; }
          else if (ph === 'in') { st.phase = 'weave'; st.trail.length = 0; }
          else st.phase = ORDER[ORDER.indexOf(ph) + 1];
        }
      }

      function breath(l, t) { return AM.reducedMotion ? 1 : 0.84 + 0.16 * (0.5 + 0.5 * Math.sin(t * 1.25 - l * 1.15)); }

      // ------------------------------------------------------------ render
      function frame(t, dt) {
        st.lastT = t;
        advance(dt || 0);
        render(t, dt || 0);
      }

      function render(t, dt) {
        const { g, w, h } = cv;
        if (!w || !G || !bg) return;
        const ph = st.phase, p = st.pt;
        const full = st.full || ph === 'in' || ph === 'out';
        st.full = false;
        if (full) cv.clear();
        else g.clearRect(0, G.ry0, w, G.ry1 - G.ry0);

        // background: warps and words
        let bgA = 1;
        if (ph === 'in') bgA = easeOut(clamp(p / DUR.in));
        if (ph === 'out') bgA = 1 - smooth(0.3, 1, p / DUR.out);
        if (full) drawUpper(g, G, bgA);
        g.globalAlpha = bgA;
        g.drawImage(bg.c, 0, bg.y0, w, bg.h);
        g.globalAlpha = 1;

        const act = st.hvI;
        const hvA = act >= 0 ? st.hvA : 0;
        let fe = ph === 'focus' && data.focus ? smooth(0, 0.7, p) * (1 - smooth(DUR.focus - 0.8, DUR.focus, p)) * (1 - hvA) : 0;
        let dim = 1 - 0.8 * hvA - 0.6 * fe;
        if (AM.reducedMotion && data.focus && ph === 'rest') { fe = 1 - hvA; dim = 1 - 0.8 * hvA - 0.35 * fe; }

        // layers of weft
        let shuttle = null, shuttleL = 0, reedA = 0;
        for (let l = 0; l < NL; l++) {
          const B = bands[l];
          if (ph === 'in') blitBand(g, B, GHOST * easeOut(clamp(p / DUR.in)));
          else if (ph === 'weave') {
            const s0 = l * (DUR.band + DUR.swoop);
            if (p < s0) blitBand(g, B, GHOST);
            else if (p < s0 + DUR.band) {
              blitBand(g, B, GHOST);
              const u = (p - s0) / DUR.band;
              shuttle = weaveBand(g, l, u, t); shuttleL = l;
              reedA = Math.sin(Math.PI * clamp(u));
            } else {
              // a layer just finished: bloom once, then settle into the breathing cloth
              const since = p - s0 - DUR.band;
              blitBand(g, B, breath(l, t));
              const flash = 0.55 * Math.max(0, 1 - since / 0.7);
              if (flash > 0.01) { g.globalCompositeOperation = 'lighter'; blitBand(g, B, flash); g.globalCompositeOperation = 'source-over'; }
            }
          } else if (ph === 'out') {
            const q = clamp((p - (NL - 1 - l) * 0.12) / (DUR.out - 0.3));
            const e = easeIO(q);
            blitBand(g, B, (1 - e) * breath(l, t), 1 - 0.88 * e, l);
          } else blitBand(g, B, breath(l, t) * dim);
        }
        // swoop between layers
        if (ph === 'weave' && !shuttle) {
          for (let l = 0; l < NL - 1; l++) {
            const s0 = l * (DUR.band + DUR.swoop) + DUR.band;
            if (p >= s0 && p < s0 + DUR.swoop) {
              const u = easeIO((p - s0) / DUR.swoop);
              const xEnd = l % 2 === 0 ? G.xs[data.T - 1] + G.sp * 0.5 : G.xs[0] - G.sp * 0.5;
              const yA = G.yb[l] - G.bandH * 0.5, yB = G.yb[l + 1] - G.bandH * 0.5;
              const bulge = (l % 2 === 0 ? 1 : -1) * Math.sin(Math.PI * u) * Math.min(26, G.sp * 0.4);
              shuttle = { x: xEnd + bulge, y: M.lerp(yA, yB, u) };
              shuttleL = l + 1;
            }
          }
        }

        // the residual stream: slow beads of light climbing each warp through the layers
        if (!AM.reducedMotion && ph !== 'in' && ph !== 'out') {
          g.globalCompositeOperation = 'lighter';
          const yTop = G.loomTop - G.bandH * 0.1;
          for (let i = 0; i < data.T; i++) {
            const u = (t * 0.13 + ((i * 0.618034) % 1)) % 1;
            const y0 = G.tokY[i] - G.fs * 0.95;
            const y = M.lerp(y0, yTop, u);
            const a = Math.pow(Math.sin(Math.PI * u), 1.4) * 0.5 * (1 - 0.6 * hvA);
            const r = G.s * 5;
            g.globalAlpha = a;
            g.drawImage(sprites.linen, G.xs[i] - r, y - r, 2 * r, 2 * r);
          }
          g.globalAlpha = 1;
          g.globalCompositeOperation = 'source-over';
        }

        // particles: information carried along the threads
        const woven = ph === 'hold' || ph === 'focus' || ph === 'rest' || ph === 'out';
        if (!AM.reducedMotion && woven) {
          let pa = 1;
          if (ph === 'hold') pa = smooth(0, 0.8, p);
          if (ph === 'out') pa = 1 - smooth(0, 0.5, p);
          drawParticles(g, dt, pa * (1 - 0.5 * fe), act >= 0 && hvA > 0.05 ? act : -1);
        }

        // focus: one thread picked out, with the prediction it supports
        if (fe > 0.01 && data.focus) {
          const f = data.focus, col = AM.headColor(f.h);
          const pulse = AM.reducedMotion ? 1 : 0.85 + 0.15 * Math.sin(t * 3.2);
          warpGlow(g, f.q, AM.col.linen, 0.6 * fe, G.loomTop);
          if (f.th) {
            paintThread(g, f.th, { alpha: fe * pulse, widthMul: 1.5, glow: true });
            if (!AM.reducedMotion) {
              g.globalCompositeOperation = 'lighter';
              for (let j = 0; j < 5; j++) {
                const tt = ((t * 0.38 + j / 5) % 1);
                const pos = bezAt(f.th.p, tt);
                const r = G.s * 4.4;
                g.globalAlpha = fe * Math.sin(Math.PI * tt);
                g.drawImage(sprites[f.h], pos.x - r, pos.y - r, 2 * r, 2 * r);
              }
              g.globalAlpha = 1;
              g.globalCompositeOperation = 'source-over';
            }
          }
          wordGlow(g, f.k, col, fe, false);
          wordGlow(g, f.q, AM.dye.weld, fe, true);
          drawPred(g, f.q, f.pred, f.pp, fe);
        }

        // inspecting one word: its threads to earlier words, its beads, its prediction
        if (act >= 0 && hvA > 0.01) {
          const ths = data.byQ[act];
          warpGlow(g, act, AM.col.linen, 0.7 * hvA, G.loomTop);
          for (const th of ths) paintThread(g, th, { alpha: hvA, widthMul: 1.15, glow: th.w > 0.3 });
          for (const kn of data.knots) if (kn.i === act) paintKnot(g, kn, hvA, 1.25);
          const best = new Map();
          for (const th of ths) if (!best.has(th.k) || best.get(th.k).w < th.w) best.set(th.k, th);
          for (const [k, th] of best) wordGlow(g, k, AM.headColor(th.h), hvA * (0.45 + 0.55 * th.w));
          wordGlow(g, act, AM.dye.weld, hvA, true);
          const pr = data.preds[act];
          drawPred(g, act, pr.token, pr.p, hvA);
        }

        if (shuttle && !AM.reducedMotion) drawShuttle(g, shuttle, shuttleL, reedA);

        // words for the readout
        if (act >= 0 && st.hvA > 0.5) setRead(readWord(act));
        else if (ph === 'focus' && fe > 0.05) setRead(readFocus());
        else if (st.phase === 'rest' && AM.reducedMotion && data.focus) setRead(readFocus());
        else setRead(DEFAULT_READ());
      }

      // ------------------------------------------------------------ lifecycle
      cv.onResize(() => { rebuild(); render(st.lastT, 0); });
      if (typeof ResizeObserver !== 'undefined') {
        let pending = 0;
        new ResizeObserver(() => {
          if (pending) return;
          pending = requestAnimationFrame(() => { pending = 0; if (cv.w) { cv.resize(); rebuild(); render(st.lastT, 0); } });
        }).observe(space);
      }
      ctx.onVisible(() => { st.full = true; });
      ctx.loop((t, dt) => frame(t, dt));
    },
  });
})();
