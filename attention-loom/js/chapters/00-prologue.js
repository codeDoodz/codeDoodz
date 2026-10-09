/* Prologue: the hero loom.
   Enhances the static hero in index.html (eyebrow, title, subtitle stay as
   they are) with a full-bleed canvas. A real sentence is read by the real
   tiny transformer that ships with the page (tinyworld: 3 layers × 4 heads).
   Its words hang as warp threads; every attention weight of every head is
   woven as a silk weft arc in that head's dye, layer 0 nearest the words and
   layer 2 on top. A shuttle weaves each layer in, a focus thread is picked
   out with the model's own next-word prediction, then the cloth unweaves and
   the next sentence goes on the loom.

   Every arc and bead is computed live from AM.model.get('tinyworld').run():
   arcs for weights ≥ 6%, beads for a head putting ≥ 10% on the word itself.
   The shuttle, the travelling sparks and the climbing beads are decoration.

   Rendering: a static "cloth" canvas (ground light, warps, layer bars, words)
   sits under the animated canvas and fades with CSS opacity; each woven layer
   is cached offscreen and revealed by a clip as the shuttle passes, so a frame
   is a few blits plus a handful of sprites. */
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
  const GHOST = 0.3;     // the not-yet-woven pattern, faintly drawn on the cloth
  const DUR = { in: 0.85, band: 1.35, swoop: 0.32, hold: 1.5, focus: 5.2, rest: 1.6, out: 1.05 };
  const WEAVE = 3 * DUR.band + 2 * DUR.swoop;
  const ORDER = ['in', 'weave', 'hold', 'focus', 'rest', 'out'];

  const clamp = M.clamp;
  const smooth = M.smoothstep;
  const easeIO = M.ease.inOut;
  const easeOut = M.ease.out;
  const pct = (w) => (w >= 0.995 ? '100%' : w < 0.005 ? '<1%' : Math.round(w * 100) + '%');
  /** A predicted token for display: punctuation gets quotes so a lone full stop is not lost. */
  const predText = (t) => (/^[.,]$/.test(t) ? `“${t}”` : t);
  const headName = (l, h) => `L${l}·H${h}`;
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const ORD = ['first', 'second', 'third', 'fourth', 'fifth'];

  // ---------------------------------------------------------------- bezier helpers
  function bezAt(P, t) {
    const u = 1 - t, a = u * u * u, b = 3 * u * u * t, c = 3 * u * t * t, d = t * t * t;
    return { x: a * P[0] + b * P[2] + c * P[4] + d * P[6], y: a * P[1] + b * P[3] + c * P[5] + d * P[7] };
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
    #ch-${ID} .pl-back .pl-cloth { position: absolute; left: 0; top: 0; pointer-events: none; }
    #ch-${ID} .pl-back .pl-main { position: relative; }
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
      #ch-${ID} .pl-read { font-size: 0.8125rem; min-height: 6.2em; }
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
      // the static cloth (ground light, warps, layer bars, words) lives on its own canvas
      // underneath, drawn once per sentence and faded with CSS opacity
      const cloth = document.createElement('canvas');
      cloth.className = 'pl-cloth';
      cloth.setAttribute('aria-hidden', 'true');
      cv.wrap.insertBefore(cloth, cv.canvas);
      cv.canvas.classList.add('pl-main');
      let clothA = -1;
      function setClothAlpha(a) {
        a = a >= 0.999 ? 1 : Math.max(0, a);
        if (Math.abs(a - clothA) < 0.004 && !(a === 1 && clothA !== 1) && !(a === 0 && clothA !== 0)) return;
        clothA = a;
        cloth.style.opacity = a === 1 ? '' : a.toFixed(3);
      }

      const ring = el('div', {
        class: 'pl-ring', tabindex: '0', role: 'group', id: 'pl-loom',
        'aria-label': 'The loom. Use the left and right arrow keys (or Home and End) to inspect what each word attends to, and Escape to clear.',
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
        const byL = Array.from({ length: NL }, (_, l) => threads.filter((t) => t.l === l));
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
        return (cache[si] = { si, tokens: r.tokens, T, attn: r.attn, threads, knots, byQ, byL, focus, preds, reads: [] });
      }

      // ------------------------------------------------------------ state
      const reduced0 = AM.reducedMotion;
      const st = {
        idx: 0, phase: reduced0 ? 'rest' : 'weave', pt: 0, next: null,
        auto: !reduced0, hover: -1, sel: -1, hvA: 0, hvI: -1, lastPoke: 0,
        wipe: true, trail: [], lastT: 0, outFrom: null, pauseHold: false,
      };
      let data = analyse(0);
      let G = null;            // geometry for the current sentence
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
        // horizontally, everything animated stays within the content column plus room for the shuttle
        const dx0 = Math.max(0, Math.floor((Mz.cL - 56) * dp) / dp);
        const dx1 = Math.min(cv.w, Math.ceil((Mz.cR + 64) * dp) / dp);
        const span = Math.max(1, xs[T - 1] - xs[0]);
        const hoff = Array.from({ length: NH }, (_, h) => (h - (NH - 1) / 2) * (phone ? 2.2 : 3));
        const geo = { phone, s, fs, labW, x0, x1, sp, xs, tw, stagger, tokH, rowGap, tokY, yb, bandH, maxLift, loomTop, predH, ry0, ry1, dx0, dx1, span, hoff, Mz, cL: Mz.cL, cR: Mz.cR };
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
      /** An offscreen layer covering [x0, x0 + wCss] × [y0, y0 + hCss] in canvas CSS px, reusing an old canvas when there is one. */
      function offscreen(old, x0, y0, wCss, hCss) {
        const c = old || document.createElement('canvas');
        const W = Math.max(1, Math.round(wCss * cv.dpr)), H = Math.max(1, Math.round(hCss * cv.dpr));
        if (c.width !== W || c.height !== H) { c.width = W; c.height = H; }
        const g = c.getContext('2d');
        g.setTransform(1, 0, 0, 1, 0, 0);
        g.globalAlpha = 1; g.globalCompositeOperation = 'source-over';
        g.clearRect(0, 0, W, H);
        g.setTransform(cv.dpr, 0, 0, cv.dpr, -x0 * cv.dpr, -y0 * cv.dpr);
        return { c, g, x0, y0, w: wCss, h: hCss };
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
      /** The static cloth for one sentence: ground light, warps hanging from the top, layer bars and labels, words. */
      function paintCloth(d, Q) {
        if (cloth.width !== cv.canvas.width || cloth.height !== cv.canvas.height) {
          cloth.width = cv.canvas.width; cloth.height = cv.canvas.height;
        }
        cloth.style.width = cv.w + 'px';
        cloth.style.height = cv.h + 'px';
        const g = cloth.getContext('2d');
        g.setTransform(1, 0, 0, 1, 0, 0);
        g.clearRect(0, 0, cloth.width, cloth.height);
        g.setTransform(cv.dpr, 0, 0, cv.dpr, 0, 0);
        const phone = Q.phone;
        g.fillStyle = groundGlow(g, Q);
        g.fillRect(0, 0, cv.w, cv.h);
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
        // warp threads hanging from the top of the hero, brightest in the loom, ending in a bead above each word
        g.beginPath();
        Q.xs.forEach((x, i) => { const xx = Math.round(x) + 0.5; g.moveTo(xx, 0); g.lineTo(xx, Q.tokY[i] - Q.fs * 0.95); });
        g.strokeStyle = warpGradient(g, Q);
        g.lineWidth = 1; g.stroke();
        g.globalAlpha = 0.35; g.lineWidth = 4; g.stroke();
        g.globalAlpha = 1;
        Q.xs.forEach((x, i) => {
          const y = Q.tokY[i] - Q.fs * 0.95;
          g.drawImage(sprites.linen, x - 6, y - 6, 12, 12);
        });
        g.font = AM.font(Q.fs, 'body', 600);
        g.textAlign = 'center';
        g.textBaseline = 'middle';
        g.fillStyle = AM.col.linenDim;
        d.tokens.forEach((t, i) => g.fillText(t, Q.xs[i], Q.tokY[i] + 1));
      }

      /** One woven layer, cached offscreen (cropped to the animated region). */
      function buildBand(d, Q, slot, l) {
        const y0 = Math.floor(Q.yb[l] - Q.maxLift * 1.22 - 6);
        const y1 = Math.ceil(Q.yb[l] + 10);
        if (!slot.bands) slot.bands = [];
        const old = slot.bands[l];
        const B = offscreen(old && old.c, Q.dx0, y0, Q.dx1 - Q.dx0, y1 - y0);
        for (const th of d.byL[l]) paintThread(B.g, th);
        for (const kn of d.knots) if (kn.l === l) paintKnot(B.g, kn);
        slot.bands[l] = B;
      }

      function blitBand(g, B, a, squash = 1, l = 0) {
        if (a <= 0.004 || !B) return;
        g.globalAlpha = Math.min(1, a);
        if (squash >= 0.999) g.drawImage(B.c, B.x0, B.y0, B.w, B.h);
        else {
          const base = G.yb[l];
          const top = base - (base - B.y0) * squash;
          g.drawImage(B.c, B.x0, top, B.w, B.h * squash);
        }
        g.globalAlpha = 1;
      }
      /** A band blitted only left (side < 0) or right (side > 0) of x. */
      function blitBandSide(g, B, a, x, side) {
        if (a <= 0.004 || !B) return;
        const L = side < 0 ? B.x0 : Math.max(B.x0, x), R = side < 0 ? Math.min(B.x0 + B.w, x) : B.x0 + B.w;
        if (R - L < 0.5) return;
        g.save();
        g.beginPath(); g.rect(L, B.y0, R - L, B.h); g.clip();
        blitBand(g, B, a);
        g.restore();
      }

      // Sentences are prepared (model run, geometry, cached layers) ahead of time,
      // one stage per frame while the loom rests, so switching is only a swap.
      let geoVer = 0;
      const slots = [{ bands: null }, { bands: null }];
      let cur = 0;               // the slot holding the installed sentence's layers
      const PREP_DONE = 2 + NL;
      const prep = { si: -1, ver: -1, slot: -1, stage: 0 };
      function ensureSprites() {
        if (sprites) return;
        sprites = Array.from({ length: NH }, (_, h) => makeSprite(AM.headColor(h)));
        sprites.linen = makeSprite(AM.col.linen);
        sprites.weld = makeSprite(AM.dye.weld);
      }
      function prepare(si) {
        const slot = 1 - cur, d = analyse(si), Q = layout(d);
        for (let l = 0; l < NL; l++) buildBand(d, Q, slots[slot], l);
        return { si, ver: geoVer, slot, stage: PREP_DONE, d, G: Q };
      }
      function stepPrep(si) {
        if (prep.si !== si || prep.ver !== geoVer || prep.slot !== 1 - cur) { prep.si = si; prep.ver = geoVer; prep.slot = 1 - cur; prep.stage = 0; }
        if (prep.stage === 0) prep.d = analyse(si);
        else if (prep.stage === 1) prep.G = layout(prep.d);
        else if (prep.stage < PREP_DONE) buildBand(prep.d, prep.G, slots[prep.slot], prep.stage - 2);
        else return;
        prep.stage++;
      }
      const prepReady = (i) => prep.si === i && prep.ver === geoVer && prep.slot === 1 - cur && prep.stage === PREP_DONE;
      function install(P) {
        cur = P.slot;
        data = P.d; G = P.G; bands = slots[cur].bands;
        prep.si = -1;
        paintCloth(data, G);
        // the keyboard focus ring hugs the loom
        const sr = space.getBoundingClientRect(), cr = cv.canvas.getBoundingClientRect();
        const ox = cr.left - sr.left, oy = cr.top - sr.top;
        const rTop = G.ry0 - 2, rBot = G.Mz.bottom + 4;
        ring.style.left = (G.x0 - G.labW + ox - 6) + 'px';
        ring.style.top = (rTop + oy) + 'px';
        ring.style.width = (G.x1 - G.x0 + G.labW + 12) + 'px';
        ring.style.height = (rBot - rTop) + 'px';
        cv.canvas.setAttribute('aria-label', `The sentence “${SENTENCES[P.si].text}”, as read by the live model. Each word hangs as a vertical warp thread. Coloured arcs, one dye per head number, show how much attention each of the model’s ${NL * NH} heads (${NL} layers × ${NH} heads) pays from a word to each earlier word: thicker, brighter arcs carry more weight, and arc height only shows distance. Weights under 6% are not drawn. A bead on a word’s own thread marks a head putting 10% or more on the word itself.`);
        resetParticles();
        st.trail.length = 0;
        st.wipe = true;
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

      // ------------------------------------------------------------ the weave (one layer revealed behind the shuttle)
      function weaveBand(g, l, u) {
        const B = bands[l];
        const dir = l % 2 === 0 ? 1 : -1;
        const xa = G.xs[0] - G.sp * 0.5, xb = G.xs[data.T - 1] + G.sp * 0.5;
        const e = easeIO(clamp(u));
        const sx = dir > 0 ? M.lerp(xa, xb, e) : M.lerp(xb, xa, e);
        // behind the shuttle the layer is woven; ahead of it the pattern is still a ghost
        blitBandSide(g, B, 1, sx, -dir);
        blitBandSide(g, B, GHOST, sx, dir);
        g.globalCompositeOperation = 'lighter';
        // needle tips riding with the shuttle on every thread it is crossing
        for (const th of data.byL[l]) {
          if (sx <= th.p[0] || sx >= th.p[6]) continue;
          const pos = bezAt(th.p, tAtX(th.p, sx));
          const r = G.s * (2.4 + 4 * th.w);
          g.globalAlpha = 0.5 + 0.5 * th.w;
          g.drawImage(sprites[th.h], pos.x - r, pos.y - r, r * 2, r * 2);
        }
        // beads (a head attending to its own word) flare as the shuttle passes
        for (const kn of data.knots) {
          if (kn.l !== l) continue;
          const passed = (sx - kn.x) * dir;
          if (passed <= 0 || passed > 60) continue;
          const f = 1 - passed / 60, r = kn.r * (1 + 4 * f);
          g.globalAlpha = 0.8 * f;
          g.drawImage(sprites[kn.h], kn.x - r, kn.y - r, r * 2, r * 2);
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
      const canHover = !!(window.matchMedia && window.matchMedia('(hover: hover)').matches);
      const W = (t) => `<span class="pl-w">${esc(t)}</span>`;
      const HD = (l, h) => `<span class="pl-hd" style="--c:${AM.headColor(h)}">${headName(l, h)}</span>`;
      const DYES = Array.from({ length: NH }, (_, h) => `<span class="pl-hd" style="--c:${AM.headColor(h)}">H${h}</span>`).join(' ');
      const DEFAULT_READ = `Real attention from the tiny transformer on this page, ${NL} layers × ${NH} heads, dyed by head number: ${DYES}. Each arc links a word to an earlier word it looks at; thicker means more attention. Arcs under 6% are left out. ${canHover ? 'Hover' : 'Tap'} a word.`;
      /** A word in quotes, with "the first / second …" when the sentence repeats it. */
      function wordRef(i) {
        const t = data.tokens[i];
        let n = 0, k = 0;
        for (let j = 0; j < data.T; j++) if (data.tokens[j] === t) { if (j < i) k++; n++; }
        return n > 1 ? `the ${ORD[k] || '#' + (k + 1)} ${W('“' + t + '”')}` : W('“' + t + '”');
      }
      const cap = (s) => (s.startsWith('the ') ? 'The' + s.slice(3) : s);
      function readFocus() {
        const f = data.focus;
        if (!f) return DEFAULT_READ;
        return `At ${wordRef(f.q)}, head ${HD(f.l, f.h)} puts <b>${pct(f.w)}</b> of its attention on ${wordRef(f.k)}. The model’s top guess for the next word: <span class="pl-pred">${esc(predText(f.pred))}</span> <b>${pct(f.pp)}</b>.`;
      }
      function readWord(i) {
        // narrow captions list two earlier words instead of three, so the readout keeps its height
        const many = read.clientWidth >= 520 ? 3 : 2, key = i * 4 + many;
        if (data.reads[key]) return data.reads[key];
        const pr = data.preds[i];
        const tail = ` Next-word guess: <span class="pl-pred">${esc(predText(pr.token))}</span> <b>${pct(pr.p)}</b>.`;
        let s;
        if (i === 0) s = `${W('“' + data.tokens[0] + '”')} is the first word. Attention only looks back, so all ${NL * NH} heads can only attend to it (100%).${tail}`;
        else {
          // the earlier words it looks at most, each with its strongest head and how many other heads join in
          const keys = [];
          for (let k = 0; k < i; k++) {
            let best = null, n = 0;
            for (let l = 0; l < NL; l++) for (let h = 0; h < NH; h++) {
              const w = data.attn[l][h][i][k];
              if (w < THRESH) continue;
              n++;
              if (!best || w > best.w) best = { l, h, w };
            }
            if (best) keys.push({ k, n, ...best });
          }
          keys.sort((a, b) => b.w - a.w);
          const top = keys.slice(0, many);
          if (!top.length) s = `From ${wordRef(i)}, no head puts 6% or more on any earlier word.${tail}`;
          else {
            const part = (x) => `${wordRef(x.k)} (${HD(x.l, x.h)} <b>${pct(x.w)}</b>${x.n > 1 ? `, ${x.n - 1} more head${x.n > 2 ? 's' : ''}` : ''})`;
            const list = top.length === 1 ? part(top[0]) : top.slice(0, -1).map(part).join(', ') + ' and ' + part(top[top.length - 1]);
            const sink = top.some((x) => x.k === 0 && x.l >= 1) ? ' Layers 1 and 2 often rest on the first word.' : '';
            s = `${cap(wordRef(i))} looks back at ${list}.${sink}${tail}`;
          }
        }
        return (data.reads[key] = s);
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

      const RM_STEP = 9;  // reduced motion + play: seconds per sentence, switched without animation
      function setAuto(on) {
        st.auto = on;
        playBtn.innerHTML = on ? ICON_PAUSE : ICON_PLAY;
        playBtn.setAttribute('aria-label', on ? 'Pause the loom on this sentence' : 'Play: cycle through the sentences');
        playBtn.setAttribute('aria-pressed', String(!on));
        playBtn.title = on ? 'Pause' : 'Play';
        // pausing while the cloth is unweaving keeps this sentence: it weaves back in and stays
        if (!on && st.phase === 'out' && st.next == null) { st.next = st.idx; st.pauseHold = true; }
        if (on && st.pauseHold) { st.next = null; st.pauseHold = false; }
        if (on && AM.reducedMotion) st.pt = 0;
      }
      setAuto(st.auto);

      /** How woven each layer is right now (null = fully), so an interrupted weave unweaves from where it was. */
      function wovenNow() {
        const ph = st.phase, p = st.pt;
        if (ph === 'in') { const e = easeOut(clamp(p / DUR.in)); return { bg: e, L: Array(NL).fill(GHOST * e) }; }
        if (ph === 'weave') {
          return {
            bg: 1,
            L: Array.from({ length: NL }, (_, l) => {
              const s0 = l * (DUR.band + DUR.swoop);
              return p < s0 ? GHOST : p < s0 + DUR.band ? GHOST + (1 - GHOST) * easeIO((p - s0) / DUR.band) : 1;
            }),
          };
        }
        return null;
      }
      function choose(i) {
        if (i === st.idx && st.phase !== 'out') return;
        pipBtns.forEach((b, j) => b.setAttribute('aria-pressed', String(j === i)));
        st.next = i; st.pauseHold = false;
        st.sel = -1; st.hover = -1;
        if (AM.reducedMotion) { switchTo(i); st.phase = 'rest'; st.pt = 0; frame(st.lastT, 0); return; }
        if (st.phase !== 'out') { st.outFrom = wovenNow(); st.phase = 'out'; st.pt = 0; }
      }
      function switchTo(i) {
        st.idx = i; st.next = null; st.outFrom = null; st.pauseHold = false;
        st.sel = -1; st.hover = -1; st.hvA = 0; st.hvI = -1;
        pipBtns.forEach((b, j) => b.setAttribute('aria-pressed', String(j === i)));
        if (!cv.w) return;
        ensureSprites();
        install(prepReady(i) ? Object.assign({}, prep) : prepare(i));
      }

      // ------------------------------------------------------------ timeline
      function advance(dt) {
        if (AM.reducedMotion) {
          if (st.phase !== 'rest') { st.phase = 'rest'; st.pt = 0; st.outFrom = null; st.next = null; }
          const a = st.sel >= 0 ? st.sel : st.hover;
          st.hvI = a; st.hvA = a >= 0 ? 1 : 0;
          // play: step to the next sentence every few seconds, with no animation
          if (st.auto && a < 0) {
            st.pt += dt;
            if (st.pt >= RM_STEP) { st.pt = 0; switchTo((st.idx + 1) % SENTENCES.length); }
          }
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
        // get the next sentence ready (one small stage per frame) while the loom is calm
        if ((ph === 'focus' || ph === 'rest' || ph === 'out') && (st.auto || st.next != null)) {
          const ni = st.next != null ? st.next : (st.idx + 1) % SENTENCES.length;
          if (!prepReady(ni)) stepPrep(ni);
        }
        // hovering or pinning a word freezes the cloth; pausing only stops it moving on to the next sentence
        const timed = ph === 'in' || ph === 'weave' || ph === 'out' || st.next != null || (act < 0 && (ph !== 'rest' || st.auto));
        if (timed) st.pt += dt;
        const dur = ph === 'weave' ? WEAVE : DUR[ph];
        if (st.pt >= dur) {
          st.pt = 0;
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
        const { g, w } = cv;
        if (!w || !G || !bands.length) return;
        const ph = st.phase, p = st.pt;
        // only the loom region is ever drawn on this canvas; the cloth canvas below holds the rest
        if (st.wipe) { cv.clear(); st.wipe = false; }
        else g.clearRect(G.dx0, G.ry0, G.dx1 - G.dx0, G.ry1 - G.ry0);
        g.save();
        g.beginPath(); g.rect(G.dx0, G.ry0, G.dx1 - G.dx0, G.ry1 - G.ry0); g.clip();

        const OF = ph === 'out' ? st.outFrom : null;
        let bgA = 1;
        if (ph === 'in') bgA = easeOut(clamp(p / DUR.in));
        if (ph === 'out') bgA = (OF ? OF.bg : 1) * (1 - smooth(0.3, 1, p / DUR.out));
        setClothAlpha(bgA);

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
              const u = (p - s0) / DUR.band;
              shuttle = weaveBand(g, l, u); shuttleL = l;
              reedA = Math.sin(Math.PI * clamp(u));
            } else {
              // a layer just finished: bloom once, then settle into the breathing cloth
              const since = p - s0 - DUR.band;
              blitBand(g, B, breath(l, t));
              const flash = 0.55 * Math.max(0, 1 - since / 0.7);
              if (flash > 0.01) { g.globalCompositeOperation = 'lighter'; blitBand(g, B, flash); g.globalCompositeOperation = 'source-over'; }
            }
          } else if (ph === 'out') {
            // layers flatten into their baselines from the top down, starting from however woven they were
            const q = clamp((p - (NL - 1 - l) * 0.12) / (DUR.out - 0.3));
            const e = easeIO(q);
            const base = OF ? OF.L[l] : 1;
            blitBand(g, B, (1 - e) * base * (base >= 0.999 ? breath(l, t) : 1), 1 - 0.88 * e, l);
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

        // decoration: slow beads of light climbing each warp through the layers
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

        // sparks travelling along the threads, from the earlier word to the later one, more often on strong ones
        const woven = ph === 'hold' || ph === 'focus' || ph === 'rest' || (ph === 'out' && !OF);
        if (!AM.reducedMotion && woven) {
          let pa = 1;
          if (ph === 'hold') pa = smooth(0, 0.8, p);
          if (ph === 'out') pa = 1 - smooth(0, 0.5, p);
          drawParticles(g, dt, pa * (1 - 0.5 * fe), act >= 0 && hvA > 0.05 ? act : -1);
        }

        // focus: one thread picked out, next to the model's prediction at that word
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
          drawPred(g, f.q, predText(f.pred), f.pp, fe);
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
          drawPred(g, act, predText(pr.token), pr.p, hvA);
        }

        if (shuttle && !AM.reducedMotion) drawShuttle(g, shuttle, shuttleL, reedA);
        g.restore();

        // words for the readout
        if (act >= 0 && st.hvA > 0.5) setRead(readWord(act));
        else if (ph === 'focus' && fe > 0.05) setRead(readFocus());
        else if (ph === 'rest' && AM.reducedMotion && data.focus) setRead(readFocus());
        else setRead(DEFAULT_READ);
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
      ctx.onVisible(() => { st.wipe = true; });
      ctx.loop((t, dt) => frame(t, dt));
    },
  });
})();
