/* The Attention Loom — core runtime.
   Global namespace AM: chapter registry, lifecycle (mount / visibility / loops),
   DPR-aware canvases, scrollytelling steps, colours and small math helpers. */
(() => {
  const AM = (window.AM = window.AM || {});

  // ------------------------------------------------------------------ motion
  const mq = window.matchMedia ? window.matchMedia('(prefers-reduced-motion: reduce)') : null;
  AM.reducedMotion = !!(mq && mq.matches);
  if (mq && mq.addEventListener) mq.addEventListener('change', (e) => { AM.reducedMotion = e.matches; });

  // ------------------------------------------------------------------ colour
  const css = getComputedStyle(document.documentElement);
  const v = (name, fallback) => (css.getPropertyValue(name).trim() || fallback);

  /** The seven natural dyes. Use these (and only these) as hues. */
  AM.dye = {
    weld: v('--weld', '#e9c24a'),
    madder: v('--madder', '#e3573f'),
    woad: v('--woad', '#6fa0f0'),
    cochineal: v('--cochineal', '#dd519e'),
    verdigris: v('--verdigris', '#40c4a3'),
    saffron: v('--saffron', '#f2973b'),
    lichen: v('--lichen', '#a286e9'),
  };
  /** Canonical order for categorical series (e.g. attention head 0, 1, 2 …). */
  AM.dyeList = ['weld', 'woad', 'madder', 'verdigris', 'cochineal', 'saffron', 'lichen'].map((k) => AM.dye[k]);
  AM.headColor = (h) => AM.dyeList[h % AM.dyeList.length];

  /** Neutrals. */
  AM.col = {
    ink: v('--ink', '#0d1121'),
    ink2: v('--ink-2', '#131a30'),
    ink3: v('--ink-3', '#1b2341'),
    rule: v('--rule', '#2a3355'),
    ruleStrong: v('--rule-strong', '#3a4673'),
    linen: v('--linen', '#ede5d3'),
    linenDim: v('--linen-dim', '#bdb6a8'),
    mist: v('--mist', '#8d94ae'),
  };

  const hexToRgb = (hex) => {
    let h = String(hex).replace('#', '').trim();
    if (h.length === 3) h = h.split('').map((c) => c + c).join('');
    const n = parseInt(h, 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  };
  AM.hexToRgb = hexToRgb;
  /** 'rgba(…)' string from a hex colour and alpha. */
  AM.rgba = (hex, a = 1) => {
    const [r, g, b] = hexToRgb(hex);
    return `rgba(${r},${g},${b},${a})`;
  };
  /** Linear mix of two hex colours → 'rgb(…)'. */
  AM.mix = (a, b, t) => {
    const A = hexToRgb(a), B = hexToRgb(b);
    const m = (i) => Math.round(A[i] + (B[i] - A[i]) * t);
    return `rgb(${m(0)},${m(1)},${m(2)})`;
  };
  const ramp = (stops, t) => {
    t = Math.max(0, Math.min(1, t));
    const n = stops.length - 1;
    const i = Math.min(n - 1, Math.floor(t * n));
    return AM.mix(stops[i], stops[i + 1], t * n - i);
  };
  AM.color = {
    /** Sequential "dye vat" ramp: ink → woad → weld → linen. t ∈ [0,1]. */
    seq: (t) => ramp([AM.col.ink3, '#2f4a8a', AM.dye.woad, AM.dye.weld, '#fbf1cf'], t),
    /** Warm ramp for attention weights: ink → madder → weld → cream. */
    heat: (t) => ramp([AM.col.ink3, '#6a2a3a', AM.dye.madder, AM.dye.weld, '#fff4d6'], t),
    /** Diverging: madder (negative) ← ink → woad (positive). v ∈ [-max, max]. */
    div: (v, max = 1) => {
      const t = Math.max(-1, Math.min(1, v / (max || 1)));
      return t < 0 ? AM.mix(AM.col.ink3, AM.dye.madder, -t) : AM.mix(AM.col.ink3, AM.dye.woad, t);
    },
    ramp,
  };

  // ------------------------------------------------------------------ math
  /** Deterministic PRNG (mulberry32). AM.math.rng(seed)() → [0,1). */
  const rng = (seed = 1) => {
    let a = seed >>> 0;
    return () => {
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  };
  AM.math = {
    rng,
    clamp: (x, a = 0, b = 1) => Math.max(a, Math.min(b, x)),
    lerp: (a, b, t) => a + (b - a) * t,
    invLerp: (a, b, x) => (x - a) / (b - a),
    smoothstep: (a, b, x) => { const t = Math.max(0, Math.min(1, (x - a) / (b - a))); return t * t * (3 - 2 * t); },
    ease: {
      inOut: (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2),
      out: (t) => 1 - Math.pow(1 - t, 3),
      in: (t) => t * t * t,
      outBack: (t) => { const c = 1.70158, c3 = c + 1; return 1 + c3 * Math.pow(t - 1, 3) + c * Math.pow(t - 1, 2); },
    },
    /** Standard normal sample from a uniform rng. */
    randn: (r = Math.random) => {
      let u = 0, w = 0;
      while (u === 0) u = r();
      while (w === 0) w = r();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * w);
    },
    range: (n) => Array.from({ length: n }, (_, i) => i),
    sum: (a) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i]; return s; },
    dot: (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; },
    norm: (a) => Math.sqrt(AM.math.dot(a, a)),
    add: (a, b) => a.map((x, i) => x + b[i]),
    sub: (a, b) => a.map((x, i) => x - b[i]),
    scale: (a, s) => a.map((x) => x * s),
    cosine: (a, b) => AM.math.dot(a, b) / (AM.math.norm(a) * AM.math.norm(b) || 1),
    argmax: (a) => { let bi = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[bi]) bi = i; return bi; },
    /** Numerically stable softmax with temperature. Returns a plain array. */
    softmax: (logits, temperature = 1) => {
      const T = Math.max(1e-6, temperature);
      let m = -Infinity;
      for (let i = 0; i < logits.length; i++) if (logits[i] > m) m = logits[i];
      const out = new Array(logits.length);
      let s = 0;
      for (let i = 0; i < logits.length; i++) {
        const e = logits[i] === -Infinity ? 0 : Math.exp((logits[i] - m) / T);
        out[i] = e; s += e;
      }
      for (let i = 0; i < out.length; i++) out[i] /= s;
      return out;
    },
    /** y = x · W where x is length n and W is n×m (array of rows). */
    vecMat: (x, W) => {
      const m = W[0].length, y = new Array(m).fill(0);
      for (let i = 0; i < x.length; i++) { const xi = x[i], row = W[i]; for (let j = 0; j < m; j++) y[j] += xi * row[j]; }
      return y;
    },
    /** C = A · B for arrays of rows. */
    matMul: (A, B) => A.map((row) => AM.math.vecMat(row, B)),
    transpose: (A) => A[0].map((_, j) => A.map((row) => row[j])),
    /** LayerNorm without learned gain/bias (for illustrations). */
    layerNorm: (x, eps = 1e-5) => {
      const n = x.length; let mu = 0; for (const a of x) mu += a; mu /= n;
      let v = 0; for (const a of x) v += (a - mu) * (a - mu); v /= n;
      const s = 1 / Math.sqrt(v + eps);
      return x.map((a) => (a - mu) * s);
    },
    gelu: (x) => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x))),
  };

  // ------------------------------------------------------------------ DOM
  /** Tiny DOM builder. AM.el('div', {class:'x', onclick: fn, html:'<b>…</b>'}, child, 'text') */
  AM.el = (tag, attrs, ...children) => {
    const isSvg = ['svg', 'g', 'path', 'rect', 'circle', 'line', 'text', 'polyline', 'polygon', 'defs', 'linearGradient', 'stop', 'tspan', 'ellipse'].includes(tag);
    const node = isSvg ? document.createElementNS('http://www.w3.org/2000/svg', tag) : document.createElement(tag);
    if (attrs) {
      for (const [k, val] of Object.entries(attrs)) {
        if (val == null || val === false) continue;
        if (k === 'html') node.innerHTML = val;
        else if (k === 'text') node.textContent = val;
        else if (k === 'style' && typeof val === 'object') Object.assign(node.style, val);
        else if (k.startsWith('on') && typeof val === 'function') node.addEventListener(k.slice(2), val);
        else if (k === 'class') node.setAttribute('class', val);
        else node.setAttribute(k, val === true ? '' : val);
      }
    }
    for (const c of children.flat(Infinity)) {
      if (c == null || c === false) continue;
      node.appendChild(typeof c === 'string' || typeof c === 'number' ? document.createTextNode(String(c)) : c);
    }
    return node;
  };
  /** Append a <style> block (chapter-scoped CSS: prefix every selector with #ch-<id>). */
  AM.css = (text) => { const s = document.createElement('style'); s.textContent = text; document.head.appendChild(s); return s; };

  /** Canvas font string for a role. */
  AM.font = (size, role = 'body', weight = role === 'display' ? 500 : role === 'mono' ? 400 : 500, italic = false) => {
    const fam = role === 'display' ? "'Bodoni Moda', 'Bodoni 72', Didot, Georgia, serif"
      : role === 'mono' ? "'Martian Mono', ui-monospace, Menlo, monospace"
        : "'Figtree', system-ui, -apple-system, 'Segoe UI', sans-serif";
    return `${italic ? 'italic ' : ''}${weight} ${size}px ${fam}`;
  };

  /** Smallest canvas text size D.text / D.measure / D.token will draw (px). Pass
      {minSize: 0} to opt out for one label. Touch screens get a slightly higher floor. */
  AM.minText = (window.matchMedia && window.matchMedia('(pointer: coarse)').matches) ? 8.5 : 8;

  // Web fonts load without blocking first paint: index.html links the Google Fonts
  // stylesheet with media="print" and flips it to "all" once it arrives. The inline
  // onload does that; this is the fallback for hosts whose CSP drops inline handlers.
  const fontLink = document.querySelector('link[data-am-fonts]');
  if (fontLink && fontLink.media !== 'all') {
    const apply = () => { fontLink.media = 'all'; };
    if (fontLink.sheet) apply(); else fontLink.addEventListener('load', apply, { once: true });
  }
  /** Resolves once the three web font families are usable (or failed, or after 10 s).
      Canvas code that caches text measurements can re-measure when it settles. */
  AM.fontsReady = new Promise((resolve) => {
    const fs = document.fonts;
    if (!fs || !fs.load) { resolve(); return; }
    const faces = ["500 16px 'Figtree'", "400 16px 'Martian Mono'", "500 16px 'Bodoni Moda'"];
    const loadFaces = () => Promise.all(faces.map((f) => fs.load(f).catch(() => null)))
      .then(() => fs.ready).then(() => resolve(), () => resolve());
    // After the stylesheet applies, wait one frame so its @font-face rules are registered.
    const afterSheet = () => requestAnimationFrame(() => setTimeout(loadFaces, 0));
    if (!fontLink || (fontLink.sheet && fontLink.media === 'all')) afterSheet();
    else {
      fontLink.addEventListener('load', afterSheet, { once: true });
      fontLink.addEventListener('error', () => resolve(), { once: true });
    }
    setTimeout(resolve, 10000);
  });

  // ------------------------------------------------------------------ registry & lifecycle
  const defs = [];
  const live = []; // mounted chapter contexts
  AM.chapters = defs;

  /** Register a chapter. See docs/CONTRACT.md §3. */
  AM.chapter = (def) => {
    if (!def || !def.id || typeof def.mount !== 'function') { console.error('AM.chapter: bad definition', def); return; }
    defs.push(def);
    if (booted) mountOne(def); // late registration still works
  };

  let booted = false;

  function makeCtx(def, root) {
    const ctx = {
      id: def.id,
      def,
      root,
      visible: false,
      _loops: [],
      _onVisible: [],
      _onHidden: [],
      _resizers: [],
      t0: performance.now(),
    };

    ctx.el = AM.el;

    ctx.header = () => {
      const head = AM.el('header', { class: 'ch-head' },
        def.num != null ? AM.el('div', { class: 'ch-num', 'aria-hidden': 'true' }, String(def.num).padStart(2, '0')) : null,
        def.kicker ? AM.el('div', { class: 'ch-kicker' }, def.kicker) : null,
        AM.el('h2', { class: 'ch-title', html: def.title || '' }),
        def.lede ? AM.el('p', { class: 'ch-lede', html: def.lede }) : null,
        def.where && AM.ui && AM.ui.glyph ? AM.el('div', { class: 'ch-glyph' }, AM.ui.glyph(def.where)) : null,
      );
      root.insertBefore(head, root.firstChild);
      return head;
    };

    ctx.loop = (fn) => { ctx._loops.push(fn); if (ctx.visible) ensureTicking(); return () => { const i = ctx._loops.indexOf(fn); if (i >= 0) ctx._loops.splice(i, 1); }; };
    ctx.onVisible = (fn) => { ctx._onVisible.push(fn); if (ctx.visible) fn(); };
    ctx.onHidden = (fn) => { ctx._onHidden.push(fn); };

    /** DPR-aware canvas that tracks its container width.
        opts.maxHeight may be a number, a function of width, or 'stage': when the
        chapter's split is stacked (phones and tablets), the whole sticky stage is
        then kept within --stage-max (half the screen, at most 560px). */
    ctx.canvas = (parent, opts = {}) => {
      const wrap = AM.el('div', { class: 'stage-canvas' });
      const canvas = AM.el('canvas');
      if (opts.label) { canvas.setAttribute('role', 'img'); canvas.setAttribute('aria-label', opts.label); }
      wrap.appendChild(canvas);
      if (parent) parent.appendChild(wrap);
      const g = canvas.getContext('2d');
      const api = { canvas, wrap, g, w: 0, h: 0, dpr: 1, _cbs: [] };
      api.onResize = (fn) => { api._cbs.push(fn); if (api.w) fn(api.w, api.h); return api; };
      /** Re-measure (cw: a known content width, which skips the layout read). */
      api.resize = (cw) => {
        const w = Math.max(1, Math.round((typeof cw === 'number' && cw > 0 ? cw : 0) || wrap.clientWidth || parent?.clientWidth || 600));
        let h;
        if (typeof opts.height === 'function') h = opts.height(w);
        else if (typeof opts.height === 'number') h = opts.height;
        else h = w / (opts.aspect || 16 / 9);
        if (opts.minHeight) h = Math.max(opts.minHeight, h);
        if (opts.maxHeight === 'stage') h = Math.min(stageCap(wrap, api.h), h);
        else if (opts.maxHeight) h = Math.min(typeof opts.maxHeight === 'function' ? opts.maxHeight(w) : opts.maxHeight, h);
        h = Math.round(h);
        const dpr = Math.min(window.devicePixelRatio || 1, opts.maxDpr || 2);
        if (w === api.w && h === api.h && dpr === api.dpr) return;
        api.w = w; api.h = h; api.dpr = dpr;
        canvas.width = Math.round(w * dpr);
        canvas.height = Math.round(h * dpr);
        canvas.style.height = h + 'px';
        g.setTransform(dpr, 0, 0, dpr, 0, 0);
        for (const fn of api._cbs) { try { fn(w, h); } catch (e) { console.error(`[${def.id}] resize`, e); } }
      };
      /** Clear to transparent (or a colour). */
      api.clear = (color) => {
        g.save(); g.setTransform(1, 0, 0, 1, 0, 0);
        if (color) { g.fillStyle = color; g.fillRect(0, 0, canvas.width, canvas.height); } else g.clearRect(0, 0, canvas.width, canvas.height);
        g.restore();
      };
      /** Pointer position in CSS px relative to the canvas. */
      api.pointer = (ev) => { const r = canvas.getBoundingClientRect(); return { x: ev.clientX - r.left, y: ev.clientY - r.top }; };
      // The observer reports the wrap's width itself (no forced layout), including its
      // first size once attached and laid out, so no extra per-canvas rAF pass is needed.
      if (typeof ResizeObserver !== 'undefined') {
        new ResizeObserver((entries) => {
          const e = entries[entries.length - 1];
          if (e && e.contentRect.width > 0) api.resize(e.contentRect.width);
        }).observe(wrap);
      } else {
        window.addEventListener('resize', () => api.resize());
        requestAnimationFrame(() => api.resize());
      }
      ctx._resizers.push(api);
      api.resize(); // synchronous first size: chapters may draw straight after creating it
      return api;
    };

    /** A scrollytelling step card: "n · label", a title and body HTML (or nodes).
        n counts up per chapter unless given. */
    ctx._stepN = 0;
    ctx.step = ({ n, label = '', title, html, body, cls } = {}) => {
      const num = n != null ? n : ++ctx._stepN;
      if (n != null) ctx._stepN = n;
      const card = AM.el('div', { class: 'step' + (cls ? ' ' + cls : '') },
        AM.el('div', { class: 'step-label' }, label ? `${num} · ${label}` : String(num)),
        title ? AM.el('h3', { html: title }) : null);
      if (html) { const t = document.createElement('template'); t.innerHTML = html; card.appendChild(t.content); }
      if (body) card.append(...[body].flat(Infinity).filter(Boolean));
      return card;
    };

    /** Opening of a section after the scrollytelling: gold eyebrow + h3 (+ optional lead). */
    ctx.subhead = (eyebrow, title, lead) => AM.el('header', { class: 'subhead' },
      eyebrow ? AM.el('p', { class: 'subhead-eyebrow' }, eyebrow) : null,
      AM.el('h3', { html: title || '' }),
      lead ? AM.el('p', { class: 'subhead-lead', html: lead }) : null);

    /** Scrollytelling: onStep(i, el) when the i-th element crosses the middle of the viewport. */
    ctx.steps = (els, onStep) => {
      els = Array.from(els);
      let current = -1;
      const set = (i) => {
        if (i === current) return;
        current = i;
        els.forEach((el, j) => el.classList.toggle('is-active', j === i));
        try { onStep(i, els[i]); } catch (e) { console.error(`[${def.id}] step`, e); }
      };
      // The active step is the last one whose top edge has crossed a line a little
      // below mid-screen. Computed from layout on scroll (not from intersection
      // events) so it stays right after instant jumps, rail clicks and resizes.
      // When the split is stacked (stage sticky on top, about as wide as the split),
      // the line moves down to just below the stage, so a step only activates once
      // its label and first lines have cleared the picture.
      const split = els[0] ? els[0].closest('.ch-split') : null;
      const stage = split ? Array.from(split.children).find((c) => c.classList.contains('ch-stage')) : null;
      const pick = () => {
        const vh = window.innerHeight;
        let line = vh * 0.58;
        if (stage) {
          const sr = stage.getBoundingClientRect();
          if (sr.width > split.getBoundingClientRect().width * 0.8 && sr.height > 0) {
            line = Math.min(vh * 0.9, Math.max(line, sr.bottom + 32));
          }
        }
        let idx = 0;
        for (let j = 0; j < els.length; j++) if (els[j].getBoundingClientRect().top < line) idx = j;
        set(idx);
      };
      stepPickers.push({ ctx, pick });
      set(0);
      ctx.onVisible(pick);
      return { set, get current() { return current; } };
    };

    return ctx;
  }

  // Sticky stage height per split (--stage-h), so focused links in the prose can keep
  // clear of a stacked stage (see scroll-margin-top in base.css).
  const stageRO = typeof ResizeObserver !== 'undefined' ? new ResizeObserver((entries) => {
    for (const e of entries) {
      const split = e.target.parentElement;
      if (split) split.style.setProperty('--stage-h', Math.round(e.target.getBoundingClientRect().height) + 'px');
    }
  }) : null;

  /** Max canvas height for opts.maxHeight === 'stage' (see ctx.canvas). */
  const stackedMQ = window.matchMedia ? window.matchMedia('(max-width: 900px)') : null;
  function stageCap(wrap, curH) {
    if (!stackedMQ || !stackedMQ.matches) return Infinity;
    const stage = wrap.closest('.ch-stage');
    if (!stage) return Infinity;
    const budget = Math.min(window.innerHeight * 0.5, 560);
    // Everything in the stage that is not this canvas: titles, controls, legends, padding.
    const other = Math.max(0, stage.offsetHeight - (wrap.offsetHeight || curH || 0));
    return Math.max(200, budget - other);
  }

  function mountOne(def) {
    const root = document.querySelector(`[data-chapter="${def.id}"]`);
    if (!root) { console.warn(`AM: no <section data-chapter="${def.id}">`); return; }
    if (root._amMounted) return;
    root._amMounted = true;
    if (!root.id) root.id = 'ch-' + def.id;
    const ctx = makeCtx(def, root);
    try {
      def.mount(root, ctx);
    } catch (e) {
      console.error(`[${def.id}] mount failed`, e);
      root.appendChild(AM.el('div', { class: 'ch-error' }, `Chapter "${def.id}" failed to load: ${e && e.message}`));
    }
    live.push(ctx);
    if (stageRO) root.querySelectorAll('.ch-split > .ch-stage').forEach((s) => stageRO.observe(s));
    if (io) io.observe(root);
  }

  // Boot mounts the hero at once, then one chapter per task in page order, so the
  // page paints and scrolls while the rest is still being built. Deep links, in-page
  // link clicks and scroll restoration build what they need synchronously.
  let pending = [];
  let pumpTimer = 0;
  let resolveMounted;
  /** Resolves when every registered chapter has mounted. */
  AM.whenMounted = new Promise((r) => { resolveMounted = r; });
  AM.mountedAll = false;
  function allMounted() {
    if (AM.mountedAll) return;
    AM.mountedAll = true;
    AM.mountedAt = performance.now();
    resolveMounted();
    try { document.dispatchEvent(new CustomEvent('am:mounted')); } catch (e) { /* old browsers */ }
  }
  function pump() {
    pumpTimer = 0;
    const def = pending.shift();
    if (def) mountOne(def);
    if (pending.length) pumpTimer = setTimeout(pump, 0);
    else allMounted();
  }
  /** Mount, right now and in order, every pending chapter up to and including `id`. */
  function mountThrough(id) {
    const i = pending.findIndex((d) => d.id === id);
    if (i < 0) return false;
    pending.splice(0, i + 1).forEach(mountOne);
    if (!pending.length) { clearTimeout(pumpTimer); pumpTimer = 0; allMounted(); }
    return true;
  }
  function mountAll() {
    const all = pending; pending = [];
    clearTimeout(pumpTimer); pumpTimer = 0;
    all.forEach(mountOne);
    allMounted();
  }
  AM.mountAll = mountAll;

  // Scroll-driven step pickers (see ctx.steps), throttled to one pass per frame.
  const stepPickers = [];
  let stepRaf = 0;
  const runPickers = () => {
    stepRaf = 0;
    for (const p of stepPickers) if (p.ctx.visible) { try { p.pick(); } catch (e) { console.error(e); } }
  };
  window.addEventListener('scroll', () => { if (!stepRaf) stepRaf = requestAnimationFrame(runPickers); }, { passive: true });
  window.addEventListener('resize', () => { if (!stepRaf) stepRaf = requestAnimationFrame(runPickers); });

  // Visibility tracking: one observer for all chapters.
  let io = null;
  if (typeof IntersectionObserver !== 'undefined') {
    io = new IntersectionObserver((entries) => {
      for (const en of entries) {
        const ctx = live.find((c) => c.root === en.target);
        if (!ctx) continue;
        const was = ctx.visible;
        ctx.visible = en.isIntersecting;
        ctx.root.classList.toggle('is-onscreen', ctx.visible);
        if (ctx.visible && !was) {
          if (ctx._fontsDirty) refireResizers(ctx);
          ctx._onVisible.forEach((f) => { try { f(); } catch (e) { console.error(e); } });
          ensureTicking();
        }
        if (!ctx.visible && was) ctx._onHidden.forEach((f) => { try { f(); } catch (e) { console.error(e); } });
      }
    }, { rootMargin: '15% 0px 15% 0px', threshold: 0 });
  }

  // One shared animation loop; only visible chapters tick.
  let ticking = false;
  let last = performance.now();
  let lastReduced = 0;
  function ensureTicking() {
    if (ticking) return;
    ticking = true;
    last = performance.now();
    requestAnimationFrame(tick);
  }
  function tick(now) {
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;
    let any = false;
    const reduced = AM.reducedMotion;
    const runThisFrame = !reduced || now - lastReduced > 500;
    if (runThisFrame && reduced) lastReduced = now;
    for (const ctx of live) {
      if (!ctx.visible || !ctx._loops.length) continue;
      any = true;
      if (document.hidden || !runThisFrame) continue;
      const t = (now - ctx.t0) / 1000;
      for (const fn of ctx._loops) {
        try { fn(t, reduced ? 0.5 : dt); } catch (e) { console.error(`[${ctx.id}] loop`, e); ctx._loops.splice(ctx._loops.indexOf(fn), 1); }
      }
    }
    if (any) requestAnimationFrame(tick);
    else ticking = false;
  }

  // ------------------------------------------------------------------ navigation rail + progress
  function buildRail() {
    const sorted = defs.filter((d) => document.querySelector(`[data-chapter="${d.id}"]`)).sort((a, b) => (a.num ?? -1) - (b.num ?? -1));
    const numbered = sorted.filter((d) => d.num != null);
    if (!numbered.length) return;
    const fill = AM.el('span', { class: 'rail-fill', 'aria-hidden': 'true' });
    const links = numbered.map((d) => AM.el('a', { href: '#ch-' + d.id, 'aria-label': `${d.num}. ${d.kicker || d.id}` },
      String(d.num).padStart(2, '0'),
      AM.el('span', { class: 'rail-label', 'aria-hidden': 'true' }, d.kicker || d.id)));
    const nav = AM.el('nav', { class: 'rail', 'aria-label': 'Chapters' }, AM.el('div', { class: 'rail-track' }, fill, links));
    const top = AM.el('div', { class: 'topthread', 'aria-hidden': 'true' });
    document.body.appendChild(nav);
    document.body.appendChild(top);
    const sections = numbered.map((d) => document.querySelector(`[data-chapter="${d.id}"]`));
    let raf = 0;
    const update = () => {
      raf = 0;
      const mid = window.innerHeight * 0.4;
      let cur = -1;
      sections.forEach((s, i) => { if (s.getBoundingClientRect().top < mid) cur = i; });
      links.forEach((a, i) => { a.classList.toggle('is-current', i === cur); a.classList.toggle('is-past', i < cur); });
      const track = nav.querySelector('.rail-track');
      if (cur >= 0 && links[cur]) {
        const r = links[cur].getBoundingClientRect(), tr = track.getBoundingClientRect();
        fill.style.height = (r.top + r.height / 2 - tr.top) + 'px';
      } else fill.style.height = '0px';
      const max = document.documentElement.scrollHeight - window.innerHeight;
      top.style.width = (max > 0 ? (window.scrollY / max) * 100 : 0) + '%';
    };
    window.addEventListener('scroll', () => { if (!raf) raf = requestAnimationFrame(update); }, { passive: true });
    window.addEventListener('resize', update);
    update();
  }

  // Contents: the whole journey at a glance, right under the hero.
  function buildToc() {
    const nav = document.getElementById('toc');
    if (!nav) return;
    const items = defs.filter((d) => d.num != null && document.querySelector(`[data-chapter="${d.id}"]`)).sort((a, b) => a.num - b.num);
    if (!items.length) return;
    nav.appendChild(AM.el('p', { class: 'toc-label' }, 'The journey, in the order data flows through the model'));
    nav.appendChild(AM.el('ol', { class: 'toc-list' }, items.map((d) => AM.el('li', {},
      AM.el('a', { href: '#ch-' + d.id },
        AM.el('span', { class: 'toc-num', 'aria-hidden': 'true' }, String(d.num).padStart(2, '0')),
        AM.el('span', { class: 'toc-text' },
          AM.el('span', { class: 'toc-kicker', html: d.kicker || '' }),
          AM.el('span', { class: 'toc-title', html: d.title || d.id })))))));
  }

  // When web fonts arrive, canvases redraw so their text uses the right face. Chapters
  // on screen redraw at once; the rest are marked and redraw when they next appear.
  function refireResizers(c) {
    c._fontsDirty = false;
    c._resizers.forEach((r) => { const w = r.w; r.w = 0; r.resize(); if (!r.w) r.w = w; });
  }
  let fontTimer = 0;
  function fontsChanged() {
    clearTimeout(fontTimer);
    fontTimer = setTimeout(() => {
      for (const c of live) { if (c.visible) refireResizers(c); else c._fontsDirty = true; }
    }, 80);
  }

  AM.boot = () => {
    if (booted) return;
    booted = true;
    // The single-file fragment build loses <html lang>; screen readers need it.
    if (!document.documentElement.lang) document.documentElement.lang = 'en';
    const order = defs.slice().sort((a, b) => (a.num ?? -1) - (b.num ?? -1));
    buildRail();
    buildToc();

    pending = order.filter((d) => document.querySelector(`[data-chapter="${d.id}"]`));
    // Before jumping to a fragment, build every chapter above (and holding) its target.
    const buildFor = (frag) => {
      if (!pending.length || !frag) return;
      let id = frag; try { id = decodeURIComponent(frag); } catch (e) { /* keep raw */ }
      const target = document.getElementById(id);
      if (!target) { mountAll(); return; } // probably inside a chapter not built yet
      const sec = target.closest('[data-chapter]');
      if (sec) { mountThrough(sec.getAttribute('data-chapter')); return; }
      const above = pending.filter((d) => {
        const s = document.querySelector(`[data-chapter="${d.id}"]`);
        return s && (s.compareDocumentPosition(target) & Node.DOCUMENT_POSITION_FOLLOWING);
      });
      if (above.length) mountThrough(above[above.length - 1].id);
    };
    const nav = performance.getEntriesByType ? performance.getEntriesByType('navigation')[0] : null;
    if (nav && (nav.type === 'reload' || nav.type === 'back_forward')) {
      mountAll(); // the browser is about to restore an old scroll position
    } else {
      pending.splice(0, 1).forEach(mountOne); // the hero; chapter 1 follows in the next task
      // Deep link: build what lies above the target, then land on it at once (the
      // browser's own fragment scroll would glide there from the top).
      const frag = location.hash.slice(1);
      if (frag) {
        buildFor(frag);
        let target = null; try { target = document.getElementById(decodeURIComponent(frag)); } catch (e) { /* bad escape */ }
        if (target && target.id !== 'top') { target.scrollIntoView({ behavior: 'instant', block: 'start' }); requestAnimationFrame(() => AM.settleOn && AM.settleOn(target)); }
      }
      if (pending.length) pumpTimer = setTimeout(pump, 0);
      else allMounted();
    }
    // Clicks on in-page links (contents, rail, cross-references) build up to the target
    // first, so the jump lands on the finished layout.
    document.addEventListener('click', (ev) => {
      if (!pending.length) return;
      const a = ev.target && ev.target.closest ? ev.target.closest('a[href^="#"]') : null;
      if (a) buildFor(a.getAttribute('href').slice(1));
    }, true);
    window.addEventListener('hashchange', () => buildFor(location.hash.slice(1)));

    // Some chapters add content the first time they come into view, which can push an
    // in-page jump's target down after the browser has landed. Once scrolling settles,
    // re-align on the target for a few seconds, unless the reader takes over.
    const settleOn = (target) => {
      if (!target || target.id === 'top') return;
      let cancelled = false, last = performance.now();
      const start = last;
      const user = () => { cancelled = true; };
      const moved = () => { last = performance.now(); };
      const evs = ['wheel', 'touchstart', 'keydown'];
      evs.forEach((e) => window.addEventListener(e, user, { passive: true }));
      window.addEventListener('scroll', moved, { passive: true });
      const done = () => { evs.forEach((e) => window.removeEventListener(e, user)); window.removeEventListener('scroll', moved); };
      const margin = parseFloat(getComputedStyle(target).scrollMarginTop) || 0;
      const tick = () => {
        if (cancelled) { done(); return; }
        const now = performance.now();
        if (now - last > 140) {
          const off = target.getBoundingClientRect().top - margin;
          const room = document.documentElement.scrollHeight - window.innerHeight - window.scrollY;
          if (Math.abs(off) > 4 && !(off > 0 && room < 2)) { window.scrollTo({ top: window.scrollY + off, behavior: 'instant' }); last = now; }
        }
        if (now - start < 3000) requestAnimationFrame(tick); else done();
      };
      requestAnimationFrame(tick);
    };
    AM.settleOn = settleOn;
    document.addEventListener('click', (ev) => {
      const a = ev.target && ev.target.closest ? ev.target.closest('a[href^="#"]') : null;
      if (!a || ev.defaultPrevented || ev.button !== 0 || ev.metaKey || ev.ctrlKey || ev.shiftKey) return;
      let id = a.getAttribute('href').slice(1); try { id = decodeURIComponent(id); } catch (e) { /* keep raw */ }
      const target = id && document.getElementById(id);
      if (target) setTimeout(() => settleOn(target), 0);
    });

    // Keyboard focus must not land under a stacked sticky stage (WCAG 2.4.11). The
    // browser does not scroll an element that is inside the viewport, even when the
    // stage covers it, so nudge the page until the element sits below the stage.
    document.addEventListener('focusin', (ev) => {
      const t = ev.target;
      if (!t || !t.closest || !t.closest('.ch-prose')) return;
      try { if (!t.matches(':focus-visible')) return; } catch (e) { /* old browsers: carry on */ }
      const split = t.closest('.ch-split');
      const stage = split && Array.from(split.children).find((c) => c.classList.contains('ch-stage'));
      if (!stage) return;
      const clear = () => {
        const sr = stage.getBoundingClientRect();
        if (sr.width < split.getBoundingClientRect().width * 0.8) return false; // side by side
        const tr = t.getBoundingClientRect();
        if (!(tr.top < sr.bottom + 8 && tr.bottom > sr.top)) return false;
        window.scrollBy({ top: tr.top - sr.bottom - 16, behavior: 'instant' });
        return true;
      };
      // Check again after the step change the scroll may trigger (stages change height).
      if (clear()) requestAnimationFrame(() => requestAnimationFrame(() => { if (document.activeElement === t) clear(); }));
    });

    AM.fontsReady.then(fontsChanged);
    if (document.fonts && document.fonts.addEventListener) document.fonts.addEventListener('loadingdone', fontsChanged);
  };

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', () => AM.boot());
  else setTimeout(() => AM.boot(), 0);
})();
