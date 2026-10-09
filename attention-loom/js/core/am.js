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
  AM.fontsReady = (document.fonts && document.fonts.ready) ? document.fonts.ready.catch(() => {}) : Promise.resolve();

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

    /** DPR-aware canvas that tracks its container width. */
    ctx.canvas = (parent, opts = {}) => {
      const wrap = AM.el('div', { class: 'stage-canvas' });
      const canvas = AM.el('canvas');
      if (opts.label) { canvas.setAttribute('role', 'img'); canvas.setAttribute('aria-label', opts.label); }
      wrap.appendChild(canvas);
      if (parent) parent.appendChild(wrap);
      const g = canvas.getContext('2d');
      const api = { canvas, wrap, g, w: 0, h: 0, dpr: 1, _cbs: [] };
      api.onResize = (fn) => { api._cbs.push(fn); if (api.w) fn(api.w, api.h); return api; };
      api.resize = () => {
        const w = Math.max(1, Math.round(wrap.clientWidth || parent?.clientWidth || 600));
        let h;
        if (typeof opts.height === 'function') h = opts.height(w);
        else if (typeof opts.height === 'number') h = opts.height;
        else h = w / (opts.aspect || 16 / 9);
        if (opts.minHeight) h = Math.max(opts.minHeight, h);
        if (opts.maxHeight) h = Math.min(typeof opts.maxHeight === 'function' ? opts.maxHeight(w) : opts.maxHeight, h);
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
      if (typeof ResizeObserver !== 'undefined') new ResizeObserver(() => api.resize()).observe(wrap);
      else window.addEventListener('resize', api.resize);
      ctx._resizers.push(api);
      requestAnimationFrame(() => api.resize());
      api.resize();
      return api;
    };

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
      const pick = () => {
        const line = window.innerHeight * 0.58;
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
    if (io) io.observe(root);
  }

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
        if (ctx.visible && !was) { ctx._onVisible.forEach((f) => { try { f(); } catch (e) { console.error(e); } }); ensureTicking(); }
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

  AM.boot = () => {
    if (booted) return;
    booted = true;
    defs.slice().sort((a, b) => (a.num ?? -1) - (b.num ?? -1)).forEach(mountOne);
    buildRail();
    // When web fonts arrive, re-fire canvas resize callbacks so text redraws in the right face.
    AM.fontsReady.then(() => live.forEach((c) => c._resizers.forEach((r) => { const w = r.w; r.w = 0; r.resize(); if (!r.w) r.w = w; })));
  };

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', () => AM.boot());
  else setTimeout(() => AM.boot(), 0);
})();
