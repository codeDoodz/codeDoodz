/* The Attention Loom — shared UI widgets. All return plain DOM nodes (or small
   handles with .el plus setters) styled by css/base.css. */
(() => {
  const AM = window.AM;
  const el = AM.el;
  const ui = (AM.ui = AM.ui || {});

  const BADGE_LABEL = { live: 'Live model', toy: 'Toy numbers', illustration: 'Illustration' };
  const BADGE_TITLE = {
    live: 'Computed right now by the real tiny transformer that ships with this page',
    toy: 'Small hand-built numbers chosen to show the mechanism clearly',
    illustration: 'An artistic depiction, not literal numbers',
  };
  /**
   * Honesty badge: kind 'live' | 'toy' | 'illustration'. The canonical label always
   * shows, so a badge reads as one of three things at a glance; `note` is an optional
   * qualifier drawn after a thin divider: ui.badge('toy', 'exact formula').
   * (A note that starts with the canonical label, like 'Live model · the pattern',
   * has that prefix dropped.)
   */
  ui.badge = (kind = 'illustration', note) => {
    const label = BADGE_LABEL[kind] || kind;
    let q = note == null ? '' : String(note).trim();
    const lower = q.toLowerCase(), canon = label.toLowerCase();
    if (lower === canon) q = '';
    else if (lower.startsWith(canon)) q = q.slice(label.length).replace(/^\s*[·:|,\-–]\s*/, '');
    else if (kind === 'live' && /^live\s+/i.test(q)) q = q.replace(/^live\s+/i, '');
    if (/^[A-Z][a-z]/.test(q)) q = q[0].toLowerCase() + q.slice(1); // 'Exact formula' → 'exact formula'; 'BPE' stays
    const title = (BADGE_TITLE[kind] || '') + (q ? ` (${q})` : '');
    return el('span', { class: `badge badge-${kind}`, title: title || null },
      label, q ? el('span', { class: 'badge-note' }, q) : null);
  };

  /**
   * <figure> with a top row (title + badge), body content and caption.
   * framed: true puts the body in a bordered --ink-2 frame with the title row
   * outside it. Frame every sticky stage and free-play figure; leave inline
   * diagrams inside prose unframed.
   */
  ui.figure = ({ title, badge, caption, cls, framed } = {}, ...content) => {
    const top = (title || badge) ? el('div', { class: 'fig-top' },
      title ? el('span', { class: 'fig-title' }, title) : el('span'),
      badge ? (typeof badge === 'string' ? ui.badge(badge) : badge) : null) : null;
    const body = framed ? el('div', { class: 'fig-frame' }, ...content) : content;
    return el('figure', { class: 'fig' + (framed ? ' is-framed' : '') + (cls ? ' ' + cls : '') }, top, body,
      caption ? el('figcaption', { html: caption }) : null);
  };

  /**
   * Example chips ("try this sentence"). options: strings or {value, label, title}.
   * opts: {id, label (aria group label), selected, onPick(value, i)}.
   * Returns {el, chips, get(), set(value, fire)}.
   */
  ui.chips = (options = [], { id, label = 'Examples', selected, onPick } = {}) => {
    const opts = options.map((o) => (typeof o === 'object' ? o : { value: o, label: String(o) }));
    let cur = selected;
    const chips = opts.map((o, i) => el('button', {
      type: 'button', class: 'chip', id: id ? `${id}-${i}` : null, title: o.title || null,
      'aria-pressed': String(o.value === cur),
      onclick: () => api.set(o.value, true),
    }, o.label));
    const api = {
      el: el('div', { class: 'chips', role: 'group', 'aria-label': label }, chips), chips,
      get: () => cur,
      set: (v, fire = false) => {
        cur = v;
        chips.forEach((c, i) => c.setAttribute('aria-pressed', String(opts[i].value === v)));
        if (fire && onPick) onPick(v, opts.findIndex((o) => o.value === v));
      },
    };
    return api;
  };

  /** Range slider. Returns {el, input, get(), set(v, fire)}. */
  ui.slider = ({ id, label, min = 0, max = 1, step = 0.01, value = 0, format = (x) => String(x), onInput } = {}) => {
    const out = el('span', { class: 'ctl-value' });
    const input = el('input', { type: 'range', id, min, max, step, value, 'aria-label': label });
    const wrap = el('label', { class: 'ctl ctl-range', for: id }, el('span', { class: 'ctl-label' }, el('span', {}, label), out), input);
    const paint = () => {
      const v = parseFloat(input.value);
      out.textContent = format(v);
      input.style.setProperty('--fill', ((v - min) / (max - min)) * 100 + '%');
    };
    input.addEventListener('input', () => { paint(); onInput && onInput(parseFloat(input.value)); });
    paint();
    return {
      el: wrap, input,
      get: () => parseFloat(input.value),
      set: (v, fire = false) => { input.value = v; paint(); if (fire && onInput) onInput(parseFloat(input.value)); },
    };
  };

  /** Segmented control. options: [{value, label}] or strings. Returns {el, get(), set(v, fire)}. */
  ui.segmented = ({ id, label, options = [], value, onChange } = {}) => {
    const opts = options.map((o) => (typeof o === 'object' ? o : { value: o, label: String(o) }));
    let cur = value ?? opts[0]?.value;
    const btns = opts.map((o, i) => el('button', {
      type: 'button', id: id ? `${id}-${i}` : null, 'aria-pressed': String(o.value === cur),
      onclick: () => api.set(o.value, true),
    }, o.label));
    const group = el('div', { class: 'seg', role: 'group', 'aria-label': label || null }, btns);
    const wrap = label ? el('div', { class: 'ctl' }, el('span', { class: 'ctl-label' }, label), group) : group;
    const api = {
      el: wrap,
      get: () => cur,
      set: (v, fire = false) => {
        cur = v;
        btns.forEach((b, i) => b.setAttribute('aria-pressed', String(opts[i].value === v)));
        if (fire && onChange) onChange(v);
      },
    };
    return api;
  };

  /** Toggle switch. Returns {el, get(), set(b, fire)}. */
  ui.toggle = ({ id, label, checked = false, onChange } = {}) => {
    const input = el('input', { type: 'checkbox', id, role: 'switch' });
    input.checked = checked;
    input.addEventListener('change', () => onChange && onChange(input.checked));
    const wrap = el('label', { class: 'toggle', for: id }, input, el('span', { class: 'toggle-track', 'aria-hidden': 'true' }), el('span', {}, label));
    return { el: wrap, input, get: () => input.checked, set: (b, fire = false) => { input.checked = b; if (fire && onChange) onChange(b); } };
  };

  /** Button. kind: 'primary' | undefined. */
  ui.button = ({ id, label, onClick, kind, title } = {}) =>
    el('button', { type: 'button', id, class: 'btn' + (kind === 'primary' ? ' btn-primary' : ''), title, onclick: onClick }, label);

  /**
   * Row of token chips. tokens: string[] (or {text, id, unknown}).
   * opts: {selected, onSelect(i), colors: (i)=>css colour, showIds}
   * Returns {el, chips, select(i)}.
   */
  ui.tokens = (tokens, { selected = -1, onSelect, colors, showIds = false, label = 'Tokens' } = {}) => {
    const items = tokens.map((t) => (typeof t === 'object' ? t : { text: t }));
    const chips = items.map((t, i) => {
      const chip = el('button', {
        type: 'button', class: 'tok' + (t.unknown ? ' is-unknown' : ''), 'aria-pressed': String(i === selected),
        onclick: () => { api.select(i); onSelect && onSelect(i); },
      }, t.text, showIds && t.id != null ? el('span', { class: 'tok-id' }, String(t.id)) : null);
      if (colors) chip.style.setProperty('--tok-color', colors(i));
      return chip;
    });
    const row = el('div', { class: 'toks', role: 'group', 'aria-label': label }, chips);
    const api = {
      el: row, chips,
      select: (i) => chips.forEach((c, j) => c.setAttribute('aria-pressed', String(i === j))),
    };
    return api;
  };

  /**
   * Numeric matrix / heatmap as a DOM grid.
   * opts: {data: number[][], rowLabels, colLabels, format, color(v,i,j), textColor(v,i,j), onHover(i,j), corner}
   * Returns {el, update(data), highlight(i,j)}.
   */
  ui.matrix = ({ data, rowLabels, colLabels, format = (x) => x.toFixed(2), color, textColor, onHover, corner = '' } = {}) => {
    const R = data.length, C = data[0].length;
    const hasR = !!rowLabels, hasC = !!colLabels;
    const grid = el('div', { class: 'matrix', role: 'table' });
    grid.style.gridTemplateColumns = `${hasR ? 'auto ' : ''}repeat(${C}, minmax(34px, auto))`;
    if (hasC) {
      if (hasR) grid.appendChild(el('div', { class: 'mx-head' }, corner));
      for (let j = 0; j < C; j++) grid.appendChild(el('div', { class: 'mx-head', role: 'columnheader' }, colLabels[j]));
    }
    const cells = [];
    for (let i = 0; i < R; i++) {
      if (hasR) grid.appendChild(el('div', { class: 'mx-head row', role: 'rowheader' }, rowLabels[i]));
      cells.push([]);
      for (let j = 0; j < C; j++) {
        const c = el('div', { class: 'mx-cell', role: 'cell' });
        if (onHover) {
          c.addEventListener('pointerenter', () => onHover(i, j));
          c.addEventListener('pointerleave', () => onHover(-1, -1));
        }
        cells[i].push(c);
        grid.appendChild(c);
      }
    }
    const api = {
      el: el('div', { class: 'matrix-wrap' }, grid), cells,
      update: (d) => {
        for (let i = 0; i < R; i++) for (let j = 0; j < C; j++) {
          const val = d[i][j], c = cells[i][j];
          c.textContent = val == null || Number.isNaN(val) ? '' : (val === -Infinity ? '−∞' : format(val, i, j));
          if (color) c.style.background = color(val, i, j);
          if (textColor) c.style.color = textColor(val, i, j);
        }
      },
      highlight: (hi, hj) => { for (let i = 0; i < R; i++) for (let j = 0; j < C; j++) cells[i][j].classList.toggle('is-hot', i === hi && (hj == null || j === hj)); },
    };
    api.update(data);
    return api;
  };

  /**
   * A vector drawn as a woven strip of coloured bars (the "thread barcode").
   * Returns {el, update(vec)}. Positive → woad, negative → madder; height ∝ |v|.
   */
  ui.vector = (vec, { width = 220, height = 34, max, label } = {}) => {
    const c = el('canvas', { role: 'img', 'aria-label': label || 'vector' });
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    c.width = width * dpr; c.height = height * dpr;
    c.style.width = width + 'px'; c.style.height = height + 'px'; c.style.maxWidth = '100%';
    const g = c.getContext('2d');
    const draw = (v) => {
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      g.clearRect(0, 0, width, height);
      const n = v.length, m = max || Math.max(1e-6, ...Array.from(v, Math.abs));
      const bw = width / n, mid = height / 2;
      g.fillStyle = AM.rgba(AM.col.linen, 0.08);
      g.fillRect(0, mid - 0.5, width, 1);
      for (let i = 0; i < n; i++) {
        const t = Math.max(-1, Math.min(1, v[i] / m));
        g.fillStyle = t >= 0 ? AM.dye.woad : AM.dye.madder;
        g.globalAlpha = 0.35 + 0.65 * Math.abs(t);
        const hh = Math.abs(t) * (mid - 1);
        g.fillRect(i * bw + bw * 0.12, t >= 0 ? mid - hh : mid, Math.max(1, bw * 0.76), Math.max(1, hh));
      }
      g.globalAlpha = 1;
    };
    draw(vec);
    return { el: c, update: draw };
  };

  // ---------------------------------------------------------------- architecture glyph
  // Vertical (desktop) and horizontal (phone) diagrams of a decoder-only transformer.
  // Keys: tokens, embed, pos, attn, heads, mlp, resid, norm, stack, unembed, sample, train, lab, all
  const ON = {
    tokens: ['tokens'], embed: ['embed'], pos: ['pos', 'embed'], attn: ['attn'], heads: ['attn'], mlp: ['mlp'],
    resid: ['resid', 'attn', 'mlp'], norm: ['resid'], stack: ['stack', 'attn', 'mlp', 'resid'], unembed: ['unembed', 'sample'], sample: ['sample', 'unembed'],
  };
  const ALL = ['tokens', 'embed', 'pos', 'attn', 'mlp', 'resid', 'stack', 'unembed', 'sample'];

  ui.glyph = (where = 'none') => {
    const on = new Set(where === 'all' || where === 'train' || where === 'lab' ? ALL : (ON[where] || []));
    const cls = (k) => 'g-node' + (on.has(k) ? ' is-on' : '');
    const wcls = (k) => 'g-wire' + (on.has(k) ? ' is-on' : '');
    const node = (k, x, y, w, h, label) => el('g', { class: cls(k) },
      el('rect', { x, y, width: w, height: h, rx: 3 }),
      el('text', { x: x + w / 2, y: y + h / 2 + 2.3, 'text-anchor': 'middle' }, label));

    // Vertical: data flows bottom → top, as in the original paper's figure.
    const V = el('svg', { class: 'glyph glyph-v', viewBox: '0 0 92 168', role: 'img', 'aria-label': `Transformer diagram, highlighting: ${where}` },
      el('line', { class: wcls('tokens'), x1: 46, y1: 156, x2: 46, y2: 146 }),
      node('tokens', 18, 156, 56, 11, 'tokens'),
      node('embed', 14, 134, 50, 12, 'embed'),
      node('pos', 68, 134, 20, 12, 'pos'),
      el('line', { class: wcls('embed'), x1: 39, y1: 134, x2: 39, y2: 118 }),
      el('rect', { class: 'g-brace' + (on.has('stack') ? ' is-on' : ''), x: 6, y: 52, width: 70, height: 68, rx: 6 }),
      el('text', { class: 'g-x', x: 80, y: 90 }, '×N'),
      // residual stream: the long thread through the block
      el('line', { class: wcls('resid'), x1: 18, y1: 118, x2: 18, y2: 54 }),
      node('attn', 23, 96, 50, 14, 'attention'),
      node('mlp', 23, 66, 50, 14, 'mlp'),
      el('path', { class: wcls('resid'), d: 'M18 112 H48 V110 M48 96 V88 H18 M18 82 H48 V80 M48 66 V58 H18' }),
      el('line', { class: wcls('unembed'), x1: 39, y1: 52, x2: 39, y2: 40 }),
      node('unembed', 14, 26, 50, 12, 'unembed'),
      el('line', { class: wcls('sample'), x1: 39, y1: 26, x2: 39, y2: 16 }),
      node('sample', 10, 2, 58, 13, 'next token'),
    );

    // Horizontal: data flows left → right (phones).
    const H = el('svg', { class: 'glyph glyph-h', viewBox: '0 0 300 40', role: 'img', 'aria-label': `Transformer diagram, highlighting: ${where}` },
      node('tokens', 2, 13, 40, 14, 'tokens'),
      el('line', { class: wcls('embed'), x1: 42, y1: 20, x2: 50, y2: 20 }),
      node('embed', 50, 6, 38, 13, 'embed'),
      node('pos', 50, 22, 38, 12, 'pos'),
      el('line', { class: wcls('embed'), x1: 88, y1: 20, x2: 98, y2: 20 }),
      el('rect', { class: 'g-brace' + (on.has('stack') ? ' is-on' : ''), x: 98, y: 3, width: 108, height: 34, rx: 6 }),
      el('line', { class: wcls('resid'), x1: 100, y1: 33, x2: 204, y2: 33 }),
      node('attn', 102, 9, 50, 16, 'attention'),
      node('mlp', 156, 9, 44, 16, 'mlp'),
      el('text', { class: 'g-x', x: 186, y: 1.5 }, '×N'),
      el('line', { class: wcls('unembed'), x1: 206, y1: 20, x2: 214, y2: 20 }),
      node('unembed', 214, 13, 38, 14, 'unembed'),
      el('line', { class: wcls('sample'), x1: 252, y1: 20, x2: 258, y2: 20 }),
      node('sample', 258, 13, 40, 14, 'next'),
    );
    return el('div', { class: 'glyph-box' }, V, H);
  };

  AM.css(`
    .glyph-box .glyph-h { display: none; }
    @media (max-width: 760px) {
      .glyph-box .glyph-v { display: none; }
      .glyph-box .glyph-h { display: block; width: 100%; max-width: 420px; height: auto; }
    }
  `);

  /** Small legend: items [{color, label}]. */
  ui.legend = (items) => el('div', { class: 'legend', style: { display: 'flex', flexWrap: 'wrap', gap: '6px 16px' } },
    items.map((it) => el('span', { style: { display: 'inline-flex', alignItems: 'center', gap: '7px', fontFamily: 'var(--font-mono)', fontSize: '10.5px', letterSpacing: '0.08em', textTransform: 'uppercase', color: 'var(--mist)' } },
      el('span', { style: { width: '14px', height: '3px', borderRadius: '2px', background: it.color, boxShadow: `0 0 6px ${it.color}` } }), it.label)));
})();
