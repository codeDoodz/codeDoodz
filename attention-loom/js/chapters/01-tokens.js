/* Chapter 01 · Tokenization ("Shattering Text").

   Every token, ID and count in this chapter comes from a real byte-level
   Byte-Pair Encoding tokenizer that is trained in the browser, at load time,
   on the small original corpus below. Only the glass physics in the opening
   figure is decorative.

   Sections of this file:
     1. corpus            the toy training text (original prose)
     2. BPE               training, encoding, tracing merges
     3. glyph drawing     shared token-tile renderer for every canvas
     4. hero              the shattering glass strip
     5. stage scenes      five scrollytelling scenes
     6. workshop          the free-play merge loom
     7. chapter mount     DOM, CSS, wiring                                   */
(() => {
  const AM = window.AM;
  const D = AM.draw;
  const M = AM.math;
  const CH = '#ch-tokens';

  // =================================================================== 1. corpus
  // A short story written for this page. The tokenizer learns from nothing else.
  const CORPUS = [
    'In a small town at the edge of the hills there lived a weaver named Ada. Every morning she sat at her loom and listened to the threads. The warp threads stood still, tall and patient, and the weft threads ran between them, over and under, over and under, row after row.',
    '"A cloth is a conversation," Ada told her apprentice. "Each thread is a word. Alone it means very little. Woven together, the threads hold each other in place, and the pattern appears."',
    'The apprentice was a curious boy named Theo. He counted everything. He counted the threads in the warp, the rows in the weft, and the strawberries in the garden. He wanted to know why the shuttle flew, why the colours held, and why the cloth did not fall apart.',
    '"Watch the shuttle," said Ada. "It carries the weft from one side to the other. Then the beater presses the new row against the rows that came before. Every new row leans on the old ones. That is the whole secret of weaving."',
    'Ada kept a pattern book. Each page held a small square of cloth and a line of notes beside it: the colours, the count of threads, the order of the rows. When a pattern worked, she wrote it down. When a pattern failed, she wrote that down too, because a failed pattern teaches as much as a good one.',
    'In the autumn they dyed the yarn. They boiled madder roots for red, weld for yellow and woad leaves for blue. They tried straw for a pale gold, elderberries for purple and blackberries for a deep blue. The strawberry dye was the worst of all, a pink that faded in the sun within a week. Ada\'s elderberry ink lasted for years, and a blackberry stain on Theo\'s sleeve never came out. Theo kept notes on every dye bath: the weight of the yarn, the heat of the water, the minutes in the pot. Some of the colours were unbelievably bright. Others were remarkably dull, and Ada laughed at them kindly.',
    'In winter the weavers of the town gathered in the long hall. There were 30 looms, and the sound of 30 shuttles was like rain on a roof. The older weavers taught the younger weavers, and the younger weavers taught the children. Patterns travelled from hand to hand, from loom to loom, from one winter to the next.',
    'Theo learned that in 1804 a weaver in France had built a loom that read its pattern from a chain of punched cards. Each card was a row. A hole meant lift the thread; no hole meant leave it down. The cards could be copied, so a pattern could be shared, stored and woven again. Theo thought this was the most wonderful thing he had ever heard. A pattern written as holes. A picture made of numbers. He wrote 1804 on the first page of his notebook and drew a small card under it.',
    '"Could a loom learn a pattern by itself?" he asked.',
    'Ada thought about it while the shuttle flew. "Perhaps," she said. "If it watched enough cloth. If it noticed which threads like to sit together. It would start with single threads, then pairs, then whole patterns, until it could weave a story without being told each stitch."',
    'That night Theo untangled a knot of straw-coloured yarn and unwound it onto a spool. He sorted his threads into pairs, then the pairs into bundles, and the bundles into a long braid. It was not a loom that learned, but it was a start.',
    'Theo wrote that down too. Then he went back to counting the strawberries, which were not threads at all, and which he could count perfectly well, one by one.',
  ].join('\n');

  // =================================================================== 2. BPE
  const BPE = (() => {
    const enc = new TextEncoder();
    const strict = new TextDecoder('utf-8', { fatal: true });
    // GPT-2's pre-tokenizer: contractions, words with their leading space,
    // runs of digits, runs of punctuation, whitespace. Merges never cross these.
    const PRE = /'(?:s|t|re|ve|m|ll|d)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
    const pretok = (t) => t.match(PRE) || [];
    const key = (a, b) => a * 65536 + b;
    // Replace every non-overlapping (a, b), left to right, with id.
    const mergePair = (ids, a, b, id) => {
      const out = [];
      for (let i = 0; i < ids.length; i++) {
        if (i + 1 < ids.length && ids[i] === a && ids[i + 1] === b) { out.push(id); i++; } else out.push(ids[i]);
      }
      return out;
    };

    // --- training: word frequencies, then greedy pair merges ------------------
    const wordCount = new Map();
    for (const w of pretok(CORPUS)) wordCount.set(w, (wordCount.get(w) || 0) + 1);
    const words = [...wordCount].map(([w, c]) => ({ w, c, ids: Array.from(enc.encode(w)) }));
    const pieces = [];                       // id -> byte array
    for (let i = 0; i < 256; i++) pieces.push([i]);
    const merges = [];                       // rank -> {a, b, id, count}
    const corpusBytes = words.reduce((s, W) => s + W.c * W.ids.length, 0);
    const totals = [corpusBytes];            // corpus length in tokens after k merges
    for (;;) {
      const pc = new Map();
      for (const W of words) {
        const s = W.ids;
        for (let i = 0; i + 1 < s.length; i++) { const k = key(s[i], s[i + 1]); pc.set(k, (pc.get(k) || 0) + W.c); }
      }
      // most frequent pair; ties go to the pair seen first; it must occur at least twice
      let best = -1, bc = 1;
      for (const [k, c] of pc) if (c > bc) { best = k; bc = c; }
      if (best < 0) break;
      const a = Math.floor(best / 65536), b = best % 65536, id = 256 + merges.length;
      merges.push({ a, b, id, count: bc });
      pieces.push(pieces[a].concat(pieces[b]));
      let tot = 0;
      for (const W of words) { W.ids = mergePair(W.ids, a, b, id); tot += W.c * W.ids.length; }
      totals.push(tot);
    }
    const N = merges.length;
    const rank = new Map(merges.map((m, r) => [key(m.a, m.b), r]));

    // --- encoding: replay merges by rank ---------------------------------------
    // Repeatedly merge the adjacent pair whose merge was learned earliest (lowest
    // rank), using only the first k merges, until no learned pair remains.
    const wordCache = new Map();
    function encodeWord(w, k = N) {
      const ck = k + '\u0001' + w;
      const hit = wordCache.get(ck);
      if (hit) return hit;
      let ids = Array.from(enc.encode(w));
      for (;;) {
        let br = Infinity, bi = -1;
        for (let i = 0; i + 1 < ids.length; i++) {
          const r = rank.get(key(ids[i], ids[i + 1]));
          if (r !== undefined && r < k && r < br) { br = r; bi = i; }
        }
        if (bi < 0) break;
        const m = merges[br];
        ids = mergePair(ids, m.a, m.b, m.id);
      }
      if (wordCache.size > 20000) wordCache.clear();
      wordCache.set(ck, ids);
      return ids;
    }
    /** Tokenize text → [{id, start, end}] with byte offsets. */
    function encode(text, k = N) {
      const out = [];
      let byte = 0;
      for (const w of pretok(text)) {
        for (const id of encodeWord(w, k)) { const n = pieces[id].length; out.push({ id, start: byte, end: byte + n }); byte += n; }
      }
      return out;
    }
    /** Same algorithm, but records the merge tree (for the ply diagram). */
    function trace(w, k = N) {
      const bytes = Array.from(enc.encode(w));
      const nodes = bytes.map((b, i) => ({ id: b, lo: i, hi: i + 1, level: 0, kids: null, step: -1, rank: -1 }));
      let seq = nodes.map((_, i) => i);
      let step = 0;
      for (;;) {
        let br = Infinity;
        for (let i = 0; i + 1 < seq.length; i++) {
          const r = rank.get(key(nodes[seq[i]].id, nodes[seq[i + 1]].id));
          if (r !== undefined && r < k && r < br) br = r;
        }
        if (br === Infinity) break;
        const m = merges[br];
        const next = [];
        for (let i = 0; i < seq.length; i++) {
          const A = nodes[seq[i]], B = i + 1 < seq.length ? nodes[seq[i + 1]] : null;
          if (B && A.id === m.a && B.id === m.b) {
            nodes.push({ id: m.id, lo: A.lo, hi: B.hi, level: Math.max(A.level, B.level) + 1, kids: [seq[i], seq[i + 1]], step, rank: br });
            next.push(nodes.length - 1); i++;
          } else next.push(seq[i]);
        }
        seq = next; step++;
      }
      return { bytes, nodes, roots: seq, steps: step };
    }
    /** Adjacent-pair counts over the whole corpus after k merges, most frequent first. */
    const pairCache = new Map();
    function topPairs(k, n = 5) {
      if (!pairCache.has(k)) {
        const pc = new Map();
        for (const W of wordCount.keys()) {
          const ids = encodeWord(W, k), c = wordCount.get(W);
          for (let i = 0; i + 1 < ids.length; i++) { const kk = key(ids[i], ids[i + 1]); pc.set(kk, (pc.get(kk) || 0) + c); }
        }
        // stable sort keeps first-seen order on ties, matching training
        pairCache.set(k, [...pc].sort((x, y) => y[1] - x[1]).slice(0, 12).map(([kk, c]) => ({ a: Math.floor(kk / 65536), b: kk % 65536, count: c })));
      }
      return pairCache.get(k).slice(0, n);
    }
    const decode = (bytes) => { try { return strict.decode(Uint8Array.from(bytes)); } catch { return null; } };
    const madeBy = new Map(merges.map((m, r) => [m.id, r]));
    return { N, merges, pieces, totals, corpusBytes, wordCount, pretok, encode, encodeWord, trace, topPairs, decode, madeBy, enc };
  })();

  // =================================================================== 3. glyph drawing
  const FONT = {
    body: (s, w = 600) => AM.font(s, 'body', w),
    mono: (s, w = 400) => AM.font(s, 'mono', w),
    disp: (s, w = 500, it = false) => AM.font(s, 'display', w, it),
  };
  const mctx = document.createElement('canvas').getContext('2d');
  let wCache = new Map();
  let runCache = new Map();
  /** Text width, cached. Cache is reset on every resize so late web fonts re-measure. */
  const tw = (str, font) => {
    const k = font + '\u0001' + str;
    let v = wCache.get(k);
    if (v === undefined) { mctx.font = font; v = mctx.measureText(str).width; wCache.set(k, v); }
    return v;
  };
  const resetMeasure = () => { wCache = new Map(); runCache = new Map(); };
  // Web fonts arrive late: drop cached widths before the core re-fires every canvas resize.
  let fontEpoch = 0;
  AM.fontsReady.then(() => { resetMeasure(); fontEpoch++; });
  const hex2 = (b) => b.toString(16).toUpperCase().padStart(2, '0');
  // Learned tokens are dyed by ID; weld (gold) stays reserved for highlights.
  const TOK_DYES = AM.dyeList.filter((c) => c !== AM.dye.weld);
  const dyeOf = (id) => (id < 256 ? AM.col.mist : TOK_DYES[(id - 256) % TOK_DYES.length]);

  /** One glyph per byte: a character, a space mark, or a hex byte when the bytes
      inside this token do not form a whole UTF-8 character. 'z' = zero-width
      continuation byte of a character already drawn. */
  function glyphsOf(bytes) {
    const out = [];
    for (let i = 0; i < bytes.length;) {
      const b = bytes[i];
      if (b === 32) { out.push({ t: 'sp' }); i++; continue; }
      if (b === 10) { out.push({ t: 'nl' }); i++; continue; }
      if (b < 0x80) { out.push(b < 32 || b === 127 ? { t: 'hex', s: hex2(b) } : { t: 'c', s: String.fromCharCode(b) }); i++; continue; }
      const L = b >= 0xf0 ? 4 : b >= 0xe0 ? 3 : b >= 0xc0 ? 2 : 0;
      if (L && i + L <= bytes.length) {
        const s = BPE.decode(bytes.slice(i, i + L));
        if (s != null) { out.push({ t: 'c', s }); for (let j = 1; j < L; j++) out.push({ t: 'z' }); i += L; continue; }
      }
      out.push({ t: 'hex', s: hex2(b) }); i++;
    }
    return out;
  }
  /** Glyph run of a token id at size s (Figtree 600): {gl, xs (per byte), w}. */
  function runOf(id, s) {
    const k = id + '|' + s;
    let r = runCache.get(k);
    if (r) return r;
    const gl = glyphsOf(BPE.pieces[id]);
    const xs = [];
    let x = 0;
    for (const G of gl) {
      xs.push(x);
      G.w = G.t === 'c' ? tw(G.s, FONT.body(s)) : G.t === 'sp' ? s * 0.5 : G.t === 'nl' ? s * 0.62
        : G.t === 'hex' ? tw(G.s, FONT.mono(s * 0.66)) + s * 0.22 : 0;
      x += G.w;
    }
    r = { gl, xs, w: x };
    runCache.set(k, r);
    return r;
  }
  /** The leading-space mark (an open box ␣), drawn so no font fallback is needed. */
  function spaceMark(g, x, y, w, s, color, alpha = 1) {
    if (alpha <= 0.01) return;
    g.save();
    g.globalAlpha *= alpha;
    g.strokeStyle = color;
    g.lineWidth = Math.max(1, s * 0.08);
    g.lineCap = 'round'; g.lineJoin = 'round';
    const by = y + s * 0.3, th = s * 0.2;
    g.beginPath();
    g.moveTo(x + w * 0.18, by - th); g.lineTo(x + w * 0.18, by); g.lineTo(x + w * 0.82, by); g.lineTo(x + w * 0.82, by - th);
    g.stroke();
    g.restore();
  }
  /** Draw one glyph of a run, vertically centred on y. */
  function drawGlyph(g, G, x, y, s, color, spColor) {
    if (G.t === 'c') { g.fillStyle = color; g.font = FONT.body(s); g.textAlign = 'left'; g.textBaseline = 'middle'; g.fillText(G.s, x, y + s * 0.05); }
    else if (G.t === 'sp') spaceMark(g, x, y, G.w, s, spColor || AM.dye.weld, 0.85);
    else if (G.t === 'nl') {
      g.save(); g.strokeStyle = spColor || AM.dye.weld; g.lineWidth = Math.max(1, s * 0.08); g.beginPath();
      g.moveTo(x + G.w * 0.75, y - s * 0.25); g.lineTo(x + G.w * 0.75, y + s * 0.15); g.lineTo(x + G.w * 0.25, y + s * 0.15); g.stroke(); g.restore();
    } else if (G.t === 'hex') {
      // a raw byte that is only part of a character: shown as two hex digits
      g.save();
      g.fillStyle = AM.mix(AM.col.linenDim, AM.dye.lichen, 0.45); g.font = FONT.mono(s * 0.66); g.textAlign = 'center'; g.textBaseline = 'middle';
      g.fillText(G.s, x + G.w / 2, y + s * 0.04);
      g.restore();
    }
  }
  function tileColors(id) {
    if (id < 256) return { fill: AM.rgba(AM.col.ink3, 0.96), stroke: AM.col.ruleStrong, line: null, text: AM.col.linenDim };
    const d = dyeOf(id);
    return { fill: AM.mix(AM.col.ink2, d, 0.16), stroke: AM.rgba(d, 0.62), line: d, text: AM.col.linen };
  }
  /** A token tile, left edge at x, vertically centred on cy. Returns its width.
      o: {pad, h, alpha, hl (0..1 weld highlight), glow, id: draw ID under, idSize} */
  function drawTile(g, id, x, cy, s, o = {}) {
    const r = runOf(id, s);
    const pad = o.pad ?? s * 0.42, h = o.h ?? s * 1.8;
    const w = r.w + pad * 2;
    const c = tileColors(id);
    g.save();
    if (o.alpha != null) g.globalAlpha *= o.alpha;
    const hl = o.hl || 0;
    if (hl > 0.01) {
      g.save(); g.globalAlpha *= hl;
      g.shadowColor = AM.rgba(AM.dye.weld, 0.8); g.shadowBlur = 16;
      D.roundRect(g, x, cy - h / 2, w, h, s * 0.36); g.fillStyle = AM.rgba(AM.dye.weld, 0.12); g.fill();
      g.restore();
    }
    D.roundRect(g, x, cy - h / 2, w, h, s * 0.36);
    g.fillStyle = c.fill; g.fill();
    g.lineWidth = 1;
    g.strokeStyle = hl > 0.01 ? AM.mix(AM.col.ruleStrong, AM.dye.weld, Math.min(1, hl)) : c.stroke;
    g.stroke();
    if (c.line) { g.fillStyle = AM.rgba(c.line, 0.9); D.roundRect(g, x + s * 0.4, cy + h / 2 - s * 0.24, w - s * 0.8, Math.max(1.5, s * 0.1), 1); g.fill(); }
    for (let i = 0; i < r.gl.length; i++) drawGlyph(g, r.gl[i], x + pad + r.xs[i], cy, s, c.text);
    if (o.id) {
      g.font = FONT.mono(o.idSize || Math.max(9, s * 0.62)); g.fillStyle = AM.dye.weld; g.textAlign = 'center'; g.textBaseline = 'top';
      g.fillText(String(id), x + w / 2, cy + h / 2 + 5);
    }
    g.restore();
    return w;
  }
  const tileW = (id, s, pad = s * 0.42) => runOf(id, s).w + pad * 2;
  /** Flow tiles into rows inside [x0, x1]. Returns {items:[{id,x,row,w}], rows, widths per row}. */
  function flow(ids, x0, x1, s, gap, center = true) {
    const items = [];
    const rowW = [0];
    let x = 0, row = 0;
    for (const id of ids) {
      const w = tileW(id, s);
      if (x > 0 && x + w > x1 - x0) { rowW[row] = x - gap; row++; x = 0; rowW.push(0); }
      items.push({ id, x, row, w });
      x += w + gap;
    }
    rowW[row] = x - gap;
    for (const it of items) it.x += x0 + (center ? (x1 - x0 - rowW[it.row]) / 2 : 0);
    return { items, rows: row + 1, rowW };
  }
  /** DOM label for a token: text with ▁ drawn as a CSS mark. */
  function labelNodes(id) {
    const out = [];
    for (const G of glyphsOf(BPE.pieces[id])) {
      if (G.t === 'sp') out.push(AM.el('span', { class: 'tk-sp', role: 'img', 'aria-label': 'space' }));
      else if (G.t === 'c') out.push(G.s);
      else if (G.t === 'hex') out.push(AM.el('span', { class: 'tk-hex' }, G.s));
      else if (G.t === 'nl') out.push('↵');
    }
    return out;
  }
  const labelText = (id) => glyphsOf(BPE.pieces[id]).map((G) => (G.t === 'sp' ? '▁' : G.t === 'c' ? G.s : G.t === 'hex' ? `<${G.s}>` : G.t === 'nl' ? '\\n' : '')).join('');
  const charCount = (str) => Array.from(str).length;
  const hashStr = (s) => { let h = 2166136261; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); } return h >>> 0; };
  /** Tracks whether an element is near the viewport, so off-screen canvases skip drawing. */
  function onScreen(node, onEnter) {
    const st = { on: true };
    if (typeof IntersectionObserver !== 'undefined') {
      new IntersectionObserver((es) => {
        for (const e of es) { st.on = e.isIntersecting; if (st.on && onEnter) onEnter(); }
      }, { rootMargin: '120px 0px' }).observe(node);
    }
    return st;
  }
  const ease = M.ease;
  const clamp = M.clamp;
  const lerp = M.lerp;

  /** Point list helpers for the shard → tile morph. */
  function resample(pts, n) {
    // closed polyline → n points evenly spaced by arc length, starting at pts[0]
    const segs = [];
    let total = 0;
    for (let i = 0; i < pts.length; i++) {
      const a = pts[i], b = pts[(i + 1) % pts.length];
      const l = Math.hypot(b[0] - a[0], b[1] - a[1]);
      segs.push(l); total += l;
    }
    const out = [];
    let si = 0, acc = 0;
    for (let k = 0; k < n; k++) {
      const d = (k / n) * total;
      while (si < segs.length - 1 && acc + segs[si] < d) { acc += segs[si]; si++; }
      const a = pts[si], b = pts[(si + 1) % pts.length];
      const t = segs[si] ? (d - acc) / segs[si] : 0;
      out.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]);
    }
    return out;
  }
  function roundRectPts(x, y, w, h, r) {
    const pts = [];
    const arc = (cx, cy, a0) => { for (let i = 0; i <= 6; i++) { const a = a0 + (i / 6) * (Math.PI / 2); pts.push([cx + Math.cos(a) * r, cy + Math.sin(a) * r]); } };
    pts.push([x + r, y]);
    arc(x + w - r, y + r, -Math.PI / 2);
    arc(x + w - r, y + h - r, 0);
    arc(x + r, y + h - r, Math.PI / 2);
    arc(x + r, y + r, Math.PI);
    return pts;
  }
  /** Sub-curve of a cubic from 0..t (de Casteljau), for threads that grow. */
  function cubicTo(g, x0, y0, x1, y1, x2, y2, x3, y3, t) {
    if (t >= 0.999) { g.bezierCurveTo(x1, y1, x2, y2, x3, y3); return; }
    const l = (a, b) => a + (b - a) * t;
    const ax = l(x0, x1), ay = l(y0, y1), bx = l(x1, x2), by = l(y1, y2), cx = l(x2, x3), cy = l(y2, y3);
    const dx = l(ax, bx), dy = l(ay, by), ex = l(bx, cx), ey = l(by, cy);
    g.bezierCurveTo(ax, ay, dx, dy, l(dx, ex), l(dy, ey));
  }
  /** Cross-stitch flourish where two tokens fuse: one small cross on the tile's top
      edge and one on its bottom edge, at the seam x, plus a bead of light.
      s = tile font size, half = half the tile height, p = 0..1 life. */
  function stitch(g, x, y, s, p, half, color = AM.dye.weld) {
    const a = p < 0.2 ? p / 0.2 : 1 - (p - 0.2) / 0.8;
    if (a <= 0.01) return;
    const k = s * 0.24 * (0.5 + 0.5 * ease.outBack(Math.min(1, p / 0.3)));
    const path = () => {
      g.beginPath();
      for (const oy of [-half, half]) {
        g.moveTo(x - k * 0.7, y + oy - k); g.lineTo(x + k * 0.7, y + oy + k);
        g.moveTo(x + k * 0.7, y + oy - k); g.lineTo(x - k * 0.7, y + oy + k);
      }
      g.moveTo(x, y - half + k); g.lineTo(x, y + half - k);
    };
    g.save();
    g.globalAlpha *= a;
    g.lineCap = 'round';
    g.strokeStyle = AM.rgba(color, 0.3); g.lineWidth = 3.4; path(); g.stroke();
    g.strokeStyle = '#fff3c4'; g.lineWidth = 1.1; g.globalAlpha *= 0.95; path(); g.stroke();
    g.restore();
    D.glowDot(g, x, y - half, 1.4 * a, color, a * 0.8);
    D.glowDot(g, x, y + half, 1.4 * a, color, a * 0.8);
  }
  /** Mono micro label (canvas). */
  function micro(g, str, x, y, o = {}) {
    D.text(g, str.toUpperCase(), x, y, { size: o.size || 10, role: 'mono', color: o.color || AM.col.mist, align: o.align || 'left', baseline: o.baseline || 'alphabetic', alpha: o.alpha, letterSpacing: '0.12em' });
  }

  /** Mono text in which '▁' is drawn as the space mark (no font fallback needed).
      y is the text's vertical centre. Returns the width drawn. */
  function monoText(g, str, x, y, o = {}) {
    const size = o.size || 11;
    const font = FONT.mono(size, o.weight || 400);
    const spW = tw('n', font);
    const parts = String(str).split('▁');
    let total = 0;
    parts.forEach((p, i) => { total += tw(p, font) + (i < parts.length - 1 ? spW : 0); });
    let cx = o.align === 'right' ? x - total : o.align === 'center' ? x - total / 2 : x;
    g.save();
    if (o.alpha != null) g.globalAlpha *= o.alpha;
    g.font = font; g.fillStyle = o.color || AM.col.linen; g.textAlign = 'left'; g.textBaseline = 'middle';
    parts.forEach((p, i) => {
      if (p) { g.fillText(p, cx, y + size * 0.04); cx += tw(p, font); }
      if (i < parts.length - 1) { spaceMark(g, cx, y - size * 0.05, spW, size, o.spColor || AM.dye.weld, 0.9); cx += spW; }
    });
    g.restore();
    return total;
  }

  // woven texture used inside the glass
  const weavePattern = (() => {
    const c = document.createElement('canvas');
    c.width = 8; c.height = 8;
    const x = c.getContext('2d');
    x.fillStyle = 'rgba(237,229,211,0.06)';
    x.fillRect(0, 0, 4, 1); x.fillRect(4, 4, 4, 1);
    x.fillStyle = 'rgba(111,160,240,0.07)';
    x.fillRect(0, 0, 1, 4); x.fillRect(4, 4, 1, 4);
    x.fillStyle = 'rgba(237,229,211,0.035)';
    x.fillRect(2, 2, 1, 1); x.fillRect(6, 6, 1, 1);
    return c;
  })();

  // =================================================================== 4. hero
  const HERO_DEFAULT = "The weaver's loom kept looming over 1805 threads.";

  function buildHero(ctx, host, onTokens) {
    let text = HERO_DEFAULT;
    let L = null;                  // layout (depends on width + text)
    let phase = 'intact';          // intact | crack | fly | settled | reform | heal
    let pt = 0;                    // seconds in current phase
    let impact = { x: 0, y: 0 };
    let pendingStrike = null;
    let splinters = [];
    let shake = 0;
    let glassIn = 1;
    let struckOnce = false;
    let pattern = null;
    let dirty = true;              // text changed: rebuild the layout on the next resize

    const cv = ctx.canvas(host, {
      label: 'A sentence on a strip of glass that shatters into token tiles, each stamped with its ID.',
      // The core calls this on every resize check, so only rebuild when something changed;
      // a rebuild is always followed by the onResize callback below.
      height: (w) => {
        if (dirty || !L || L.w !== w || L.epoch !== fontEpoch) { L = layout(w); L.epoch = fontEpoch; dirty = false; }
        return L.h;
      },
    });
    pattern = cv.g.createPattern(weavePattern, 'repeat');

    function heroTokens(str) {
      return BPE.encode(str).map((t) => {
        const bytes = BPE.pieces[t.id];
        const lead = bytes[0] === 32;
        const rest = lead ? bytes.slice(1) : bytes;
        let s = BPE.decode(rest);
        let mono = false;
        if (s == null) { s = rest.map(hex2).join(' '); mono = true; }
        return { id: t.id, lead, str: s.replace(/\s/g, ' '), mono };
      });
    }

    function layout(w) {
      const wide = w >= 640;
      const toks = heroTokens(text);
      const m = wide ? 40 : 16;
      const avail = w - 2 * m;
      const Fmax = wide ? 46 : 32, Fmin = wide ? 30 : 23;
      const runW = (tk, F) => (tk.lead ? F * 0.27 : 0) + (tk.mono ? tw(tk.str, FONT.mono(F * 0.5)) + F * 0.3 : tw(tk.str, FONT.disp(F)));
      const padX = (F) => F * 0.62;
      let F = Fmax;
      const total = toks.reduce((s, tk) => s + runW(tk, F), 0) + 2 * padX(F);
      if (total > avail) F = clamp(F * (avail / total) * 0.985, Fmin, Fmax);
      // Line wrap between words (a word starts at a token with a leading space), balanced so
      // the last line is not left with a lone token. A word longer than a line breaks between tokens.
      const words = [];
      for (const tk of toks) { if (!words.length || tk.lead) words.push([tk]); else words[words.length - 1].push(tk); }
      const wordW = words.map((wd) => wd.reduce((a, tk) => a + runW(tk, F), 0));
      const wrapAt = (limit) => {
        const out = [[]];
        let lw = 0;
        const put = (tk, rw) => {
          if (out[out.length - 1].length && lw + rw > limit) { out.push([]); lw = 0; }
          out[out.length - 1].push(tk); lw += rw;
        };
        words.forEach((wd, i) => {
          if (wordW[i] > limit) { for (const tk of wd) put(tk, runW(tk, F)); return; }
          if (out[out.length - 1].length && lw + wordW[i] > limit) { out.push([]); lw = 0; }
          out[out.length - 1].push(...wd); lw += wordW[i];
        });
        return out;
      };
      const maxLine = avail - 2 * padX(F);
      let lines = wrapAt(maxLine);
      if (lines.length > 1) {
        let lo = Math.min(maxLine, Math.max(...wordW)), hi = maxLine;
        for (let it = 0; it < 16; it++) { const mid = (lo + hi) / 2; if (wrapAt(mid).length <= lines.length) hi = mid; else lo = mid; }
        lines = wrapAt(hi);
      }
      const lineH = F * 1.5, lineGap = F * 0.34;
      const stripH = lines.length * lineH + (lines.length - 1) * lineGap;

      // target tiles: same glyphs, scaled down to the tile font
      let Ft = wide ? 24 : 18;
      const tileGap = wide ? 12 : 8;
      const padL = F * 0.4, tileHL = F * 1.4;
      // each tile gets a slot at least as wide as the ID stamped under it, so IDs never touch
      const idFont = FONT.mono(wide ? 13 : 11, 500);
      const slotW = (tk, s) => Math.max((runW(tk, F) + 2 * padL) * s, tw(String(tk.id), idFont) + (wide ? 6 : 5));
      const tileRows = (ft) => {
        const s = ft / F;
        let rows = 1, x = 0;
        for (const tk of toks) { const tw_ = slotW(tk, s); if (x > 0 && x + tw_ > avail) { rows++; x = 0; } x += tw_ + tileGap; }
        return rows;
      };
      while (Ft > 13 && tileRows(Ft) > (wide ? 2 : 4)) Ft -= 1;
      const s = Ft / F;
      const tileH = tileHL * s;
      const rowH = tileH + (wide ? 36 : 30);

      // assign tile positions (rows centred)
      const placed = [];
      {
        let row = [], x = 0;
        const rows = [];
        for (const tk of toks) {
          const wT = slotW(tk, s);
          if (row.length && x + wT > avail) { rows.push({ row, width: x - tileGap }); row = []; x = 0; }
          row.push({ tk, wT, x }); x += wT + tileGap;
        }
        rows.push({ row, width: x - tileGap });
        rows.forEach((R, ri) => R.row.forEach((p) => placed.push({ tk: p.tk, cx: (w - R.width) / 2 + p.x + p.wT / 2, ri, wT: p.wT })));
        placed.nRows = rows.length;
      }
      const tilesH = placed.nRows * rowH;
      const single = placed.nRows === 1 && wide;
      const top = wide ? 56 : 48;
      const h = Math.round(Math.max(wide ? 360 : 330, single ? top + tilesH + 160 : top + Math.max(stripH, tilesH) + 70, stripH + 210));
      const cy = h / 2;                     // the glass strip hangs in the middle of the frame
      // tiles land in a row near the top (leaving room for warp threads) or centred when they wrap
      const tilesCy = single ? top + tilesH / 2 : Math.max(top + tilesH / 2, h / 2 + 8);

      // glass strips and crack lines
      const rnd = M.rng(hashStr(text) ^ Math.round(w));
      const shards = [];
      const cracks = [];
      const strips = [];
      let ti = 0;
      lines.forEach((ln, li) => {
        const lwid = ln.reduce((a, tk) => a + runW(tk, F), 0);
        const x0 = w / 2 - lwid / 2 - padX(F), x1 = w / 2 + lwid / 2 + padX(F);
        const ymid = cy - stripH / 2 + li * (lineH + lineGap) + lineH / 2;
        const y0 = ymid - lineH / 2, y1 = ymid + lineH / 2;
        strips.push({ x0, x1, y0, y1, ymid });
        // crack polylines at the start of every token's run (its leading space included)
        let x = w / 2 - lwid / 2;
        const starts = ln.map((tk) => { const sx = x; x += runW(tk, F); return sx; });
        const edges = starts.map((sx, j) => {
          if (j === 0) return null;
          const pts = [];
          const n = 4;
          for (let k = 0; k <= n; k++) {
            const yy = y0 + (k / n) * (y1 - y0);
            const jit = k === 0 || k === n ? (rnd() - 0.5) * F * 0.16 : (rnd() - 0.5) * F * 0.34;
            pts.push([sx + jit, yy]);
          }
          cracks.push({ pts, x: sx, y0, y1 });
          return pts;
        });
        const bev = F * 0.16;
        ln.forEach((tk, j) => {
          const rw = runW(tk, F);
          const ax = starts[j] + rw / 2, ay = ymid;
          const left = edges[j];          // null for the first token of a line
          const right = edges[j + 1] || [[x1 - bev, y0], [x1, y0 + bev], [x1, y1 - bev], [x1 - bev, y1]];
          // clockwise from the top-left: across the top, down the right, back along the bottom, up the left
          let poly;
          if (edges[j]) poly = [left[0], ...right, ...left.slice().reverse().slice(0, -1)];
          else poly = [[x0 + bev, y0], ...right, [x0 + bev, y1], [x0, y1 - bev], [x0, y0 + bev]];
          const local = resample(poly, 48).map(([px, py]) => [px - ax, py - ay]);
          const tw_ = rw + 2 * padL;
          const target = resample(roundRectPts(-tw_ / 2, -tileHL / 2, tw_, tileHL, F * 0.3), 48);
          const P = placed[ti];
          const ty = tilesCy - tilesH / 2 + P.ri * rowH + tileH / 2 + (rowH - tileH) * 0.18;
          shards.push({
            tk, i: ti, ax, ay, rw, local, target,
            tx: P.cx, ty, wT: P.wT,
            x: ax, y: ay, a: 0, sc: 1, u: 0, vx: 0, vy: 0, va: 0, a0: 0, aT: 0,
          });
          ti++;
        });
      });
      return { w, h, wide, F, s, toks, lines, strips, cracks, shards, lineH, tileH, tileHL, padL, single, cy, tilesCy, stripH, tilesH, runW, top, rowH };
    }

    // --- timeline constants (seconds)
    const T_CRACK = 0.42, T_FREE = 0.62, T_SETTLE = 1.15, T_STAG = 0.045, T_STAMP = 0.38;
    const settleStart = (sh) => T_FREE + sh.i * T_STAG;
    const stampStart = (sh) => settleStart(sh) + T_SETTLE + 0.02;

    function placeSettled() { for (const sh of L.shards) { sh.x = sh.tx; sh.y = sh.ty; sh.a = 0; sh.sc = L.s; sh.u = 1; } }
    function placeIntact() { for (const sh of L.shards) { sh.x = sh.ax; sh.y = sh.ay; sh.a = 0; sh.sc = 1; sh.u = 0; } }

    function strike(px, py) {
      if (!L) return;
      struckOnce = true;
      const sh0 = L.strips[Math.floor((L.strips.length - 1) / 2)];
      impact = { x: px ?? lerp(sh0.x0, sh0.x1, 0.38), y: py ?? sh0.ymid };
      if (AM.reducedMotion) { phase = 'settled'; pt = 99; placeSettled(); return; }
      phase = 'crack'; pt = 0; shake = 0;
      for (const c of L.cracks) c.delay = Math.min(0.26, Math.abs(c.x - impact.x) / 2600);
    }
    const DRAG = 3.4;                         // 1/s, air drag on shards
    const gravity = () => (L.wide ? 170 : 140);
    function burst() {
      const sp = L.wide ? 1 : 0.62;
      const G = gravity();
      const n = L.shards.length;
      for (const sh of L.shards) {
        const dx = sh.ax - impact.x, dy = sh.ay - impact.y;
        const d = Math.hypot(dx, dy) || 1;
        // Art direction: choose where each shard should drift to before it settles
        // (pushed away from the impact, scattered into height bands), then launch it
        // so that air drag brings it to rest there.
        const hw = sh.rw / 2 + L.F * 0.5, hh = L.F * 0.85;
        const even = L.w * 0.08 + ((sh.i + 0.5) / n) * L.w * 0.84;
        const push = Math.sign(dx || 1) * (30 + Math.random() * 90) * (L.wide ? 1 : 0.5) * Math.min(1, 140 / (d + 40) + 0.4);
        const sx = clamp(lerp(sh.ax, even, 0.35) + push, hw + 6, L.w - hw - 6);
        // neighbours go to different height bands, so the cloud of shards spreads out
        const band = [0, 2, 1, 3, 2, 0, 3, 1][sh.i % 8];
        const sy = clamp(lerp(L.h * 0.12, L.h * 0.88, (band + 0.5 + (Math.random() - 0.5) * 0.7) / 4), hh + 6, L.h - hh - 6);
        // With linear drag k, a shard launched at v0 = (S - A)·k glides to rest at S.
        sh.vx = (sx - sh.ax) * DRAG;
        sh.vy = (sy - sh.ay) * DRAG - G / DRAG * 0.5;
        sh.va = ((Math.random() - 0.5) * 5.5 + Math.sign(dx || 1) * 1.3) * (L.wide ? 1 : 0.85) * DRAG * 0.35;
        sh.x = sh.px = sh.ax; sh.y = sh.py = sh.ay; sh.a = sh.pa = 0; sh.sc = 1; sh.u = 0; sh.settling = false;
      }
      // glass splinters along every crack and around the impact
      splinters = [];
      const add = (x, y, n, spd) => {
        for (let i = 0; i < n && splinters.length < 260; i++) {
          const ang = Math.random() * Math.PI * 2, v = (0.3 + Math.random()) * spd * sp;
          splinters.push({ x, y, vx: Math.cos(ang) * v, vy: Math.sin(ang) * v - 160 * sp, a: Math.random() * 6, va: (Math.random() - 0.5) * 14,
            r: 1.5 + Math.random() * 4, life: 0, max: 0.9 + Math.random() * 1.1, glint: Math.random() < 0.25 });
        }
      };
      for (const c of L.cracks) for (const p of c.pts) add(p[0], p[1], 3, 360);
      add(impact.x, impact.y, 46, 520);
      shake = 1;
    }
    function reform(px, py) {
      if (AM.reducedMotion) { placeIntact(); phase = 'intact'; pendingStrike = null; strike(px, py); return; }
      phase = 'reform'; pt = 0; pendingStrike = { x: px, y: py };
    }
    function trigger(px, py) {
      if (phase === 'intact') strike(px, py);
      else if (phase === 'settled') reform(px, py);
    }

    function setText(str) {
      text = (str || '').replace(/\s+/g, ' ').trim().slice(0, 90) || HERO_DEFAULT;
      dirty = true;
      cv.w = 0; cv.resize();   // re-layout at the same width (height may change)
      placeIntact();
      phase = 'intact'; pt = 0; glassIn = 0; pendingStrike = { auto: true };
      onTokens && onTokens(L.toks, text);
      describe();
    }
    function describe() {
      const list = L.toks.map((t) => `${t.lead ? '(space)' : ''}${t.str} ${t.id}`).join(', ');
      cv.canvas.setAttribute('aria-label', `The sentence “${text}” on a strip of glass that shatters into ${L.toks.length} token tiles: ${list}.`);
    }

    cv.onResize(() => {
      // a resize mid-animation simply lands on the nearest resting state
      if (phase === 'settled' || phase === 'fly') { phase = 'settled'; placeSettled(); }
      else { placeIntact(); if (phase !== 'intact') { phase = 'intact'; } }
      pattern = cv.g.createPattern(weavePattern, 'repeat');
      draw(0);
    });

    cv.canvas.addEventListener('pointerdown', (e) => {
      const p = cv.pointer(e);
      if (phase === 'intact' || phase === 'settled') trigger(p.x, p.y);
    });
    cv.canvas.style.cursor = 'pointer';

    // ------------------------------------------------------------- update
    function update(dt) {
      pt += dt;
      if (glassIn < 1) glassIn = Math.min(1, glassIn + dt * 2.4);
      shake = Math.max(0, shake - dt * 4);
      if (phase === 'intact' && pendingStrike && pendingStrike.auto && glassIn >= 1 && pt > 0.7) { pendingStrike = null; strike(); }
      if (phase === 'crack' && pt >= T_CRACK) { phase = 'fly'; pt = 0; burst(); }
      if (phase === 'fly') {
        const G = gravity();
        let done = true;
        for (const sh of L.shards) {
          const s0 = settleStart(sh);
          // free flight (still integrated while settling, but damped)
          if (pt < s0 + T_SETTLE) {
            // drag + a little gravity; stronger damping once the shard starts to settle
            const damp = Math.exp(-(pt > s0 ? 8 : DRAG) * dt);
            sh.vx *= damp; sh.vy = sh.vy * damp + G * dt; sh.va *= damp;
            sh.px = (sh.px ?? sh.ax) + sh.vx * dt;
            sh.py = (sh.py ?? sh.ay) + sh.vy * dt;
            sh.pa = (sh.pa ?? 0) + sh.va * dt;
            // soft walls so shards stay in frame
            const hw = sh.rw / 2 * 0.6 + 10;
            if (sh.px < hw || sh.px > L.w - hw) { sh.vx *= -0.5; sh.px = clamp(sh.px, hw, L.w - hw); }
            if (sh.py > L.h - 22) { sh.vy = -Math.abs(sh.vy) * 0.4; sh.py = L.h - 22; }
            if (sh.py < 18) { sh.vy = Math.abs(sh.vy) * 0.4; sh.py = 18; }
          }
          if (pt <= s0) { sh.x = sh.px; sh.y = sh.py; sh.a = sh.pa; sh.u = 0; sh.sc = 1; done = false; continue; }
          if (!sh.settling) { sh.settling = true; sh.a0 = sh.pa; sh.aT = Math.round(sh.pa / (Math.PI * 2)) * Math.PI * 2; }
          const u = clamp((pt - s0) / T_SETTLE);
          const e = ease.inOut(u);
          sh.x = lerp(sh.px, sh.tx, e);
          sh.y = lerp(sh.py, sh.ty, e);
          sh.a = lerp(sh.pa, sh.aT, e);
          sh.sc = lerp(1, L.s, e);
          sh.u = M.smoothstep(0.1, 0.95, u);
          if (pt < stampStart(sh) + T_STAMP) done = false;
        }
        if (done) { phase = 'settled'; pt = 0; for (const sh of L.shards) { sh.px = sh.py = sh.pa = undefined; } placeSettled(); }
      }
      if (phase === 'reform') {
        let done = true;
        for (const sh of L.shards) {
          const u = clamp((pt - sh.i * 0.025) / 0.85);
          if (u < 1) done = false;
          const e = 1 - ease.inOut(u);
          sh.x = lerp(sh.ax, sh.tx, e); sh.y = lerp(sh.ay, sh.ty, e); sh.a = 0; sh.sc = lerp(1, L.s, e); sh.u = e;
        }
        if (done) { phase = 'heal'; pt = 0; placeIntact(); }
      }
      if (phase === 'heal' && pt > 0.45) {
        phase = 'intact'; pt = 0;
        const p = pendingStrike; pendingStrike = null;
        strike(p && p.x, p && p.y);
      }
      // splinters
      if (splinters.length) {
        const G = 900;
        for (const s of splinters) { s.life += dt; s.vy += G * dt; s.vx *= Math.exp(-0.8 * dt); s.x += s.vx * dt; s.y += s.vy * dt; s.a += s.va * dt; }
        splinters = splinters.filter((s) => s.life < s.max && s.y < L.h + 20);
      }
    }

    // ------------------------------------------------------------- draw
    function glassPath(g, pts) {
      g.beginPath();
      g.moveTo(pts[0][0], pts[0][1]);
      for (let i = 1; i < pts.length; i++) g.lineTo(pts[i][0], pts[i][1]);
      g.closePath();
    }
    function drawRun(g, tk, x0, y, F, alphaSpace, color) {
      // x0 = left of the run (where its leading space begins)
      let x = x0;
      if (tk.lead) { spaceMark(g, x - F * 0.02, y - F * 0.04, F * 0.3, F, AM.dye.weld, alphaSpace); x += F * 0.27; }
      g.fillStyle = color;
      g.textAlign = 'left'; g.textBaseline = 'alphabetic';
      if (tk.mono) { g.font = FONT.mono(F * 0.5); g.fillText(tk.str, x + F * 0.15, y + F * 0.18); }
      else { g.font = FONT.disp(F); g.fillText(tk.str, x, y + F * 0.33); }
    }
    function drawStrip(g, t, alpha) {
      const F = L.F;
      g.save();
      g.globalAlpha *= alpha;
      for (const S of L.strips) {
        const wS = S.x1 - S.x0, hS = S.y1 - S.y0;
        // halo
        const halo = g.createRadialGradient((S.x0 + S.x1) / 2, S.ymid, 0, (S.x0 + S.x1) / 2, S.ymid, wS * 0.6);
        halo.addColorStop(0, AM.rgba(AM.dye.woad, 0.10)); halo.addColorStop(1, AM.rgba(AM.dye.woad, 0));
        g.fillStyle = halo; g.fillRect(S.x0 - wS * 0.2, S.y0 - hS * 1.5, wS * 1.4, hS * 4);
        D.roundRect(g, S.x0, S.y0, wS, hS, F * 0.18);
        const gr = g.createLinearGradient(0, S.y0, 0, S.y1);
        gr.addColorStop(0, AM.rgba(AM.col.linen, 0.13)); gr.addColorStop(0.5, AM.rgba(AM.dye.woad, 0.06)); gr.addColorStop(1, AM.rgba(AM.col.linen, 0.04));
        g.fillStyle = gr; g.fill();
        g.save(); g.clip();
        if (pattern) { g.fillStyle = pattern; g.fillRect(S.x0, S.y0, wS, hS); }
        // travelling sheen
        const sx = S.x0 - 160 + ((t * 0.22) % 1.6) * (wS + 320);
        const sg = g.createLinearGradient(sx - 60, S.y0, sx + 60, S.y1);
        sg.addColorStop(0, 'rgba(255,248,230,0)'); sg.addColorStop(0.5, 'rgba(255,248,230,0.10)'); sg.addColorStop(1, 'rgba(255,248,230,0)');
        g.fillStyle = sg; g.fillRect(S.x0, S.y0, wS, hS);
        g.restore();
        g.strokeStyle = AM.rgba(AM.col.linen, 0.42); g.lineWidth = 1; D.roundRect(g, S.x0 + 0.5, S.y0 + 0.5, wS - 1, hS - 1, F * 0.18); g.stroke();
        g.strokeStyle = 'rgba(255,248,230,0.32)'; g.beginPath(); g.moveTo(S.x0 + F * 0.3, S.y0 + 2.5); g.lineTo(S.x1 - F * 0.3, S.y0 + 2.5); g.stroke();
      }
      for (const sh of L.shards) drawRun(g, sh.tk, sh.ax - sh.rw / 2, sh.ay, F, 0, AM.col.linen);
      g.restore();
    }
    function drawCracks(g, alpha) {
      g.save();
      g.lineCap = 'round'; g.lineJoin = 'round';
      for (const c of L.cracks) {
        const p = phase === 'heal' ? 1 : clamp((pt - c.delay) / 0.11);
        if (p <= 0) continue;
        const n = c.pts.length - 1, upto = p * n;
        const path = () => {
          g.beginPath(); g.moveTo(c.pts[0][0], c.pts[0][1]);
          for (let k = 1; k <= Math.ceil(upto); k++) {
            const f = Math.min(1, upto - (k - 1));
            const a = c.pts[k - 1], b = c.pts[k];
            g.lineTo(a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f);
          }
        };
        g.strokeStyle = AM.rgba(AM.dye.weld, 0.4 * alpha); g.lineWidth = 6; path(); g.stroke();
        g.strokeStyle = AM.rgba('#fff8e6', 0.95 * alpha); g.lineWidth = 1.3; path(); g.stroke();
      }
      // star fracture at the impact
      if (phase === 'crack') {
        const p = clamp(pt / 0.12);
        const rnd = M.rng(7);
        for (let i = 0; i < 9; i++) {
          const a = (i / 9) * Math.PI * 2 + rnd() * 0.5, len = (16 + rnd() * 30) * p * (L.F / 40);
          g.strokeStyle = AM.rgba('#fff8e6', 0.8 * alpha); g.lineWidth = 1;
          g.beginPath(); g.moveTo(impact.x, impact.y);
          g.lineTo(impact.x + Math.cos(a) * len * 0.5 + (rnd() - 0.5) * 4, impact.y + Math.sin(a) * len * 0.5 + (rnd() - 0.5) * 4);
          g.lineTo(impact.x + Math.cos(a) * len, impact.y + Math.sin(a) * len); g.stroke();
        }
      }
      g.restore();
    }
    function drawShard(g, sh, t, band) {
      const F = L.F;
      const u = sh.u;
      g.save();
      g.translate(sh.x, sh.y);
      g.rotate(sh.a);
      g.scale(sh.sc, sh.sc);
      const pts = sh.local.map((p, i) => [lerp(p[0], sh.target[i][0], u), lerp(p[1], sh.target[i][1], u)]);
      glassPath(g, pts);
      // glass body
      const gr = g.createLinearGradient(0, -L.tileHL / 2, 0, L.tileHL / 2);
      gr.addColorStop(0, AM.rgba(AM.col.linen, 0.15)); gr.addColorStop(0.55, AM.rgba(AM.dye.woad, 0.07)); gr.addColorStop(1, AM.rgba(AM.col.linen, 0.05));
      g.fillStyle = gr; g.fill();
      // as it lands, the glass takes the token's dye
      if (u > 0.01) {
        const c = tileColors(sh.tk.id);
        g.globalAlpha = u;
        g.fillStyle = c.fill; g.fill();
        g.globalAlpha = 1;
      }
      // a slow band of light drifts across the resting tiles
      if (band != null) {
        const f = Math.exp(-Math.pow((sh.x - band) / 90, 2));
        if (f > 0.02) { g.fillStyle = AM.rgba('#fff8e6', 0.1 * f); g.fill(); }
      }
      if (pattern) { g.save(); g.clip(); g.globalAlpha = 0.9 * (1 - u * 0.5); g.fillStyle = pattern; g.fillRect(-sh.rw, -F, sh.rw * 2, F * 2); g.restore(); }
      g.lineWidth = 1 / sh.sc;
      const d = dyeOf(sh.tk.id);
      g.strokeStyle = u > 0.5 ? AM.rgba(d, 0.35 + 0.35 * u) : AM.rgba('#fff8e6', 0.55 - 0.2 * u);
      g.stroke();
      if (u < 0.6) {
        g.strokeStyle = AM.rgba(AM.dye.weld, 0.16 * (1 - u)); g.lineWidth = 5 / sh.sc; g.stroke();
      }
      // dye underline once landed
      if (u > 0.6 && sh.tk.id >= 256) {
        g.globalAlpha = (u - 0.6) / 0.4;
        g.fillStyle = d;
        D.roundRect(g, -sh.rw / 2, L.tileHL / 2 - F * 0.16, sh.rw, F * 0.06, 1); g.fill();
        g.globalAlpha = 1;
      }
      drawRun(g, sh.tk, -sh.rw / 2, 0, F, M.smoothstep(0.5, 1, u), AM.col.linen);
      g.restore();
    }
    function drawStamp(g, sh, p, t) {
      if (p <= 0) return;
      const e = ease.out(clamp(p));
      const x = sh.tx, y = sh.ty + L.tileH / 2 + (L.wide ? 17 : 14);
      const size = L.wide ? 13 : 11;
      g.save();
      if (p < 1) {
        g.strokeStyle = AM.rgba(AM.dye.weld, 0.5 * (1 - e)); g.lineWidth = 1.2;
        g.beginPath(); g.arc(x, y, 4 + 22 * e, 0, Math.PI * 2); g.stroke();
      }
      g.translate(x, y);
      const sc = 1 + 0.9 * (1 - e);
      g.scale(sc, sc);
      g.globalAlpha = clamp(p * 1.6);
      g.font = FONT.mono(size, 500); g.textAlign = 'center'; g.textBaseline = 'middle';
      g.fillStyle = AM.dye.weld;
      g.fillText(String(sh.tk.id), 0, 0);
      g.restore();
    }
    function drawWarp(g, t, alpha) {
      // single-row layout: every ID hangs on a warp thread that runs down into the loom
      const yTop = L.shards[0].ty + L.tileH / 2 + 30;
      const yBot = L.h - 30;
      g.save();
      g.globalAlpha *= alpha;
      for (const sh of L.shards) {
        const d = dyeOf(sh.tk.id);
        const sway = Math.sin(t * 1.1 + sh.i * 0.9) * 2.2;
        const grd = g.createLinearGradient(0, yTop, 0, yBot);
        grd.addColorStop(0, AM.rgba(d, 0.55)); grd.addColorStop(1, AM.rgba(d, 0.04));
        g.strokeStyle = grd; g.lineWidth = 1.2;
        g.beginPath(); g.moveTo(sh.tx, yTop); g.bezierCurveTo(sh.tx + sway, yTop + 30, sh.tx - sway, yBot - 30, sh.tx + sway * 0.5, yBot); g.stroke();
        const f = (t * 0.32 + sh.i * 0.137) % 1;
        D.glowDot(g, sh.tx + sway * (f - 0.5) * 0.8, lerp(yTop, yBot, f), 1.6, d, Math.sin(f * Math.PI) * 0.9);
      }
      micro(g, 'only these integers enter the model', L.w / 2, L.h - 10, { align: 'center', color: AM.col.mist });
      g.restore();
    }
    function drawSplinters(g) {
      for (const s of splinters) {
        const a = 1 - s.life / s.max;
        g.save(); g.translate(s.x, s.y); g.rotate(s.a);
        g.fillStyle = s.glint ? AM.rgba('#fff8e6', 0.85 * a) : AM.rgba(AM.dye.woad, 0.45 * a);
        g.beginPath(); g.moveTo(0, -s.r); g.lineTo(s.r * 0.7, s.r * 0.6); g.lineTo(-s.r * 0.6, s.r * 0.4); g.closePath(); g.fill();
        g.restore();
        if (s.glint && a > 0.4) D.glowDot(g, s.x, s.y, 1.2, AM.dye.weld, a * 0.6);
      }
    }
    function draw(t) {
      const g = cv.g;
      if (!L) return;
      cv.clear();
      g.save();
      if (shake > 0) g.translate((Math.random() - 0.5) * 7 * shake, (Math.random() - 0.5) * 5 * shake);
      if (phase === 'crack') g.translate((Math.random() - 0.5) * 2.2, (Math.random() - 0.5) * 1.6);
      // header line
      const nT = L.toks.length, nC = charCount(text);
      if (phase === 'settled' || (phase === 'fly' && pt > 1.4)) {
        const a = phase === 'settled' ? 1 : clamp((pt - 1.4) / 0.5);
        micro(g, `${nC} characters → ${nT} tokens → ${nT} integers`, L.w / 2, L.wide ? 30 : 26, { align: 'center', color: AM.dye.weld, alpha: a });
      } else if (phase === 'intact' && struckOnce && !(pendingStrike && pendingStrike.auto)) {
        micro(g, 'tap the glass', L.w / 2, L.wide ? 30 : 26, { align: 'center', alpha: 0.8 * glassIn });
      }
      if (phase === 'intact' || phase === 'crack' || phase === 'heal') {
        drawStrip(g, t, glassIn);
        if (phase !== 'intact') drawCracks(g, phase === 'heal' ? 1 - pt / 0.45 : 1);
      } else {
        const band = phase === 'settled' ? (((t * 0.14) % 1.6) - 0.3) * L.w : null;
        for (const sh of L.shards) drawShard(g, sh, t, band);
        if (phase === 'fly' || phase === 'settled') {
          for (const sh of L.shards) drawStamp(g, sh, phase === 'settled' ? 1 : (pt - stampStart(sh)) / T_STAMP, t);
        }
        if (L.single && (phase === 'settled' || phase === 'fly')) {
          const a = phase === 'settled' ? Math.min(1, pt / 0.8) : 0;
          if (a > 0) drawWarp(g, t, a);
        }
      }
      // impact flash + shock ring
      if (phase === 'crack') {
        const f = 1 - clamp(pt / 0.35);
        const gr = g.createRadialGradient(impact.x, impact.y, 0, impact.x, impact.y, 90);
        gr.addColorStop(0, AM.rgba('#fff8e6', 0.55 * f)); gr.addColorStop(0.3, AM.rgba(AM.dye.weld, 0.22 * f)); gr.addColorStop(1, AM.rgba(AM.dye.weld, 0));
        g.fillStyle = gr; g.fillRect(impact.x - 90, impact.y - 90, 180, 180);
      }
      if (phase === 'fly' && pt < 0.7) {
        const e = ease.out(pt / 0.7);
        g.strokeStyle = AM.rgba(AM.col.linen, 0.35 * (1 - e)); g.lineWidth = 2 * (1 - e) + 0.5;
        g.beginPath(); g.arc(impact.x, impact.y, 20 + e * Math.max(L.w, L.h) * 0.6, 0, Math.PI * 2); g.stroke();
      }
      drawSplinters(g);
      g.restore();
    }

    const vis = onScreen(cv.wrap);
    ctx.loop((t, dt) => { if (!vis.on) return; update(dt); draw(t); });
    ctx.onVisible(() => { if (!struckOnce && phase === 'intact') { pendingStrike = { auto: true }; pt = 0; } });

    describe();
    onTokens && onTokens(L.toks, text);
    draw(0);
    return { setText, strike: () => trigger(), get text() { return text; } };
  }

  // =================================================================== 5. stage scenes
  /* Each scene: {title, rest, layout(w,h), enter(), draw(g, w, h, t, alpha)}; t = seconds since entered.
     rest = a time at which the scene reads completely; used for the first frame and for reduced motion. */

  // ---- Scene 1: characters vs words vs subword tokens
  function sceneGranularity() {
    const S = 'The shuttle flew while the weaver hummed.';
    const WORD_RE = /\w+|[^\w\s]/g;
    const vocabWords = new Set(CORPUS.match(WORD_RE));
    const chars = Array.from(S);
    const words = S.match(WORD_RE).map((w) => ({ w, unk: !vocabWords.has(w) }));
    const bpe = BPE.encode(S).map((t) => t.id);
    const STEP = 0.13;
    let rows = [], L = null;
    const defs = [
      { label: 'Characters (bytes)', note: `${chars.length} tokens · vocab 256`, n: chars.length, kind: 'char' },
      { label: 'Whole words', note: `${words.length} tokens · vocab: every word`, n: words.length, kind: 'word' },
      { label: 'BPE subwords', note: `${bpe.length} tokens · vocab ${256 + BPE.N}`, n: bpe.length, kind: 'bpe' },
    ];
    function itemW(kind, i, s) {
      if (kind === 'char') return (chars[i] === ' ' ? s * 0.55 : tw(chars[i], FONT.body(s))) + s * 0.6;
      if (kind === 'word') return (words[i].unk ? tw('[UNK]', FONT.mono(s * 0.74)) : tw(words[i].w, FONT.body(s))) + s * 0.84;
      return tileW(bpe[i], s);
    }
    return {
      title: 'Three ways to cut a sentence',
      rest: chars.length * STEP + 0.9,
      layout(w, h, sc = 1) {
        const wide = w >= 480;
        const m = wide ? 26 : 12;
        const sizes = { char: (wide ? 15 : 11) * sc, word: (wide ? 18 : 12.5) * sc, bpe: (wide ? 18 : 12.5) * sc };
        const gaps = { char: wide ? 3 : 2.5, word: wide ? 7 : 4, bpe: wide ? 7 : 4 };
        const labelH = wide ? 28 : 22, secGap = wide ? 40 : 18;
        let y = 0;
        rows = defs.map((d) => {
          const s = sizes[d.kind], gap = gaps[d.kind];
          const th = s * 1.8, rowH = th + (d.kind === 'char' ? 5 : 8);
          const items = [];
          let x = 0, r = 0;
          for (let i = 0; i < d.n; i++) {
            const iw = itemW(d.kind, i, s);
            if (x > 0 && x + iw > w - 2 * m - 18) { r++; x = 0; }
            items.push({ x: m + x, r, w: iw }); x += iw + gap;
          }
          const R = { ...d, s, th, rowH, items, nRows: r + 1, y0: y };
          y += labelH + R.nRows * rowH + secGap;
          return R;
        });
        const totalH = y - secGap;
        if (totalH > h - 16 && sc > 0.7) return this.layout(w, h, sc - 0.08);   // shrink until it fits
        const off = Math.max(10, (h - totalH) / 2);
        for (const R of rows) R.y0 += off;
        L = { wide, m, labelH, fs: wide ? 11 : 10 };
      },
      enter() {},
      draw(g, w, h, t, alpha) {
        if (!L) return;
        const { m } = L;
        g.save();
        g.globalAlpha *= alpha;
        const cycle = chars.length * STEP + 1.8;
        const tc = t % cycle;
        rows.forEach((R, ri) => {
          const appear = clamp((t - ri * 0.15) / 0.5);
          g.globalAlpha = alpha * appear;
          micro(g, R.label, m, R.y0 + 11, { color: AM.col.linenDim, size: L.fs });
          micro(g, R.note, w - m, R.y0 + 11, { align: 'right', color: R.kind === 'bpe' ? AM.dye.weld : AM.col.mist, size: L.fs });
          const cur = Math.min(R.n, Math.floor(tc / STEP));
          R.items.forEach((it, i) => {
            const cy = R.y0 + L.labelH + it.r * R.rowH + R.th / 2;
            const read = i < cur, here = i === cur && cur < R.n;
            const dim = read || here ? 1 : 0.6;
            if (R.kind === 'bpe') drawTile(g, bpe[i], it.x, cy, R.s, { hl: here ? 1 : 0, alpha: dim });
            else {
              const unk = R.kind === 'word' && words[i].unk;
              g.save();
              g.globalAlpha *= dim;
              D.roundRect(g, it.x, cy - R.th / 2, it.w, R.th, R.s * 0.32);
              g.fillStyle = unk ? AM.mix(AM.col.ink2, AM.dye.madder, 0.12) : AM.rgba(AM.col.ink3, 0.96); g.fill();
              g.lineWidth = 1;
              if (unk) g.setLineDash([3, 3]);
              g.strokeStyle = here ? AM.dye.weld : unk ? AM.rgba(AM.dye.madder, 0.8) : AM.col.ruleStrong; g.stroke();
              g.setLineDash([]);
              if (here) { g.shadowColor = AM.rgba(AM.dye.weld, 0.7); g.shadowBlur = 12; g.stroke(); g.shadowBlur = 0; }
              if (R.kind === 'char') {
                if (chars[i] === ' ') spaceMark(g, it.x + R.s * 0.3, cy, R.s * 0.55, R.s, AM.dye.weld, 0.8);
                else D.text(g, chars[i], it.x + it.w / 2, cy + R.s * 0.05, { size: R.s, weight: 600, align: 'center', baseline: 'middle', color: AM.col.linenDim });
              } else if (unk) D.text(g, '[UNK]', it.x + it.w / 2, cy + 1, { size: R.s * 0.74, role: 'mono', align: 'center', baseline: 'middle', color: AM.dye.madder });
              else D.text(g, words[i].w, it.x + it.w / 2, cy + R.s * 0.05, { size: R.s, weight: 600, align: 'center', baseline: 'middle', color: AM.col.linen });
              g.restore();
            }
            if (here) D.glowDot(g, it.x + it.w / 2, cy - R.th / 2 - 6, 2.2, AM.dye.weld, 1);
          });
          // a check mark once this row has been read to the end
          const lastIt = R.items[R.items.length - 1];
          const ly = R.y0 + L.labelH + (R.nRows - 1) * R.rowH + R.th / 2;
          if (cur >= R.n) D.text(g, '✓', lastIt.x + lastIt.w + 7, ly + 1, { size: 13, color: AM.dye.verdigris, baseline: 'middle' });
        });
        g.restore();
      },
    };
  }

  // ---- Scene 2: BPE training, live on the corpus
  function sceneTraining() {
    const WORDS = [' the', ' weavers', ' threads', ' and', ' pattern'].filter((w) => BPE.wordCount.has(w));
    const K = 12, P = 1.9;
    const states = [];
    for (let k = 0; k <= K; k++) states.push({ words: WORDS.map((w) => BPE.encodeWord(w, k)), pairs: BPE.topPairs(k, 5) });
    let L = null;
    // per-byte glyph x of a word's pieces laid out as tiles (relative to row start)
    function byteXs(ids, s, gap) {
      const xs = [], tiles = [];
      let x = 0, b = 0;
      for (const id of ids) {
        const r = runOf(id, s), pad = s * 0.42;
        tiles.push({ id, x, w: r.w + 2 * pad, b0: b, b1: b + r.xs.length });
        r.xs.forEach((bx) => { xs.push(x + pad + bx); b++; });
        x += r.w + 2 * pad + gap;
      }
      return { xs, tiles };
    }
    return {
      title: 'Byte-pair encoding, learning',
      rest: K * P + 0.6,
      layout(w, h) {
        const wide = w >= 480;
        const m = wide ? 26 : 12;
        const s = wide ? 17 : 12.5;
        const gap = wide ? 5 : 3.5;
        const fs = wide ? 11 : 10;
        const colW = wide ? w * 0.52 : w - 2 * m;
        const rowH = s * (wide ? 2.5 : 2.3);
        const nW = wide ? WORDS.length : 4;
        const nBars = wide ? 5 : 3;
        const barH = wide ? 42 : 27;
        const logLines = wide ? 3 : 1;
        const head = 24;
        const wordsH = head + nW * rowH;
        const barsH = head + nBars * barH;
        const logH = head + logLines * 19;
        const total = wide ? Math.max(wordsH, barsH) + 30 + logH : wordsH + 18 + barsH + 14 + logH;
        const top = Math.max(8, (h - total) / 2);
        const wordsTop = top + head;
        const barsX = wide ? m + colW + 30 : m;
        const barsTop = wide ? wordsTop : wordsTop + nW * rowH + 18 + head;
        const barsW = wide ? w - barsX - m : w - 2 * m;
        const logTop = (wide ? top + Math.max(wordsH, barsH) + 30 : barsTop + nBars * barH + 14) + 11;
        L = { wide, m, s, gap, fs, colW, rowH, wordsTop, nW, barsX, barsTop, barsW, nBars, barH, logTop, logLines, head };
        L.lay = states.map((st) => st.words.map((ids) => byteXs(ids, s, gap)));
      },
      enter() {},
      draw(g, w, h, t, alpha) {
        if (!L) return;
        const { m, s, wide, fs } = L;
        const cycle = K * P + 2.4;
        const tc = t % cycle;
        const k = Math.min(K, Math.floor(tc / P));
        const tau = k < K ? tc - k * P : P;
        const resetFade = Math.min(clamp((cycle - tc) / 0.5), clamp(tc / 0.4));
        g.save();
        g.globalAlpha *= alpha * resetFade;
        const mg = k < K ? BPE.merges[k] : null;
        const hl = k < K ? M.smoothstep(0.15, 0.55, tau) * (1 - M.smoothstep(1.0, 1.2, tau)) : 0;
        const fuse = k < K ? ease.inOut(clamp((tau - 0.6) / 0.42)) : 0;
        const after = k < K && tau >= 1.02;
        const kk = after ? k + 1 : k;

        micro(g, 'Words in the corpus', m, L.wordsTop - L.head + 8, { color: AM.col.linenDim, size: fs });
        micro(g, `vocab ${256 + kk}`, wide ? m + L.colW : w - m, L.wordsTop - L.head + 8, { align: 'right', color: AM.dye.weld, size: fs });
        for (let wi = 0; wi < L.nW; wi++) {
          const cy = L.wordsTop + wi * L.rowH + L.rowH / 2;
          const A = L.lay[k][wi], B = L.lay[Math.min(K, k + 1)][wi];
          const ids = after ? states[kk].words[wi] : states[k].words[wi];
          const tiles = (after ? B : A).tiles;
          const x0 = m;
          // byte positions glide from layout k to layout k+1 while the pair fuses
          const xs = A.xs.map((x, b) => (after ? B.xs[b] : lerp(x, B.xs[b], fuse)));
          // which tiles take part in this merge (non-overlapping, left to right)
          const inPair = new Set();
          if (!after && mg) for (let i = 0; i + 1 < ids.length; i++) if (ids[i] === mg.a && ids[i + 1] === mg.b) { inPair.add(i); inPair.add(i + 1); i++; }
          ids.forEach((id, ti) => {
            const T = tiles[ti];
            const r = runOf(id, s), pad = s * 0.42;
            const xL = x0 + xs[T.b0] - pad;
            const xR = x0 + xs[T.b1 - 1] + (r.gl[r.gl.length - 1].w || 0) + pad;
            const c = tileColors(id);
            const hlv = inPair.has(ti) ? hl : 0;
            g.save();
            if (hlv > 0.02) { g.shadowColor = AM.rgba(AM.dye.weld, 0.8 * hlv); g.shadowBlur = 14; }
            D.roundRect(g, xL, cy - s * 0.9, xR - xL, s * 1.8, s * 0.36);
            g.fillStyle = c.fill; g.fill();
            g.shadowBlur = 0;
            g.lineWidth = 1; g.strokeStyle = hlv > 0.02 ? AM.mix(AM.col.ruleStrong, AM.dye.weld, hlv) : c.stroke; g.stroke();
            if (c.line) { g.fillStyle = c.line; D.roundRect(g, xL + s * 0.4, cy + s * 0.9 - s * 0.24, xR - xL - s * 0.8, 1.6, 1); g.fill(); }
            g.restore();
            r.gl.forEach((G, j) => drawGlyph(g, G, x0 + xs[T.b0 + j], cy, s, c.text));
          });
          if (!after && mg && fuse > 0) {
            for (let i = 0; i + 1 < ids.length; i++) if (ids[i] === mg.a && ids[i + 1] === mg.b) {
              stitch(g, x0 + xs[tiles[i + 1].b0] - 1, cy, s, clamp((tau - 0.75) / 0.9), s * 0.9);
              i++;
            }
          }
          monoText(g, `×${BPE.wordCount.get(WORDS[wi])}`, wide ? m + L.colW : w - m, cy, { align: 'right', color: AM.col.mist, size: fs });
        }

        // pair-count bars
        micro(g, 'Most frequent pairs', L.barsX, L.barsTop - L.head + 8, { color: AM.col.linenDim, size: fs });
        const pairs = states[kk].pairs.slice(0, L.nBars);
        const maxC = states[0].pairs[0].count;
        const grow = after ? ease.out(clamp((tau - 1.02) / 0.45)) : 1;
        const bfs = wide ? 13 : 11;
        pairs.forEach((p, i) => {
          const y = L.barsTop + i * L.barH;
          const isTop = i === 0;
          monoText(g, `${labelText(p.a)} + ${labelText(p.b)}`, L.barsX, y + bfs * 0.6, { size: bfs, color: isTop ? AM.col.linen : AM.col.linenDim });
          monoText(g, String(p.count), L.barsX + L.barsW, y + bfs * 0.6, { size: bfs, align: 'right', color: isTop ? AM.dye.weld : AM.col.mist });
          const bw = (L.barsW * p.count / maxC) * grow;
          const by = y + bfs + (wide ? 9 : 6);
          g.fillStyle = AM.rgba(AM.col.linen, 0.07); g.fillRect(L.barsX, by, L.barsW, 3);
          g.save();
          g.fillStyle = isTop ? AM.dye.weld : AM.rgba(AM.dye.woad, 0.75);
          if (isTop && hl > 0) { g.shadowColor = AM.dye.weld; g.shadowBlur = 12 * hl; }
          g.fillRect(L.barsX, by, bw, 3);
          g.restore();
        });

        // merge log
        micro(g, 'Merges learned', m, L.logTop - 3, { color: AM.col.linenDim, size: fs });
        const lfs = wide ? 12 : 10.5;
        const from = Math.max(0, kk - L.logLines);
        for (let r = from; r < kk; r++) {
          const mm = BPE.merges[r];
          const y = L.logTop + 16 + (r - from) * 19;
          const fresh = r === kk - 1 && after ? clamp((tau - 1.02) / 0.3) : 1;
          const last = r === kk - 1;
          let x = m;
          x += monoText(g, `#${r + 1}`, x, y, { size: lfs, color: last ? AM.dye.weld : AM.col.mist, alpha: fresh }) + 12;
          x += monoText(g, `${labelText(mm.a)} + ${labelText(mm.b)} → ${labelText(mm.id)}`, x, y, { size: lfs, color: last ? AM.col.linen : AM.col.linenDim, alpha: fresh }) + 12;
          if (wide || L.barsW > 300) monoText(g, `id ${mm.id} · ${mm.count}×`, w - m, y, { size: lfs, align: 'right', color: AM.col.mist, alpha: fresh });
        }
        if (kk === 0) monoText(g, 'none yet: every byte is its own token', m, L.logTop + 16, { size: lfs, color: AM.col.mist });
        g.restore();
      },
    };
  }

  // ---- Scene 3: replaying merges on new words, drawn as threads plying into yarn
  function scenePly() {
    const WORDS = [' weavers', ' looming', ' unweaving'];
    const traces = WORDS.map((w) => ({ w, tr: BPE.trace(w), seen: BPE.wordCount.get(w) || 0 }));
    const EV = 0.42, HOLD = 2.8;
    let L = null;
    const durOf = (T) => 0.6 + T.tr.steps * EV + 1.0 + HOLD;
    function lay(T, w, h) {
      const wide = w >= 480;
      const n = T.tr.bytes.length;
      const m = wide ? 30 : 14;
      const cw = Math.min(wide ? 58 : 34, (w - 2 * m) / n);
      const x0 = w / 2 - (cw * n) / 2;
      const leafY = h - (wide ? 58 : 38);
      const s = wide ? 18 : 13;
      const tileY = (wide ? 62 : 54) + s * 0.9;
      const maxLv = Math.max(1, ...T.tr.nodes.map((nd) => nd.level));
      const topNodeY = tileY + s * 0.9 + (wide ? 64 : 44);
      const dy = (leafY - topNodeY) / maxLv;
      const leafX = (i) => x0 + (i + 0.5) * cw;
      const pos = T.tr.nodes.map((nd) => ({ x: (leafX(nd.lo) + leafX(nd.hi - 1)) / 2, y: leafY - nd.level * dy }));
      const ids = T.tr.roots.map((ri) => T.tr.nodes[ri].id);
      const F = flow(ids, m, w - m, s, wide ? 12 : 6);
      const rootOf = new Array(T.tr.nodes.length).fill(-1);
      T.tr.roots.forEach((ri, k) => { const mark = (i) => { rootOf[i] = k; const nd = T.tr.nodes[i]; if (nd.kids) nd.kids.forEach(mark); }; mark(ri); });
      return { wide, s, cw, leafY, tileY, pos, F, rootOf, ids, m, fs: wide ? 11 : 9.5 };
    }
    return {
      title: 'Replaying the merges, word by word',
      // rest: ' looming' (never seen in training) fully replayed, its two tiles in place
      rest: durOf(traces[0]) + 0.6 + traces[1].tr.steps * EV + 2.0,
      layout(w, h) { L = traces.map((T) => lay(T, w, h)); },
      enter() {},
      draw(g, w, h, t, alpha) {
        if (!L) return;
        const total = traces.reduce((a, T) => a + durOf(T), 0);
        let tt = t % total, wi = 0;
        while (tt > durOf(traces[wi])) { tt -= durOf(traces[wi]); wi++; }
        const T = traces[wi], Y = L[wi];
        const dur = durOf(T);
        const fade = Math.min(clamp(tt / 0.35), clamp((dur - tt) / 0.35));
        g.save();
        g.globalAlpha *= alpha * fade;
        const nodes = T.tr.nodes;
        const stepT = (st) => 0.6 + st * EV;
        const rootDye = (k) => dyeOf(Y.ids[k]);
        const nTok = T.tr.roots.length;
        // heading: the word, whether training saw it, and the outcome
        let hx = Y.m;
        hx += monoText(g, T.w.replace(/ /g, '▁'), Y.m, 16, { size: Y.wide ? 14 : 12, color: AM.col.linen }) + 12;
        micro(g, T.seen ? `seen ${T.seen}× in training` : 'never seen in training', hx, 20, { color: T.seen ? AM.dye.verdigris : AM.dye.saffron, size: Y.fs });
        const seenTxt = T.seen ? `seen ${T.seen}× in training` : 'never seen in training';
        let outcome = `${nTok} token${nTok > 1 ? 's' : ''} · ${T.tr.steps} merge step${T.tr.steps === 1 ? '' : 's'}`;
        const microW = (str) => str.length * Y.fs * 0.74;     // mono advance + letter-spacing
        if (hx + microW(seenTxt) + 18 + microW(outcome) > w - Y.m) outcome = `${nTok} token${nTok > 1 ? 's' : ''}`;
        if (Y.wide) micro(g, outcome, w - Y.m, 20, { align: 'right', color: AM.dye.weld, size: Y.fs });
        // threads: child → parent, revealed in the order the merges are replayed.
        // Piece labels are collected and drawn after every thread, so no strand crosses a label.
        const tRise = stepT(T.tr.steps) + 0.15;
        const tp = clamp((tt - tRise - 0.45) / 0.4);
        const rootSet = new Set(T.tr.roots);
        const labels = [];
        g.lineCap = 'round';
        for (let i = 0; i < nodes.length; i++) {
          const nd = nodes[i];
          if (!nd.kids) continue;
          const p = clamp((tt - stepT(nd.step)) / (EV * 0.9));
          if (p <= 0) continue;
          const P = Y.pos[i];
          const col = rootDye(Y.rootOf[i]);
          for (const ci of nd.kids) {
            const C = Y.pos[ci], cn = nodes[ci];
            const wdt = 1.1 + 0.9 * Math.sqrt(cn.hi - cn.lo);
            const midY = (C.y + P.y) / 2;
            const path = () => { g.beginPath(); g.moveTo(C.x, C.y); cubicTo(g, C.x, C.y, C.x, midY, P.x, midY, P.x, P.y, ease.inOut(p)); };
            g.strokeStyle = AM.rgba(col, 0.16); g.lineWidth = wdt * 4; path(); g.stroke();
            g.strokeStyle = AM.rgba(col, 0.88); g.lineWidth = wdt; path(); g.stroke();
            // ply twist: a lighter dashed strand that slowly travels upward
            g.setLineDash([2.5, 3.5]); g.lineDashOffset = -t * 9;
            g.strokeStyle = AM.rgba('#fff8e6', 0.45); g.lineWidth = Math.max(0.8, wdt * 0.45); path(); g.stroke();
            g.setLineDash([]);
          }
          if (p >= 1) {
            const age = tt - stepT(nd.step) - EV * 0.9;
            D.glowDot(g, P.x, P.y, 2 + 2 * Math.max(0, 1 - age * 2), col, 1);
            // piece label, if there is room beside the node
            const lab = labelText(nd.id);
            const lw = tw(lab, FONT.mono(Y.fs)) + 10;
            const span = (nd.hi - nd.lo) * Y.cw;
            // a root's label fades once its finished tile appears above
            const la = clamp(age * 3) * (rootSet.has(i) ? 1 - tp : 1);
            if (lw < span + Y.cw * 0.7 && la > 0.01) labels.push({ lab, x: P.x, y: P.y, lw, col, a: la });
          }
        }
        // leaves: the raw bytes
        const lf = Y.wide ? 19 : 14;
        T.tr.bytes.forEach((b, i) => {
          const P = Y.pos[i];
          const a = clamp((tt - i * 0.04) / 0.3);
          const col = rootDye(Y.rootOf[i]);
          D.glowDot(g, P.x, P.y, 1.5, col, a * 0.85);
          const G = glyphsOf([b])[0];
          g.save(); g.globalAlpha *= a;
          if (G.t === 'sp') spaceMark(g, P.x - lf * 0.28, P.y + lf * 1.0, lf * 0.56, lf, AM.dye.weld, 0.9);
          else if (G.t === 'c') D.text(g, G.s, P.x, P.y + lf * 1.05, { size: lf, weight: 600, align: 'center', baseline: 'middle', color: AM.col.linenDim });
          g.restore();
        });
        // roots rise to the final token tiles, which get their IDs
        const rp = clamp((tt - tRise) / 0.6);
        if (rp > 0) {
          T.tr.roots.forEach((ri, k) => {
            const P = Y.pos[ri], it = Y.F.items[k];
            const cx = it.x + it.w / 2, ty = Y.tileY + Y.s * 0.9;
            const col = rootDye(k);
            const midY = (P.y + ty) / 2;
            g.strokeStyle = AM.rgba(col, 0.82); g.lineWidth = 1.2 + 0.9 * Math.sqrt(nodes[ri].hi - nodes[ri].lo);
            g.beginPath(); g.moveTo(P.x, P.y); cubicTo(g, P.x, P.y, P.x, midY, cx, midY, cx, ty, ease.inOut(rp)); g.stroke();
          });
          if (tp > 0) {
            Y.F.items.forEach((it) => drawTile(g, it.id, it.x, Y.tileY, Y.s, { alpha: tp }));
            Y.F.items.forEach((it) => monoText(g, String(it.id), it.x + it.w / 2, Y.tileY - Y.s * 0.9 - 10, { size: Y.wide ? 12 : 10, align: 'center', color: AM.dye.weld, alpha: tp }));
            if (!Y.wide) micro(g, outcome, w - Y.m, 20, { align: 'right', color: AM.dye.weld, size: Y.fs, alpha: tp });
          }
        }
        for (const B of labels) {
          g.save();
          g.globalAlpha *= B.a;
          D.roundRect(g, B.x - B.lw / 2, B.y - Y.fs - 16, B.lw, Y.fs + 9, 5);
          g.fillStyle = AM.rgba(AM.col.ink, 0.96); g.fill();
          g.strokeStyle = AM.rgba(B.col, 0.55); g.lineWidth = 1; g.stroke();
          monoText(g, B.lab, B.x, B.y - 11.5 - Y.fs / 2, { size: Y.fs, align: 'center', color: AM.col.linen });
          g.restore();
        }
        g.restore();
      },
    };
  }

  // ---- Scene 4: how the model sees "strawberry"
  function sceneStrawberry() {
    const WORD = ' strawberry';
    const toks = BPE.encode(WORD);
    const letters = Array.from('strawberry');
    const ids = toks.map((t) => t.id);
    let L = null;
    const CYC = 10.5;
    return {
      title: 'How the model sees “strawberry”',
      rest: 8.6,
      layout(w, h) {
        const wide = w >= 480;
        const S = Math.min(wide ? 66 : 44, (w - 36) / 6.0);
        const sp = S * 0.14;
        const ws = letters.map((c) => tw(c, FONT.disp(S)));
        const tot = ws.reduce((a, b) => a + b, 0) + sp * (letters.length - 1);
        let x = w / 2 - tot / 2;
        const lx = ws.map((wd) => { const v = x; x += wd + sp; return v; });
        const cy = h * 0.4;
        // token capsules: the same letters packed into tiles
        const S2 = S * 0.84;
        const pad = S2 * 0.36, gap = S2 * 0.4, spW = S2 * 0.32;
        const caps = toks.map((tk) => {
          const bytes = BPE.pieces[tk.id];
          const lead = bytes[0] === 32;
          const chars = Array.from(BPE.decode(lead ? bytes.slice(1) : bytes));
          const cws = chars.map((c) => tw(c, FONT.disp(S2)));
          return { id: tk.id, lead, chars, cws, w: pad * 2 + (lead ? spW : 0) + cws.reduce((a, b) => a + b, 0) };
        });
        const capTot = caps.reduce((a, c) => a + c.w, 0) + gap * (caps.length - 1);
        let cx = w / 2 - capTot / 2;
        const target = [];
        caps.forEach((c) => { c.x = cx; let gx = cx + pad + (c.lead ? spW : 0); c.cws.forEach((cw) => { target.push(gx); gx += cw; }); cx += c.w + gap; });
        L = { wide, S, S2, lx, ws, cy, caps, target, pad, spW, capH: S2 * 1.55, w, h };
      },
      enter() {},
      draw(g, w, h, t, alpha) {
        if (!L) return;
        const tc = t % CYC;
        const fade = Math.min(clamp(tc / 0.4), clamp((CYC - tc) / 0.5));
        g.save();
        g.globalAlpha *= alpha * fade;
        const { S, S2, cy, wide } = L;
        const countStart = 0.7, perL = 0.24;
        const nCounted = clamp(Math.floor((tc - countStart) / perL) + 1, 0, letters.length);
        let rs = 0;
        for (let i = 0; i < nCounted; i++) if (letters[i] === 'r') rs++;
        const gather = ease.inOut(clamp((tc - 3.4) / 1.0));
        const frost = ease.inOut(clamp((tc - 4.6) / 0.9));
        const stamp = clamp((tc - 5.3) / 0.5);
        const capTop = cy - Math.max(L.capH / 2, S * 0.75);
        micro(g, 'what you see', w / 2, capTop - 26, { align: 'center', alpha: 1 - clamp((tc - 3.4) / 0.4), color: AM.col.linenDim, size: wide ? 11 : 10 });
        micro(g, 'what the model gets', w / 2, capTop - 26, { align: 'center', alpha: clamp((tc - 4.8) / 0.4), color: AM.dye.weld, size: wide ? 11 : 10 });
        // capsules
        if (gather > 0) {
          L.caps.forEach((c) => {
            const d = dyeOf(c.id);
            g.save();
            g.globalAlpha *= gather;
            D.roundRect(g, c.x, cy - L.capH / 2, c.w, L.capH, S2 * 0.3);
            const gr = g.createLinearGradient(0, cy - L.capH / 2, 0, cy + L.capH / 2);
            gr.addColorStop(0, AM.rgba(AM.col.linen, 0.12)); gr.addColorStop(1, AM.rgba(AM.dye.woad, 0.05));
            g.fillStyle = gr; g.fill();
            // frosting: the capsule turns opaque and hides its letters
            g.fillStyle = AM.mix(AM.col.ink3, d, 0.22); g.globalAlpha *= frost; g.fill();
            g.restore();
            g.save(); g.globalAlpha *= gather;
            D.roundRect(g, c.x, cy - L.capH / 2, c.w, L.capH, S2 * 0.3);
            g.strokeStyle = AM.rgba(d, 0.4 + 0.4 * frost); g.lineWidth = 1.2; g.stroke();
            if (c.lead) spaceMark(g, c.x + L.pad - S2 * 0.04, cy - S2 * 0.05, L.spW, S2, AM.dye.weld, 1 - frost * 0.85);
            g.restore();
          });
        }
        // letters
        letters.forEach((ch, i) => {
          const x = lerp(L.lx[i], L.target[i], gather);
          const size = lerp(S, S2, gather);
          const lit = ch === 'r' && i < nCounted ? 1 - gather * 0.7 : 0;
          const a = (1 - frost * 0.93) * clamp((tc - i * 0.05) / 0.3);
          const col = lit > 0 ? AM.mix(AM.col.linen, AM.dye.madder, lit) : AM.col.linen;
          D.text(g, ch, x, cy + size * 0.34, { size, role: 'display', color: col, alpha: a });
          if (lit > 0.3 && gather < 0.2) D.glowDot(g, x + L.ws[i] / 2, cy + S * 0.64, 2.2, AM.dye.madder, lit);
        });
        // counting cursor
        if (tc > countStart && tc < 3.3 && nCounted > 0) {
          const i = nCounted - 1;
          D.glowDot(g, L.lx[i] + L.ws[i] / 2, cy - S * 0.78, 2.6, AM.dye.weld, 1 - clamp((tc - 3.0) / 0.3));
        }
        monoText(g, `r × ${rs}`, w / 2, cy + S * 1.25, { size: wide ? 16 : 13, align: 'center', color: AM.dye.madder, alpha: (1 - gather) * clamp((tc - 0.8) / 0.3) });
        // IDs stamped onto the frosted capsules
        if (stamp > 0) {
          L.caps.forEach((c, k) => {
            const p = clamp(stamp * 1.4 - k * 0.3);
            if (p <= 0) return;
            const e = ease.outBack(p);
            g.save();
            g.translate(c.x + c.w / 2, cy);
            g.scale(0.6 + 0.4 * e, 0.6 + 0.4 * e);
            g.globalAlpha *= p;
            D.text(g, String(c.id), 0, 1, { size: S2 * 0.62, role: 'mono', weight: 500, align: 'center', baseline: 'middle', color: AM.dye.weld });
            g.restore();
          });
          const fs = wide ? 16 : 13;
          const y1 = cy + L.capH / 2 + fs * 2.4;
          monoText(g, `[ ${ids.join(', ')} ]`, w / 2, y1, { size: fs, align: 'center', color: AM.col.linen, alpha: clamp((tc - 6.2) / 0.5) });
          D.text(g, `How many r’s are in ${ids.join(' ')}?`, w / 2, y1 + (wide ? 46 : 36), { size: wide ? 26 : 18, role: 'display', italic: true, align: 'center', color: AM.col.linenDim, alpha: clamp((tc - 6.9) / 0.5) });
        }
        g.restore();
      },
    };
  }

  // ---- Scene 5: numbers and other scripts
  function sceneEdges() {
    const NUMS = [' 1804', ' 1805', ' 2025', ' 300'];
    const LANGS = [
      { tag: 'English', text: ' cloth' },
      { tag: 'French', text: ' tissu' },
      { tag: 'Japanese', text: ' 布' },
      { tag: 'Russian', text: ' ткань' },
    ];
    // Each example is tokenized the way it appears mid-sentence, after a space. The space is
    // counted as a character and drawn in the label, so both sides of "chars → tokens" match.
    const mk = (tag, text) => {
      const ids = BPE.encode(text).map((t) => t.id), chars = charCount(text);
      return { tag, word: text.replace(/ /g, '▁'), ids, chars, note: `${chars} char${chars > 1 ? 's' : ''} → ${ids.length} token${ids.length > 1 ? 's' : ''}` };
    };
    const numRows = NUMS.map((n) => mk(null, n));
    const langRows = LANGS.map((l) => mk(l.tag, l.text));
    let L = null;
    return {
      title: 'Numbers and other scripts',
      rest: 3,
      layout(w, h) {
        const wide = w >= 480;
        const m = wide ? 26 : 12;
        const s = wide ? 16 : 11.5;
        const nfs = wide ? 11 : 9.5;
        const noteW = Math.max(...numRows.map((R) => tw(R.note, FONT.mono(nfs)))) + 12;
        const numTagW = Math.max(...numRows.map((R) => tw(R.word.replace('▁', 'n'), FONT.mono(12)))) + 10;
        const labW = wide ? 172 : numTagW;
        let numRowH = wide ? s * 2.75 : s * 2.4;
        let langRowH = wide ? s * 2.75 : s * 2.0 + 19;
        const head = wide ? 26 : 22, secGap = wide ? 38 : 14;
        let total = head * 2 + numRowH * 4 + langRowH * 4 + secGap;
        // squeeze row heights if the canvas is short
        const room = h - 10;
        if (total > room) { const f = (room - head * 2 - secGap) / (numRowH * 4 + langRowH * 4); numRowH *= f; langRowH *= f; total = room; }
        const top = Math.max(4, (h - total) / 2);
        L = { wide, m, s, nfs, labW, noteW, numRowH, langRowH, head, top, top2: top + head + numRowH * 4 + secGap };
      },
      enter() {},
      draw(g, w, h, t, alpha) {
        if (!L) return;
        const { m, s, labW, noteW, wide, nfs } = L;
        g.save();
        g.globalAlpha *= alpha;
        const section = (title, rows, y0, isLang, delay) => {
          micro(g, title, m, y0 + 8, { color: AM.col.linenDim, size: nfs });
          const rowH = isLang ? L.langRowH : L.numRowH;
          const stack = !wide && isLang;      // phones: label line above the tiles
          rows.forEach((R, i) => {
            const rowTop = y0 + L.head + i * rowH;
            const ap = clamp((t - delay - i * 0.12) / 0.4);
            g.save(); g.globalAlpha *= ap;
            const noteC = R.ids.length > R.chars ? AM.dye.saffron : AM.dye.weld;
            const tag = (x, y) => {
              if (!isLang) return monoText(g, R.word, x, y, { size: wide ? 15 : 12, color: AM.col.linen });
              const fs = wide ? 14 : 12, head = R.tag + ' · ';
              D.text(g, head, x, y + 1, { size: fs, weight: 500, baseline: 'middle', color: AM.col.linenDim });
              return monoText(g, R.word, x + tw(head, FONT.body(fs, 500)), y, { size: fs - 1, color: AM.col.linen });
            };
            let cy, x, maxX;
            if (wide) {
              cy = rowTop + rowH / 2;
              tag(m, cy - 8); monoText(g, R.note, m, cy + 10, { size: nfs, color: noteC });
              x = m + labW; maxX = w - m;
            } else if (stack) {
              tag(m, rowTop + 7); monoText(g, R.note, w - m, rowTop + 7, { size: nfs, align: 'right', color: noteC });
              cy = rowTop + 17 + s; x = m; maxX = w - m;
            } else {
              cy = rowTop + rowH / 2;
              tag(m, cy); monoText(g, R.note, w - m, cy, { size: nfs, align: 'right', color: noteC });
              x = m + labW; maxX = w - m - noteW;
            }
            const gap = wide ? 4 : 3;
            // shrink tiles if the row would overflow
            let ss = s;
            const need = R.ids.reduce((a, id) => a + tileW(id, ss) + gap, 0);
            if (need > maxX - x) ss = Math.max(7, s * (maxX - x) / need);
            R.ids.forEach((id, j) => {
              const sweep = 0.5 + 0.5 * Math.sin(t * 2.2 - j * 0.5 - i);
              x += drawTile(g, id, x, cy, ss, { hl: id < 256 && isLang && j > 0 ? 0.18 * sweep : 0 }) + gap;
            });
            g.restore();
          });
        };
        section('Numbers, each after a space', numRows, L.top, false, 0);
        section('The word “cloth”, after a space', langRows, L.top2, true, 0.5);
        g.restore();
      },
    };
  }

  function buildStage(ctx, host, titleEl) {
    const scenes = [sceneGranularity(), sceneTraining(), scenePly(), sceneStrawberry(), sceneEdges()];
    const stacked = () => window.innerWidth <= 900;
    const cv = ctx.canvas(host, {
      label: 'Tokenization scene',
      height: (w) => {
        const vh = window.innerHeight || 800;
        return stacked() ? Math.min(w * 0.98, vh * 0.48, 460) : Math.min(w * 0.86, vh * 0.76);
      },
    });
    // the first scene starts at its resting frame, so the stage is never blank
    let cur = 0, prev = -1, fadeP = 1, tCur = scenes[0].rest, tPrev = 0;
    const labels = [
      'Three rows compare cutting the sentence into characters, whole words, and BPE subword tokens; gold beads read one token per step.',
      'Byte-pair encoding training on the toy corpus: the most frequent adjacent pair is merged, step by step.',
      'Merge trees showing how learned merges are replayed on words; rare words end as several tokens.',
      'The word strawberry: its letters are packed into two opaque token capsules, leaving only their ID numbers.',
      'How numbers and non-English words are split by the toy tokenizer.',
    ];
    cv.onResize((w, h) => { for (const s of scenes) s.layout(w, h); frame(0); });
    function set(i) {
      if (i === cur || i < 0) return;
      prev = cur; tPrev = tCur; cur = i; tCur = 0; fadeP = 0;
      scenes[cur].enter();
      titleEl.textContent = scenes[cur].title;
      cv.canvas.setAttribute('aria-label', labels[cur]);
      if (AM.reducedMotion) { fadeP = 1; frame(0); }   // switch at once instead of waiting for a slow tick
    }
    titleEl.textContent = scenes[0].title;
    cv.canvas.setAttribute('aria-label', labels[0]);
    function frame(dt) {
      tCur += dt; tPrev += dt;
      fadeP = Math.min(1, fadeP + dt / 0.45);
      cv.clear();
      const g = cv.g;
      // reduced motion: every scene holds still on its resting frame
      const still = AM.reducedMotion;
      if (fadeP < 1 && prev >= 0) scenes[prev].draw(g, cv.w, cv.h, still ? scenes[prev].rest : tPrev, 1 - ease.inOut(fadeP));
      g.save();
      g.translate(0, (1 - ease.out(fadeP)) * 10);
      scenes[cur].draw(g, cv.w, cv.h, still ? scenes[cur].rest : tCur, ease.inOut(fadeP));
      g.restore();
    }
    const vis = onScreen(cv.wrap);
    ctx.loop((t, dt) => { if (vis.on) frame(dt); });
    frame(0);
    return { set };
  }

  // =================================================================== 6. workshop
  const PRESETS = [
    { value: 'corpus', label: 'Corpus words', text: 'The weavers taught the children a new pattern, and the shuttle flew from loom to loom.' },
    { value: 'new', label: 'New words', text: "Unravelling patterns, the weavers' apprentices kept looming over threadbare looms." },
    { value: 'straw', label: 'strawberry', text: "How many r's are in strawberry?" },
    { value: 'num', label: 'Numbers', text: 'In 1804, 1805 and 2025 the digits split oddly: 30, 300, 3.14159.' },
    { value: 'intl', label: 'Other scripts', text: 'The cloth: tissu, 布, ткань.' },
  ];

  function buildWorkshop(ctx, parts) {
    const { canvasHost, curveHost, inspector, mergeList, stats, slider, playBtn, textarea, presets } = parts;
    let text = PRESETS[0].text;
    let bytes = Array.from(BPE.enc.encode(text));
    let k = AM.reducedMotion ? BPE.N : 0;
    let toks = [];         // [{id, start, end, P:{x,y}, Q:{x,y}, o:[{x,y}], f:[x], flash}]
    let abs = [];          // current absolute glyph position per byte
    let stitches = [];             // {b: byte just after the seam, life 0..1}
    let tokOf = new Int32Array(0);
    let hover = -1;
    let playing = false, playT = 0, played = false;
    let textCurve = [];
    let G = null;          // geometry for current width

    const geom = (w) => {
      const wide = w >= 640;
      const s = wide ? 17 : 14;
      return { wide, s, pad: s * 0.36, gap: s * 0.3, lineH: s * 2.3, m: wide ? 22 : 12, w };
    };
    // flow tokens of a given k into lines; returns per-token tile x/line
    function flowToks(list, g0) {
      const out = [];
      let x = 0, line = 0;
      const maxW = g0.w - 2 * g0.m;
      for (const t of list) {
        const r = runOf(t.id, g0.s);
        const wT = r.w + 2 * g0.pad;
        if (x > 0 && x + wT > maxW) { x = 0; line++; }
        out.push({ x: g0.m + x, line, wT, r });
        x += wT + g0.gap;
      }
      return { pos: out, lines: line + 1 };
    }
    const cv = ctx.canvas(canvasHost, {
      label: 'Your text as BPE tokens',
      height: (w) => {
        G = geom(w);
        // tallest case: no merges at all (one tile per byte)
        const worst = flowToks(BPE.encode(text, 0), G).lines;
        return Math.round(Math.max(G.wide ? 150 : 130, worst * G.lineH + 2 * G.m + 8));
      },
    });
    const curve = ctx.canvas(curveHost, { label: 'Line chart of tokens per byte against merges learned, for the whole corpus and for your text, with the current merge count marked.', height: (w) => (w > 420 ? 132 : 118) });

    function layoutTargets(snap) {
      const F = flowToks(toks, G);
      const nL = F.lines;
      const top = (cv.h - nL * G.lineH) / 2 + G.lineH / 2;
      toks.forEach((t, i) => {
        const p = F.pos[i];
        t.Q = { x: p.x, y: top + p.line * G.lineH };
        t.wT = p.wT;
        t.r = p.r;
        t.f = p.r.xs.map((bx) => G.pad + bx);
        if (snap || !t.P) { t.P = { ...t.Q }; t.o = t.f.map((fx) => ({ x: fx, y: 0 })); }
      });
    }
    function computeAbs() {
      abs = new Array(bytes.length);
      for (const t of toks) for (let b = t.start; b < t.end; b++) { const o = t.o[b - t.start]; abs[b] = { x: t.P.x + o.x, y: t.P.y + o.y }; }
    }
    /** Re-tokenize at merge count k, keeping every glyph where it currently is. */
    function retokenize(newK, opts = {}) {
      const oldK = k;
      k = newK;
      const old = toks;
      if (old.length && !opts.snap) computeAbs();
      const oldStart = new Set(old.map((t) => t.start));
      toks = BPE.encode(text, k).map((t) => ({ ...t }));
      // any old stitches stay where they are; new ones go on every seam that closed
      for (const t of toks) {
        if (abs.length && !opts.snap && abs[t.start]) {
          const pad = G.pad;
          t.P = { x: abs[t.start].x - pad, y: abs[t.start].y };
          t.o = [];
          for (let b = t.start; b < t.end; b++) t.o.push({ x: abs[b].x - t.P.x, y: abs[b].y - t.P.y });
        } else t.P = null;
        t.flash = 0;
        if (!opts.snap && newK > oldK) {
          for (let b = t.start + 1; b < t.end; b++) {
            if (oldStart.has(b) && stitches.length < 140) { t.flash = 1; stitches.push({ b, life: 0 }); }
          }
        }
      }
      layoutTargets(!!opts.snap);
      if (opts.snap) computeAbs();
      else {
        // A token whose glyphs would have to cross to another line does not slide diagonally
        // over its neighbours: it reappears in place and fades in.
        const far = G.lineH * 0.5;
        for (const t of toks) {
          if (Math.abs(t.P.y - t.Q.y) > far || t.o.some((o) => Math.abs(o.y) > far)) {
            t.P = { ...t.Q }; t.o = t.f.map((fx) => ({ x: fx, y: 0 })); t.pop = 1;
          }
        }
      }
      // byte → token index, so each stitch can follow the glyph after its seam
      tokOf = new Int32Array(bytes.length);
      toks.forEach((t, i) => { for (let b = t.start; b < t.end; b++) tokOf[b] = i; });
      if (hover >= toks.length) hover = -1;
      refreshDom();
    }
    function setText(str, fromPreset) {
      text = str;
      bytes = Array.from(BPE.enc.encode(text));
      stitches = [];
      cv.w = 0; cv.resize();
      retokenize(k, { snap: true });
      computeTextCurve();
      if (!fromPreset) presets.set(null);
      draw(0); drawCurve();
    }
    function computeTextCurve() {
      textCurve = [];
      const nb = Math.max(1, bytes.length);
      for (let kk = 0; kk <= BPE.N; kk++) textCurve.push(BPE.encode(text, kk).length / nb);
    }

    // ------------------------------------------------------------- DOM side
    function refreshDom() {
      slider.set(k);
      // merge list window
      mergeList.replaceChildren();
      const from = Math.max(0, k - 6), to = Math.min(BPE.N, k + 2);
      if (k === 0) mergeList.appendChild(AM.el('li', { class: 'tk-m-empty' }, 'No merges yet: every byte is its own token.'));
      for (let r = from; r < to; r++) {
        const mm = BPE.merges[r];
        const state = r < k ? (r === k - 1 ? 'is-last' : 'is-done') : 'is-next';
        mergeList.appendChild(AM.el('li', { class: 'tk-m ' + state },
          AM.el('span', { class: 'tk-m-n' }, '#' + (r + 1)),
          AM.el('span', { class: 'tk-m-rule' },
            AM.el('span', { class: 'tk-chip' }, labelNodes(mm.a)), ' + ', AM.el('span', { class: 'tk-chip' }, labelNodes(mm.b)),
            ' → ', AM.el('span', { class: 'tk-chip tk-chip-new' }, labelNodes(mm.id))),
          AM.el('span', { class: 'tk-m-c' }, r < k ? `id ${mm.id} · ${mm.count}×` : 'next')));
      }
      const nChars = charCount(text);
      const cpt = toks.length ? nChars / toks.length : 0;
      stats.k.textContent = `${k} / ${BPE.N}`;
      stats.vocab.textContent = `${(256 + k).toLocaleString('en-US')}`;
      stats.tok.textContent = `${toks.length}`;
      stats.cpt.textContent = cpt.toFixed(2);
      stats.corpus.textContent = `${BPE.totals[k].toLocaleString('en-US')}`;
      if (!playing) playBtn.textContent = k >= BPE.N ? 'Replay merges' : 'Play merges';
      cv.canvas.setAttribute('aria-label', `Your text as ${toks.length} tokens after ${k} merges: ${toks.map((t) => labelText(t.id).replace(/▁/g, '(space)')).join(' | ')}. Use the arrow keys to inspect each token.`);
      updateInspector();
    }
    function updateInspector() {
      inspector.replaceChildren();
      if (hover < 0 || !toks[hover]) {
        inspector.append(AM.el('span', { class: 'tk-insp-hint' }, 'Hover or tap a token to inspect it, or tab to the tiles and use the arrow keys.'));
        return;
      }
      const id = toks[hover].id;
      const same = toks.filter((t) => t.id === id).length;
      const chip = AM.el('span', { class: 'tk-chip tk-chip-new' }, labelNodes(id));
      if (id < 256) {
        const part = id >= 0x80 ? ', one piece of a character that takes several bytes in UTF-8' : '';
        inspector.append(chip, AM.el('span', {}, ` ID ${id} · a single raw byte (0x${hex2(id)})${part}. In this tokenizer, IDs 0 to 255 are the bytes themselves.`));
      } else {
        const r = BPE.madeBy.get(id), mm = BPE.merges[r];
        inspector.append(chip, AM.el('span', {}, ` ID ${id} · made by merge #${r + 1}: `), AM.el('span', { class: 'tk-chip' }, labelNodes(mm.a)), ' + ',
          AM.el('span', { class: 'tk-chip' }, labelNodes(mm.b)), AM.el('span', {}, ` · that pair appeared ${mm.count}× in the corpus when it was learned`));
      }
      if (same > 1) inspector.append(AM.el('span', { class: 'tk-insp-same' }, ` · ${same}× in your text, same ID each time`));
    }

    // ------------------------------------------------------------- canvas side
    function tokAt(p) {
      for (let i = 0; i < toks.length; i++) {
        const t = toks[i];
        if (p.x >= t.Q.x - 2 && p.x <= t.Q.x + t.wT + 2 && Math.abs(p.y - t.Q.y) <= G.s * 0.95) return i;
      }
      return -1;
    }
    cv.canvas.addEventListener('pointermove', (e) => { if (e.pointerType === 'touch') return; const h = tokAt(cv.pointer(e)); if (h !== hover) { hover = h; updateInspector(); } });
    cv.canvas.addEventListener('pointerleave', (e) => { if (e.pointerType === 'touch') return; hover = -1; updateInspector(); });
    cv.canvas.addEventListener('pointerdown', (e) => { const h = tokAt(cv.pointer(e)); hover = h === hover && e.pointerType === 'touch' ? -1 : h; updateInspector(); });
    // keyboard: focus the canvas, then step through the tokens with the arrow keys
    cv.canvas.tabIndex = 0;
    cv.canvas.id = 'tk-ws-tokens';
    cv.canvas.addEventListener('keydown', (e) => {
      const n = toks.length;
      if (!n) return;
      let h = hover;
      if (e.key === 'ArrowRight' || e.key === 'ArrowDown') h = hover < 0 ? 0 : Math.min(n - 1, hover + 1);
      else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') h = hover < 0 ? n - 1 : Math.max(0, hover - 1);
      else if (e.key === 'Home') h = 0;
      else if (e.key === 'End') h = n - 1;
      else if (e.key === 'Escape') h = -1;
      else return;
      e.preventDefault();
      if (h !== hover) { hover = h; updateInspector(); if (AM.reducedMotion) draw(0); }
    });
    cv.canvas.addEventListener('blur', () => { if (hover >= 0) { hover = -1; updateInspector(); } });

    function step(dt) {
      // play: k grows slowly at first (each early merge is visible), faster later
      if (playing) {
        playT += dt;
        const T = 11;
        const nk = Math.min(BPE.N, Math.round(BPE.N * Math.pow(playT / T, 2.2)));
        if (nk !== k) retokenize(nk);
        if (nk >= BPE.N) setPlaying(false);
      }
      const aP = 1 - Math.exp(-dt * 11), aO = 1 - Math.exp(-dt * 15);
      for (const t of toks) {
        t.P.x += (t.Q.x - t.P.x) * aP; t.P.y += (t.Q.y - t.P.y) * aP;
        for (let j = 0; j < t.o.length; j++) { const o = t.o[j]; o.x += (t.f[j] - o.x) * aO; o.y += (0 - o.y) * aO; }
        t.flash = Math.max(0, t.flash - dt * 1.4);
        if (t.pop) t.pop = Math.max(0, t.pop - dt * 4);
      }
      for (const s of stitches) s.life += dt / 0.95;
      stitches = stitches.filter((s) => s.life < 1);
    }
    // faint warp threads behind the tiles, cached per size
    let bg = null;
    function makeBg() {
      bg = document.createElement('canvas');
      bg.width = Math.max(1, Math.round(cv.w * cv.dpr)); bg.height = Math.max(1, Math.round(cv.h * cv.dpr));
      const b = bg.getContext('2d');
      b.setTransform(cv.dpr, 0, 0, cv.dpr, 0, 0);
      for (let x = 4.5; x < cv.w; x += 7) {
        const gr = b.createLinearGradient(0, 0, 0, cv.h);
        gr.addColorStop(0, AM.rgba(AM.col.linen, 0)); gr.addColorStop(0.5, AM.rgba(AM.col.linen, 0.045)); gr.addColorStop(1, AM.rgba(AM.col.linen, 0));
        b.strokeStyle = gr; b.lineWidth = 1;
        b.beginPath(); b.moveTo(x, 0); b.lineTo(x, cv.h); b.stroke();
      }
    }
    function draw(t) {
      const g = cv.g;
      cv.clear();
      if (!G) return;
      if (bg) g.drawImage(bg, 0, 0, cv.w, cv.h);
      const s = G.s;
      const hid = hover >= 0 && toks[hover] ? toks[hover].id : -1;
      for (let i = 0; i < toks.length; i++) {
        const T = toks[i];
        const r = T.r;
        const last = T.o.length - 1;
        const xL = T.P.x;
        const xR = T.P.x + T.o[last].x + (r.gl[last].w || 0) + G.pad;
        const cy = T.P.y;
        const c = tileColors(T.id);
        const hl = Math.max(T.flash, T.id === hid ? 0.9 : 0);
        const fadeIn = T.pop ? 1 - 0.85 * T.pop : 1;
        g.save();
        g.globalAlpha = fadeIn;
        if (hl > 0.02) { g.shadowColor = AM.rgba(AM.dye.weld, 0.75 * hl); g.shadowBlur = 14; }
        D.roundRect(g, xL, cy - s * 0.9, Math.max(4, xR - xL), s * 1.8, s * 0.34);
        g.fillStyle = c.fill; g.fill();
        g.shadowBlur = 0;
        g.lineWidth = 1;
        g.strokeStyle = hl > 0.02 ? AM.mix(AM.col.ruleStrong, AM.dye.weld, Math.min(1, hl)) : c.stroke;
        g.stroke();
        if (c.line) { g.fillStyle = AM.rgba(c.line, 0.9); D.roundRect(g, xL + s * 0.35, cy + s * 0.9 - s * 0.22, Math.max(1, xR - xL - s * 0.7), 1.6, 1); g.fill(); }
        g.restore();
        g.globalAlpha = fadeIn;
        for (let j = 0; j < r.gl.length; j++) drawGlyph(g, r.gl[j], T.P.x + T.o[j].x, cy + T.o[j].y, s, c.text);
        g.globalAlpha = 1;
      }
      for (const st of stitches) {
        const T = toks[tokOf[st.b]];
        if (!T) continue;
        const o = T.o[st.b - T.start];
        stitch(g, T.P.x + o.x - 0.5, T.P.y + o.y, s, st.life, s * 0.9);
      }
      // gentle hint at rest
      if (k === 0 && !playing && !played) micro(g, 'every byte is a token · press play', cv.w - G.m, cv.h - 8, { align: 'right', alpha: 0.7 });
    }
    function drawCurve() {
      const g = curve.g, w = curve.w, h = curve.h;
      curve.clear();
      const l = 30, r = 12, tp = 14, b = 22;
      const X = (kk) => l + (kk / BPE.N) * (w - l - r);
      const Y = (v) => tp + (1 - v) * (h - tp - b);
      // grid
      g.strokeStyle = AM.rgba(AM.col.linen, 0.07); g.lineWidth = 1;
      [0.25, 0.5, 0.75, 1].forEach((v) => { g.beginPath(); g.moveTo(l, Y(v) + 0.5); g.lineTo(w - r, Y(v) + 0.5); g.stroke(); });
      D.text(g, '1.0', l - 6, Y(1) + 3, { size: 9, role: 'mono', align: 'right', color: AM.col.mist });
      D.text(g, '0.5', l - 6, Y(0.5) + 3, { size: 9, role: 'mono', align: 'right', color: AM.col.mist });
      D.text(g, '0', l - 6, Y(0) + 3, { size: 9, role: 'mono', align: 'right', color: AM.col.mist });
      D.text(g, '0', l, h - 6, { size: 9, role: 'mono', align: 'center', color: AM.col.mist });
      D.text(g, `${BPE.N} merges`, w - r, h - 6, { size: 9, role: 'mono', align: 'right', color: AM.col.mist });
      D.text(g, 'tokens per byte', l + 4, Y(1) - 4 + 0, { size: 9, role: 'mono', color: AM.col.mist });
      const line = (vals, col, width) => {
        g.beginPath();
        vals.forEach((v, i) => { const x = X(i), y = Y(v); if (i) g.lineTo(x, y); else g.moveTo(x, y); });
        g.strokeStyle = col; g.lineWidth = width; g.stroke();
      };
      const corpusVals = BPE.totals.map((v) => v / BPE.corpusBytes);
      // soft area under the corpus curve
      g.beginPath();
      corpusVals.forEach((v, i) => { const x = X(i), y = Y(v); if (i) g.lineTo(x, y); else g.moveTo(x, y); });
      g.lineTo(X(BPE.N), Y(0)); g.lineTo(X(0), Y(0)); g.closePath();
      const ag = g.createLinearGradient(0, tp, 0, h - b);
      ag.addColorStop(0, AM.rgba(AM.dye.weld, 0.16)); ag.addColorStop(1, AM.rgba(AM.dye.weld, 0));
      g.fillStyle = ag; g.fill();
      line(corpusVals, AM.dye.weld, 1.6);
      if (textCurve.length) line(textCurve, AM.dye.woad, 1.4);
      // the current merge count
      const x = X(k);
      g.setLineDash([3, 3]); g.strokeStyle = AM.rgba(AM.col.linen, 0.35); g.beginPath(); g.moveTo(x + 0.5, tp); g.lineTo(x + 0.5, h - b); g.stroke(); g.setLineDash([]);
      D.glowDot(g, x, Y(corpusVals[k]), 2.4, AM.dye.weld, 1);
      if (textCurve.length) D.glowDot(g, x, Y(textCurve[k]), 2.2, AM.dye.woad, 1);
    }

    function setPlaying(on) {
      playing = on;
      playBtn.textContent = on ? 'Pause' : (k >= BPE.N ? 'Replay merges' : 'Play merges');
      playBtn.setAttribute('aria-pressed', String(on));
      if (on) {
        played = true;
        if (k >= BPE.N) { retokenize(0, { snap: false }); }
        // resume from the current k on the same curve
        playT = 11 * Math.pow(k / BPE.N, 1 / 2.2);
      }
    }

    cv.onResize(() => { G = geom(cv.w); makeBg(); if (toks.length) layoutTargets(true); else retokenize(k, { snap: true }); stitches = []; draw(0); });
    curve.onResize(() => drawCurve());
    retokenize(k, { snap: true });
    computeTextCurve();

    slider.input.addEventListener('input', () => {
      played = true;                    // a reader who moves the slider has taken over; no autoplay later
      if (playing) setPlaying(false);
      retokenize(slider.get());
      playBtn.textContent = k >= BPE.N ? 'Replay merges' : 'Play merges';
      if (AM.reducedMotion) { step(1); draw(0); drawCurve(); }
    });
    playBtn.addEventListener('click', () => setPlaying(!playing));
    let deb = 0;
    const fit = () => { textarea.style.height = 'auto'; textarea.style.height = textarea.scrollHeight + 2 + 'px'; };
    textarea.addEventListener('input', () => { fit(); clearTimeout(deb); deb = setTimeout(() => setText(textarea.value.slice(0, 240) || ' '), 120); });
    window.addEventListener('resize', fit);
    requestAnimationFrame(fit);

    // autoplay the merges once, shortly after the figure first scrolls into view
    const vis = onScreen(canvasHost, () => {
      if (played || AM.reducedMotion) return;
      setTimeout(() => { if (!played && vis.on) setPlaying(true); }, 700);
    });
    ctx.loop((t, dt) => { if (!vis.on) return; step(dt); draw(t); drawCurve(); });
    draw(0); drawCurve();
    return {
      preset(v) { const p = PRESETS.find((x) => x.value === v); if (!p) return; textarea.value = p.text; fit(); setText(p.text, true); },
    };
  }

  // =================================================================== 7. chapter mount
  AM.css(`
    ${CH} .tk-sp { display: inline-block; width: 0.5em; height: 0.32em; margin: 0 0.06em; border: 1.5px solid var(--weld); border-top: 0; border-radius: 0 0 2px 2px; vertical-align: 0.02em; opacity: 0.9; }
    ${CH} .tk-hex { font-family: var(--font-mono); font-size: 0.72em; padding: 0 0.25em; border: 1px solid color-mix(in srgb, var(--lichen) 55%, transparent); border-radius: 3px; color: var(--linen-dim); margin: 0 0.05em; }
    ${CH} .tk-chip { display: inline-flex; align-items: baseline; justify-content: center; min-width: 1.6em; padding: 0 0.42em; border-radius: 5px; background: var(--ink-3); border: 1px solid var(--rule-strong); font-family: var(--font-body); font-weight: 600; color: var(--linen); white-space: pre; line-height: 1.5; }
    ${CH} .tk-chip::before { content: '\\200B'; }
    ${CH} .tk-chip-new { border-color: color-mix(in srgb, var(--weld) 55%, var(--rule)); background: color-mix(in srgb, var(--weld) 12%, var(--ink-2)); }
    ${CH} .tk-hero canvas { cursor: pointer; }
    ${CH} .tk-hero-ctl { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
    ${CH} .tk-hero-ctl .text-input { flex: 1 1 280px; min-width: 0; }
    ${CH} .tk-ids { font-family: var(--font-mono); font-size: 12px; line-height: 1.75; color: var(--linen-dim); overflow-wrap: anywhere; }
    ${CH} .tk-ids .tk-ids-label { color: var(--mist); letter-spacing: 0.12em; text-transform: uppercase; font-size: 10.5px; margin-right: 8px; }
    ${CH} .tk-ids b { font-weight: 500; color: var(--weld); }
    ${CH} .tk-ids .tk-ids-count { white-space: nowrap; }
    ${CH} .tk-stage-fig .fig-top { min-height: 24px; }
    ${CH} .tk-ws { display: grid; gap: var(--space-4); }
    ${CH} .tk-ws-head { display: flex; flex-wrap: wrap; gap: var(--space-3) var(--space-5); align-items: flex-end; justify-content: space-between; }
    ${CH} .tk-ws-head .ctl-range { width: min(440px, 100%); }
    ${CH} .tk-ws-canvas { border-radius: var(--radius-sm); background: color-mix(in srgb, var(--ink) 70%, transparent); border: 1px solid var(--rule); }
    ${CH} .tk-insp { min-height: 2.2em; font-size: var(--fs-small); color: var(--linen-dim); line-height: 1.7; }
    ${CH} .tk-insp-hint { color: var(--mist); }
    ${CH} .tk-insp-same { color: var(--weld); }
    ${CH} .tk-ws-input { display: grid; gap: var(--space-3); }
    ${CH} .tk-ws-input textarea { resize: none; overflow: hidden; min-height: 44px; line-height: 1.45; }
    ${CH} .tk-ws-input .seg { max-width: 100%; }
    ${CH} .tk-ws-lower { display: grid; grid-template-columns: minmax(0, 1.25fr) minmax(0, 1fr); gap: var(--space-5); }
    @media (max-width: 760px) { ${CH} .tk-ws-lower { grid-template-columns: minmax(0, 1fr); } }
    ${CH} .tk-sub { font-family: var(--font-mono); font-size: var(--fs-micro); letter-spacing: 0.12em; text-transform: uppercase; color: var(--mist); margin-bottom: 8px; }
    ${CH} .tk-merges { list-style: none; margin: 0; padding: 0; display: grid; gap: 4px; font-size: 13px; }
    ${CH} .tk-m { display: grid; grid-template-columns: 3.4em minmax(0, 1fr) auto; gap: 8px; align-items: center; padding: 4px 8px; border-radius: 6px; border: 1px solid transparent; color: var(--linen-dim); }
    ${CH} .tk-m-n { font-family: var(--font-mono); font-size: 11px; color: var(--mist); }
    ${CH} .tk-m-rule { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    ${CH} .tk-m-c { font-family: var(--font-mono); font-size: 10.5px; color: var(--mist); white-space: nowrap; }
    ${CH} .tk-m.is-last { border-color: color-mix(in srgb, var(--weld) 50%, var(--rule)); background: color-mix(in srgb, var(--weld) 8%, var(--ink-2)); }
    ${CH} .tk-m.is-last .tk-m-n, ${CH} .tk-m.is-last .tk-m-c { color: var(--weld); }
    ${CH} .tk-m.is-next { opacity: 0.45; border-style: dashed; border-color: var(--rule); }
    ${CH} .tk-m-empty { color: var(--mist); font-size: var(--fs-small); padding: 4px 8px; }
    ${CH} .tk-stats { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin-bottom: 10px; }
    ${CH} .tk-stat { padding: 8px 10px; border-radius: var(--radius-sm); background: var(--ink); border: 1px solid var(--rule); min-width: 0; }
    ${CH} .tk-stat .tk-stat-l { font-family: var(--font-mono); font-size: 10px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--mist); }
    ${CH} .tk-stat .tk-stat-v { font-family: var(--font-display); font-size: 1.6rem; line-height: 1.1; color: var(--linen); font-variant-numeric: lining-nums tabular-nums; }
    ${CH} .tk-stat .tk-stat-v.is-gold { color: var(--weld); }
    ${CH} .tk-rule { display: flex; flex-wrap: wrap; gap: 6px 18px; font-family: var(--font-mono); font-size: 11px; color: var(--mist); }
    ${CH} .step .tk-chip { font-size: 0.92em; }
  `);

  AM.chapter({
    id: 'tokens',
    num: 1,
    kicker: 'Tokenization',
    title: 'Shattering <em>Text</em>',
    lede: 'A model never reads letters or words. Text is first broken into tokens, pieces from a fixed vocabulary, and each piece is swapped for an integer ID.',
    where: 'tokens',
    mount(root, ctx) {
      ctx.header();
      const el = ctx.el;
      const ui = AM.ui;
      const SP = '<span class="tk-sp" role="img" aria-label="space"></span>';
      const chip = (html) => `<span class="tk-chip">${html}</span>`;
      const idOf = (w) => { const t = BPE.encode(w); return t.map((x) => x.id); };

      // ------------------------------------------------ hero
      const heroHost = el('div');
      const heroInput = el('input', { type: 'text', class: 'text-input', id: 'tk-hero-text', maxlength: '90', value: HERO_DEFAULT, 'aria-label': 'Sentence to shatter', spellcheck: 'false', autocomplete: 'off' });
      const heroBtn = ui.button({ id: 'tk-hero-go', label: 'Shatter', kind: 'primary' });
      const idsLine = el('p', { class: 'tk-ids', 'aria-live': 'polite' });
      const heroBadge = ui.badge('illustration', 'Illustration · toy-BPE IDs');
      heroBadge.title = 'The glass is decorative. The cut points and ID numbers come from the real BPE tokenizer trained on this page.';
      const heroFig = ui.figure({
        title: 'Shatter a sentence',
        badge: heroBadge,
        cls: 'tk-hero',
        caption: `The cracks fall exactly on the token boundaries chosen by the tokenizer trained further down this page. ${SP} marks a space. Most tokenizers glue a word’s leading space onto the word: “${SP}loom” is one token (ID ${idOf(' loom').join('')}), while “loom” with no space before it becomes ${idOf('loom').length} tokens. GPT-2’s tokenizer prints that space as Ġ and SentencePiece prints it as ▁; we use the open-box symbol because it is easier to read. Tap the glass, or type your own sentence.`,
      }, heroHost, el('div', { class: 'tk-hero-ctl' }, heroInput, heroBtn), idsLine);

      const body = el('div', { class: 'ch-body' });
      root.appendChild(body);
      body.appendChild(heroFig);

      const renderIds = (toks, txt) => {
        idsLine.replaceChildren(el('span', { class: 'tk-ids-label' }, 'Model input'), '[ ',
          ...toks.flatMap((t, i) => [el('b', {}, String(t.id)), i < toks.length - 1 ? ', ' : '\u00a0]']),
          el('span', { class: 'tk-ids-label tk-ids-count', style: { marginLeft: '12px' } }, `${charCount(txt)} chars · ${toks.length} tokens`));
      };
      const hero = buildHero(ctx, heroHost, renderIds);
      heroBtn.addEventListener('click', () => { if (heroInput.value.trim() !== hero.text) hero.setText(heroInput.value); else hero.strike(); });
      heroInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); hero.setText(heroInput.value); } });

      // ------------------------------------------------ intro prose
      body.appendChild(el('div', { class: 'prose', html: `
        <p>A transformer only does arithmetic, so text has to become numbers before anything else can happen. A <span class="term">tokenizer</span> cuts the text into <span class="term">tokens</span> and replaces each one with its <span class="term">token ID</span>: a row number in a fixed <span class="term">vocabulary</span> that was settled before the model was trained.</p>
        <p>Real vocabularies are large. GPT-2’s has <strong>50,257</strong> entries; recent models use between about 100,000 and a few hundred thousand. The IDs above come from a much smaller vocabulary of <strong>${(256 + BPE.N).toLocaleString('en-US')}</strong> tokens, learned in your browser a moment ago from a ${CORPUS.split(/\s+/).length}-word story about a weaver. Here is how it was built.</p>
      ` }));

      // ------------------------------------------------ scrollytelling
      const stageTitle = el('span', { class: 'fig-title' }, '');
      const stageBadge = ui.badge('toy', 'Real BPE · toy corpus');
      stageBadge.title = 'A real byte-level BPE tokenizer, trained in your browser on the short story used throughout this chapter.';
      const stageCanvasHost = el('div');
      const stageFig = el('figure', { class: 'fig tk-stage-fig' }, el('div', { class: 'fig-top' }, stageTitle, stageBadge), stageCanvasHost);
      const stage = el('div', { class: 'ch-stage' }, stageFig);
      const [idThe] = idOf(' the');
      const strawIds = idOf(' strawberry');
      const steps = [
        { label: '1 · How fine to cut', html: `
          <h3>Letters are too fine, words too coarse</h3>
          <p>Feed a model single characters and every sentence becomes a long sequence. Attention compares every token with every other token, so the work grows with the square of the length. The gold beads read one token per step.</p>
          <p>Use whole words instead and the vocabulary never ends. Any word missing from it, like <em>hummed</em> here, collapses into an unknown <code>[UNK]</code>. <strong>Subword</strong> tokens sit in between: common words stay whole, rare ones are assembled from pieces.</p>` },
        { label: '2 · Learning the pieces', html: `
          <h3>Byte-Pair Encoding: merge the most frequent pair</h3>
          <p>Start with every word spelled as raw bytes. There are only 256 possible bytes, so nothing is ever unknown. Count every adjacent pair inside each word, across the whole corpus, then merge the most frequent pair into a new token and repeat. Each merge adds one entry to the vocabulary.</p>
          <p>On our story, ${chip(SP)} + ${chip('t')} wins first, then ${chip('h')} + ${chip('e')}, and soon ${chip(SP + 't')} + ${chip('he')} gives ${chip(SP + 'the')} (ID ${idThe}). Our tokenizer stops when no pair appears twice, after <strong>${BPE.N}</strong> merges. Real tokenizers keep merging until the vocabulary reaches a chosen size: <span style="white-space:nowrap">GPT-2’s</span> 50,257 is 256 bytes, 50,000 merges and one special end-of-text token.</p>` },
        { label: '3 · Tokenizing new text', html: `
          <h3>Replay the merges, in order</h3>
          <p>Training happens once. To tokenize new text, split it into words, spell each word as bytes, then keep applying the learned merge with the lowest rank (the one learned earliest) until none applies. Each merge here twists two strands into one yarn.</p>
          <p>${chip(SP + 'weavers')} is common enough to end as one token. ${chip(SP + 'looming')} never appeared in the story, so it comes out as ${chip(SP + 'loom')} + ${chip('ing')}, two pieces the model knows well.</p>` },
        { label: '4 · A blind spot', html: `
          <h3>How many r’s are in strawberry?</h3>
          <p>Chatbots famously stumble on this. They never see the letters. Our toy tokenizer turns ${chip(SP + 'strawberry')} into ${strawIds.length === 2 ? 'two' : strawIds.length} IDs, ${strawIds.join(' and ')}. GPT-4’s tokenizer cuts “strawberry”, with no space before it, into three: <em style="white-space:nowrap">str · aw · berry</em>.</p>
          <p>To count letters, a model must have learned the spelling of each token as a separate fact, and then count across pieces. Spelling, rhyming and reversing words are hard for the same reason.</p>` },
        { label: '5 · Rough edges', html: `
          <h3>Odd numbers, expensive languages</h3>
          <p>Digits are merged wherever the corpus happened to repeat them. The story mentions 1804 twice, so ${chip(SP + '1804')} is a single token, while 1805 becomes ${chip(SP + '180')} + ${chip('5')} and 2025 falls apart into digits. That patchiness is one reason arithmetic is awkward for language models; some tokenizers now split numbers into single digits, or into groups of up to three digits.</p>
          <p>Scripts the tokenizer rarely saw fall back to raw bytes: the Japanese 布 costs three tokens for one character. Production tokenizers train on many languages, so the gap is smaller than here, but the same text usually costs more tokens outside English.</p>` },
      ];
      const stepEls = steps.map((s) => el('div', { class: 'step' }, el('div', { class: 'step-label' }, s.label), el('div', { html: s.html, style: { display: 'grid', gap: '12px' } })));
      body.appendChild(el('div', { class: 'ch-split' }, stage, el('div', { class: 'ch-prose' }, stepEls)));
      const st = buildStage(ctx, stageCanvasHost, stageTitle);
      ctx.steps(stepEls, (i) => st.set(i));

      // ------------------------------------------------ workshop
      body.appendChild(el('div', { class: 'prose', html: `
        <p>Now drive the tokenizer yourself. The slider replays its <strong>${BPE.N}</strong> merges one at a time. At zero, every byte is its own token. Each step to the right stitches one more pair together everywhere it occurs. Type anything below and it is tokenized with whatever merges have been learned so far.</p>
      ` }));
      const wsCanvasHost = el('div', { class: 'tk-ws-canvas' });
      const curveHost = el('div');
      const inspector = el('p', { class: 'tk-insp', 'aria-live': 'polite' });
      const mergeList = el('ol', { class: 'tk-merges', 'aria-label': 'Learned merges around the current step' });
      const mkStat = (label, gold) => { const v = el('div', { class: 'tk-stat-v' + (gold ? ' is-gold' : '') }, '–'); return { node: el('div', { class: 'tk-stat' }, el('div', { class: 'tk-stat-l' }, label), v), v }; };
      const sK = mkStat('Merges learned'), sV = mkStat('Vocabulary size', true), sT = mkStat('Tokens in your text'), sC = mkStat('Chars per token', true);
      const sCorpus = el('span', {}, '');
      const slider = ui.slider({ id: 'tk-ws-k', label: 'Merges learned', min: 0, max: BPE.N, step: 1, value: 0, format: (v) => `${v} / ${BPE.N}` });
      const playBtn = ui.button({ id: 'tk-ws-play', label: 'Play merges', kind: 'primary' });
      const textarea = el('textarea', { class: 'text-input', id: 'tk-ws-text', rows: '1', maxlength: '240', 'aria-label': 'Text to tokenize', spellcheck: 'false' });
      textarea.value = PRESETS[0].text;
      let ws = null;
      const presets = ui.segmented({ id: 'tk-ws-preset', label: 'Try', options: PRESETS.map((p) => ({ value: p.value, label: p.label })), value: 'corpus', onChange: (v) => ws && ws.preset(v) });
      const wsBadge = ui.badge('toy', 'Real BPE · toy corpus');
      wsBadge.title = 'A real byte-level BPE tokenizer, trained in your browser on the short story used throughout this chapter. Every number here is computed live.';
      const wsFig = ui.figure({
        title: 'The merge loom',
        badge: wsBadge,
        cls: 'tk-ws ch-wide',
        caption: `Grey tiles are raw bytes (IDs 0 to 255); dyed tiles are learned tokens (IDs from 256 up, in the order they were learned). Our vocabulary is tiny, so even fully trained it averages about ${(BPE.corpusBytes / BPE.totals[BPE.N]).toFixed(1)} characters per token on its own corpus. Real tokenizers, with vocabularies a hundred times larger, average about <strong>4 characters</strong>, or <strong>¾ of an English word</strong>, per token. The ID numbers belong to this toy vocabulary; GPT-2 or any other real tokenizer numbers its pieces differently.`,
      },
      el('div', { class: 'tk-ws-head' }, slider.el, playBtn),
      wsCanvasHost,
      inspector,
      el('div', { class: 'tk-ws-input' }, textarea, presets.el),
      el('div', { class: 'tk-ws-lower' },
        el('div', {}, el('div', { class: 'tk-sub' }, 'Merge list, in learned order'), mergeList),
        el('div', {},
          el('div', { class: 'tk-stats' }, sK.node, sV.node, sT.node, sC.node),
          el('div', { class: 'tk-sub' }, 'Compression as merges are learned'),
          curveHost,
          el('div', { class: 'tk-rule' }, ui.legend([{ color: AM.dye.weld, label: 'whole corpus' }, { color: AM.dye.woad, label: 'your text' }]), el('span', {}, 'corpus tokens: ', sCorpus)))));
      body.appendChild(wsFig);
      ws = buildWorkshop(ctx, {
        canvasHost: wsCanvasHost, curveHost, inspector, mergeList, slider, playBtn, textarea, presets,
        stats: { k: sK.v, vocab: sV.v, tok: sT.v, cpt: sC.v, corpus: sCorpus },
      });

      // ------------------------------------------------ key idea
      body.appendChild(el('div', { class: 'callout' },
        el('div', { class: 'callout-label' }, 'Key idea'),
        el('p', { html: 'A transformer never sees text. It sees a list of token IDs, produced by replaying a fixed list of learned merges. Common words cost one token; rare words, numbers and unfamiliar scripts cost several. Anything that depends on individual letters has to be learned around the tokens. Next, each ID is swapped for a long list of numbers the model can actually compute with.' })));
    },
  });
})();
