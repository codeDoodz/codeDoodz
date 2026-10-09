/* The Attention Loom — shared canvas drawing primitives.
   Everything takes a 2D context `g` already scaled to CSS pixels (see ctx.canvas). */
(() => {
  const AM = window.AM;
  const D = (AM.draw = AM.draw || {});

  /** Rounded rectangle path (does not fill/stroke). */
  D.roundRect = (g, x, y, w, h, r = 6) => {
    g.beginPath();
    if (!(w > 0 && h > 0)) return; // degenerate box (e.g. mid-resize): empty path
    r = Math.max(0, Math.min(r, w / 2, h / 2));
    g.moveTo(x + r, y);
    g.arcTo(x + w, y, x + w, y + h, r);
    g.arcTo(x + w, y + h, x, y + h, r);
    g.arcTo(x, y + h, x, y, r);
    g.arcTo(x, y, x + w, y, r);
    g.closePath();
  };

  /** Text helper. opts: {size, role:'body'|'mono'|'display', weight, italic, color, align, baseline, alpha, maxWidth} */
  D.text = (g, str, x, y, o = {}) => {
    g.save();
    g.font = AM.font(o.size || 14, o.role || 'body', o.weight, o.italic);
    g.fillStyle = o.color || AM.col.linen;
    g.textAlign = o.align || 'left';
    g.textBaseline = o.baseline || 'alphabetic';
    if (o.alpha != null) g.globalAlpha *= o.alpha;
    if (o.letterSpacing && 'letterSpacing' in g) g.letterSpacing = o.letterSpacing;
    g.fillText(str, x, y, o.maxWidth);
    g.restore();
  };

  /** Measure text width for a font role/size. */
  D.measure = (g, str, size = 14, role = 'body', weight) => {
    g.save(); g.font = AM.font(size, role, weight); const w = g.measureText(str).width; g.restore(); return w;
  };

  /** Soft glowing dot (a bead of light). */
  D.glowDot = (g, x, y, r, color, alpha = 1) => {
    g.save();
    const grd = g.createRadialGradient(x, y, 0, x, y, r * 3);
    grd.addColorStop(0, AM.rgba(color, 0.9 * alpha));
    grd.addColorStop(0.25, AM.rgba(color, 0.35 * alpha));
    grd.addColorStop(1, AM.rgba(color, 0));
    g.fillStyle = grd;
    g.beginPath(); g.arc(x, y, r * 3, 0, Math.PI * 2); g.fill();
    g.fillStyle = AM.rgba('#fff8e6', 0.85 * alpha);
    g.beginPath(); g.arc(x, y, r * 0.45, 0, Math.PI * 2); g.fill();
    g.restore();
  };

  /**
   * A silk thread between two points: a cubic curve that sags/arcs, with an
   * optional soft glow. opts: {color, width, alpha, bend, glow, dash, lift}
   *  bend: arc height as a fraction of the distance (negative arcs the other way)
   *  lift: absolute extra arc height in px
   */
  D.thread = (g, x1, y1, x2, y2, o = {}) => {
    const color = o.color || AM.dye.weld;
    const width = o.width ?? 1.5;
    const alpha = o.alpha ?? 1;
    if (alpha <= 0.003 || width <= 0) return;
    const dx = x2 - x1, dy = y2 - y1;
    const dist = Math.hypot(dx, dy) || 1;
    const bend = o.bend ?? 0.35;
    const lift = (o.lift ?? 0) + dist * bend;
    // normal vector (rotate direction by -90°) so positive bend arcs "up" for left→right
    const nx = dy / dist, ny = -dx / dist;
    const sgn = dx < 0 ? -1 : 1;
    const c1x = x1 + dx * 0.25 + nx * lift * sgn, c1y = y1 + dy * 0.25 + ny * lift * sgn;
    const c2x = x1 + dx * 0.75 + nx * lift * sgn, c2y = y1 + dy * 0.75 + ny * lift * sgn;
    g.save();
    g.lineCap = 'round';
    if (o.dash) g.setLineDash(o.dash);
    if (o.glow !== false && width > 0.6) {
      g.strokeStyle = AM.rgba(color, 0.18 * alpha);
      g.lineWidth = width * 4.5;
      g.beginPath(); g.moveTo(x1, y1); g.bezierCurveTo(c1x, c1y, c2x, c2y, x2, y2); g.stroke();
    }
    g.strokeStyle = AM.rgba(color, alpha);
    g.lineWidth = width;
    g.beginPath(); g.moveTo(x1, y1); g.bezierCurveTo(c1x, c1y, c2x, c2y, x2, y2); g.stroke();
    g.restore();
    return { c1x, c1y, c2x, c2y };
  };

  /** Point on the cubic thread at parameter t (for particles travelling along a thread). */
  D.threadPoint = (x1, y1, x2, y2, t, o = {}) => {
    const dx = x2 - x1, dy = y2 - y1;
    const dist = Math.hypot(dx, dy) || 1;
    const lift = (o.lift ?? 0) + dist * (o.bend ?? 0.35);
    const nx = dy / dist, ny = -dx / dist, sgn = dx < 0 ? -1 : 1;
    const c1x = x1 + dx * 0.25 + nx * lift * sgn, c1y = y1 + dy * 0.25 + ny * lift * sgn;
    const c2x = x1 + dx * 0.75 + nx * lift * sgn, c2y = y1 + dy * 0.75 + ny * lift * sgn;
    const u = 1 - t;
    return {
      x: u * u * u * x1 + 3 * u * u * t * c1x + 3 * u * t * t * c2x + t * t * t * x2,
      y: u * u * u * y1 + 3 * u * u * t * c1y + 3 * u * t * t * c2y + t * t * t * y2,
    };
  };

  /**
   * Token tile: rounded label for a word/token. Returns {x, y, w, h} of the box.
   * opts: {size, role, weight, pad, fill, stroke, color, selected, alpha, align:'center'|'left', underline}
   */
  D.token = (g, str, x, y, o = {}) => {
    const size = o.size || 15;
    const role = o.role || 'body';
    const weight = o.weight || 600;
    const padX = o.padX ?? Math.round(size * 0.65), padY = o.padY ?? Math.round(size * 0.42);
    g.save();
    g.font = AM.font(size, role, weight);
    const tw = g.measureText(str).width;
    const w = o.w || tw + padX * 2, h = o.h || size + padY * 2;
    const bx = o.align === 'left' ? x : x - w / 2;
    const by = y - h / 2;
    if (o.alpha != null) g.globalAlpha *= o.alpha;
    if (o.selected) {
      g.shadowColor = AM.rgba(o.glowColor || AM.dye.weld, 0.6);
      g.shadowBlur = 18;
    }
    D.roundRect(g, bx, by, w, h, o.radius ?? 7);
    g.fillStyle = o.fill || (o.selected ? AM.mix(AM.col.ink2, AM.dye.weld, 0.16) : AM.col.ink2);
    g.fill();
    g.shadowBlur = 0;
    g.lineWidth = 1;
    g.strokeStyle = o.stroke || (o.selected ? AM.dye.weld : AM.col.ruleStrong);
    g.stroke();
    if (o.underline) {
      g.fillStyle = o.underline;
      D.roundRect(g, bx + 7, by + h - 4, w - 14, 2, 1);
      g.fill();
    }
    g.fillStyle = o.color || AM.col.linen;
    g.textAlign = 'center';
    g.textBaseline = 'middle';
    g.fillText(str, bx + w / 2, by + h / 2 + size * 0.04);
    g.restore();
    return { x: bx, y: by, w, h, cx: bx + w / 2, cy: by + h / 2 };
  };

  /** Heatmap of a matrix into a rectangle. opts: {color(v,i,j), gap, radius} */
  D.heatmap = (g, x, y, w, h, M, o = {}) => {
    const R = M.length, C = M[0].length;
    const gap = o.gap ?? 1.5;
    const cw = (w - gap * (C - 1)) / C, ch = (h - gap * (R - 1)) / R;
    const color = o.color || ((v) => AM.color.heat(v));
    g.save();
    for (let i = 0; i < R; i++) for (let j = 0; j < C; j++) {
      const val = M[i][j];
      if (val == null) continue;
      g.fillStyle = color(val, i, j);
      if (o.radius) { D.roundRect(g, x + j * (cw + gap), y + i * (ch + gap), cw, ch, o.radius); g.fill(); }
      else g.fillRect(x + j * (cw + gap), y + i * (ch + gap), cw, ch);
    }
    g.restore();
    return { cw, ch, gap };
  };

  /** Vector as vertical bars around a midline (thread barcode). opts: {max, pos, neg, alpha} */
  D.vectorBars = (g, x, y, w, h, vec, o = {}) => {
    const n = vec.length;
    const m = o.max || Math.max(1e-6, ...Array.from(vec, Math.abs));
    const bw = w / n, mid = y + h / 2;
    g.save();
    for (let i = 0; i < n; i++) {
      const t = Math.max(-1, Math.min(1, vec[i] / m));
      g.globalAlpha = (o.alpha ?? 1) * (0.35 + 0.65 * Math.abs(t));
      g.fillStyle = t >= 0 ? (o.pos || AM.dye.woad) : (o.neg || AM.dye.madder);
      const hh = Math.abs(t) * (h / 2);
      g.fillRect(x + i * bw + bw * 0.12, t >= 0 ? mid - hh : mid, Math.max(1, bw * 0.76), Math.max(0.75, hh));
    }
    g.restore();
  };

  /** Arrow from (x1,y1) to (x2,y2). opts: {color, width, head, alpha, dash} */
  D.arrow = (g, x1, y1, x2, y2, o = {}) => {
    const color = o.color || AM.col.linenDim, width = o.width || 1.5, head = o.head || 7;
    const a = Math.atan2(y2 - y1, x2 - x1);
    g.save();
    g.globalAlpha *= o.alpha ?? 1;
    g.strokeStyle = color; g.fillStyle = color; g.lineWidth = width; g.lineCap = 'round';
    if (o.dash) g.setLineDash(o.dash);
    g.beginPath(); g.moveTo(x1, y1); g.lineTo(x2 - Math.cos(a) * head * 0.8, y2 - Math.sin(a) * head * 0.8); g.stroke();
    g.setLineDash([]);
    g.beginPath();
    g.moveTo(x2, y2);
    g.lineTo(x2 - Math.cos(a - 0.42) * head, y2 - Math.sin(a - 0.42) * head);
    g.lineTo(x2 - Math.cos(a + 0.42) * head, y2 - Math.sin(a + 0.42) * head);
    g.closePath(); g.fill();
    g.restore();
  };

  /** Faint woven texture into a rect (for stage backgrounds). */
  D.weave = (g, x, y, w, h, o = {}) => {
    const step = o.step || 6, alpha = o.alpha ?? 0.05;
    g.save();
    g.beginPath(); g.rect(x, y, w, h); g.clip();
    g.strokeStyle = AM.rgba(AM.col.linen, alpha);
    g.lineWidth = 1;
    g.beginPath();
    for (let xx = x + 0.5; xx < x + w; xx += step) { g.moveTo(xx, y); g.lineTo(xx, y + h); }
    g.stroke();
    g.strokeStyle = AM.rgba(AM.col.linen, alpha * 0.6);
    g.beginPath();
    for (let yy = y + 0.5; yy < y + h; yy += step) { g.moveTo(x, yy); g.lineTo(x + w, yy); }
    g.stroke();
    g.restore();
  };

  /** Lay out tokens in a row centred in [x0, x1]; returns centres. Wraps scale if too wide. */
  D.layoutRow = (g, tokens, x0, x1, o = {}) => {
    const size = o.size || 15, gap = o.gap ?? 10, role = o.role || 'body', weight = o.weight || 600;
    const pad = Math.round(size * 0.65) * 2;
    const widths = tokens.map((t) => D.measure(g, t, size, role, weight) + pad);
    const total = widths.reduce((a, b) => a + b, 0) + gap * (tokens.length - 1);
    const scale = Math.min(1, (x1 - x0) / total);
    let x = x0 + ((x1 - x0) - total * scale) / 2;
    return widths.map((w) => { const c = x + (w * scale) / 2; x += (w + gap) * scale; return { cx: c, w: w * scale, scale }; });
  };
})();
