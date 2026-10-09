# The Attention Loom — build contract

A scroll-driven, art-forward, *interactive* explainer of how transformers work.
Everything is vanilla JS + Canvas/SVG, no build step required to view
(`index.html` opens straight from disk via `file://`). A tiny bundler
(`tools/build.mjs`) inlines everything into one HTML file for publishing.

This file is the contract every contributor (human or agent) follows.

---

## 1. Art direction — "the loom"

Metaphor that ties the whole piece together: **a transformer is a loom that
weaves context into words.**

- Tokens are **warp threads** (vertical, fixed in order).
- Attention draws **weft threads** between tokens — silk-like bezier curves whose
  brightness/width = attention weight.
- Each attention head has its own **dye colour** (madder, weld, woad, cochineal,
  verdigris, saffron, lichen). Use `AM.dye` — never invent new hues.
- The **residual stream** is the main thread running up through the layers;
  attention and MLP blocks "read from it and add to it".
- Ground is a deep **indigo-dyed cloth** with a faint woven texture.
  Text is unbleached **linen**. Gold (**weld**) is the primary accent thread.

The page is deliberately dark-only (one visual world). All colours come from
CSS custom properties in `css/base.css` (`--ink`, `--linen`, `--weld`, …) or,
in canvas code, from `AM.dye` / `AM.col` (which read the same values).

Typography: **Bodoni Moda** (display, used with restraint — chapter titles,
big numerals), **Figtree** (body), **Martian Mono** (labels, numbers,
matrices, uppercase micro-labels with letter-spacing).

Tone of copy: plain, warm, precise. Short sentences. A knowledgeable friend
explaining at a whiteboard. No hype, no "delve", no em-dash asides, no
"not X but Y" constructions. Real terms of art are introduced and then used
(query, key, value, logit, softmax, residual stream, d_model…).

**Honesty labels.** Every visualization carries a badge made by
`AM.ui.badge(kind)`:
- `live` — "LIVE MODEL": numbers computed right now by the real tiny transformer
  shipped with the page (`AM.model`).
- `toy` — "TOY NUMBERS": small hand-built vectors chosen to show the mechanism.
- `illustration` — "ILLUSTRATION": artistic depiction, not literal numbers.

The canonical label always shows. A qualifier goes in the second argument and is
drawn after a thin divider in mist: `AM.ui.badge('toy', 'exact formula')` →
"TOY NUMBERS | exact formula". Never replace the label with free text.

Never present toy numbers as if they came from a real model.

## 2. Files

```
attention-loom/
  index.html                 shell: <section data-chapter="…"> placeholders, script tags in order
  css/base.css               design tokens + shared components (owned by core)
  js/core/am.js              global AM namespace: registry, lifecycle, canvas, math, colours
  js/core/ui.js              shared UI widgets (sliders, chips, matrix, badge, map glyph…)
  js/core/draw.js            shared drawing primitives (threads, tokens, glow, vectors)
  js/model/*.js              tiny transformer: tensor lib, model, weights, API (see §6)
  js/chapters/NN-name.js     one file per chapter (owned by that chapter's author)
  tools/build.mjs            bundles everything into dist/attention-loom.html
  tools/shoot.mjs            Playwright screenshot + console-error checker
  docs/CONTRACT.md           this file
  docs/model-notes.md        what the real model learned (written after training)
```

**Ownership rule:** a chapter author edits ONLY their own `js/chapters/NN-*.js`
file (and may add chapter-scoped CSS *inside that JS file* via
`AM.css(\`...\`)`, with every selector prefixed by `#ch-<id>`). Do not edit
core files, other chapters, or `index.html`. If you need a helper that does not
exist, write it locally inside your chapter file.

## 3. Chapter API

```js
AM.chapter({
  id: 'attention',               // matches <section data-chapter="attention">
  num: 4,                        // order in the journey (shown as numeral)
  kicker: 'Self-attention',      // the term of art
  title: 'Threads of Attention', // evocative title (Bodoni)
  lede: 'One sentence: what this part of the machine does.',
  where: 'attn',                 // which part of the architecture glyph to light up
  mount(root, ctx) { ... }       // build DOM + start visuals. root = <section>
});
```

`ctx` (per-chapter helpers, all lifecycle-aware):
- `ctx.header()` → renders the standard chapter header from the fields above
  (numeral, kicker, title, lede, architecture glyph). Call it first.
- `ctx.el(tag, attrs, ...children)` → DOM builder (`attrs.class`, `attrs.html`,
  event handlers via `onclick` etc.).
- `ctx.canvas(parent, {aspect, height, maxHeight})` → DPR-aware canvas.
  Returns `{canvas, g, w, h, dpr, onResize(fn)}` (`g` = 2D context in CSS px).
- `ctx.loop(fn)` → `fn(t, dt)` each animation frame **only while the chapter is
  on screen** (t in seconds since mount, dt seconds). Respect
  `AM.reducedMotion`: when true, loops are still called but at ~2 fps; design
  so a single frame reads well.
- `ctx.step({label, title, html | body})` → a step card in the house layout:
  "`n · label`" micro label, an `h3` title (always give one) and the body. `n`
  counts up per chapter (pass `n` to set it). Use it for every step.
- `ctx.subhead(eyebrow, title, lead?)` → the opening of a section after the
  scrollytelling: gold mono eyebrow + Figtree `h3` (+ optional lead paragraph).
  Use it for every second-level section (free play, side topics).
- `ctx.steps(stepEls, onStep)` → scrollytelling: calls `onStep(i)` when the
  i-th step element reaches mid-screen.
  When the split is stacked (≤900px, stage sticky on top) a step activates only
  once its top edge has cleared the stage.
- `ctx.onVisible(fn)` / `ctx.onHidden(fn)`. The section also carries the class
  `is-onscreen` while visible (CSS animations can key off it).
- `ctx.canvas(..., {maxHeight: 'stage'})` keeps the whole sticky stage within
  `--stage-max` (half the screen, at most 560px) when the split is stacked. Use
  it for stage canvases; tall stages leave no room to read the steps.

Mount order: the hero mounts at once, then one chapter per task in page order
(deep links and in-page link clicks build what they need first). Never assume
another chapter is already mounted. `AM.whenMounted` resolves when all are.

Layouts provided by `base.css` (compose them, don't reinvent):
- `.ch-body` — content wrapper (max ~1180px).
- `.ch-split` — two columns: `.ch-stage` (sticky visual, left/top) +
  `.ch-prose` (scrolling text steps, right). Stacks on phones (stage on top,
  sticky at reduced height).
- `.ch-wide` — full-width stage block.
- `.prose` — readable text column (~65ch). `.step` — a scrollytelling step card.
- `.panel` — a framed instrument panel (use sparingly: one per figure).
- `.controls` — row of controls under a stage. `.caption` — figure caption.
- `.math` — inline formula styling (mono). `.kbd`, `.term` (defined term).
- `.callout` — aside box (the "Key idea"), kept to the prose measure.
  `.grid-2`, `.grid-3` — responsive grids.
- `.subhead` (see `ctx.subhead`). Headings: `h3` is Figtree `--fs-h3`; the one
  `h4` role is Figtree 17px semibold (`.prose h4`, `.step h4`, `.panel h4`).
  Do not restyle them per chapter.
- Figures: `AM.ui.figure({title, badge, caption, framed})`. Frame (`framed:
  true`) every sticky stage and every free-play figure: the title row sits
  outside, the body gets the `--ink-2` frame. Inline diagrams inside prose stay
  unframed. Do not wrap a framed figure in `.panel`.
- Example pickers ("try this sentence"): `AM.ui.chips(options, {onPick,
  selected})` → `.chip` pills. Use them instead of chapter-made chips.

## 4. Rules for visuals

1. **Complete at rest.** Draw a meaningful first frame synchronously in
   `mount` (or immediately on first resize). Never hide text waiting for scroll.
2. **Pause off-screen.** Only animate via `ctx.loop`. No raw
   `requestAnimationFrame` loops, no `setInterval` animation.
3. **Phones.** Must work at 380px wide. No horizontal page scroll. Canvases
   resize with their container. Touch works (use pointer events).
4. **Performance.** Target 60fps on a laptop. Keep per-frame work small:
   cache static layers in offscreen canvases, cap particle counts (~600 max).
5. **Interactive.** Every chapter has at least one thing to poke: a slider,
   clickable tokens, a drag, a toggle. Controls get visible focus and `id`s.
6. **Accurate.** The math must be right. When you show a formula, the
   visual must actually compute it. Softmax rows sum to 1. Causal masks
   are lower-triangular (query i sees keys ≤ i). √d_k scaling is applied.
7. **Accessible.** Canvas figures get `role="img"` + `aria-label` describing
   what is shown. Text contrast ≥ 4.5:1 on `--ink`.
8. **Legible canvas text.** `D.text`, `D.measure` and `D.token` never draw
   below `AM.minText` (8px, 8.5px on touch screens). Fit labels with shorter
   strings, not smaller type; pass `{minSize: 0}` only for decorative text.
   Text drawn with `AM.font` directly, or under a scaling transform, must follow
   the same floor.
9. **No external fetches.** No images from the web, no fetch/XHR. Fonts come
   from Google Fonts (already linked, loaded without blocking the first paint).
   Everything else is inline. Canvases redraw when the fonts arrive; code that
   caches text measurements should re-measure when `AM.fontsReady` settles.
10. **No new globals** except your chapter registration. Wrap your file in an
   IIFE: `(() => { ... })();`.

## 5. Testing your chapter

```
node attention-loom/tools/shoot.mjs --chapter <id> [--width 1280] [--height 900] \
     [--wait 1200] [--out /path/to/dir] [--click "#selector"] [--scroll-steps]
```
Writes PNG screenshots (desktop + phone by default) and prints any console
errors / page errors (exit code 1 if there were errors). Look at the PNGs.
Iterate until the chapter looks deliberate and beautiful at both widths and
there are zero console errors.

## 6. The real tiny model (`AM.model`)

A real decoder-only transformer, trained offline in this repo, runs in the
browser. Chapters may use it for "LIVE MODEL" panels.

```js
const m = AM.model.get('tinyworld');   // word-level toy-English model
// also: 'reverse', 'sort' (algorithmic char-level models used by the Lab)
m.config   // {n_layer, n_head, d_model, d_ff, n_ctx, vocab_size}
m.vocab    // array of token strings
m.encode('the king lost his crown')   // → {ids:number[], tokens:string[], unknown:string[]}
m.decode(ids)                          // → string[]
const r = m.run(ids, {capture: true})  // full forward pass with introspection:
r.tokens                // string[]
r.logits[t]             // Float32Array(vocab) for position t
r.probs[t]              // softmax(logits[t])
r.attn[layer][head][q]  // Float32Array(T): attention of query q over keys (causal; sums to 1)
r.resid[l][t]           // Float32Array(d_model): residual stream entering layer l (l = n_layer → final)
r.residMid[l][t]        // after layer l's attention, before its MLP
r.q[l][h][t], r.k[l][h][t], r.v[l][h][t]   // per-head vectors (Float32Array(d_head))
r.mlp[l][t]             // Float32Array(d_ff): post-GELU MLP activations
r.lens[l][t]            // Float32Array(vocab): logit-lens probs from resid[l][t]
m.topk(probsRow, k)     // → [{id, token, p}]
m.sample(probsRow, {temperature, topK, topP, rng})  // → id
m.generate(ids, {maxNew, temperature, topK, topP, rng, stopAt:'.'}) // → ids
AM.model.ready          // Promise resolving when all weights are decoded
```

Details of what the trained model actually learned (which heads do what,
curated example sentences that show crisp patterns, accuracies) live in
`docs/model-notes.md`. Use those examples; do not claim behaviours the notes
don't support.
