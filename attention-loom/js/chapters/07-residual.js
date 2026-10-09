/* PLACEHOLDER — to be replaced by the chapter author. */
(() => {
  AM.chapter({
    id: 'residual',
    num: 7,
    kicker: 'Residual stream &amp; LayerNorm',
    title: 'The River <em>Thread</em>',
    lede: 'Each block reads from a shared stream of numbers and adds its contribution back in, so nothing is overwritten.',
    where: 'resid',
    mount(root, ctx) {
      ctx.header();
      root.appendChild(ctx.el('div', { class: 'ch-body' }, ctx.el('p', { class: 'caption' }, 'This chapter is being woven.')));
    },
  });
})();
