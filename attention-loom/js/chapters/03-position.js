/* PLACEHOLDER — to be replaced by the chapter author. */
(() => {
  AM.chapter({
    id: 'position',
    num: 3,
    kicker: 'Positional encoding',
    title: 'The Clockwork of <em>Order</em>',
    lede: 'Attention has no sense of order on its own, so every position gets a unique signature added to its vector.',
    where: 'pos',
    mount(root, ctx) {
      ctx.header();
      root.appendChild(ctx.el('div', { class: 'ch-body' }, ctx.el('p', { class: 'caption' }, 'This chapter is being woven.')));
    },
  });
})();
