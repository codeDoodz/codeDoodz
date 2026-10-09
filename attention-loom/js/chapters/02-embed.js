/* PLACEHOLDER — to be replaced by the chapter author. */
(() => {
  AM.chapter({
    id: 'embed',
    num: 2,
    kicker: 'Embeddings',
    title: 'Constellations of <em>Meaning</em>',
    lede: 'Each token ID becomes a long list of numbers, a point in a space where similar meanings sit close together.',
    where: 'embed',
    mount(root, ctx) {
      ctx.header();
      root.appendChild(ctx.el('div', { class: 'ch-body' }, ctx.el('p', { class: 'caption' }, 'This chapter is being woven.')));
    },
  });
})();
