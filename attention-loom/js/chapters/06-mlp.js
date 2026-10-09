/* PLACEHOLDER — to be replaced by the chapter author. */
(() => {
  AM.chapter({
    id: 'mlp',
    num: 6,
    kicker: 'Feed-forward network',
    title: 'The Memory <em>Vaults</em>',
    lede: 'Between rounds of attention, each token passes alone through a wide network that stores much of what the model knows.',
    where: 'mlp',
    mount(root, ctx) {
      ctx.header();
      root.appendChild(ctx.el('div', { class: 'ch-body' }, ctx.el('p', { class: 'caption' }, 'This chapter is being woven.')));
    },
  });
})();
