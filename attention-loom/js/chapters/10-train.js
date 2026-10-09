/* PLACEHOLDER — to be replaced by the chapter author. */
(() => {
  AM.chapter({
    id: 'train',
    num: 10,
    kicker: 'Training',
    title: 'Learning by <em>Falling</em>',
    lede: 'A model starts as random numbers. Billions of tiny nudges downhill on the error turn it into something that predicts.',
    where: 'train',
    mount(root, ctx) {
      ctx.header();
      root.appendChild(ctx.el('div', { class: 'ch-body' }, ctx.el('p', { class: 'caption' }, 'This chapter is being woven.')));
    },
  });
})();
