/* PLACEHOLDER — to be replaced by the chapter author. */
(() => {
  AM.chapter({
    id: 'predict',
    num: 9,
    kicker: 'Unembedding &amp; sampling',
    title: 'Rolling the <em>Dice</em>',
    lede: 'The final vector is turned into a score for every word in the vocabulary, and the next word is drawn from those odds.',
    where: 'unembed',
    mount(root, ctx) {
      ctx.header();
      root.appendChild(ctx.el('div', { class: 'ch-body' }, ctx.el('p', { class: 'caption' }, 'This chapter is being woven.')));
    },
  });
})();
