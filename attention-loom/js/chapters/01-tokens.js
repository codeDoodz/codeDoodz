/* PLACEHOLDER — to be replaced by the chapter author. */
(() => {
  AM.chapter({
    id: 'tokens',
    num: 1,
    kicker: 'Tokenization',
    title: 'Shattering <em>Text</em>',
    lede: 'Before a model can read, text is broken into tokens: words and pieces of words, each with an ID number.',
    where: 'tokens',
    mount(root, ctx) {
      ctx.header();
      root.appendChild(ctx.el('div', { class: 'ch-body' }, ctx.el('p', { class: 'caption' }, 'This chapter is being woven.')));
    },
  });
})();
