/* PLACEHOLDER — to be replaced by the chapter author. */
(() => {
  AM.chapter({
    id: 'attention',
    num: 4,
    kicker: 'Self-attention',
    title: 'Threads of <em>Attention</em>',
    lede: 'Every token asks a question, every token offers an answer, and information flows along the threads that match.',
    where: 'attn',
    mount(root, ctx) {
      ctx.header();
      root.appendChild(ctx.el('div', { class: 'ch-body' }, ctx.el('p', { class: 'caption' }, 'This chapter is being woven.')));
    },
  });
})();
