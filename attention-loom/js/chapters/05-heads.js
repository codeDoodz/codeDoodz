/* PLACEHOLDER — to be replaced by the chapter author. */
(() => {
  AM.chapter({
    id: 'heads',
    num: 5,
    kicker: 'Multi-head attention',
    title: 'The <em>Prism</em>',
    lede: 'Several attention heads run side by side, each free to look for a different kind of relationship.',
    where: 'heads',
    mount(root, ctx) {
      ctx.header();
      root.appendChild(ctx.el('div', { class: 'ch-body' }, ctx.el('p', { class: 'caption' }, 'This chapter is being woven.')));
    },
  });
})();
