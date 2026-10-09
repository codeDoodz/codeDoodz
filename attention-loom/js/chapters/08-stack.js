/* PLACEHOLDER — to be replaced by the chapter author. */
(() => {
  AM.chapter({
    id: 'stack',
    num: 8,
    kicker: 'Depth &amp; the logit lens',
    title: 'The Tower of <em>Layers</em>',
    lede: 'Stack the block again and again. Peek between layers and watch a guess sharpen into an answer.',
    where: 'stack',
    mount(root, ctx) {
      ctx.header();
      root.appendChild(ctx.el('div', { class: 'ch-body' }, ctx.el('p', { class: 'caption' }, 'This chapter is being woven.')));
    },
  });
})();
