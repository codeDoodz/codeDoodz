/* PLACEHOLDER — to be replaced by the chapter author. */
(() => {
  AM.chapter({
    id: 'epilogue',
    num: 12,
    kicker: 'Scale &amp; recap',
    title: 'From Toy to <em>Titan</em>',
    lede: 'The same loom, scaled up a million times, is what writes your chatbot\'s replies.',
    where: 'all',
    mount(root, ctx) {
      ctx.header();
      root.appendChild(ctx.el('div', { class: 'ch-body' }, ctx.el('p', { class: 'caption' }, 'This chapter is being woven.')));
    },
  });
})();
