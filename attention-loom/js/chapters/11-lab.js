/* PLACEHOLDER — to be replaced by the chapter author. */
(() => {
  AM.chapter({
    id: 'lab',
    num: 11,
    kicker: 'Live training lab',
    title: 'Watch a Mind <em>Form</em>',
    lede: 'Train a real transformer from scratch in your browser and watch its attention crystallise out of noise.',
    where: 'lab',
    mount(root, ctx) {
      ctx.header();
      root.appendChild(ctx.el('div', { class: 'ch-body' }, ctx.el('p', { class: 'caption' }, 'This chapter is being woven.')));
    },
  });
})();
