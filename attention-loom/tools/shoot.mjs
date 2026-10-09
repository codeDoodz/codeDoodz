#!/usr/bin/env node
/* Screenshot + console-error checker for The Attention Loom.

   node tools/shoot.mjs --chapter attention            desktop + phone shots of one chapter
        [--width 1280 --height 900]                    desktop viewport (phone is 390x844)
        [--only desktop|phone]                         skip one of the two
        [--wait 1200]                                  ms to let animations run before each shot
        [--scroll-steps]                               one shot per .step (scrollytelling states)
        [--positions 0,0.5,1]                          shots at fractions through the section
        [--click "#sel"] (repeatable)                  click elements (in order) before shooting
        [--eval "js"] (repeatable)                     run JS in the page before shooting
        [--perf]                                       measure animation fps while chapter is visible
        [--full]                                       also a full-page screenshot (desktop)
        [--file dist/attention-loom.html]              test another HTML file
        [--out DIR]                                    where PNGs go (default: $TMPDIR/loom-shots)

   Prints console errors, page errors, failed loads, horizontal overflow.
   Exit code 1 if any errors occurred. */
import { createRequire } from 'node:module';
import path from 'node:path';
import fs from 'node:fs';
import os from 'node:os';
import { fileURLToPath, pathToFileURL } from 'node:url';

const require = createRequire(import.meta.url);
let playwright;
try { playwright = require('playwright'); } catch { playwright = require('/opt/node22/lib/node_modules/playwright'); }

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '..');

const args = process.argv.slice(2);
const opt = { click: [], eval: [] };
for (let i = 0; i < args.length; i++) {
  const a = args[i];
  if (!a.startsWith('--')) continue;
  const k = a.slice(2);
  const nxt = args[i + 1];
  const takesValue = nxt !== undefined && !nxt.startsWith('--');
  if (k === 'click' || k === 'eval') { opt[k].push(nxt); i++; }
  else if (takesValue) { opt[k] = nxt; i++; }
  else opt[k] = true;
}

const chapter = opt.chapter;
const W = parseInt(opt.width || '1280', 10), H = parseInt(opt.height || '900', 10);
const wait = parseInt(opt.wait || '1200', 10);
const outDir = path.resolve(opt.out || path.join(process.env.TMPDIR || os.tmpdir(), 'loom-shots'));
fs.mkdirSync(outDir, { recursive: true });
const file = path.resolve(root, opt.file || 'index.html');
const url = pathToFileURL(file).href;

const viewports = [];
if (opt.only !== 'phone') viewports.push({ name: 'desktop', width: W, height: H });
if (opt.only !== 'desktop') viewports.push({ name: 'phone', width: 390, height: 844, isMobile: true, hasTouch: true });

let errorCount = 0;
const shots = [];

const browser = await playwright.chromium.launch();
try {
  for (const vp of viewports) {
    const context = await browser.newContext({
      viewport: { width: vp.width, height: vp.height },
      deviceScaleFactor: 1,
      isMobile: !!vp.isMobile,
      hasTouch: !!vp.hasTouch,
    });
    const page = await context.newPage();
    const errors = [];
    page.on('console', (m) => { if (m.type() === 'error') errors.push('console.error: ' + m.text()); });
    page.on('pageerror', (e) => errors.push('pageerror: ' + (e.stack || e.message)));
    page.on('requestfailed', (r) => {
      const u = r.url();
      if (u.startsWith('https://fonts.')) return; // fonts may be unreachable offline; fallbacks apply
      errors.push('requestfailed: ' + u + ' ' + (r.failure()?.errorText || ''));
    });

    await page.goto(url, { waitUntil: 'load' });
    await page.evaluate(() => (document.fonts ? Promise.race([document.fonts.ready, new Promise((r) => setTimeout(r, 2500))]) : null));
    await page.waitForTimeout(300);

    const tag = (s) => `${chapter || 'page'}-${vp.name}${s ? '-' + s : ''}.png`;

    if (chapter) {
      const exists = await page.evaluate((id) => !!document.querySelector(`[data-chapter="${id}"]`), chapter);
      if (!exists) { errors.push(`no section [data-chapter="${chapter}"]`); }
      else {
        const sectionTop = async () => page.evaluate((id) => {
          const s = document.querySelector(`[data-chapter="${id}"]`);
          const r = s.getBoundingClientRect();
          return { top: r.top + window.scrollY, height: r.height };
        }, chapter);

        // Initial: scroll section top into view
        let { top, height } = await sectionTop();
        await page.evaluate((y) => window.scrollTo({ top: y, behavior: 'instant' }), top);
        await page.waitForTimeout(wait);

        for (const sel of opt.click) {
          try { await page.click(sel, { timeout: 3000 }); await page.waitForTimeout(250); }
          catch (e) { errors.push(`click failed: ${sel}: ${e.message.split('\n')[0]}`); }
        }
        for (const js of opt.eval) {
          try { await page.evaluate(js); await page.waitForTimeout(250); }
          catch (e) { errors.push(`eval failed: ${e.message.split('\n')[0]}`); }
        }

        if (opt.perf) {
          const fps = await page.evaluate(() => new Promise((res) => {
            let n = 0; const t0 = performance.now(); let worst = 0; let prev = t0;
            const f = (t) => { n++; worst = Math.max(worst, t - prev); prev = t; if (t - t0 < 2000) requestAnimationFrame(f); else res({ fps: n / ((t - t0) / 1000), worstFrameMs: worst }); };
            requestAnimationFrame(f);
          }));
          console.log(`[${vp.name}] perf: ${fps.fps.toFixed(1)} fps, worst frame ${fps.worstFrameMs.toFixed(1)} ms`);
        }

        const p0 = path.join(outDir, tag('top'));
        await page.screenshot({ path: p0 });
        shots.push(p0);

        ({ top, height } = await sectionTop());
        if (opt['scroll-steps']) {
          const n = await page.evaluate((id) => document.querySelectorAll(`[data-chapter="${id}"] .step`).length, chapter);
          for (let i = 0; i < n; i++) {
            await page.evaluate(([id, i]) => {
              const el = document.querySelectorAll(`[data-chapter="${id}"] .step`)[i];
              const r = el.getBoundingClientRect();
              window.scrollTo({ top: window.scrollY + r.top + r.height / 2 - window.innerHeight / 2, behavior: 'instant' });
            }, [chapter, i]);
            await page.waitForTimeout(wait);
            const p = path.join(outDir, tag('step' + i));
            await page.screenshot({ path: p });
            shots.push(p);
          }
        }
        const positions = opt.positions ? String(opt.positions).split(',').map(Number)
          : (height > vp.height * 1.3 && !opt['scroll-steps'] ? [0.5, 1] : []);
        for (const f of positions) {
          const y = top + Math.max(0, height - vp.height) * f;
          await page.evaluate((y) => window.scrollTo({ top: y, behavior: 'instant' }), y);
          await page.waitForTimeout(wait);
          const p = path.join(outDir, tag('at' + Math.round(f * 100)));
          await page.screenshot({ path: p });
          shots.push(p);
        }
      }
    } else {
      await page.waitForTimeout(wait);
      const p = path.join(outDir, tag('top'));
      await page.screenshot({ path: p });
      shots.push(p);
    }

    if (opt.full && vp.name === 'desktop') {
      const p = path.join(outDir, tag('full'));
      await page.screenshot({ path: p, fullPage: true });
      shots.push(p);
    }

    const overflow = await page.evaluate(() => {
      const de = document.documentElement;
      if (de.scrollWidth <= window.innerWidth + 1) return null;
      const wide = [];
      document.querySelectorAll('body *').forEach((el) => {
        const r = el.getBoundingClientRect();
        if (r.right > window.innerWidth + 1 && r.width > 0) {
          let p = el.parentElement, clipped = false;
          while (p) { const cs = getComputedStyle(p); if (/(auto|scroll|hidden|clip)/.test(cs.overflowX)) { clipped = true; break; } p = p.parentElement; }
          if (!clipped) wide.push(`${el.tagName.toLowerCase()}${el.id ? '#' + el.id : ''}${el.className && typeof el.className === 'string' ? '.' + el.className.split(' ').join('.') : ''} right=${Math.round(r.right)}`);
        }
      });
      return { scrollWidth: de.scrollWidth, innerWidth: window.innerWidth, offenders: wide.slice(0, 8) };
    });
    if (overflow) errors.push(`HORIZONTAL OVERFLOW at ${vp.name}: scrollWidth ${overflow.scrollWidth} > ${overflow.innerWidth}; offenders: ${overflow.offenders.join(' | ')}`);

    if (errors.length) {
      errorCount += errors.length;
      console.log(`[${vp.name}] ${errors.length} problem(s):`);
      for (const e of errors) console.log('  - ' + e);
    } else console.log(`[${vp.name}] no console errors`);
    await context.close();
  }
} finally {
  await browser.close();
}
console.log('screenshots:\n' + shots.map((s) => '  ' + s).join('\n'));
process.exit(errorCount ? 1 : 0);
