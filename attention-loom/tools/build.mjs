#!/usr/bin/env node
/* Bundle The Attention Loom into a single self-contained HTML file.

   node tools/build.mjs                → dist/attention-loom.html          (full standalone document)
   node tools/build.mjs --fragment     → dist/attention-loom.fragment.html (no doctype/html/head/body:
                                          for hosts that wrap the page in their own skeleton)

   Inlines every local <link rel="stylesheet"> and <script src>, keeps remote
   stylesheet links (Google Fonts). */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '..');
const fragment = process.argv.includes('--fragment');

let html = fs.readFileSync(path.join(root, 'index.html'), 'utf8');

const isLocal = (u) => !/^(https?:)?\/\//.test(u);

html = html.replace(/<link\s+rel="stylesheet"\s+href="([^"]+)"\s*\/?>/g, (m, href) => {
  if (!isLocal(href)) return m;
  const css = fs.readFileSync(path.join(root, href), 'utf8');
  return `<style>/* ${href} */\n${css}\n</style>`;
});

html = html.replace(/<script\s+src="([^"]+)"\s*><\/script>/g, (m, src) => {
  if (!isLocal(src)) return m;
  const p = path.join(root, src);
  if (!fs.existsSync(p)) { console.warn('missing script, skipped:', src); return ''; }
  const js = fs.readFileSync(p, 'utf8').replace(/<\/script/gi, '<\\/script');
  return `<script>/* ${src} */\n${js}\n</script>`;
});

// Drop HTML comments that only served as markers.
html = html.replace(/<!--\s*(model:begin|model:end|core|chapters)\s*-->\n?/g, '');

if (fragment) {
  const title = (html.match(/<title>[\s\S]*?<\/title>/) || [''])[0];
  const head = (html.match(/<head>([\s\S]*?)<\/head>/) || ['', ''])[1];
  const links = head.match(/<link[^>]+>/g) || [];
  const styles = head.match(/<style>[\s\S]*?<\/style>/g) || [];
  const body = (html.match(/<body[^>]*>([\s\S]*)<\/body>/) || ['', html])[1];
  html = [title, ...links.filter((l) => !/preconnect/.test(l)), ...styles, body.trim()].join('\n');
}

const outDir = path.join(root, 'dist');
fs.mkdirSync(outDir, { recursive: true });
const out = path.join(outDir, fragment ? 'attention-loom.fragment.html' : 'attention-loom.html');
fs.writeFileSync(out, html);
console.log(`wrote ${path.relative(process.cwd(), out)} (${(Buffer.byteLength(html) / 1024).toFixed(0)} KB)`);
