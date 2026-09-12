// Audit every variant's rendered text against CONTENT.md rules.
//
//   node scripts/check-content.mjs [slug ...]
//
// For each variant route it fetches the rendered HTML from the dev server, strips scripts,
// styles and tags, and reports: numbers not on the §9 allowlist, banned words from §10, and
// mandatory items that are missing. Flags are for human review, not automatic failure: a
// lens may legitimately mention "sixteen beats" or "78 rpm" as its own world, never as a
// project claim.
import { readdirSync } from 'node:fs';

const BASE = process.env.SITE_BASE ?? 'http://127.0.0.1:4321';

// §9 allowlist, as the digit tokens that may appear in page text.
const ALLOWED_NUMBERS = new Set([
  '4', '22', '6', '3', '5', '0', '20', '8', '10', '50', '0.67', '1.43', '0.8', '1.25', '1.4',
  '65.6', '87', '53.8', '141', '138', '2,500', '2500', '88.5', '37', '14', '23', '41', '38', '48',
  '432', '26', '114', '2026', '09', '02', '05', '2026-09-02', '2026-09-05',
  '2.2', '2.9', '1.9', '4.3', '3.9', '3.5', '4.6', '2.8', // clip lengths
  '1', '2', // "one", "two" as digits are tolerable in labels
]);

const BANNED = [
  'embedding', 'vector', 'cosine', 'latent', 'pca', 'vendi', 'ecapa', 'drift', 'corpus', 'tokenizer',
  'token', 'inference', 'codec', 'asr', 'mos', 'gmm', 'revolutionary', 'cutting-edge', 'ai-powered',
  'state-of-the-art', 'seamless', 'unleash', 'supercharge', 'next-generation', 'leverage',
  'game-changing', 'effortless', 'magic', 'try it now', 'sign up', 'get started',
];

// Mandatory items, as phrases at least one of which must be present per item.
const MANDATORY = {
  'repo link': ['github.com/PranavMishra17/alaap'],
  'name in Devanagari': ['आलाप'],
  'the example description': ['very deep voice, very rough and gravelly'],
  'meaning of alaap': ['opening improvisation'],
  'differentiator: voice you keep': ['keep', 'stored', 'persistent', 'same character', 'same person'],
  'differentiator: who vs what': ['exactly zero', 'identical'],
  'differentiator: no recording': ['no voice cloning', 'no recording', 'never accepts a recording', 'does not need a recording', 'no user uploads', 'no upload'],
  'differentiator: knows its limits': ['roughly twenty', 'wrong on the first', 'caught by a'],
  'limit: not a product': ['not a product', 'no signup', 'no hosted service', 'no api'],
  'limit: emotion': ['apologetically', 'emotion'],
  'limit: catalogue small': ['tens of distinct voices', 'not hundreds', '22', '41'],
  'limit: licensing': ['licens', 'research-only', 'research only'],
  'limit: little listened': ['listened', 'one listener'],
  'limit: no commercial comparison': ['commercial'],
  'limit: clips are English': ['english'],
  'caveat: watermark': ['watermark'],
  'caveat: emotion setting on clips': ['experimental emotion setting', 'experimental', 'not been validated'],
  'clip text: mentor line 0': ['mountains remember every footstep'],
};

function stripHtml(html) {
  return html
    .replace(/<script[\s\S]*?<\/script>/gi, ' ')
    .replace(/<style[\s\S]*?<\/style>/gi, ' ')
    .replace(/<!--[\s\S]*?-->/g, ' ')
    .replace(/<[^>]+>/g, ' ')
    .replace(/&#123;/g, '{').replace(/&#125;/g, '}')
    .replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"').replace(/&#39;/g, "'")
    .replace(/\s+/g, ' ');
}

function listVariants() {
  return readdirSync(new URL('../src/pages/variants/', import.meta.url))
    .filter((f) => f.endsWith('.astro') && !f.startsWith('_'))
    .map((f) => f.replace(/\.astro$/, ''))
    .sort();
}

const slugs = process.argv.slice(2).length ? process.argv.slice(2) : listVariants();
let anyFlag = false;

for (const slug of slugs) {
  const url = `${BASE}/variants/${slug}`;
  let html;
  try {
    const res = await fetch(url);
    if (!res.ok) { console.log(`\n## ${slug}\n  HTTP ${res.status} — page did not render`); anyFlag = true; continue; }
    html = await res.text();
  } catch (e) {
    console.log(`\n## ${slug}\n  fetch failed: ${e.message}`); anyFlag = true; continue;
  }
  const text = stripHtml(html).replace(/CC[- ]BY[- ]4\.0|Apache[- ]2\.0|Qwen3[- ]TTS/g, ' ');
  const lower = text.toLowerCase();

  const numbers = [...text.matchAll(/(?<![\w.])\d(?:[\d,]*\d)?(?:\.\d+)?(?![\w.])/g)].map((m) => m[0]);
  const badNumbers = [...new Set(numbers.filter((n) => !ALLOWED_NUMBERS.has(n)))];
  const contexts = badNumbers.map((n) => {
    const i = text.indexOf(n);
    return `${n}  …${text.slice(Math.max(0, i - 40), i + n.length + 40).trim()}…`;
  });

  const banned = BANNED.filter((w) => new RegExp(`(?<![a-z])${w.replace(/[-\s]/g, '[-\\s]')}(?![a-z])`, 'i').test(lower));
  const htmlLower = html.toLowerCase();
  const missing = Object.entries(MANDATORY)
    .filter(([k, alts]) => !alts.some((a) => (k === 'repo link' ? htmlLower : lower).includes(a.toLowerCase())))
    .map(([k]) => k);
  const words = text.split(' ').filter(Boolean).length;
  const externals = [...html.matchAll(/(?:src|href)=["'](https?:\/\/[^"']+)/g)].map((m) => new URL(m[1]).host).filter((h) => !/github\.com|fonts\.googleapis\.com|fonts\.gstatic\.com/.test(h));

  console.log(`\n## ${slug}  (${words} words)`);
  if (contexts.length) { anyFlag = true; console.log('  numbers off the allowlist (review each):'); for (const c of contexts) console.log(`    ${c}`); }
  if (banned.length) { anyFlag = true; console.log(`  banned words: ${banned.join(', ')}`); }
  if (missing.length) { anyFlag = true; console.log(`  mandatory items not found: ${missing.join('; ')}`); }
  if (externals.length) { anyFlag = true; console.log(`  external hosts beyond fonts/github: ${[...new Set(externals)].join(', ')}`); }
  if (!contexts.length && !banned.length && !missing.length && !externals.length) console.log('  clean');
}
process.exit(anyFlag ? 1 : 0);
