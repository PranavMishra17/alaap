// Screenshot one page with headless Chromium and report console/page errors.
//
//   node scripts/shot.mjs <url> <out.png> [--width=1440] [--height=900] [--full] [--dark] [--reduced-motion] [--wait=800] [--hmr]
//
// By default the dev server's hot-reload client and dev toolbar are blocked, so another
// variant being saved cannot reload the page mid-capture and the toolbar pill stays out of
// the image. Pass --hmr to allow them.
//
// Scrolls through the whole page first so scroll-triggered reveals fire, then returns to the
// top before capturing. Warns if the page is wider than the viewport (horizontal overflow).
process.env.PLAYWRIGHT_BROWSERS_PATH ||= 'E:/ml-cache/ms-playwright';
const { chromium } = await import('playwright');

const args = process.argv.slice(2);
const pos = args.filter((a) => !a.startsWith('--'));
const opt = Object.fromEntries(
  args.filter((a) => a.startsWith('--')).map((a) => { const [k, v] = a.slice(2).split('='); return [k, v ?? true]; }),
);
const [url, out] = pos;
if (!url || !out) {
  console.error('usage: node scripts/shot.mjs <url> <out.png> [--width=1440] [--height=900] [--full] [--dark] [--reduced-motion] [--wait=800]');
  process.exit(2);
}
const width = Number(opt.width ?? 1440);
const height = Number(opt.height ?? 900);

const browser = await chromium.launch();
const ctx = await browser.newContext({
  viewport: { width, height },
  deviceScaleFactor: 1,
  colorScheme: opt.dark ? 'dark' : 'light',
  reducedMotion: opt['reduced-motion'] ? 'reduce' : 'no-preference',
});
const page = await ctx.newPage();
if (!opt.hmr) {
  await page.route((u) => /\/@vite\/client|\/@id\/astro:dev-toolbar|\/@id\/astro:toolbar|__vite_ping|\/@vite\/env/.test(u.href), (r) => r.abort());
}
const problems = [];
page.on('requestfailed', (r) => { if (/@vite|astro:dev-toolbar|astro:toolbar|__vite_ping/.test(r.url())) return; });
page.on('console', (m) => {
  if (m.type() !== 'error' && m.type() !== 'warning') return;
  if (!opt.hmr && /net::ERR_FAILED/.test(m.text())) return; // the blocked hot-reload client
  problems.push(`[console.${m.type()}] ${m.text()}`);
});
page.on('pageerror', (e) => problems.push(`[pageerror] ${e.message}`));
page.on('requestfailed', (r) => { if (!/@vite|astro:dev-toolbar|astro:toolbar|__vite_ping/.test(r.url())) problems.push(`[requestfailed] ${r.url()} ${r.failure()?.errorText ?? ''}`); });

const resp = await page.goto(url, { waitUntil: 'networkidle', timeout: 60000 });
if (!resp || resp.status() >= 400) console.error(`HTTP ${resp?.status()} for ${url}`);
await page.waitForTimeout(Number(opt.wait ?? 800));

const docHeight = await page.evaluate(() => document.documentElement.scrollHeight);
for (let y = 0; y < docHeight; y += Math.floor(height * 0.8)) {
  await page.evaluate((yy) => window.scrollTo(0, yy), y);
  await page.waitForTimeout(120);
}
await page.evaluate(() => window.scrollTo(0, 0));
await page.waitForTimeout(300);
await page.screenshot({ path: out, fullPage: !!opt.full });

const overflow = await page.evaluate(() => document.documentElement.scrollWidth > document.documentElement.clientWidth + 1);
console.log(`saved ${out}  viewport ${width}x${height}${opt.full ? `  full height ${docHeight}` : ''}${overflow ? '\nWARNING: horizontal overflow, page is wider than the viewport' : ''}`);
if (problems.length) { console.log('--- console / page problems:'); for (const p of problems) console.log(p); }
await browser.close();
