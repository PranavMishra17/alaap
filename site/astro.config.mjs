import { defineConfig } from 'astro/config';

// GitHub Pages under a repo path needs `base: '/alaap'`; leave default until the
// winner is chosen so dev URLs stay short. Audio paths already go through BASE_URL.
export default defineConfig({
  site: 'https://pranavmishra17.github.io',
  base: process.env.ASTRO_BASE ?? '/alaap',
  trailingSlash: 'ignore',
});
