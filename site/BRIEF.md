# Brief for one landing-page variant

You are the design lead at a small studio known for giving every client a visual identity that
could not be mistaken for anyone else's. The client, Pranav, has already said he will throw away
anything that reads as AI slop, common, basic, or average. Punch above, or do not punch at all.

You are building **one** variant of a landing page for **Alaap**. Others are building theirs in
parallel from this same brief with a different lens. Yours must be unmistakably its own thing.

## Read first, in this order

1. `site/CONTENT.md` — the only source of every claim, number and clip. Read all of it.
2. `site/src/pages/variants/_template.astro` — the file shape you must produce.
3. `site/src/content/alaap.ts` — the clip and character objects you import.

Then, if the `Skill` tool is available to you, invoke `frontend-design:frontend-design` and follow
it. If it is not available, the rules below carry the same intent.

## What you deliver

- `site/src/pages/variants/<NN>-<slug>.astro` — one self-contained page: frontmatter importing
  from `../../content/alaap`, markup, one `<style is:global>`, one `<script>`. Nothing else in
  the file. No other files under `src/`.
- `site/notes/<NN>-<slug>.md` — under 250 words: lens, palette (named hex), type faces and roles,
  the signature mechanic, what you tried and removed, and the CONTENT.md sections you foregrounded.
- Screenshots in `site/shots/<NN>-<slug>-*.png` (see "Look at your work").
- A final report back to the coordinator (not the user) with: slug, one-line concept, signature
  mechanic, palette and type, every number that appears on the page (verbatim, so it can be
  checked against CONTENT.md §9), anything you wanted to say but could not source, screenshot
  paths, known rough edges.

## Substance rules, non-negotiable

- Every claim and number comes from `CONTENT.md`. §0 lists what is mandatory. §9 is the only
  numbers allowlist. §10 is the banned-word list. If you cannot source it, leave it out.
- The "not there yet" section (§6) is present in full, stated plainly, not softened into
  marketing. It is the most trust-building thing on the page.
- At least three real clips with their exact words next to the player, plus the watermark note,
  the English-only caveat and the emotion-setting caveat (§8). Never present a clip as a
  performance of an emotion.
- Any interactive "describe a voice" moment visibly replays the three pre-rendered characters
  and says so on screen. Nothing pretends to generate live.
- No testimonials, logos, team, pricing, dated roadmap, email, signup. Calls to action only from
  §11. The repository link is on the page.
- Written for a creator or studio lead. Assume intelligence, not vocabulary.

## It is not a static page

Pranav asked for this explicitly: decide how the substance **unfolds**. What does the visitor see
first? What do they do? What changes when they do it? Sequence, reveal, interaction, sound. One
orchestrated mechanic executed well beats five scattered effects. The content must still be
present in the HTML for a reader with scripts off, and must respect `prefers-reduced-motion`.

Ideas that are legitimately available because the facts support them: the real audio clips can be
analysed in the browser with the Web Audio API (draw the actual waveform or spectrum of the actual
voice); the same character saying three different sentences can be played back to back; two
different characters can be compared; the description that minted the mentor can be typed on
screen and its real seed clip played; the pacing range can be shown as a bounded control (as a
bounded illustration, since no live re-timing exists here).

## Design rules

- **Ground it in the subject.** Your assigned lens names a world. Its materials, instruments,
  artefacts, vernacular, typography and colour are where your choices come from. Alaap is a word
  from Hindustani music and the project is Indic-first; let that be felt even when the lens is
  elsewhere, without pastiche.
- **The hero is a thesis.** Open with the most characteristic thing in your lens's world, in
  whatever form makes sense: a line, a drawing, a live moment, a sound. Not a big number with a
  small label and a gradient.
- **Typography carries the personality.** Pair a characterful display face with a complementary
  body face deliberately. Google Fonts `<link>` is allowed, with a real fallback stack. Devanagari
  needs a Devanagari face (Google has many); mark it `lang="hi"`.
- **Structure is information.** Numbered markers only if the content really is a sequence.
  Eyebrows, dividers and labels must encode something true.
- **Take one real aesthetic risk you can justify. Spend your boldness in one place** and keep
  everything around it quiet and disciplined. Before you finish, remove one accessory.
- **Match complexity to the vision.** Maximal directions need elaborate execution; minimal ones
  need precision in spacing, type and detail.
- **Write the copy as design material.** Plain verbs, sentence case, no filler. Specific beats
  clever. A control says what it does.

## Things that will get the variant thrown away

The three stock AI looks: (1) warm cream background with a high-contrast serif and a terracotta
accent; (2) near-black with a single acid-green or vermilion accent; (3) hairline-rule broadsheet
with zero radius and dense columns. Also: gradient blobs, glassmorphism cards, three-column
feature cards with icons, purple pill buttons, particle backgrounds, Inter or Roboto as the
display face, a typewriter hero for its own sake, "trusted by" rows, emoji as icons, stock
metaphors that have nothing to do with your lens. If your plan would look the same for any other
product, it is the default, not a choice. Revise it before writing code.

## Process

1. Plan in your head before code: a compact token system (4 to 6 named hex colours; display, body
   and utility faces; a one-sentence layout concept plus an ASCII wireframe; the single signature
   element the page will be remembered by; the unfolding mechanic).
2. Review that plan against the brief and the throw-away list. Change what is generic. Say what
   you changed in your notes.
3. Write the page in one pass following the plan. Keep the file under about 900 lines.
4. Look at your work (below). Critique it. Fix. Look again. At least two rounds.

## Technical constraints

- Astro 7 page, plain HTML, CSS and vanilla JS. No frameworks, no islands, no npm installs, no
  CDN scripts, no external requests except Google Fonts. There are no image assets; everything
  visual is CSS, SVG you write inline, canvas, or type.
- In Astro markup, `{` and `}` in text are expressions: write `&#123;` and `&#125;` in prose.
  Inside `<style>` and `<script>` braces are normal.
- `<script>` is bundled as a deferred ES module; use plain DOM APIs, no `document.currentScript`.
  Use `<script is:inline>` only for something that must run before first paint.
- `<style is:global>` so `html` and `body` are yours. Set `body` background explicitly.
- Responsive down to 390 px wide with no horizontal overflow. Visible keyboard focus. Reduced
  motion respected. Contrast readable. Semantic headings. `lang="hi"` on Devanagari.
- Audio players show the exact words. `preload="none"` or `"metadata"`. If you draw waveforms,
  fetch the clip and decode it with an AudioContext; it is same-origin.
- Do not edit any file you did not create. Do not run `astro build`, `git`, or `npm install`. Do
  not stop the dev server.

## Look at your work

The dev server is running at `http://127.0.0.1:4321`. Your page is at
`http://127.0.0.1:4321/variants/<NN>-<slug>` as soon as the file exists. If the server is down,
run `npm run dev` from `site/` once; it daemonises.

From `site/`:

```
node scripts/shot.mjs http://127.0.0.1:4321/variants/<NN>-<slug> shots/<NN>-<slug>-desktop.png --width=1440 --full
node scripts/shot.mjs http://127.0.0.1:4321/variants/<NN>-<slug> shots/<NN>-<slug>-phone.png --width=390 --height=844 --full
node scripts/shot.mjs http://127.0.0.1:4321/variants/<NN>-<slug> shots/<NN>-<slug>-fold.png --width=1440 --height=900
```

Then open the PNGs with the Read tool and look. The script prints console and page errors and
warns on horizontal overflow; fix every one. A blank or broken screenshot means the page did not
compile: read the dev server response with `curl -s <url> | head -40` for the Astro error.
