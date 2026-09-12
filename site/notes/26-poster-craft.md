# 26 · poster · craft

**Lens.** Variant 12's hand-painted wall poster, re-cut by the craftsman: same enamel, lettering
and slips; the work is in the edges, spacing, type scale, and what a visitor feels but never sees.

**Palette.** The poster's five enamels, gulabi for the played waveform, a dark wall. Contrast:
kajal/haldi 9.9, neel/haldi 5.2, chuna/neel 8.4, chuna/sindoor 5.0, gloss `#D9D3C3`/kajal 12.3,
`#B9B2A3`/kajal 8.7, `#3D382E`/chuna 10.1, sindoor/chuna 5.0.

**Type.** Yatra One (titles, band names), Khand 500–700 (labels, controls, boxes), Eczar 400/500
(body; the Devanagari lines at 1.3rem/1.6). Fallbacks metric-matched with `size-adjust` plus
ascent/descent overrides measured with fontTools. CLS 0.015 → 0.011.

**Signature mechanic.** The programme strip: a stencilled kajal band that sticks, follows you, and
turns the sheet over (Landing · Technical map). One hand-drawn arrow glyph for every chain.

**Tried and removed.** Dry-brush speckle on band edges. The 17rem title. A "not yet checked"
tag on eleven Hindi rows (one closing slip says it once).

**Foregrounded.** §1, §8, §4, §5, §7 as one painted diagram plus the map, §6, §11.

**Round 2.**
- Hindi plays: four voices × three lines, waveforms from the clips, `indicCaveat` beside every
  player (slip per voice, `aria-describedby` per button), guru line 0 alone marked checked; a 404
  hides the player, keeps the words. "Play the pair": mentor's English seed, then guru's Hindi.
- The map is the other side of the sheet: `TechnicalMap` reused, repainted under `.tmap`
  (brush-edged boxes, painted lettering, the poster's arrow, torn slips); `:target`/`:has` with
  scripts off, `html.view-tech` with scripts on; fourteen sections mirrored into the strip.
- Landing keeps diagram A; C folded into its note, D lives in the map; painted pointer.
- `prefers-contrast: more`; fine print at three columns ≥ 1240; 8,224 → 8,087 px at 1440.
