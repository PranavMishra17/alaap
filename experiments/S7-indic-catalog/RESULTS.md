# S7 — where the Indic catalog saturates

**Run:** 2026-09-05 · `SPRINGLab/Indic-Mio` + `MioCodec-25Hz-44.1kHz-v2` · 80 descriptions · hi, then hi+bn+ta
**Question:** S6 minted four voices and all four passed. How many can it actually hold?

---

## Result — 38 accepted of 80, and every rejection is the same one

| | |
|---|---|
| accepted | **38 / 80 (47.5%)** |
| rejected for **uniqueness** | **42** |
| rejected for drift | **0** |
| rejected for consistency | **0** |
| normalised Vendi | 0.362 → **~14 effective voices** from 38 accepted |
| drift, mean | 0.720 (floor 0.40) |
| consistency, mean | 0.722 (floor 0.43) |

Rolling acceptance over a window of 20, first to last:

```
100%  91%  55%  45%  50%  25%  40%  50%
```

**The catalog saturates fast and it saturates for exactly one reason.** Not a single
voice failed on quality — drift averaged 0.720 against a floor of 0.40, consistency
0.722 against 0.43. Every one of the 42 rejections was a *uniqueness collision*: the
description asked for a voice the catalog already had.

That is a much cleaner diagnosis than E11's English run, where failures were mixed
(9 uniqueness, 8 drift, 2 consistency). Here the message is unambiguous: **the Indic
ceiling is diversity, not fidelity.** Whatever limits this catalog, making the renders
better will not move it.

## Against the English catalog

| | Indic (S7) | English (E11) |
|---|---|---|
| minted | 80 | 42 |
| normalised Vendi | 0.362 | 0.482 |
| effective voices | **~14** | ~20 |
| drift, mean | **0.720** | 0.457 |
| consistency, mean | **0.722** | 0.564 |
| failures that were quality | **0** | 10 of 19 |

**Better voices, fewer of them.** The Indic path renders a more faithful and more
self-consistent identity than the English path does, and holds fewer distinct ones.
Both are above `RESEARCH/06`'s anti-mode-collapse floor of 0.35, but only just, and
0.362 on 80 samples is the tighter number of the two.

## The architecture showed up in the runtime

80 voices × 3 lines audited in **30 seconds**.

E11 had to run the LM once per voice per line — N voices cost N × L generations. Here
the towers are separable, so content tokens are generated **once** and decoded through
every speaker vector:

```
L LM generations + (N x L) codec decodes     not     (N x L) LM generations
```

N voices cost the same LM time as one. That is not an optimisation trick, it is the
two-tower split being real — the same property that made S5's identity test valid. It
also makes the comparison between voices exact: every voice says the identical
utterance, so a drift or consistency difference **cannot** be a content difference.

## Widening the corpus does not help — measured, not assumed

Pooling Bengali and Tamil into one described space (`--corpus hi,bn,ta`): 432 speakers
instead of 141, one binner over the pooled attributes.

| | hi only | **hi+bn+ta** |
|---|---|---|
| speakers | 141 | **432** |
| accepted | 38/80 (47.5%) | **55/80 (68.8%)** |
| rejected, all uniqueness | 42 | 25 |
| normalised Vendi | 0.362 | 0.257 |
| **effective voices** | **~14** | **~14** |
| drift / consistency | 0.720 / 0.722 | 0.723 / 0.720 |

**Three times the speakers. Acceptance up 21 points. Effective voices unchanged.**

`S9` explains why, by measuring the ceiling nobody had: 141 real Hindi speakers hold
**~38** effective voices and all 432 pooled hold **~48** — tripling the corpus bought
26% more diversity, not 200%, because Hindi, Bengali and Tamil speakers occupy heavily
overlapping regions of MioCodec's identity space.

And minting was never near that ceiling. It reaches **37%** of the 38 available, where
`S8`'s retrieval over the same speakers reaches **87%**. More anchors gave minting more
to retrieve *from* — hence the acceptance jump — without moving what it *spans*.

**So the diagnosis in the section above needs sharpening.** The Indic ceiling is not
diversity in the abstract, and it is not the corpus. It is that minting with
`novelty=0.0` interpolates inside the convex region its anchors span, and that region is
about a third of the space. See `S9` for the three fixes, cheapest first.

## The fix — 14 → 22 effective voices, and nothing traded away

`S9b` found the mechanism: minting blends `top_k` anchors in a `pca_dims`-truncated
basis. Both are already parameters; both were set badly.

| | old (`pca_dims 32, top_k 4`) | **fixed (`64, 2`)** |
|---|---|---|
| accepted | 38/80 (47.5%) | **50/80 (62.5%)** |
| normalised Vendi | 0.362 | **0.431** |
| **effective voices** | **~14** | **~22** |
| share of the ~38 bound | 37% | **58%** |
| nn distance, median | 0.159 | **0.471** |
| drift, mean | 0.720 | **0.753** |
| consistency, mean | 0.722 | **0.740** |

**Drift and consistency went UP.** The fix is not a diversity-for-quality trade — the
old setting was leaving both on the table. Acceptance rose 15 points because fewer mints
collided with voices already in the catalog.

What did **not** work, tested before this was written: rescaling minted vectors back to
the real-speaker radius. That moved Vendi 0.161 → 0.161. The radial contraction is a
*symptom* of blending, not the mechanism.

`S10` finds the same defect in the English pipeline, worse: 11% of its bound, fixed to
20%, +81%.

## What this means for the product

A catalog of 500 voices where 300 are audibly the same voice is a catalog of 200 voices
and a support problem. On this corpus the honest number is **~22 effective Indic
voices** after the `S9b` fix, against ~38 available. That is a cast, not a catalog — and the shortfall is
the method's, not the corpus's.

The constraint is not the renderer. Three candidate causes, in order of how cheaply
they can be tested:

1. ~~**The corpus.**~~ **Tested and ruled out** — see the section above. 432 speakers
   gave exactly the same ~14.
2. ~~**Blending settings.**~~ **Found and fixed** — see above, 14 → 22.
3. **The described space.** Five axes × five bins, and `speaking_rate` carries a
   measured weight of 0.10 — nearly worthless for identity. Four useful axes cannot
   separate very many people.
4. **MioCodec's global embedding itself.** 128-d, and by its own card it mixes speaker
   with recording environment and microphone.

`S8` tests a fourth possibility that turns out to matter more than any of these: not
minting at all.

## Not established

- **One corpus, one language, one seed, novelty 0.** `--novelty` was never raised;
  E11's adaptive-novelty arm exists precisely because escalating novelty on collision
  changes the curve, and that was not run here.
- **Nobody has listened to the 8 saved renders.** Machine-scored only.
- **Vendi at n=38 is noisy.** E11's 0.482 came from 42 accepted; these are comparable
  in size and both are small.
- ~~The `uniqueness` floor of 0.30 is VoicePrivacy B3's threshold applied in working
  space, a defensible choice rather than a measured one.~~ **`S11` measured it and it was
  too low** — a listener heard pairs at d=0.323 as the same person half the time, and
  everything at d≥0.506 correctly. The floor is now 0.45 for MioCodec, which drops 10 of
  50 accepted voices and 3% of effective diversity: those ten were duplicates. Numbers in
  this document above were produced at 0.30 and so **overcount the nominal catalog** —
  the effective-voice figures barely move (21.5 → 20.8).

## Reproduce

```bash
envs/qwen3/Scripts/python.exe experiments/S6-indic-mint/run_indic_mint.py --skip-render
envs/qwen3/Scripts/python.exe experiments/S7-indic-catalog/run_indic_catalog.py --n 80
```

The first builds the MioCodec embedding cache S7 reads. ~1 minute total on a 6 GB card.
