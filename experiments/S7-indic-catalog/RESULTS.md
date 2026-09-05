# S7 — where the Indic catalog saturates

**Run:** 2026-09-05 · `SPRINGLab/Indic-Mio` + `MioCodec-25Hz-44.1kHz-v2` · 80 descriptions
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

## What this means for the product

A catalog of 500 voices where 300 are audibly the same voice is a catalog of 200 voices
and a support problem. On this corpus the honest number is **~14 effective Indic
voices**. That is a cast, not a catalog.

The constraint is not the renderer. Three candidate causes, in order of how cheaply
they can be tested:

1. **The corpus.** 141 speakers of read Hindi. A mapper cannot describe a region no
   speaker occupies. → widen to bn/ta (already measured in S4) and re-run.
2. **The described space.** Five axes × five bins, and `speaking_rate` carries a
   measured weight of 0.10 — nearly worthless for identity. Four useful axes cannot
   separate very many people.
3. **MioCodec's global embedding itself.** 128-d, and by its own card it mixes speaker
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
- The `uniqueness` floor of 0.30 is VoicePrivacy B3's threshold applied in working
  space. It is a defensible choice, not a measured one for *this* space — a different
  floor gives a different effective count.

## Reproduce

```bash
envs/qwen3/Scripts/python.exe experiments/S6-indic-mint/run_indic_mint.py --skip-render
envs/qwen3/Scripts/python.exe experiments/S7-indic-catalog/run_indic_catalog.py --n 80
```

The first builds the MioCodec embedding cache S7 reads. ~1 minute total on a 6 GB card.
