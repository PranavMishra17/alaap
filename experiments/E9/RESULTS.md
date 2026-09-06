# E9 — Do extreme voices hold together as well as ordinary ones?

> **Status:** ✅ RUN AND COMPLETE · **Date:** 2026-09-05
> **Model:** `Qwen3-TTS-12Hz-0.6B-Base` *(deliberately — the cached GLOBE corpus was embedded with 0.6B; mixing encoders would invalidate the comparison)*
> **Mapper:** 2,500 GLOBE_V2 pairs, novelty 0.0 · **24 renders** (4 extreme + 4 central identities × 3 lines)
> **Scorer:** ECAPA, independent of the conditioning encoder
> **Artefacts:** `out/results.json`, `out/audio/`

---

## 0. Why this matters

RESEARCH/12 found that operating in **sparse regions of speaker space costs +60% relative WER** (10.94% vs 6.83%), and warned that *"elderly", "raspy", "very low-pitched"* are exactly those regions. RESEARCH/10 adds that **nobody has published identity-drift numbers for heavy stylisation at all** — every published figure covers the six canonical emotions instead.

Scope §5 promises **"human + heavy stylization — aged, raspy, whispered, breathy, menacing, theatrical."** If identity collapses precisely at the extremes, that promise is undeliverable, and it is much cheaper to learn that now than at S8.

---

## 1. VERDICT

> **The worry does not reproduce. Extreme voices hold together as well as central ones — and are dramatically easier to hit.**

| | EXTREME | CENTRAL | delta | |
|---|---|---|---|---|
| **identity consistency** | **0.5924** | 0.5795 | +0.0129 | ✅ equal (extreme marginally better) |
| — normalised (0 = a different speaker) | **0.786** | 0.760 | +0.026 | ✅ |
| — **worst case** | **0.4279** | 0.5008 | −0.0730 | ⚠️ **more variance at the extremes** |
| **vocoder drift** | **0.5265** | 0.5407 | −0.0142 | ✅ equal |
| — worst case | 0.4494 | 0.3724 | +0.0769 | ✅ better |
| **adherence, exact match** | **0.778** | 0.167 | **+0.611** | ⭐ **4.7× better** |
| **adherence, bin distance** | **0.583** | 1.097 | −0.514 | ⭐ far better |
| retrieval anchor similarity | 0.8732 | 0.8573 | +0.0159 | |
| render duration | 3.15 s | 3.21 s | −0.07 | no runaway generation |

*(reference: ECAPA C_same 0.6988 / C_diff 0.2011 on real speech)*

---

## 2. The surprise: extreme descriptions are the EASY case

`exact match 0.778 vs 0.167` — nearly **five times better** for extremes. That inverts the expectation the research set up, and the explanation is partly real and partly an artefact. Both halves matter:

**Real half.** An extreme description like *"a very deep voice, very rough and gravelly, speaking very slowly"* specifies the **outer bin of every attribute at once**. Retrieval then only has to find the corpus's most extreme voice in each direction, which is an easy, well-posed search. A central description asks for the **middle** bin, which requires the retrieved voice to be simultaneously unremarkable on several axes — a narrower target.

**Artefact half.** **Outer bins are absorbing; middle bins are not.** Anything beyond the 80th percentile counts as "very high-pitched", so measurement noise cannot push you out of it — but the middle bin has neighbours on both sides, so the same noise costs you an exact match. **Some of the 4.7× gap is this asymmetry, not genuine skill**, and the effect would shrink with more bins or a tolerance-based score.

> **Practical consequence, regardless of the split:** the character descriptions users will actually type — *"a gravelly old mentor"*, *"a shrill, panicked merchant"* — are **exactly the extreme case**, which is the case this system handles best. The weak case is *"a normal-sounding person"*, which is also the case nobody asks for.

---

## 3. The one genuine concern

> ⚠️ **This did not reproduce.** A re-run gives 0.4184 vs 0.3848 — the opposite sign. See the re-run section below; n=4 cannot support a worst-case claim.

**Worst-case consistency is meaningfully lower at the extremes: 0.4279 vs 0.5008.**

Means are equal, but the *tail* is worse — some extreme identities are noticeably less self-consistent across lines. With only 4 identities per group this is a weak signal, but it is the direction RESEARCH/12 predicted, and it is the one metric that would actually hurt a shipped game (one character whose voice wanders between lines).

**Action:** the closed drift loop in `service.py` already re-mints below a 0.40 floor. This suggests the *consistency* check should be part of minting too, not just drift — render 2–3 probe lines at mint time and reject an identity whose self-consistency is below the operating threshold.

---

## 5. Re-run at the fixed settings — the worst-case finding REVERSED

Re-run after `S9b`/`S10` (full basis, `top_k=2`), and after two stale breakages in this
script were fixed. Same design, same n.

| | old EXTREME | old CENTRAL | **new EXTREME** | **new CENTRAL** |
|---|---|---|---|---|
| identity consistency | 0.5924 | 0.5795 | 0.5326 | 0.5161 |
| — **worst case** | **0.4279** | **0.5008** | **0.4184** | **0.3848** |
| vocoder drift | 0.5265 | 0.5407 | 0.5279 | 0.5840 |
| — **worst case** | 0.4494 | 0.3724 | **0.2440** | **0.4425** |
| adherence exact-match | — | — | 0.583 | 0.229 |

**Both worst-case verdicts flipped sign.**

- Worst-case *consistency* was §3's headline — extremes worse by −0.073. It is now
  **+0.034 in the extremes' favour**.
- Worst-case *drift* was extremes-better by +0.077. It is now **−0.198 against them**,
  and 0.2440 sits below the 0.40 `DRIFT_FLOOR`, so that identity would be rejected and
  re-minted in the shipped path.

### n=4 was never enough for a worst case

The worst case of four samples *is* one sample. §4 already said the worst-case gap needed
more samples to be trusted — and §3 recommended an action on it anyway.

Two runs of the same design now give **opposite** worst-case verdicts. That is as clear a
statement as this design can make that these were noise. Neither direction should be
believed. **The means, which barely moved and agree across both runs, are what survives:**
extreme and central identities hold together about equally.

### The action taken on the old finding, and whether it stands

§3 recommended adding a consistency probe to minting, not just drift. That was
implemented — `CONSISTENCY_FLOOR = 0.43` in `service.py`, checked at mint time.

**It stays, and its justification changes.** It was argued for as *"extremes are riskier"*,
which does not reproduce. It is worth keeping for the reason that does survive: it rejects
genuinely unstable identities wherever they arise, and this run produced one at drift
0.2440, well under its floor. **A floor that catches real failures does not need a
directional story about which voices cause them.**

---

## 4. Caveats

- **4 identities per group, and a re-run flipped BOTH worst-case verdicts.** See §5. The worst case of four samples is one sample; the means are the part that survives.
- **The adherence gap is partly a binning artefact** (§2). Do not quote 0.778 as the system's adherence.
- **Run on 0.6B**, for corpus-consistency with the cached GLOBE embeddings. E10 showed 1.7B discriminates better but drifts the same, so the conclusion should carry.
- **"Extreme" here means extreme *within GLOBE_V2's range*.** GLOBE is diverse read speech, not character acting — a genuinely aged or theatrical voice is still outside anything the corpus contains. This tests the edge of the manifold we have, not the edge of the promise.
- **No listening test.** `out/audio/` has one render per identity, extreme and central side by side. These are the most interesting files produced tonight to actually listen to.

---

## 5. What this changes

1. **Scope §5's voice range is not obviously blocked.** The strongest available evidence says extremes are as stable as centres, and easier to target.
2. **Add a consistency probe to minting**, not just a drift probe (§3).
3. **Adherence scoring needs a tolerance-aware variant** — exact bin match rewards absorbing outer bins. Report bin *distance* as primary, which already behaves sensibly (0.583 vs 1.097).
4. **The real remaining question is corpus, not manifold.** GLOBE_V2 has no aged or theatrical voices. That is the VoicePersona-v2 argument from RESEARCH/05, now with a measured reason behind it.

---

## 6. Reproduce

```bash
envs/qwen3/Scripts/python.exe experiments/E9/run_e9.py
```

~6 min. Requires `experiments/S2/out/corpus_globe_v2_2500_1.npz`.
