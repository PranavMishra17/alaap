# PHASE 09 — Multi-Tenant & Safe on the Open Internet

> **Goal:** accounts, quotas, moderation, abuse controls, and a compliance checklist that is fully green.
> **GPU:** cloud · **Depends on:** S8
> **Evidence:** [`09`](../09-safety-and-watermarking.md) · [`08`](../08-licensing-propagation.md)

---

## 0. Exit criterion

**X9.1 — Safe to put on the open internet**, evidenced by a green compliance checklist (§3) and a written response to each abuse vector in §4.

---

## 1. Restate the safety claim honestly

Scope §15.1 claims Alaap has **"no impersonation vector at all."** That does not survive scrutiny.

**The correct, defensible claim is: "no *cloning* vector — we never accept reference audio."**

Why the broader claim fails:

| Evidence | Detail |
|---|---|
| ParaSpeechCaps trains description→voice on **594 named celebrities** | using GPT-4 as a name→attribute oracle. The capability is in the field's training data |
| Google's own novel-voice work defines success as **g2s = s2s** | i.e. generated voices sit as close to real speakers as real speakers sit to each other |
| Generated (not cloned) **master voices match 69% of women at FAR 1%** | a biometric-defeat result, not a copyright one |
| *Midler* and *Waits* | make vocal imitation actionable **without any copying** — the architecture defends against the wrong tort |
| **Periphrasis defeats name gates at >90% ASR** | "the guy from the boxing films with the gravelly voice" |

**Keep the no-upload property** — it is real, and it keeps us outside the ELVIS Act core, NO FAKES §2(c)(2)(B), and *Arijit Singh* ¶18. **Just describe it accurately.** *Lehrman v. Lovo* dismissed the copyright claims and let the **right-of-publicity** claims through — which is exactly the axis this architecture does *not* fully defend.

---

## 2. Description-layer gating

- **Named-real-person refusal** at description validation. Necessary but far from sufficient (>90% periphrasis bypass). Implement with NER + a gazetteer *and* an LLM classifier; accept that it is a speed bump.
- **False-positive risk is real** — many ordinary names collide with celebrity names. Prefer a soft challenge ("this looks like a real person's name — confirm this is a fictional character") over a hard block.
- **Dialogue-text moderation** on the render endpoint, separate from description moderation. They are different threat surfaces.

---

## 3. The compliance checklist

| Requirement | Source | Stage |
|---|---|---|
| Watermark every render (AudioSeal, presence bit) | EU Art. 50(2), live **2026-08-02** | S0 ✅ |
| Provenance log per render | Art. 50 + model-upgrade safety | S0 ✅ |
| **Signed metadata layer** (second marking layer) | Code of Practice requires **two** for audio | S6 |
| Synthetic-audio disclosure, UI + manifest | Art. 50(2) | S8 |
| **Free public detector** | Code of Practice | **S9** |
| Prohibited-use blocking | India IT Rules (10 Feb 2026) | **S9** |
| Permanent non-removable metadata | India IT Rules | **S9** |
| **"Prominently prefixed audio disclosure"** | India IT Rules | ⚠️ **COUNSEL — gates Indic public serving** |
| Per-demographic eval slices published | model card, I10 | S10 |

**Art. 2(12) carves Art. 50 out of the open-source exemption** — releasing openly does not exempt us.

---

## 4. Abuse vectors — including the ones scope §15 missed

| Vector | Severity | Response |
|---|---|---|
| **Voice-biometric defeat** | **High** — we are a ready-made master-voice generator | Rate-limit bulk minting; detect enumeration patterns; consider refusing high-volume automated mint |
| **Fraud / vishing** | **High** — $7 M FCC precedent | No permissive vishing classifier exists. Log everything; KYC above a volume threshold; clear ToS |
| Child voices | High | Refuse at description validation |
| **Watermark forgery** | Medium-High — a public detector means **we get blamed** for forgeries | Publish detector limitations alongside it; keep signing keys for the metadata layer |
| Model extraction via free mint | Medium | Quotas; the free-preview cap from S8 doubles as this control |
| Render-cache side channel | Medium | Namespace the cache per tenant |
| **Open-sourcing the mapper at S10 voids the watermark downstream** | **Medium-High** | Decide deliberately at S10 (§[`PHASE-10`](PHASE-10-public-release.md)). This is a real trade, not an oversight |
| Watermark stripping | High, trivial | **Polarity inversion drives detection to 0.18/0.00** with an inaudible one-liner. Do not claim the watermark is robust; claim it is a good-faith marker |

---

## 5. Fairness is a quality instrument

**Three independent literatures all disadvantage female voices:** watermark removal (p ≈ 2.4×10⁻⁶), master-voice matching (69% vs 38%), ASV cost (2.58×). And the corpus is 62.6% female.

Separately, **low-density regions of speaker space cost +60% relative WER** — and "elderly", "raspy", "very low-pitched" *are* low-density regions in every corpus we have.

**So per-demographic slices are not a courtesy. They are how you find quality cliffs before users do.** Report them at every stage (I10), never a single aggregate.

---

## 6. Quotas and metering

From S7's arithmetic: **meter renders to shape demand, not to cover the GPU cost** — the bill is a fixed warm-GPU floor, and below ~55 h/mo self-hosting is more expensive per minute than ElevenLabs anyway.

Practical shape: **free minting and browsing, metered rendering** — but note Tier-2 minting costs a GPU render, so cap free previews per user per day rather than making minting unlimited.

---

*Phase spec v2 · 2026-09-02 · prev: [`PHASE-08`](PHASE-08-web-app.md) · next: [`PHASE-10`](PHASE-10-public-release.md)*
