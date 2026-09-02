# 12 — Is the Speaker-Embedding Manifold Navigable by Synthesis?

> **Domain:** the decisive empirical question — do arbitrary/synthesized speaker vectors produce coherent voices, or fall off-manifold into artefacts?
> **Answers:** **A2** (the single most decisive question), and supplies the evidence base for **A1** and **A3**
> **Date:** 2026-09-02 · Pass 1
> **Cross-refs:** [00-EXECUTIVE-VERDICT.md](00-EXECUTIVE-VERDICT.md) · [01-ttv-landscape.md](01-ttv-landscape.md) · [02-identity-representation.md](02-identity-representation.md) · [03-tts-backends-english.md](03-tts-backends-english.md) · [06-evaluation-harness.md](06-evaluation-harness.md) · [11-production-api-landscape.md](11-production-api-landscape.md)

> **Verification method:** every arXiv ID below was fetched (abstract page or PDF) and the title/content confirmed. Quantitative claims are read out of actual PDF text and tables, not summaries. Where a number's provenance is ambiguous (PDF figure-label extraction), it is marked MEDIUM.

---

## 0. Bottom line

**The two-tower architecture survives. Synthesizing a speaker vector numerically produces usable, intelligible, natural speech. This is routine, production-grade, and is an official challenge baseline.** [HIGH]

But the evidence splits into two findings that must not be conflated:

1. **Synthesized vectors work.** Uniformly random points on the unit hypersphere yield speech "as natural as for seen or unseen real speakers" — naturalness MOS **3.65** (Jia et al., NeurIPS 2018). WGAN-sampled embeddings are now the official **VoicePrivacy 2026 baseline B3**. Intelligibility cost of a fully synthetic vector is roughly **+1 to +2.5 points absolute WER**. [HIGH]
2. **Naive averaging is a documented, quantified failure — and it does not fail the way scope §4.2 predicts.** It does **not** produce "bland mush." It produces **the same voice every single time**. [HIGH]

**The scope document's §4.2 diagnosis is directionally right but mechanically wrong**, and the distinction matters enormously for how you detect the failure:

| | Scope §4.2 predicts | What actually happens |
|---|---|---|
| Failure mode | MSE → conditional mean → "bland, averaged, characterless voice" | Averaging **shrinks vectors toward the origin**, because dimensions are ~zero-centred → **all identities collapse to one voice** |
| Symptom | "fine but generic" | *Literally the same voice regardless of input* |
| Measured as | — | GVD **−6.5 to −11.6 dB** |
| Fix | generative head (flow/diffusion) | **per-dimension rescaling to each dimension's observed empirical range** → GVD recovers to **≈ −0.1** |

**The single highest-leverage fix in this entire research pass is per-dimension rescaling, and it is nearly free.** [HIGH]

**The decisive variable is whether your synthetic vector matches the per-dimension distribution / norm of real embeddings — not whether it corresponds to a real person.** [HIGH]

Three further findings that constrain the design:

- **Discriminative ASV spaces resist simple generative priors.** TacoSpawn measured a GMM prior over 256-d d-vectors sampling to **g2s = 0.35 against s2s = 0.20** — sampled vectors land nearly **twice as far from any real speaker as real speakers are from each other**. A *jointly-learned* embedding space hit 0.20/0.20 exactly. [HIGH]
- **Vocoder drift: the vector you feed in is not the vector you get out.** Systematic, measurable, learnable. If the product promise is "this vector = this voice," a closed verification loop is mandatory. [HIGH]
- **Diversity and in-distribution-ness are in direct tension.** "Generating new, varied voices inherently pushes the embeddings further away from the original distribution" — MMD rises 0.005 → 0.17 as top-5 cosine similarity falls 0.80 → 0.40. This *is* the adherence↔diversity dial, and it is a real geometric constraint, not a UI choice. [HIGH]

---

## 1. Corrections to VOICEFORGE-SCOPE.md

| # | Scope claim | Verdict | Correction | Primary source | Conf. |
|---|---|---|---|---|---|
| 12-1 | §4.2: MSE loss → "the mean of many voices is a bland, averaged, characterless voice… Output sounds 'fine but generic'" | **Mechanically wrong; the real failure is worse** | Averaging does not blur toward genericness — it **collapses every identity onto one voice**. Meyer et al.: the system "produces a very similar voice **regardless of** the anonymized speaker embedding fed to the TTS." Detection differs: a bland-voice failure needs perceptual judgement; a collapse failure is caught instantly by GVD or pairwise cosine. | [arXiv:2207.04834](https://arxiv.org/abs/2207.04834) §5.1.1 | HIGH |
| 12-2 | §4.2: implies the cure is architectural (generative head) | **Incomplete — the cheapest cure is arithmetic** | Per-dimension rescaling alone recovers GVD from −6.50 to **−0.14**. Do this *before* reaching for flow matching. A generative head on an unrescaled space inherits the collapse. | [arXiv:2207.04834](https://arxiv.org/abs/2207.04834) Table 1 | HIGH |
| 12-3 | §17-A2: "do arbitrary/synthetic vectors produce coherent voices, or fall off-manifold into artefacts?" framed as unresolved | **Resolved: YES, they produce coherent voices** | Established since 2018 and now an official VoicePrivacy baseline. The question was never open in the literature — it was open in the scope doc. | [arXiv:1806.04558](https://arxiv.org/abs/1806.04558) §3.6 | HIGH |
| 12-4 | §4.3 Tier 1: "Persist the speaker embedding vector → reproducible **exactly, by construction**" | **False — vocoder drift breaks this** | The x-vector extracted from rendered audio "often differs substantially from the x-vector at the vocoder input." Tier 1 is reproducible *at the conditioning input*, not *at the perceptual output*. A closed loop (synthesize → re-extract → compare → correct) is required for the exactness claim. | Panariello et al., Interspeech 2023, [DOI 10.21437/Interspeech.2023-448](https://doi.org/10.21437/Interspeech.2023-448) | HIGH |
| 12-5 | §7: use a frozen ASV speaker encoder to define the target manifold | **Risky as stated** | ASV encoders are trained to *suppress* intra-speaker variance — "misaligned with the objective of generation." Prefer a space the generator was trained to accept; if you must use an ASV space, concatenate heterogeneous encoders or add a consistency loss. | [arXiv:2407.04291](https://arxiv.org/abs/2407.04291) | HIGH |
| 12-6 | §4.2 option 1: "Flow matching, a diffusion head, or a normalizing flow… **This is the correct answer**" | **Overconfident on the head, underspecified on the space** | Which *space* you model matters more than which *head* you use. A GMM over a learned space beat a GMM over a d-vector space decisively (g2s 0.20 vs 0.35). Space selection dominates head selection. | [arXiv:2111.05095](https://arxiv.org/abs/2111.05095) §6.2 | HIGH |
| 12-7 | §4.2 option 2: contrastive retrieval "then perturb or interpolate" | **Valid, with constraints the brief doesn't state** | Interpolation works (1,225 pairs, WER 6.42–7.35%) but: use **SLERP not LERP** (preserves unit norm), pair **nearest-neighbour and same-gender**, and **never extrapolate** — extrapolation outside the real-embedding hull is documented as "not particularly meaningful." | [arXiv:2106.05762](https://arxiv.org/abs/2106.05762), [arXiv:2508.19210](https://arxiv.org/abs/2508.19210), [arXiv:2310.03538](https://arxiv.org/abs/2310.03538) | HIGH |
| 12-8 | §10 axis 3: diversity as an anti-mode-collapse check | **Correct, and there is a ready-made metric with published baselines** | Use **GVD** (Gain of Voice Distinctiveness), an official VoicePrivacy secondary metric with a published scale: 0 dB = preserved, −10 dB = collapsed. Better than inventing one. | Noé et al., Interspeech 2020, [DOI 10.21437/Interspeech.2020-2720](https://doi.org/10.21437/Interspeech.2020-2720) | HIGH |
| 12-9 | §5: hardware/plan assumes this is unexplored research | **A whole adjacent field has solved much of it** | **Speaker anonymization** (VoicePrivacy Challenge, 4 editions, 2020–2026) is the same technical problem viewed from the privacy side: generate a pseudo-speaker vector, render speech, require it to be distinct and natural. Six years of baselines, metrics, and negative results transfer directly. The scope doc does not mention it once. | [voiceprivacychallenge.org](https://www.voiceprivacychallenge.org/) | HIGH |

---

## 2. The adjacent field the scope document missed: speaker anonymization

**VoiceForge's core technical problem has been an open benchmark since 2020.** The VoicePrivacy Challenge asks: replace a speaker's identity with a synthetic pseudo-speaker such that the output is natural, intelligible, and *distinct from other pseudo-speakers*. That last requirement is exactly VoiceForge's diversity axis, and it has a named metric and six years of results.

| Edition | Document | ID | Conf. |
|---|---|---|---|
| VPC 2020 | Evaluation Plan | [arXiv:2205.07123](https://arxiv.org/abs/2205.07123) | HIGH (PDF read) |
| VPC 2020 | Results and findings (*Computer Speech & Language*) | [arXiv:2109.00648](https://arxiv.org/abs/2109.00648) | HIGH (PDF read) |
| VPC 2022 | Evaluation Plan | [arXiv:2203.12468](https://arxiv.org/abs/2203.12468) | HIGH (PDF read) |
| VPC 2022 | Progress and Perspectives (IEEE/ACM TASLP) | [arXiv:2407.11516](https://arxiv.org/abs/2407.11516) | HIGH (PDF read) |
| VPC 2024 | Evaluation Plan | [arXiv:2404.02677](https://arxiv.org/abs/2404.02677) | HIGH |
| VPC 2024 | The Third VoicePrivacy Challenge | [arXiv:2601.11846](https://arxiv.org/abs/2601.11846) | HIGH (PDF read) |
| **VPC 2026** | Evaluation Plan | [VPC_2026_sept01.pdf](https://www.voiceprivacychallenge.org/vp2026/docs/VPC_2026_sept01.pdf) | HIGH (PDF read) |

**A 2026 edition is live.** Fourth edition, started March 2026, two tracks (English / Multilingual), workshop co-located with SPSC / Interspeech 2026 in Sydney, **26 Sept 2026** — three weeks from now. Baseline repo: [`Voice-Privacy-Challenge-2026`](https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2026).

> **Action:** the VPC 2026 results land ~3 weeks after this research pass. Re-check after 26 Sept 2026 — it will contain the current best answer to "how do you generate a good synthetic speaker vector," for free.

---

## 3. What averaging does — the failure mode, precisely

### 3.1 The canonical procedure (VPC baseline B1), verified

Sources: Fang et al. SSW10 2019 ([arXiv:1905.13561](https://arxiv.org/abs/1905.13561), [ISCA DOI](https://doi.org/10.21437/SSW.2019-28)); Srivastava et al. Interspeech 2020 ([arXiv:2005.08601](https://arxiv.org/abs/2005.08601), [ISCA DOI](https://doi.org/10.21437/Interspeech.2020-2692)); VPC 2020 Evaluation Plan §6.1 + footnote 13.

- Encoder: **512-d TDNN x-vector**, trained on VoxCeleb 1+2
- Pool: speaker-level x-vectors from LibriTTS `train-other-500` (**1,160 speakers**), disjoint from eval
- Distance: PLDA log-likelihood ratio; cosine documented as the alternative
- Select the **N = 200 farthest** pool vectors from the source speaker
- Draw **N\* = 100** of those uniformly at random
- **Arithmetic mean** → pseudo-speaker x-vector
- **Same-gender constraint**
- Synthesis: acoustic model (F0 + BN + x-vector → 80-d mel) → NSF vocoder

Srivastava, verbatim:
> "Step 2 … aims to generate a pseudo-speaker and comprises two sub-steps: 1) select N\* candidate target x-vectors from the anonymization pool; 2) **average them** to obtain the pseudo-speaker x-vector."

> "we rank all the x-vectors in the anonymization pool in increasing order of their distance from S and select either the top N (near) or the bottom N (far). To introduce some randomness, N\* < N x-vectors are selected out of these N uniformly at random. **The values of N and N\* are fixed to 200 and 100** … **We noticed a sharp decline in utility for a smaller value of N\*.**"

### 3.2 Averaged vectors produce *intelligible* speech

| Source | Condition | WER |
|---|---|---|
| VPC 2020 Plan, Table 9 | LibriSpeech-test original | **4.15%** |
| " | LibriSpeech-test B1-anonymized | **6.73%** |
| " | VCTK-test original → anonymized | **12.82% → 15.23%** |
| VPC 2024 ([2601.11846](https://arxiv.org/abs/2601.11846) Table 9) | LibriSpeech test original | **1.85%** |
| " | **B1** (averaged x-vector) | **2.91%** |
| " | **B3** (WGAN artificial embedding) | **4.35%** |
| VPC 2026 Plan, Table 7 | eval: orig / B3 / B4 / B5 / B2 | **1.84 / 4.26 / 5.90 / 4.53 / 9.11%** |

In VPC 2024, the averaged-x-vector pipeline is the **best baseline on WER** — better than the neural-codec and GAN baselines. Speech from an entirely artificial x-vector is intelligible. [HIGH]

*(B1's privacy collapsed against modern attackers — EER 6.08% vs 4.59% original — which is why it was retired. That is a privacy result, not a voice-quality result, and does not bear on VoiceForge.)*

### 3.3 …but they collapse into a single voice

**The organizers' own observation** (VPC 2020 results, §4.1.5, on voice-similarity matrices):
> "The matrices for signal processing based systems and for system K2 exhibit a distinct diagonal in M_aa, indicating that voices remain distinguishable after anonymization. **For x-vector based systems, this diagonal is much weaker.**"

**Quantified via GVD** — Gain of Voice Distinctiveness, `GVD = 10·log₁₀(D_diag(M_aa)/D_diag(M_oo))`. 0 dB = distinctiveness preserved; negative = collapse.

| System | GVD (Libri F/M) | GVD (VCTK F/M) |
|---|---|---|
| **VPC B1 (average of 100 x-vectors)** | **−10.09 / −8.95** | **−10.56 / −11.58** |
| Turner GMM-sampled x-vectors | **−12.14** | **−13.79** |
| Espinoza-Cuadros adversarial AE | **−13.60** | **−15.22** |

−10 dB means the diagonal dominance of the anonymized similarity matrix fell to ~10% of the original. **All pseudo-speakers sound alike.** [HIGH]

### 3.4 The mechanism — isolated, explained, and fixed ⭐ **the key result**

Meyer et al., Interspeech 2022, [arXiv:2207.04834](https://arxiv.org/abs/2207.04834), §5.1.1, verbatim:

> "The GVD of the **pool raw** approach (unscaled embeddings) shows that the **system is basically unusable since it produces a very similar voice regardless of the anonymized speaker embedding fed to the TTS**. Since pool anonymization works by averaging speaker embeddings and the value range of the dimensions is in most cases centered around 0, **the range of values in the anonymized embeddings becomes smaller, leading to the collapse of different speakers into one**. By simply scaling each of the dimensions in the anonymized speaker embeddings according to their usual range of values, we achieve **near perfect GVD** using the pool approach."

| Anonymization method | Libri F/M | VCTK F/M |
|---|---|---|
| VPC B1 baseline | −10.09 / −8.95 | −10.56 / −11.58 |
| **pool raw** (average, no rescaling) | **−6.50 / −7.68** | **−7.79 / −9.57** |
| **pool** (average **+ per-dim rescaling**) | **−0.14 / −0.11** | **−0.02 / −0.58** |
| **random** (each dim sampled uniformly in *its own* observed range) | **−0.13 / −0.14** | **−0.88 / −1.38** |

And the geometric reason it happens — same paper, §2.2, on a 704-d ECAPA+x-vector concatenation:

> "each dimension contains a different value distribution such that the range of values varies a lot between the dimensions. For instance, [in] the position for which the smallest value range could be found … the values range between **−0.97 and −0.11**, while the dimension with the largest range allows values between **−72.37 and 81.59**. **Ignoring these differences during the anonymization could lead to unnatural embeddings with properties that would not be found for a real speaker.**"

> **This is the single most actionable finding in the research pass.** Per-dimension scale heterogeneity spans ~3 orders of magnitude. Any operation that treats the space as isotropic — averaging, plain Gaussian noise, unweighted MSE, naive LERP — silently destroys identity diversity. The fix is a per-dimension rescale, costs nothing, and takes GVD from −6.5 to −0.14.

### 3.5 Independent confirmation

Turner, Lovisotto & Martinovic, [arXiv:2010.13457](https://arxiv.org/abs/2010.13457):
> "the challenge baseline system generates fake X-vectors which are **very similar to each other, significantly more so than those extracted from organic speakers**. This difference arises from averaging many X-vectors from a pool of speakers … causing a loss of information."
> "leads to fake X-vectors which **underutilize their multi-dimensional vector space** … **less entropy in fake X-vector space**."

Their PCA + GMM fix gained EER **up to +19.4% (male) / +11.1% (female)**. **Caveat:** GVD remained −12 to −14 dB inside the VPC pipeline — the fix improved *unlinkability*, not *distinctiveness*. [HIGH]

### 3.6 Density matters — a hard quality cliff

Srivastava et al., Table 1 (test WER):

| Selection strategy | Test WER |
|---|---|
| no anonymization | 4.15% |
| Random | 6.58% |
| PLDA / Far / same-gender | 6.71% |
| Near | 6.79% |
| Dense | 6.83% |
| **Sparse** | **10.94%** |
| Random gender | 6.88% |
| **Opposite gender** | **7.19%** |

**Averaging inside sparse (low-density) regions costs +60% relative WER.** Direct evidence that the space is *not uniformly* navigable — low-density regions synthesize measurably worse. [HIGH]

> **Consequence for VoiceForge:** "a gravelly 80-year-old woman's voice" is a *low-density region* in every corpus you will train on (see [`05-datasets-and-annotation.md`](05-datasets-and-annotation.md) on the 76%-twenties skew). This finding predicts that exactly the descriptions your users most want will land in the worst-behaved part of the space. Per-demographic eval slices are not a fairness nicety — they are a *quality* instrument.

### 3.7 Vocoder drift — you don't get the voice you asked for

Panariello, Todisco & Evans, Interspeech 2023, [DOI 10.21437/Interspeech.2023-448](https://doi.org/10.21437/Interspeech.2023-448). As summarised in the official VPC 2022 progress paper:

> "**the x-vector extracted from the anonymised utterance often differs substantially from the x-vector at the vocoder input.** This phenomenon, referred to as **vocoder drift** … It is argued that **poor control of the speaker embedding space can hinder the development of better anonymisation functions**."

Follow-up: "Vocoder drift compensation by x-vector alignment in speaker anonymisation," 3rd SPSC Symposium 2023, pp. 16–20 — drift is **systematic and correctable**, not noise.

> **This is the most important caveat for Tier-1 identity.** Scope §4.3 claims a stored vector reproduces the voice "exactly, by construction." That is true of the *conditioning input* and false of the *rendered output*. VoiceForge must either (a) accept identity is defined at the input and measure output consistency separately, or (b) run a closed loop: synthesize → re-extract → compare → correct. Option (b) costs a GPU render per mint — which erases Tier 1's main economic advantage over Tier 2 (see [`11-production-api-landscape.md`](11-production-api-landscape.md) §9.3).

### 3.8 How much of B1's unnaturalness is the vector's fault? Not much.

"Evaluation of the Speech Resynthesis Capabilities of the VoicePrivacy Challenge Baseline B1", [arXiv:2308.11337](https://arxiv.org/abs/2308.11337), 3rd SPSC Symposium 2023. Runs B1's synthesis pipeline with the **original, un-anonymized** x-vectors, isolating the vocoder/AM. MUSHRA-like, 18 subjects:

> "**both the speech representation and the vocoder introduces artifacts, causing an unnatural perception.**"

Listener reports include "a severe muffling effect, often at [utterance] beginnings, **rendering the part of the utterance completely unintelligible**" and "random impulsive artifacts, somewhat like a 'sizzling frying pan'." Conclusion: "the copy synthesis scores better than the synthesis from representations."

> **Engineering read:** a large fraction of B1's unnaturalness is attributable to a 2019-era NSF vocoder and bottleneck features — **not** to the x-vector being artificial. Modern backends (see [`03-tts-backends-english.md`](03-tts-backends-english.md)) should carry substantially less of this penalty. Do not read VPC-era naturalness numbers as the ceiling for VoiceForge.

---

## 4. Sampling from a generative model of the space

| Work | ID / venue | Finding | Conf. |
|---|---|---|---|
| **Turner et al.**, Distribution-Preserving X-Vector Generation | [arXiv:2010.13457](https://arxiv.org/abs/2010.13457), VPC2020 | **PCA + GMM** on 512-d x-vectors, sample, inverse-PCA. Restores realistic cross-similarity distribution; EER +19.4%/+11.1%. Removes the need to ship a speaker pool. GVD still ≈ −12/−14 in the VPC pipeline. | HIGH |
| **Meyer et al.**, Anonymizing Speech with GANs | [arXiv:2210.07002](https://arxiv.org/abs/2210.07002), **IEEE SLT 2022** | **WGAN-QC** trained to mimic the real speaker-embedding distribution. WER **5.90 (Libri) / 10.02 (VCTK)** — best of all systems tested, better than original on VCTK. **GVD −0.06 to +0.18** — distinctiveness fully preserved. Human study: listeners judged same/different speaker correctly **85.63%** of the time on anonymized audio → **artificial voices are internally consistent and mutually distinct.** Naturalness MOS drops ~1.2 (3.44/3.09 vs 4.67/4.33); intelligibility MOS barely moves (4.48/4.18 vs 4.74/4.5). | HIGH |
| " (architecture note) | " | **Generator architecture matters a lot:** "the use of MLP … prevents the model to converge, leading to generated embeddings that can be **easily distinguished from original ones**." **ResNet generator required.** | HIGH |
| **B3 baseline**, VPC 2024 & 2026 | [arXiv:2601.11846](https://arxiv.org/abs/2601.11846) §4.3 | The Meyer WGAN is now an **official challenge baseline**. Artificial embedding generated by WGAN with a constraint that **cosine distance from the original exceeds 0.3**. VPC2024 test: EER 27.32%, WER 4.35%. VPC2026 eval: EER 14.24%, WER 4.26%. | HIGH |
| **NouveauVoice** | [arXiv:2607.03985](https://arxiv.org/abs/2607.03985) (2026-07-04) | Hierarchical deep VAE over speaker timbre space; plug-in for FACodec / CosyVoice2. EER > 38%. Head-to-head vs **GMM** on the same space: CosyVoice2 tie (36.20/4.29 vs 36.40/4.29); FACodec **GMM WER 9.90% vs NV 7.56%** (recon floor 4.50%) — *a poorly-matched generative sample measurably degrades intelligibility in some pipelines.* Diversity: top-5 cosine — real 0.75–0.84; **GMM 0.74–0.80 (barely more diverse than real — it just reproduces the pool)**; NVAE 0.40–0.66 but MMD to real rises 0.005 → 0.17. Conclusion: "**generating new, varied voices inherently pushes the embeddings further away from the original distribution.**" | HIGH |
| **SpeakerVAE** | [arXiv:2511.07135](https://arxiv.org/abs/2511.07135) | Deep hierarchical VAE over timbre space; sampling generates "novel, unseen speakers with **quality comparable to that of the training speakers**." Plug-in, no co-training of the base VC model. | MEDIUM (abstract) |
| **SVT-assisted Matrix** | [arXiv:2405.10786](https://arxiv.org/abs/2405.10786), IEEE/ACM TASLP | Explicit critique: utterance-level ASV vectors "averaged or modified … suffer from **deterioration in the naturalness**, **degradation in speaker distinctiveness**, and severe privacy leakage." Proposes frame-level speaker vectors + SVD transform. | HIGH (abstract) |
| **VPC 2024 entry T10** (NPU-NTU) | [arXiv:2409.04173](https://arxiv.org/abs/2409.04173) | Interpolates explicitly: **s_anon = α·s̄ + (1−α)·ŝ** (pool average vs Gaussian random identity). Organizers: "A higher α puts more weight on the averaged speaker identity, typically resulting in **less anonymity but better utility** preservation, while a lower α increases randomness, enhancing anonymity but potentially decreasing utility." | HIGH |

> **⭐ Direct read for VoiceForge's CFG dial (scope §4.2 option 3):** T10's α *is* the adherence↔diversity dial, already built and characterised. The mean-voice direction is the **high-utility, low-diversity** end. This gives the dial a published precedent and a sane default region, and confirms [`01-ttv-landscape.md`](01-ttv-landscape.md)'s finding that the trade is real and bidirectional.

---

## 5. Interpolation — does it work?

| Work | ID / venue | Result | Conf. |
|---|---|---|---|
| **Gabryś et al. (Amazon)**, residual encoder + normalizing flows | [arXiv:2106.05762](https://arxiv.org/abs/2106.05762), Interspeech 2021 | **The most on-point interpolation experiment found.** 50 speakers, **all 1,225 pairs**, **polar interpolation**, 50 sentences per new voice. Interpolants **fully intelligible**: WER **6.42%** (baseline) vs **7.35%** (flow-normalized). Flow-normalized gives **13.11% lower FAR** (more distinct). Authors: "**the interpolated speakers in the baseline are less distinctive and resemble much more to one of the two actual speakers**, hence its intelligibility is naturally higher at the cost of a worse distinctiveness." Hypothesis: "the **normalized** speaker embedding space has a **denser latent distribution** … hence the generalization to new speakers is better." | HIGH |
| **Latent Filling** | [arXiv:2310.03538](https://arxiv.org/abs/2310.03538), ICASSP 2024 | ZS-TTS trained with `s̃ = λs_i + (1−λ)s_j`, λ~Beta(β,β), + Gaussian noise. Needs a bespoke **latent-filling consistency loss** because there is no ground-truth audio for an interpolated embedding. SECS 0.827→**0.836**, SMOS 3.20→**3.34**, WER 1.02→1.10%. Two decisive negatives: **(i) "we observed that extrapolation is not particularly meaningful in our case"**; **(ii) "ensuring s_j has the same language information as s_i is crucial for achieving stable performance."** Ablation: removing interpolation costs CSMOS −0.115, WER +0.25. | HIGH |
| **INSIDE** | [arXiv:2508.19210](https://arxiv.org/abs/2508.19210), APSIPA ASC 2025 | **SLERP**, "**because it better fits the hyperspherical geometry** of embedding spaces and **preserves unit norm**." Interpolants → TTS → ASV training data: **3.06–5.24% relative EER improvement**, 13.44% on gender classification. Geometry: random pairing "results in **uneven coverage** … Interpolated embeddings tend to **cluster in specific regions**, leaving peripheral areas underrepresented" → adopts layered **nearest-neighbour, same-gender** pairing. | HIGH |
| **Eigenvoice Synthesis via Model Editing** | [arXiv:2507.03377](https://arxiv.org/abs/2507.03377), Interspeech 2025 | Argues the right speaker space is the **DNN parameter space**, not the embedding: "the effective way to define this speaker space remains unclear." Found a **gender-dominant axis**. Explicitly positions against methods that "simply interpolate between two base models." | MEDIUM |
| **VoiceMe** | [arXiv:2203.15379](https://arxiv.org/abs/2203.15379), Interspeech 2022 | Humans interactively navigate a SpeakerNet embedding space to build voices matching faces/portraits/cartoons. Independent raters confirm match; gender recovered; participants **converge consistently** toward the real voice prototype. Direct evidence the space is **human-navigable by directed search**. | MEDIUM |

---

## 6. The decisive positive and negative results, side by side

### 6.1 POSITIVE — a uniformly random vector *is* a valid voice

Jia et al., NeurIPS 2018, [arXiv:1806.04558](https://arxiv.org/abs/1806.04558) §3.6 "Fictitious speakers", verbatim:

> "Bypassing the speaker encoder network and conditioning the synthesizer on **random points in the speaker embedding space** results in speech from fictitious speakers … 10 such speakers, generated from **uniformly sampled points on the surface of the unit hypersphere** … **Even though these speakers are totally fictitious, the synthesizer and the vocoder are able to generate audio as natural as for seen or unseen real speakers. The low cosine similarity to the nearest neighbor training utterances and very high EER indicate that they are indeed distinct from the training speakers.**"

Table 6: naturalness MOS **3.65 ± 0.06**; cosine similarity to nearest neighbour **0.222** (SV-EER 56.77%) / **0.245** (SV-EER 38.54%).

> **Crucial nuance:** this d-vector encoder emits **L2-normalized** 256-d embeddings, so uniform sampling on the unit sphere **matches the real norm exactly**. It does *not* match the real *direction* distribution (real speakers occupy a limited cone) — and it still worked. This is the strongest single counter-argument to "the space is mostly off-manifold," and it isolates **norm-matching** as the property that matters.

### 6.2 NEGATIVE — a parametric prior over a discriminative space samples off-manifold

TacoSpawn — Stanton et al., "Speaker Generation," [arXiv:2111.05095](https://arxiv.org/abs/2111.05095) §6.2. Mixture-of-10-Gaussians prior over speaker embeddings. Metrics: **s2s** = median d-vector distance between synthesized *training* speakers (the reference), **g2s** = generated-to-training, **g2g** = generated-to-generated. **Target: g2s ≈ g2g ≈ s2s.**

| approach | dim | s2t-same | s2t | **s2s** | **g2s** | **g2g** |
|---|---|---|---|---|---|---|
| TacoSpawn (learned embedding) | 128 | 0.23 | 0.34 | 0.20 | **0.20** | **0.20** |
| TacoSpawn (learned embedding) | 256 | 0.22 | 0.34 | 0.20 | **0.23** | **0.18** |
| **d-vector (speaker-discriminative)** | 256 | 0.23 | 0.34 | 0.20 | **0.35** | **0.27** |

Verbatim:
> "d-vectors capture the training speakers reasonably well (s2t-same), though still not as well as learned vectors … However **d-vectors appear to be much less amenable to modeling with a simple parametric prior (g2s and g2g much too large). We therefore use learned vector embeddings.**"

> **Read this carefully: g2s = 0.35 vs s2s = 0.20 means sampled d-vectors land nearly twice as far from any real speaker as real speakers are from each other.** This is the sharpest quantitative evidence that a **discriminatively-trained ASV space is not a well-behaved generative manifold under a simple prior**.

TacoSpawn also reports that with a *learned* space, generated speakers are indistinguishable from real ones in naturalness MOS (libriclean: training 3.37±0.14 vs **generated 3.54±0.14**; en1468: 3.68 vs 3.62; gb: 3.69 vs 3.51; au: 3.30 vs 3.03), matching real F0 distributions with clear male/female clustering.

### 6.3 Why: discriminative ≠ generative

Ulgen et al., "Rethinking Speaker Embeddings for Speech Generation: Sub-Center Modeling," [arXiv:2407.04291](https://arxiv.org/abs/2407.04291) (v4 Aug 2026; accepted Interspeech 2026 per comment — venue MEDIUM):

> "Speaker embeddings are commonly used to condition personalized speech systems, but they are **typically trained for speaker recognition, where intra-speaker variability is suppressed and inter-speaker separation is maximized. This objective leads to overly compact representations that may discard variations crucial for generation.**"
> "…encouraging all utterances to **collapse** [toward a single prototype] … While effective for discrimination, it is **misaligned with the objective of generation**."

Quantified fix (sub-center ECAPA-TDNN, multiple prototypes per speaker): intra/inter-class variance ratio 0.42→0.47 (VCTK), 0.66→0.91 (VoxCeleb1-E) **while EER improves** 1.71→1.55% / 1.46→1.21%. Downstream zero-shot VC: **WER 14.84→13.93%, CER 6.82→6.41%, SECS 64.04→64.59%**, F0 std 8.03→10.25, F0 range 52.37→57.09, best naturalness MOS. Conclusion: "higher variance favors intelligibility, while lower variance [favors speaker similarity]."

---

## 7. Consolidated geometry of the space

| Property | Evidence | Conf. |
|---|---|---|
| **Per-dimension scales are wildly heterogeneous** — smallest-range dim spans [−0.97, −0.11]; largest spans [−72.37, 81.59] on a 704-d ECAPA+x-vector concat | Meyer et al. IS2022 §2.2 | HIGH |
| **Averaging shrinks vectors toward the origin** (dims ~zero-centred) → "**collapse of different speakers into one**" | Meyer et al. IS2022 §5.1.1 | HIGH |
| **Averaged vectors "underutilize their multi-dimensional vector space"**; cross-cosine distribution visibly shifted vs real | Turner et al. §3.1 Fig. 1 | HIGH |
| **Gender is strong cluster structure** — "clear clustering of the two genders in x-vector space using both cosine and PLDA"; opposite-gender averaging costs WER 6.83→7.19% | Srivastava §2.5, Table 1 | HIGH |
| **Density varies enormously; low-density regions synthesize badly** — 1,160 speakers → 80 clusters of 6–36; sparse-cluster averaging costs **WER 6.83 → 10.94%** | Srivastava §2.4, Table 1 | HIGH |
| **Hyperspherical geometry** — SLERP preferred over LERP because it "preserves unit norm" | INSIDE §III-B | HIGH |
| **Interpolants cluster centrally** — random pairing gives "uneven coverage … leaving peripheral areas underrepresented" | INSIDE §III-B Fig. 2 | HIGH |
| **Extrapolation outside the real-embedding hull does not work** | Latent Filling §2.2 | HIGH |
| **Diversity ⟂ in-distribution-ness are in direct tension** — MMD 0.005 → 0.17 as top-5 cosine falls 0.80 → 0.40 | NouveauVoice §V Table II | HIGH |
| **Vocoder drift: input vector ≠ output vector** | Panariello et al. IS2023 | HIGH |
| **Concatenating heterogeneous encoders helps** — ECAPA + x-vector "**significantly increased the variance in samples generated from random speaker embeddings**" | Meyer et al. IS2022 §5.2.1 | HIGH |
| **Hubness** in speaker-embedding space | **UNVERIFIED** — no primary source; all hits were text-retrieval literature | — |

---

## 8. Zero-shot TTS degradation under OOD conditioning

| Work | ID / venue | Finding | Conf. |
|---|---|---|---|
| Cooper et al., Zero-Shot Multi-Speaker TTS with SOTA Neural Speaker Embeddings | [arXiv:1910.10838](https://arxiv.org/abs/1910.10838), ICASSP 2020 | "there remains a gap for zero-shot adaptation to unseen speakers." Also: **a better SV EER does not automatically mean a better generation embedding** — LDE + angular softmax beats x-vectors on both, but they are separable axes. | HIGH |
| Jia et al. | [arXiv:1806.04558](https://arxiv.org/abs/1806.04558) | Seen/unseen naturalness **3.89 / 4.12**, similarity **3.28 / 3.03**; embedding-table (seen-only) baseline similarity **3.70** → a **~0.4–0.7 MOS similarity penalty** the moment you condition on an embedding unseen in synthesizer training. | HIGH |
| Latent Filling | [arXiv:2310.03538](https://arxiv.org/abs/2310.03538) | Frames the core problem: "**it is impossible to calculate the [reconstruction loss] because there is no corresponding ground-truth speech for the augmented speaker embedding**." Fix = consistency loss. **Latent-space** augmentation improves SMOS +0.14/+0.12; **input-level** augmentation degrades everything. Sensitive τ: "When τ was set too high, the TTS system **generated speech that lacked coherence with the input text**." | HIGH |
| VPC 2024 B1 vs B3 | [arXiv:2601.11846](https://arxiv.org/abs/2601.11846) Table 9 | Averaged-real embedding → WER 2.91%; **fully synthetic GAN embedding → WER 4.35%** vs original 1.85%. **~1.4-point absolute WER premium for a fully synthetic conditioning vector.** | HIGH |
| VPC 2026 Track 2 BM1 | VPC2026 plan Table 8 | BM1 EER on MLS-en eval = **3.40%** vs original **4.21%** — *worse than no anonymization*. Extreme case of the pseudo-speaker failing to be a real, distinct identity. | HIGH |
| "Your Voice Cloning System is Secretly a Voice Anonymizer" | [arXiv:2608.27360](https://arxiv.org/abs/2608.27360) (2026-08-27) | Repurposes XTTSv2 (27k h) with a pseudo-speaker condition, **no retraining**: EER ≈ 0.49, "**substantially better speech quality than dedicated anonymization baselines**" across 7 languages. → **use a big modern cloning model as the renderer rather than a bespoke pipeline.** | MEDIUM (abstract) |

---

## 9. Verdict on A2, and what to build

### 9.1 The verdict

**A2 is answered YES — with three non-negotiable constraints.** The two-tower design survives. The scope document's fear ("the manifold may only be navigable by extraction, not by synthesis") is not borne out: synthesis into speaker space is an established, benchmarked technique with an official challenge baseline.

### 9.2 The three constraints

1. **Match the per-dimension distribution, not just the semantic intent.**
   Averaging zero-centred dimensions shrinks dynamic range → all voices collapse (GVD −6.5 to −11.6 dB). Meyer et al. fixed it with **per-dimension rescaling to each dimension's observed empirical range**, recovering GVD to ≈ −0.1. If the encoder is normalized, sample on the unit hypersphere. If raw x-vectors, rescale per-dimension or fit a generative model. **Highest-leverage, cheapest fix available.**
2. **Stay in dense regions, or accept a quality cliff.**
   Sparse-region averaging costs **+60% relative WER**. Interpolate between **nearest-neighbour, same-gender** speakers. **Never extrapolate.** Cross-gender and cross-language mixing are documented destabilizers.
3. **Prefer a space the generator was trained to accept over a raw ASV space.**
   TacoSpawn: d-vector prior → g2s 0.35 vs s2s 0.20 (off-manifold); learned space → 0.20/0.20. ASV encoders *suppress* the intra-speaker variance generation needs. If an ASV space is unavoidable: concatenate heterogeneous encoders, or add a consistency loss.

> **Constraint 3 is why [`03-tts-backends-english.md`](03-tts-backends-english.md)'s finding matters so much.** Qwen3-TTS's `(2048,)` speaker embedding has `enc_dim == hidden_size` — it lives in **the backbone's own embedding space**, jointly trained with the generator. That is precisely the "learned space" TacoSpawn found works, rather than the bolt-on ASV space TacoSpawn found does not. This is a stronger reason to prefer Qwen3-TTS than its licence.

### 9.3 "Design a vector" vs "generate a seed waveform and clone"

**Vector design is viable and is what the state of the art does** — VPC 2026 baseline B3 is exactly this. Two things push toward a hybrid:

- **Vocoder drift** means you do not get the voice you specified. If the product promise is "this vector = this voice," you need a **closed loop**: synthesize → re-extract → compare → correct. That is a real cost the vector path carries and the clone-from-waveform path largely does not — and it **erases Tier 1's free-CPU-mint advantage** from scope §12.2.
- **A designed vector is not human-steerable out of the box.** The only documented interpretable axes are **gender** (Srivastava; Eigenvoice's "gender-dominant axis") and **F0 range** (TacoSpawn). VoiceMe achieved interpretable control only via *human-in-the-loop search*, never by writing numbers.

### 9.4 Recommended architecture, from the evidence

1. Fit a **lightweight generative model** (GMM → WGAN → hierarchical VAE, in ascending order of cost) over the chosen encoder's **real-speaker embeddings**. Never sample raw Gaussian noise in an unnormalized x-vector space.
2. **Per-dimension rescale** every synthesized vector to each dimension's observed empirical range. Non-negotiable.
3. Enforce a **minimum cosine distance from existing identities** — B3 uses **0.3**. This is your catalog-uniqueness guarantee, free.
4. **Verify by re-extraction**: render, re-extract the embedding, compare to the intended vector, correct or reject. Log the drift.
5. Expose voice design to users as **constrained navigation within the fitted distribution**, never free numeric entry.
6. Prefer a **generator-native embedding space** (Qwen3-TTS) over a bolt-on ASV space.
7. Score diversity with **GVD**, which has a published scale and baselines, alongside the Vendi score from [`06-evaluation-harness.md`](06-evaluation-harness.md).

---

## 10. Open — must be settled by experiment

| # | Question | Cheapest experiment | Est. | Blocks |
|---|---|---|---|---|
| 12-O1 | Does Qwen3-TTS's `(2048,)` space behave like TacoSpawn's *learned* space (good) or its *d-vector* space (bad)? | Extract embeddings for ~500 LibriTTS-R speakers; fit a 10-component GMM; sample 200; compute **s2s / g2s / g2g** exactly as TacoSpawn §6.2. Target g2s ≈ s2s. | ~4 GPU-hrs | **Everything.** This is the single most important experiment in the project. |
| 12-O2 | How large is vocoder drift on Qwen3-TTS? | Render 100 identities, re-extract, measure cosine(input, output). Report mean/σ/worst. | ~2 GPU-hrs | Whether Tier 1 needs a closed loop; the mint cost model |
| 12-O3 | Does per-dimension rescaling matter for this space, or is it already normalized? | Dump 1,000 real embeddings; plot per-dim min/max/σ. If ranges span >1 order of magnitude, rescaling is mandatory. | **~20 minutes, CPU** | The mapper's output layer. Do this first. |
| 12-O4 | What is `C_same` and `C_diff` for this encoder? | 50 speakers × 20 utterances; compute intra- and inter-speaker cosine distributions. | ~1 GPU-hr | Every identity-consistency threshold in [`06`](06-evaluation-harness.md) |
| 12-O5 | Where are the low-density regions, and do they map onto our target descriptions? | Cluster the embedding corpus; check density at "elderly", "raspy", "very low-pitched" exemplars. | ~2 GPU-hrs | Whether the §5 voice range is reachable at all; per-demographic eval slices |
| 12-O6 | What do the VPC 2026 results say? | **Re-check after 2026-09-26** (Sydney workshop, SPSC/Interspeech). | free | Nothing — but likely supersedes several choices here |

---

## 11. Sources

| # | URL | Type | Used for | Conf. |
|---|---|---|---|---|
| 1 | [arXiv:1806.04558](https://arxiv.org/abs/1806.04558) | Paper, NeurIPS 2018 (PDF read) | **Random unit-hypersphere vectors → MOS 3.65** | HIGH |
| 2 | [arXiv:2207.04834](https://arxiv.org/abs/2207.04834) | Paper, Interspeech 2022 (PDF read) | **The collapse mechanism and the per-dim rescaling fix** | HIGH |
| 3 | [arXiv:2111.05095](https://arxiv.org/abs/2111.05095) | Paper, TacoSpawn (PDF read) | **d-vector space resists parametric priors (g2s 0.35 vs 0.20)** | HIGH |
| 4 | [arXiv:2210.07002](https://arxiv.org/abs/2210.07002) | Paper, IEEE SLT 2022 (PDF read) | WGAN embeddings; GVD ≈ 0; 85.63% human same/diff | HIGH |
| 5 | [arXiv:2010.13457](https://arxiv.org/abs/2010.13457) | Paper, VPC2020 (PDF read) | Fake x-vectors "underutilize their vector space" | HIGH |
| 6 | [arXiv:2005.08601](https://arxiv.org/abs/2005.08601) | Paper, Interspeech 2020 (PDF read) | B1 parameters; sparse-region WER cliff; gender clustering | HIGH |
| 7 | [arXiv:2205.07123](https://arxiv.org/abs/2205.07123) | Official VPC 2020 Evaluation Plan (PDF read) | B1 procedure; WER table | HIGH |
| 8 | [arXiv:2109.00648](https://arxiv.org/abs/2109.00648) | Official VPC 2020 results (PDF read) | Weak-diagonal similarity matrices | HIGH |
| 9 | [arXiv:2407.11516](https://arxiv.org/abs/2407.11516) | Official VPC 2022 progress, TASLP (PDF read) | Naturalness ranges; vocoder-drift summary | HIGH |
| 10 | [arXiv:2601.11846](https://arxiv.org/abs/2601.11846) | Official VPC 2024 results (PDF read) | B1 vs B3 WER; B3 cosine-0.3 constraint | HIGH |
| 11 | [VPC 2026 Evaluation Plan](https://www.voiceprivacychallenge.org/vp2026/docs/VPC_2026_sept01.pdf) | Official (PDF read) | **B1 retired**; current baselines; 2026-09-26 workshop | HIGH |
| 12 | [arXiv:2106.05762](https://arxiv.org/abs/2106.05762) | Paper, Interspeech 2021 | **1,225 pairwise interpolations; WER 6.42–7.35%** | HIGH |
| 13 | [arXiv:2310.03538](https://arxiv.org/abs/2310.03538) | Paper, ICASSP 2024 | Latent Filling; **extrapolation does not work** | HIGH |
| 14 | [arXiv:2508.19210](https://arxiv.org/abs/2508.19210) | Paper, APSIPA ASC 2025 | SLERP; nearest-neighbour same-gender pairing | HIGH |
| 15 | [arXiv:2407.04291](https://arxiv.org/abs/2407.04291) | Paper (PDF read) | **Discriminative ≠ generative**; sub-center fix | HIGH |
| 16 | [arXiv:2607.03985](https://arxiv.org/abs/2607.03985) | Paper, 2026-07 | **Diversity ⟂ in-distribution tension, quantified** | HIGH |
| 17 | [arXiv:2308.11337](https://arxiv.org/abs/2308.11337) | Paper, SPSC 2023 | Unnaturalness is the vocoder's fault, not the vector's | HIGH |
| 18 | [DOI 10.21437/Interspeech.2023-448](https://doi.org/10.21437/Interspeech.2023-448) | Paper, Interspeech 2023 (ISCA verified) | **Vocoder drift** | HIGH |
| 19 | [DOI 10.21437/Interspeech.2020-2720](https://doi.org/10.21437/Interspeech.2020-2720) | Paper, Interspeech 2020 (ISCA verified) | **GVD metric definition** | HIGH |
| 20 | [arXiv:1905.13561](https://arxiv.org/abs/1905.13561) | Paper, SSW10 2019 (ISCA verified) | Original x-vector anonymization method | HIGH |
| 21 | [arXiv:2409.04173](https://arxiv.org/abs/2409.04173) | Paper, VPC2024 entry T10 | **The explicit α adherence↔diversity knob** | HIGH |
| 22 | [arXiv:2405.10786](https://arxiv.org/abs/2405.10786) | Paper, IEEE/ACM TASLP | Critique of utterance-level averaged ASV vectors | HIGH |
| 23 | [arXiv:1910.10838](https://arxiv.org/abs/1910.10838) | Paper, ICASSP 2020 | Better SV EER ≠ better generation embedding | HIGH |
| 24 | [arXiv:2511.07135](https://arxiv.org/abs/2511.07135) | Paper | SpeakerVAE | MEDIUM |
| 25 | [arXiv:2507.03377](https://arxiv.org/abs/2507.03377) | Paper, Interspeech 2025 | Parameter-space speaker modelling; gender axis | MEDIUM |
| 26 | [arXiv:2203.15379](https://arxiv.org/abs/2203.15379) | Paper, Interspeech 2022 | VoiceMe: human-navigable by directed search | MEDIUM |
| 27 | [arXiv:2608.27360](https://arxiv.org/abs/2608.27360) | Paper, 2026-08 | Big cloning model > bespoke pipeline | MEDIUM |
| 28 | [Voice-Privacy-Challenge-2026](https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2026) | Official repo | Current baselines | HIGH |

---

## 12. Caveats on this document

- **VPC 2020 per-system subjective naturalness (B1 ≈ 0.29) and VPC 2022 Fig. 6 values** were extracted from PDF figure text where axis labels and value sequences interleave. The *gap* (roughly a halving vs unprotected) is HIGH confidence; the *per-system assignment* is **MEDIUM**. Read Fig. 16 of [2109.00648](https://arxiv.org/abs/2109.00648) and Fig. 6 of [2407.11516](https://arxiv.org/abs/2407.11516) visually before quoting these.
- **Turner et al. venue** — arXiv preprint + VPC 2020 submission; no ISCA/IEEE proceedings entry found.
- **VPC 2026 results do not exist yet.** Workshop is 2026-09-26. Only the evaluation plan and baseline numbers are available.
- **Hubness** in speaker-embedding space: **UNVERIFIED**, no primary source found.
- **Every number here comes from pipelines built on 2019–2024 vocoders.** Modern backends should perform better in absolute terms; treat these figures as *relative* evidence about geometry, not as VoiceForge's expected quality.

---

*Pass 1 · 2026-09-02 · Every arXiv ID fetched and confirmed. Quantitative claims read from PDF text and tables.*
