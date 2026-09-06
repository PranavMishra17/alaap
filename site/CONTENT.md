# Alaap landing page — content pack (the single source of truth)

Every claim, number and clip on every variant comes from this file. Sources are `file:line`
relative to the repo root (`E:\VoiceForge TTV Pipeine\v3`). `HANDOFF-WEBSITE.md` is the owner's
brief; where it supplies plain-English wording, that wording is pre-approved.

## 0. Rules for anyone writing a page from this pack

1. **Numbers.** Only the numbers in §9 may appear, in the wordings §9 allows. No other digits about
   the project anywhere on the page: not in copy, not in decorative stats, not in alt text.
2. **Claims.** Only claims from §§1–8. Rephrase for rhythm; never strengthen. When unsure, use the
   approved wording verbatim. Every "achieved" claim keeps its caveat within reach on the page.
3. **Mandatory on every page:** §1 name, meaning, plain one-liner and the example description ·
   §4 all four differentiators · §6 every item (compress, never drop) · the §7 shape of the system ·
   at least three §8 clips with their exact text, the watermark note, the emotion-setting caveat
   and the English-only caveat · the repository link · one honest call to action from §11.
4. **Banned words and register:** §10.
5. **Never fabricate:** testimonials, logos, team members, pricing, a roadmap with dates, an email
   address, a signup, or any interactive "demo" that pretends to generate a voice live. An
   interactive "describe a voice" moment is fine only if it visibly replays the pre-rendered clips
   from §8 and says so on screen.
6. The page is for a creator or studio lead, not an engineer. Assume intelligence, not vocabulary.

---

## 1. Identity

| item | approved content | source |
|---|---|---|
| Name | **Alaap** · Devanagari **आलाप** | README.md:1 · DECISIONS.md:14 |
| Meaning | "the opening improvisation in Hindustani classical music, where a voice explores its full range before the composition begins" | README.md:5 |
| Meaning, extended | "a voice mapping out its own possibility space. That is exactly what description-to-voice design does." | DECISIONS.md:16 |
| Plain one-liner | "You describe a character in ordinary language. You get back a voice that has never existed. Then you use that voice for any line of dialogue you write, and it stays the same character every time." | HANDOFF-WEBSITE.md · README.md:3 |
| Tighter one-liner | "Describe a character. Get a voice nobody has recorded. Keep it." | derived from README.md:3 |
| The example | **"a very deep voice, very rough and gravelly, speaking very slowly"** → a voice nobody has recorded → say anything you like, in Hindi or English | S6 RESULTS.md:12 · demo_out/manifest.json (the demo's "mentor" uses this exact description, so it maps to a playable clip: §8) |
| Languages proven | English, Hindi, Bengali, Tamil | README.md:16, 128 |
| Engine coverage | the Indian-language engine covers 22 Indian languages; four are measured so far | README.md:128 |
| Repository | https://github.com/PranavMishra17/alaap | brief |
| Licence | research and documentation CC-BY-4.0; code MIT; no model weights and no generated Indian-language audio ship | README.md:157, 159 |
| Dates | research pass 2026-09-02 · implementation and measurement 2026-09-05 | README.md:163 |

## 2. Who it is for

"People making games, animation, audio drama, dubbing, interactive fiction: anyone who needs many
distinct speaking characters and cannot hire a cast for each one." (HANDOFF-WEBSITE.md; grounded in
README.md:3 "quality usable in games and dramatic content".)

## 3. Why Indian languages specifically

- "Indic is the stated reason the project exists." (DECISIONS.md:136)
- The Indian-language tools that exist ship a fixed menu of voices, and every project draws from
  the same menu. (S8 RESULTS.md:10–11: "Every commercial Indic TTS ships a curated voice library
  rather than per-request speaker synthesis.")
- Approved plain wording: "The tools that do this well mostly do it in English. Indian-language
  creators either use a handful of stock voices that everyone else is also using, or they hire
  actors. This project treats Hindi, Bengali and Tamil as first-class, not as an afterthought port."
  (HANDOFF-WEBSITE.md)
- Optional: in the project's own measurements the Hindi path scored higher than the English path on
  all three identity checks (S6 RESULTS.md:47–51).
- Optional: the Indian-language speech data the project builds on holds more distinct speakers than
  any English set it has, and its licence allows commercial use (DECISIONS.md:61).

## 4. The four things that make it different — all four are mandatory

### 4.1 A voice is a thing you keep, not a setting you re-guess
"Most tools generate a voice each time you ask. Here the character's voice is stored as an identity
and re-used. The same character sounds like the same person across every line, every scene, and
every session. That consistency is measured, not hoped for."
Sources: README.md:3 (persistent, reusable identity) · DECISIONS.md:118 (every identity stores the
voice, a short seed clip, and the engine version that made it) · S6 RESULTS.md:53–62 (consistency
measured).

### 4.2 Who is speaking and what is being said are genuinely separate
"The system holds a character's identity apart from the words. That is why one character can say
anything without their voice drifting, and why changing the line never changes the person. This was
verified rather than assumed: when the same audio content is played through two different
identities, the words come out identically. The measured error between them was exactly zero on
every line tested."
Sources: README.md:30–34 (word-error spread between identities 0.000 on every line) · S6
RESULTS.md:90–105 (identical content decoded through six different identities) · README.md:31–32,
124 (checked by a separate speaker-recognition system that has never seen the generator).

### 4.3 It does not need a recording of the character
"There is no voice cloning here, and no user uploads their own audio. Voices are created from a
written description. That sidesteps the consent and impersonation problem entirely: you cannot
accidentally clone a real person if the system never accepts a recording."
Sources: DECISIONS.md:122 and README.md:139 ("No user audio upload in the public product. Ever.") ·
S6 RESULTS.md:10–12 (cloning needs a recording of every character; the product claim is different).

### 4.4 It knows what it does not know
"Every capability is measured against a control, and the honest limits are published alongside the
results. Roughly twenty findings in this project were wrong on the first attempt and caught by a
check, including one where an entire experiment produced fluent-sounding speech that was saying
different words than requested, and nothing errored. That is written up publicly rather than quietly
fixed."
Sources: README.md:96–99 · README.md:17 (every experiment states what was *not* established) · S8
RESULTS.md:105–119 and S9 RESULTS.md:49–62 (a published claim retracted, with the reasoning left in
place).
Quotable rule: "A suspiciously good or suspiciously bad number is a bug report about your setup."
(README.md:101)

## 5. What is achieved — each with its caveat

### 5.1 It works end to end today
"It works end to end today, on one laptop GPU, in English, Hindi, Bengali and Tamil. Describe a
character, get a voice, render a scene of dialogue, and out comes watermarked audio plus a file a
game engine can read." (README.md:9–16; "one 6 GB laptop GPU" README.md:11.) The demo mints three
characters from text, renders a five-line scene, watermarks everything and writes the manifest in
about four minutes (README.md:22–26). The clips in §8 are that demo's output.

### 5.2 Minting a voice from a description costs nothing in clarity
"A described voice is just as intelligible as one taken from a real recording. The measured
difference was tiny, leaned slightly in the invented voice's favour, and was smaller than the
difference between two real speakers." The project reads that as **no detectable difference**, not
as invented voices being better. The gap between two *real* speakers was six times the gap between
invented and real. (S6 RESULTS.md:96–105, 132.) The control that had to pass first: every version
must come out the same length before any score is read, or the run aborts (S6 RESULTS.md:116–121).

### 5.3 Invented voices are recognisably distinct people
"Verified by a separate speaker-recognition system that has never seen the generator: an
independent check, not the system marking its own homework." Each invented voice also sits clearly
apart from the nearest real voice the system learned from, so these are new voices, not real people
wearing a label. (S6 RESULTS.md:64 · README.md:31–32, 124.)

### 5.4 A character stays itself more consistently than a real person does
"A character stays itself across different sentences more consistently than a real person does
across different recordings, which is unsurprising once you think about it: a stored voice has no
bad-throat day, no change of microphone, no mood." (S6 RESULTS.md:53–62.)

### 5.5 A native speaker confirmed the Hindi says what it should
"A fluent Hindi speaker has confirmed one rendered line says the target sentence. They were told the
sentence and asked 'does it say this?', not 'does it sound right?'. That is the check the earlier
wrong-codec audio failed." Still to be checked: the other three voices and the other two lines.
(S6 RESULTS.md:133.) The sentence, displayable: **नमस्ते आप कैसे हैं आज मौसम बहुत अच्छा है**
(S6 RESULTS.md:110). A plain translation, which is ours and not a project claim: "Hello, how are you?
The weather is very nice today."

### 5.6 A blind listening test found and fixed a real problem
"A blind listening test with built-in controls found a real problem in how the system judged whether
two voices were the same person: pairs the system called different were heard as the same person
about half the time. The threshold was raised. That removed 10 duplicate voices from a set of 50
and cost 3 percent of measured variety." Caveat, keep nearby: the test was small, 8 pairs and one
listener with every control answered correctly; it found the problem but does not settle the exact
threshold, and the English threshold has not had its own listening test. (README.md:82–89 · S9
RESULTS.md:131–133 · DECISIONS.md:311–324.)

### 5.7 Pacing is controllable within a measured range
"You can slow a character to about two-thirds of normal speed or push them to about 1.4 times, and
they stay the same person saying the same words. Outside that range the identity fades gradually
rather than breaking." A listener noted the slow extreme starts to sound processed; the range they
called clean is narrower, about 0.8 to 1.25 times. (README.md:63–71 · DECISIONS.md:380–398.)

### 5.8 Two ways to answer a description, and the numbers for the trade-off
"There are two ways to answer a description, and the system measures both: invent a new voice, or
find the closest match in an existing library of voices. Inventing gives you a character nobody
else has; finding gives you higher accuracy and studio quality. Neither is strictly better and the
project has numbers for the trade-off."
- Finding, on a library of 141 real Hindi voices: two thirds of descriptions (65.6%) get a voice
  that matches on every trait named, and 87% land within one step. When the description names only
  a couple of traits, the way people actually write, 53.8% match exactly on the traits named. Random
  chance is about 20%. The same method on a library of 2,500 English voices reaches 88.5%. It
  replicates in Bengali within a point or two. (S8 RESULTS.md:25–28, 68–74, 126–134 · README.md:48–50.)
- Reach: finding reaches 87% of the variety the library actually holds; inventing reached 37%
  before two mis-set settings were fixed, and the fix raised the invented catalogue from 14 to 22
  distinct Hindi voices and from 23 to 41 English. (S9 RESULTS.md:38–47 · README.md:41–46 ·
  DECISIONS.md:424.)
- What finding cannot do: produce a voice the library does not contain; guarantee a new character
  is unique (two descriptions can return the same voice); and the library is someone else's, so
  every character in every project draws from the same shelf. (S8 RESULTS.md:147–154.)
- Caveat: the library measured was real recorded speakers standing in for a studio library, and no
  commercial system has been measured. (S8 RESULTS.md:157–169 · README.md:91–92.)

### 5.9 Watermarked and logged from the first render
"Every render is watermarked and logged, from the first output, not added later." (DECISIONS.md:123
· README.md:140.) The watermark is AudioSeal (README.md:125). "Output includes a manifest a game
engine can consume: this is built for a pipeline, not a demo button." (README.md:15, 26.) The
manifest's own disclosure line, quotable: "AI-generated synthetic speech. All audio carries an
AudioSeal watermark." (demo_out/manifest.json; outside the five core docs but a real artefact.)

### 5.10 Smaller facts, optional
- 114 automated invariant tests guard the rules the project learned the hard way (README.md:18).
- The identity is who the character is; each line is a separate performance (DECISIONS.md:121 ·
  README.md:138).
- The rules that matter, one line each (README.md:132–143): never accept a user's audio; watermark
  from the first render; store timbre, not performance; do not start the next stage until this one
  is measured (DECISIONS.md:124).

## 6. What is honestly not there yet — every item is mandatory; compress, never drop

1. **Not a product.** "This is a working research system, not a product. There is no signup, no
   hosted service, no API." (README.md:19 · DECISIONS.md:246.)
2. **Emotion is not solved.** "Emotional performance direction is not solved. You can control pacing
   within a measured range. You cannot yet say 'say this line apologetically' and have it work. That
   was tested thoroughly and does not work on the current engine, for reasons the project
   documented and reported upstream." Detail allowed: the engine's documented emotion tags turn out
   not to be recognised by the engine at all, and a listener heard no difference between four tagged
   versions of the same line. (README.md:52–61 · DECISIONS.md:359–367, 400–401.)
3. **The catalogue is small.** "Tens of distinct voices per language so far, not hundreds": about 22
   in Hindi and 41 in English by the project's own measure. "The limit is understood and measured
   rather than mysterious": the real speakers the system learned from hold about 38 to 48
   distinguishable voices, and the project measured exactly where its own method loses variety.
   (README.md:41–46 · S9 RESULTS.md:25–47.)
4. **Cannot ship as a product, for licensing reasons.** "It cannot be shipped as a product in its
   current form for licensing reasons, which are documented." The Indian-language engine's training
   chain includes data licensed for non-commercial use, so no model weights and no Indian-language
   audio leave the repository. "It is research-only today, deliberately." (README.md:19, 159 ·
   DECISIONS.md:230–261.)
5. **Very little has been listened to.** Most results are automatic measurements. One listener, a
   handful of pairs. (README.md:82–89.)
6. **No comparison against a commercial system.** (README.md:91–92 · S8 RESULTS.md:167–169.)
7. **The clips on this page are English.** Hindi renders exist and a native speaker has checked one,
   but the Indian-language engine's licence is research-only, so those clips do not ship, not even
   here. (README.md:159 · DECISIONS.md:261 · S6 RESULTS.md:133.)
8. **Three of the five demo lines** were rendered with an experimental emotion setting whose effect
   has not been validated on this engine. Listen to them as voice samples, not performances.
   (demo_out/manifest.json, `degradations`.)

## 7. How it works, for a non-technical reader

```
your description  →  [ the mapper ]  →  a voice identity  →  [ the voice engine ]  →  audio
                      the only part                            never modified,
                      that learns                              swappable
```
(DECISIONS.md:29: "description → [frozen text encoder + TRAINED mapper] → voice identity → [FROZEN TTS] → audio".)

Three ideas worth landing:
1. **Only one small piece is trained.** The speech engine is used as-is and never altered, so a
   better engine can be swapped in without rebuilding everything. (DECISIONS.md:29, 138.)
2. **The voice identity is a small, storable thing.** That is what makes a character persistent: it
   can be saved, versioned and handed to someone else. Each one is stored with a short seed clip and
   the engine version that made it. (DECISIONS.md:118 · README.md:135.)
3. **Two ways to answer a description, both measured.** Invent or find; see §5.8.

Also: the checker is a separate system that never sees the engine's internals (README.md:124).
Every render is watermarked and logged from the first output (§5.9). The output manifest is for a
game engine, not a demo button (§5.9).

## 8. The clips — real, watermarked, English, from the end-to-end demo run on 2026-09-05

Engine for these clips: Qwen3-TTS, Apache-2.0 (README.md:121). All eight carry the watermark
(demo_out/manifest.json). Files live in `public/audio/` and are exported from
`src/content/alaap.ts` as `clips` and `characters`, with `src` already prefixed for deployment.

Three characters, each minted from nothing but the description below. The seed clip is the short
reference recording the system stores alongside every identity when it is minted (DECISIONS.md:118).

| character | description given | seed clip | seed says |
|---|---|---|---|
| mentor | "a very deep voice, very rough and gravelly, speaking very slowly" | `mentor_seed.mp3` (2.2 s) | "This is how I sound when I speak." |
| scout | "a high voice, crisp-toned, speaking quickly and highly animated" | `scout_seed.mp3` (2.9 s) | "This is how I sound when I speak." |
| innkeeper | "a mid-range voice, warm-toned, at a steady pace" | `innkeeper_seed.mp3` (1.9 s) | "This is how I sound when I speak." |

The five-line scene:

| file | character | exact words | length | note |
|---|---|---|---|---|
| `mentor_line0.mp3` | mentor | "The mountains remember every footstep, even the ones you regret." | 4.3 s | no direction |
| `scout_line1.mp3` | scout | "There's smoke on the ridge. Two fires, maybe three." | 3.9 s | experimental emotion setting, unvalidated |
| `innkeeper_line2.mp3` | innkeeper | "Sit down, both of you. Nobody rides out on an empty stomach." | 3.5 s | no direction |
| `mentor_line3.mp3` | mentor | "Then we leave at first light, and we do not look back." | 4.6 s | experimental emotion setting, unvalidated |
| `scout_line4.mp3` | scout | "I told you exactly what would happen!" | 2.8 s | experimental emotion setting, unvalidated |

Useful pairings: the mentor's seed and two lines are one character across three different sentences
(§4.1). Mentor against scout is two invented voices from two descriptions (§5.3). The mentor's
description is the example in §1, so "the example" can be heard, not just read.

Labelling rules: every player shows the exact words. Somewhere near the players: the watermark line,
the English-only caveat (§6.7) and the emotion-setting caveat (§6.8). Never present a clip as a
performance of an emotion.

## 9. Numbers allowlist — the only numbers permitted on the page

| number | what it is | allowed plain wording | source |
|---|---|---|---|
| 4 languages: English, Hindi, Bengali, Tamil | languages measured | as listed | README.md:16, 128 |
| 22 | Indian languages the Indic engine covers | "covers 22 Indian languages; four measured so far" | README.md:128 |
| one 6 GB laptop GPU | hardware | "one laptop GPU", optionally "6 GB" | README.md:11 |
| 3 characters, 5 lines, about 4 minutes | the demo | as listed | README.md:22–26 |
| 0 / "exactly zero" | word-error spread between identities on every line | "exactly zero on every line tested" | README.md:33–34 |
| six times | gap between two real speakers vs gap between invented and real | "six times larger" | S6 RESULTS.md:103–104 |
| roughly twenty | findings wrong on first run, caught by a control | "roughly twenty" | README.md:96 |
| one line, one speaker | Hindi listening confirmation | "one rendered line", "a fluent Hindi speaker" | S6 RESULTS.md:133 |
| 8 pairs, one listener, controls all correct | the blind pilot | as listed; "about half the time" | README.md:83–89 |
| 10 of 50, 3 percent | duplicates removed, variety cost | as listed | README.md:85–86 · S9 RESULTS.md:132–133 |
| 0.67× to 1.43× | measured pacing range | "about two-thirds to about 1.4 times normal speed" | README.md:67 |
| 0.8× to 1.25× | listener's clean pacing range | "about 0.8 to 1.25 times" | DECISIONS.md:397 |
| 65.6%, 87%, 53.8%, about 20% | library search: exact match, within one step, sparse-query exact, random chance | "two thirds", "87 percent", "a bit over half", "about one in five by chance" | S8 RESULTS.md:25–28, 68–74 |
| 141 Hindi, 138 Bengali, 2,500 English | library sizes | as listed | S8 RESULTS.md:3, 126 · README.md:49 |
| 88.5% | library search on 2,500 English voices | as listed | README.md:49 |
| 87% vs 37% | share of available variety reached: finding vs inventing (before the fix) | as listed | S9 RESULTS.md:40–43 |
| 14 → 22 Hindi, 23 → 41 English | distinct voices before and after the fix | "about 22 in Hindi and 41 in English" | README.md:44–45 |
| about 38, about 48 | distinguishable voices held by 141 Hindi speakers; by 432 speakers across three languages | as listed | S9 RESULTS.md:27–30 |
| 26% | extra variety from tripling the speakers | "tripling the speakers bought about a quarter more variety" | S9 RESULTS.md:72 |
| 114 | automated invariant tests | "114 automated checks" | README.md:18 |
| 2026-09-02, 2026-09-05 | research pass; implementation and measurement; demo render date | as dates | README.md:163 · demo_out/manifest.json |
| clip lengths in §8 | durations | as listed, rounded to 0.1 s | demo_out/manifest.json |

Anything not in this table is not on the page. In particular, no similarity scores, no error rates as
decimals, no counts of experiments, no version numbers of models.

## 10. Banned words, and the register

Never use: embedding, vector, cosine, latent space, latent, PCA, Vendi, ECAPA, drift, corpus,
tokenizer, token, inference, codec, ASR, MOS, GMM, revolutionary, cutting-edge, AI-powered,
state-of-the-art, seamless, unleash, supercharge, next-generation, leverage, game-changing,
effortless, magic, "try it now", "sign up", "get started".

Say "voice engine" not "TTS backend"; "voice identity" or "the character's voice" not "speaker
embedding"; "checker" or "a separate speaker-recognition system" not "ECAPA"; "invent" and "find"
or "mint" and "retrieve" for the two methods; "the mapper" for the only trained part; "measured",
"checked", "verified" as plain verbs.

Register: "here is what we built, here is what we measured, here is what is missing." Confidence
without hype. State limits plainly and move on; do not soften them into marketing.

## 11. Calls to action — the only honest ones

- **Read the research** → https://github.com/PranavMishra17/alaap
- **Corrections welcome** → https://github.com/PranavMishra17/alaap/issues · README.md:153: "If a
  claim here is wrong, an issue with a primary source is more useful than almost anything else."
- **Get in touch** may only point at the repository or its issues. There is no email in the sources;
  do not invent one.
