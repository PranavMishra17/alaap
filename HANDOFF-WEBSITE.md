# Handoff prompt — user-facing landing page for Alaap

Copy everything below the line into a fresh agent session.

---

## Your task

Build a **makeshift user-facing landing page** for a project called **Alaap**. One page. It
explains what the project does, who it is for, how far along it is, and why anyone should
believe it works.

**I am specifying content, not style.** Design it however you think reads well. What I care
about is that the substance is right, honest, and legible to someone who is not an engineer.

### Read these first

The repository is at `https://github.com/PranavMishra17/alaap`. Read, in this order:

1. `README.md` — current status and headline results
2. `experiments/S6-indic-mint/RESULTS.md` — the core capability working
3. `experiments/S8-library-retrieval/RESULTS.md` — the alternative approach and how they compare
4. `experiments/S9-diversity-bound/RESULTS.md` — how capacity is measured honestly
5. `DECISIONS.md` — the locked design decisions and why

**Do not invent numbers.** Every figure on the page must come from those files. If you want
to say something and cannot source it, leave it out. This project's entire culture is that
unverified claims get retracted publicly, and a landing page that overstates would be
embarrassing in a way the work itself is not.

---

## What the project is, in plain terms

**You describe a character in ordinary language. You get back a voice that has never
existed. Then you use that voice for any line of dialogue you write — and it stays the same
character every time.**

> *"a very deep voice, very rough and gravelly, speaking slowly"* → a voice nobody has
> recorded → say anything you like, in Hindi or English

The word *alaap* is the opening improvisation in Hindustani classical music, where a voice
explores its full range before the composition begins.

**Who it is for:** people making games, animation, audio drama, dubbing, interactive fiction
— anyone who needs many distinct speaking characters and cannot hire a cast for each one.

**Why it matters for Indian languages specifically:** the tools that do this well mostly do
it in English. Indian-language content creators either use a handful of stock voices that
everyone else is also using, or they hire actors. This project treats Hindi, Bengali and
Tamil as first-class, not as an afterthought port.

---

## The four things that make it different — say these clearly

### 1. A voice is a thing you keep, not a setting you re-guess

Most tools generate a voice each time you ask. Here the character's voice is **stored as an
identity** and re-used. The same character sounds like the same person across every line,
every scene, and every session — that consistency is measured, not hoped for.

### 2. Who is speaking and what is being said are genuinely separate

The system holds a character's *identity* apart from the *words*. That is why one character
can say anything without their voice drifting, and why changing the line never changes the
person. This was verified rather than assumed: **when the same audio content is played
through two different identities, the words come out identically — measured error between
them was exactly zero on every line tested.**

### 3. It does not need a recording of the character

There is no voice cloning here, and no user uploads their own audio. Voices are **created
from a written description**. That sidesteps the consent and impersonation problem
entirely — you cannot accidentally clone a real person if the system never accepts a
recording.

### 4. It knows what it does not know

Every capability is measured against a control, and the honest limits are published
alongside the results. Roughly twenty findings in this project were **wrong on the first
attempt and caught by a check** — including one where an entire experiment produced
fluent-sounding speech that was saying different words than requested, and nothing errored.
That is written up publicly rather than quietly fixed.

**Put a version of that on the page.** It is the most trust-building thing here and most
projects cannot say it.

---

## How far it has actually come — the honest status

**It works end to end today**, on one laptop GPU, in **English, Hindi, Bengali and Tamil**.
Describe a character, get a voice, render a scene of dialogue, and out comes watermarked
audio plus a file a game engine can read.

Things you can state as achieved, sourced from the repo:

- **Minting a voice from a description costs nothing in clarity.** A described voice is
  just as intelligible as one taken from a real recording — the measured difference was
  slightly in the *invented* voice's favour, and smaller than the difference between two
  real speakers.
- **Invented voices are recognisably distinct people.** Verified by a separate speaker-
  recognition system that has never seen the generator — an independent check, not the
  system marking its own homework.
- **A character stays itself across different sentences** more consistently than a real
  person does across different recordings, which is unsurprising once you think about it:
  a stored voice has no bad-throat day.
- **A native speaker has confirmed the Hindi output says what it is supposed to say**, and
  a blind listening test with built-in controls found and fixed a real problem in how the
  system judged whether two voices were the same person.

### What is honestly not there yet — say this too

- **This is a working research system, not a product.** There is no signup, no hosted
  service, no API.
- **Emotional performance direction is not solved.** You can control pacing within a
  measured range. You cannot yet say *"say this line apologetically"* and have it work —
  that was tested thoroughly and does not work on the current engine, for reasons the
  project documented and reported upstream.
- **The catalogue is small.** Tens of distinct voices per language so far, not hundreds.
  The limit is understood and measured rather than mysterious.
- **It cannot be shipped as a product in its current form for licensing reasons**, which
  are documented. It is research-only today, deliberately.

**Do not soften these into marketing language.** State them plainly and move on. The
credibility of everything above depends on this section existing.

---

## The high-level design, for a non-technical reader

Explain the shape without jargon. Something like:

```
your description  →  [ the mapper ]  →  a voice identity  →  [ the voice engine ]  →  audio
                     the only part                            never modified,
                     that learns                              swappable
```

Three ideas worth landing:

1. **Only one small piece is trained.** The speech engine itself is used as-is and never
   altered, so a better engine can be swapped in without rebuilding everything.
2. **The voice identity is a small, storable thing.** That is what makes a character
   persistent — it can be saved, versioned, and handed to someone else.
3. **There are two ways to answer a description**, and the system measures both: *invent* a
   new voice, or *find* the closest match in an existing library of voices. Inventing gives
   you a character nobody else has; finding gives you higher accuracy and studio quality.
   Neither is strictly better and the project has numbers for the trade-off.

Also worth one line each, without dwelling:

- **Every render is watermarked and logged**, from the first output, not added later.
- **Output includes a manifest a game engine can consume** — this is built for a pipeline,
  not a demo button.

---

## Tone and framing

- Write for a **creator or studio lead**, not an ML engineer. Assume intelligence, not
  vocabulary.
- Avoid: embedding, vector, cosine, latent space, PCA, Vendi, ECAPA, drift, corpus,
  tokenizer, inference. If a concept needs one of those words, find another way to say it
  or cut it.
- Numbers are welcome **when they mean something to a reader**. "A native speaker confirmed
  the Hindi is correct" lands. "ECAPA 0.7129 normalised +1.03" does not — but *"it stays
  more consistent than a real person does between recordings"* is the same fact and does.
- **Do not use the word "revolutionary", "cutting-edge", or "AI-powered".** The work is
  interesting on its own terms.
- Confidence without hype. The correct register is *"here is what we built, here is what we
  measured, here is what is missing."*

---

## Practical constraints

- **One self-contained page.** No build step, no framework, no external services.
- Must be readable on a phone.
- Include a clear link to the GitHub repository.
- If you want a call-to-action, the honest one is *"read the research"* or *"get in touch"*
  — **not** "sign up" or "try it now", because neither exists.
- Do not fabricate testimonials, logos, pricing, team members, or a roadmap with dates.

## When you are done

Say which claims you put on the page and where each came from, so they can be checked. Flag
anything you were tempted to say but could not source.
