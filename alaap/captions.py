"""
Grounded caption generation — the "caption second" half of measure-first.

RESEARCH/05, on why VoicePersona v1's captions are unusable: they were
produced by asking an audio-LM to describe a voice directly. Audio-LMs are
fluent and weakly grounded, so a mapper trained on them learns the caption
model's biases rather than the voice. Worse, such captions are NOT
REVERSIBLE -- you cannot check a generated voice against them.

Here the LLM (or, in v1, a template) never hears the audio. It only sees
MEASURED BINS. That makes every caption checkable by re-measurement, which
is what turns eval axis 1 from an LLM-judge problem into arithmetic.

    audio -> measure -> bin -> phrase from bins -> caption
                          ^                          |
                          +----- verify by re-measuring

Two layers, mirroring the Data-Speech / ParaSpeechCaps split:

  GROUNDED   (this module, deterministic)  everything that was measured
  IMPRESSION (optional, an LLM)            abstract qualities that cannot be
                                           measured -- menacing, world-weary,
                                           kindly. ParaSpeechCaps validates
                                           these against a fixed 59-tag
                                           taxonomy rather than free-form,
                                           and so should we.

Templates are deliberately varied so a mapper cannot learn to key off a
single fixed sentence shape -- that would be the same failure as training on
a caption model's tics, just with a smaller vocabulary.
"""
from __future__ import annotations

import random
from typing import Optional

from .acoustics import Attributes, Binner, BIN_LABELS

# ---------------------------------------------------------------- vocabulary
# Each bin label maps to several interchangeable phrasings.
PHRASES = {
    "f0_mean": {
        "very low-pitched":    ["a very deep voice", "an extremely low voice",
                                "a bass-heavy voice", "a profoundly deep voice"],
        "low-pitched":         ["a low voice", "a deep voice", "a low-set voice"],
        "moderately pitched":  ["a mid-range voice", "a moderately pitched voice",
                                "a middle-register voice"],
        "high-pitched":        ["a high voice", "a bright, high-pitched voice",
                                "a light, high voice"],
        "very high-pitched":   ["a very high voice", "an extremely high-pitched voice",
                                "a piercingly high voice"],
    },
    "f0_cv": {
        "monotone":            ["almost monotone", "flat and unvarying in pitch",
                                "with very little pitch movement"],
        "slightly varied":     ["with slight pitch variation", "gently inflected",
                                "with restrained intonation"],
        "moderately expressive": ["moderately expressive", "with natural intonation",
                                  "with an even melodic range"],
        "expressive":          ["expressive", "with lively intonation",
                                "with a wide melodic range"],
        "highly animated":     ["highly animated", "with dramatic pitch swings",
                                "extremely expressive in pitch"],
    },
    "speaking_rate": {
        "very slow":  ["speaking very slowly", "with a very deliberate pace",
                       "unhurried almost to the point of stillness"],
        "slow":       ["speaking slowly", "at a measured, slow pace", "taking its time"],
        "measured":   ["at a steady pace", "at a natural, even tempo",
                       "neither rushed nor slow"],
        "quick":      ["speaking quickly", "at a brisk pace", "moving briskly"],
        "rapid":      ["speaking very rapidly", "at a hurried, rapid clip",
                       "racing through the words"],
    },
    "hnr_db": {
        "very rough":     ["very rough and gravelly", "harsh and rasping",
                           "heavily textured with grit"],
        "rough":          ["rough-edged", "with a gravelly texture", "slightly hoarse"],
        "slightly rough": ["with a faint rasp", "lightly textured",
                           "with a trace of roughness"],
        "clear":          ["clear-toned", "with a clean tone", "smooth and clear"],
        "very clear":     ["very clear and pure", "with a strikingly clean tone",
                           "crystalline in tone"],
    },
    "spectral_tilt": {
        "very dark":  ["very dark and warm in timbre", "muffled and warm",
                       "with a heavily shadowed timbre"],
        "dark":       ["dark in timbre", "warm-toned", "with a mellow, dark colour"],
        "balanced":   ["balanced in timbre", "neither bright nor dark",
                       "with an even tonal colour"],
        "bright":     ["bright in timbre", "with a forward, bright tone",
                       "crisp-toned"],
        "very bright": ["very bright and edgy", "with a sharp, brilliant timbre",
                        "piercingly bright"],
    },
    "shimmer": {
        "very steady":       ["rock-steady in volume", "with unwavering support"],
        "steady":            ["steady", "well-supported"],
        "slightly unsteady": ["with a slight tremor", "faintly unsteady"],
        "unsteady":          ["unsteady", "with a noticeable waver"],
        "very unsteady":     ["with a pronounced tremor", "markedly shaky"],
    },
}

# Varied deliberately. S2 found template captions give the text encoder too
# narrow a signal, so retrieval keys off almost nothing. More surface variety
# without changing what is GROUNDED underneath.
OPENERS = [
    "{subject} has {desc}.",
    "{subject} speaks with {desc}.",
    "The voice is {desc_bare}.",
    "{subject}: {desc_bare}.",
    "A voice that is {desc_bare}.",
    "You hear {desc}.",
    "{subject} sounds like this: {desc_bare}.",
    "Picture {desc}.",
    "{desc_bare} — that is how {subject_lower} sounds.",
    "Imagine {desc}.",
    "The speaker has {desc}.",
    "This is {desc}.",
]

# Which attributes to include, in the order they read most naturally.
ORDER = ["f0_mean", "hnr_db", "spectral_tilt", "f0_cv", "speaking_rate", "shimmer"]


def caption_from_bins(bins: dict[str, str], subject: str = "This speaker",
                      max_attrs: int = 5, seed: int | None = None,
                      impressions: Optional[list[str]] = None) -> str:
    """
    Deterministic, grounded caption. Every clause traces to a measured bin.

    `impressions` is the optional ungrounded layer (menacing, world-weary).
    It is kept SEPARATE and appended, never blended, so that adherence
    scoring can ignore it -- you cannot re-measure "world-weary".
    """
    rng = random.Random(seed)
    parts = []
    for f in ORDER:
        if f not in bins or f not in PHRASES:
            continue
        opts = PHRASES[f].get(bins[f])
        if opts:
            parts.append(rng.choice(opts))
        if len(parts) >= max_attrs:
            break
    if not parts:
        return f"{subject} has an unremarkable voice."

    head = parts[0]
    tail = parts[1:]
    if tail:
        desc_bare = head + ", " + ", ".join(tail[:-1]) + (", " if len(tail) > 1 else "") \
                    + ("and " + tail[-1] if len(tail) >= 1 else "")
    else:
        desc_bare = head
    desc_bare = desc_bare.replace(", and ", " and ").replace(",,", ",")

    # shuffle attribute order sometimes, so position is not a fixed cue
    tpl = rng.choice(OPENERS)
    text = tpl.format(subject=subject, desc=desc_bare, desc_bare=desc_bare,
                      subject_lower=subject[0].lower() + subject[1:])
    text = text[0].upper() + text[1:]
    if impressions:
        text += " It sounds " + ", ".join(impressions[:2]) + "."
    return text


def caption_attributes(attrs: Attributes, binner: Binner, **kw) -> tuple[str, dict]:
    """measure -> bin -> caption, returning both so the pair stays traceable."""
    bins = binner.bin_one(attrs)
    return caption_from_bins(bins, **kw), bins


def target_bins_from_text(text: str) -> dict[str, str]:
    """
    Parse a free-form description back into target bins.

    This is the REVERSIBILITY step, and it is deliberately simple: match the
    known phrasings. It exists so a human-written description ("a gravelly
    old mentor, speaking slowly") can be scored against a rendered voice
    with adherence_error(). Unmatched wording is silently ignored rather
    than guessed at -- inventing a target bin would fabricate a score.
    """
    import re
    t = " " + text.lower().strip() + " "

    def has(phrase: str) -> bool:
        """
        Whole-word match. Substring matching is WRONG here and was a real bug:
        'unhurried' contains 'hurried', so 'a low, smooth voice, unhurried'
        was parsed as speaking_rate=rapid -- the exact opposite of what it
        says. That corrupted S2's adherence scores before it was caught.
        """
        return re.search(r"(?<![a-z])" + re.escape(phrase.lower()) +
                         r"(?![a-z])", t) is not None

    out: dict[str, str] = {}
    for field, by_label in PHRASES.items():
        best, best_len = None, 0
        for label, opts in by_label.items():
            for o in opts:
                if has(o) and len(o) > best_len:
                    best, best_len = label, len(o)
        if best:
            out[field] = best

    # Natural synonyms people actually type. Ordered LONGEST-FIRST within each
    # field so that "very deep" beats "deep" and "very clear" beats "clear".
    SYN = {
        "f0_mean": [("very deep", "very low-pitched"), ("extremely low", "very low-pitched"),
                    ("bass", "very low-pitched"), ("very high", "very high-pitched"),
                    ("squeaky", "very high-pitched"), ("deep", "low-pitched"),
                    ("low", "low-pitched"), ("high", "high-pitched"),
                    ("mid-range", "moderately pitched"),
                    # E15b: only 22% of realistic descriptions parsed. These are
                    # additions whose ACOUSTIC mapping is defensible. Words like
                    # "old", "young", "giant" or "child" are deliberately NOT
                    # here: mapping an age or a body to a formant is an
                    # inference, and inventing a target bin fabricates a score.
                    ("booming", "very low-pitched"), ("rumbling", "very low-pitched"),
                    ("baritone", "low-pitched"), ("bassy", "very low-pitched"),
                    ("shrill", "very high-pitched"), ("piping", "very high-pitched"),
                    ("reedy", "high-pitched"), ("light", "high-pitched")],
        "hnr_db": [("very rough", "very rough"), ("very clear", "very clear"),
                   ("crystalline", "very clear"), ("gravelly", "very rough"),
                   ("raspy", "very rough"), ("harsh", "very rough"),
                   ("rasping", "very rough"), ("faint rasp", "slightly rough"),
                   ("slight rasp", "slightly rough"), ("rasp", "rough"),
                   ("hoarse", "rough"), ("rough", "rough"),
                   ("clean", "very clear"), ("smooth", "clear"),
                   ("clear", "clear"),
                   # Breathiness and whisper are noise in the harmonic ratio,
                   # which is exactly what HNR measures -- so these map to the
                   # rough end without inferring anything.
                   ("breathy", "rough"), ("whispery", "very rough"),
                   ("whispered", "very rough"), ("whisper", "very rough"),
                   ("husky", "rough"), ("scratchy", "very rough"),
                   ("grating", "very rough"), ("croaky", "very rough"),
                   ("silky", "very clear"), ("velvety", "clear"),
                   ("pure", "very clear"), ("polished", "very clear")],
        "speaking_rate": [("very slow", "very slow"), ("very slowly", "very slow"),
                          ("very rapidly", "rapid"), ("unhurried", "slow"),
                          ("deliberate", "slow"), ("slowly", "slow"),
                          ("slow", "slow"), ("measured", "measured"),
                          ("steady pace", "measured"), ("racing", "rapid"),
                          ("rapid", "rapid"), ("rapidly", "rapid"),
                          ("hurried", "rapid"), ("quickly", "quick"),
                          ("quick", "quick"), ("brisk", "quick"),
                          ("fast", "quick"), ("gabbling", "rapid"),
                          ("breakneck", "rapid"), ("clipped", "quick"),
                          ("languid", "very slow"), ("ponderous", "very slow"),
                          ("drawling", "very slow"), ("leisurely", "slow"),
                          ("measured pace", "measured"), ("even tempo", "measured")],
        # Was keyed "f0_std" until 2026-09-05. f0_std is a DEAD axis -- S2 run 4
        # replaced it with f0_cv because f0_std in Hz correlates r=+0.72 with
        # f0_mean. BIN_LABELS has no f0_std, and adherence_error skips any
        # target bin it cannot find, so every natural synonym for
        # expressiveness -- including the bare word "monotone" -- was parsed
        # into an axis that was then silently dropped and never scored.
        "f0_cv": [("highly animated", "highly animated"), ("monotone", "monotone"),
                   ("flat", "monotone"), ("gently inflected", "slightly varied"),
                   ("animated", "highly animated"), ("expressive", "expressive"),
                   ("lively", "expressive"),
                   ("singsong", "highly animated"), ("melodic", "expressive"),
                   ("deadpan", "monotone"), ("flat and dull", "monotone"),
                   ("droning", "monotone"), ("unvarying", "monotone"),
                   ("emotive", "expressive")],
        "spectral_tilt": [("very bright", "very bright"), ("very dark", "very dark"),
                          ("dark-timbred", "dark"), ("shrill", "very bright"),
                          ("piercing", "very bright"), ("warm", "dark"),
                          ("mellow", "dark"), ("dark", "dark"),
                          ("bright", "bright"), ("crisp", "bright"),
                          ("balanced", "balanced"),
                          ("muffled", "very dark"), ("boomy", "very dark"),
                          ("plummy", "dark"), ("nasal", "bright"),
                          ("tinny", "very bright"), ("sharp", "bright"),
                          ("edgy", "very bright"), ("round", "dark"),
                          ("resonant", "dark")],
    }
    for field, pairs in SYN.items():
        if field in out:
            continue
        for word, label in pairs:
            if has(word):
                out[field] = label
                break
    return out
