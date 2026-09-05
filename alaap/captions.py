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
    "f0_std": {
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

OPENERS = ["{subject} has {desc}.", "{subject} speaks with {desc}.",
           "The voice is {desc_bare}.", "{subject}: {desc_bare}."]

# Which attributes to include, in the order they read most naturally.
ORDER = ["f0_mean", "hnr_db", "spectral_tilt", "f0_std", "speaking_rate", "shimmer"]


def caption_from_bins(bins: dict[str, str], subject: str = "This speaker",
                      max_attrs: int = 4, seed: int | None = None,
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

    tpl = rng.choice(OPENERS)
    text = tpl.format(subject=subject, desc=desc_bare, desc_bare=desc_bare)
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
    t = text.lower()
    out: dict[str, str] = {}
    for field, by_label in PHRASES.items():
        best, best_len = None, 0
        for label, opts in by_label.items():
            for o in opts:
                if o.lower() in t and len(o) > best_len:
                    best, best_len = label, len(o)
        if best:
            out[field] = best
    # a few natural synonyms people actually type
    SYN = {
        "f0_mean": [("deep", "low-pitched"), ("bass", "very low-pitched"),
                    ("high", "high-pitched"), ("squeaky", "very high-pitched")],
        "hnr_db": [("gravelly", "very rough"), ("raspy", "very rough"),
                   ("rasp", "rough"), ("hoarse", "rough"), ("smooth", "clear"),
                   ("clean", "very clear")],
        "speaking_rate": [("slow", "slow"), ("deliberate", "slow"),
                          ("fast", "quick"), ("rapid", "rapid"),
                          ("hurried", "rapid")],
        "f0_std": [("monotone", "monotone"), ("flat", "monotone"),
                   ("animated", "highly animated"), ("expressive", "expressive")],
        "spectral_tilt": [("warm", "dark"), ("dark", "dark"),
                          ("bright", "bright"), ("shrill", "very bright")],
    }
    for field, pairs in SYN.items():
        if field in out:
            continue
        for word, label in pairs:
            if word in t:
                out[field] = label
                break
    return out
