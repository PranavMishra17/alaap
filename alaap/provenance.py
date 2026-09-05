"""
The gate on TRAINING data, and the attribution that has to ship with it.

`renderer.SERVABLE` guards one question: may we serve these weights? RESEARCH/08
section 8.5 points out that this leaves the more expensive question unguarded:

    "public_servable guards Q2. NOTHING currently guards Q3 -- and Q3 is the
     one that costs a retrain."

Q3 is whether weights trained on a given corpus may be released. Getting it
wrong is not a config change, it is a retrain: the defect is baked into the
parameters. This module is that gate, built to section 8.5's shape.

It matters right now because ADR-006 concluded that **route C -- train our own
Indic tower on IndicVoices-R -- is the only Indic route no third party can
veto**. The moment that route is taken, this is the thing standing between a
clean release and an unreleasable checkpoint.

Two rules, both from section 8.5, both enforced in code rather than prose:

  1. `assert_trainable(mix)` runs BEFORE the first optimiser step. A corpus
     absent from the registry fails closed -- an unlicensed artefact and an
     unknown artefact carry the same risk (section 8.4 rule 1).

  2. `emit_attribution(mix)` generates ATTRIBUTION.md **from the mix itself**,
     so the attribution file cannot drift from what was actually trained on.
     A hand-maintained NOTICE file is wrong the first time the mix changes.

CC-BY-4.0 section 3(a)(1) is what rule 2 discharges: retain creator
identification, a copyright notice, a licence reference, a disclaimer
reference, a link to the material, and **an indication that it was modified**.
Training a model on a corpus is modification, so that last clause is not
optional for us.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Iterable, Optional


class TrainingLicenceError(RuntimeError):
    """Raised when a training mix contains something we could not release."""


@dataclass(frozen=True)
class Corpus:
    """
    One training input and the terms attached to it.

    `train_releasable` is deliberately NOT derivable from `licence`. CC-BY-4.0
    is releasable with attribution; CC-BY-NC-4.0 is not, for a commercial
    release; and a bespoke EULA has to be read. The field records a decision
    somebody made and can point at, which is the whole purpose.
    """
    name: str
    licence: str                    # SPDX id, or "other:<name>"
    train_releasable: bool          # may weights trained on this ship publicly?
    source_url: str
    attribution: Optional[str] = None   # required NOTICE line, if any
    note: str = ""                      # why, and where the reasoning lives

    def __post_init__(self):
        if self.train_releasable and self.licence.startswith("CC-BY") \
                and not self.attribution:
            raise ValueError(
                f"{self.name}: {self.licence} requires attribution under "
                f"section 3(a)(1), so `attribution` cannot be empty. If weights "
                f"ship without a NOTICE line the licence condition is unmet.")


# ---------------------------------------------------------------- registry
#
# Every entry traces to RESEARCH/08. Adding a corpus without tracing its chain
# violates invariant I4, and `assert_trainable` fails closed on anything absent.

REGISTRY: dict[str, Corpus] = {
    "indicvoices_r": Corpus(
        name="IndicVoices-R",
        licence="CC-BY-4.0",
        train_releasable=True,
        source_url="https://huggingface.co/datasets/ai4bharat/indicvoices_r",
        attribution=("IndicVoices-R (AI4Bharat), CC BY 4.0 — "
                     "https://huggingface.co/datasets/ai4bharat/indicvoices_r"),
        note="ADR-006 route C's input. 1,704 h / 10,496 speakers / 22 "
             "languages. Gated (gated=auto) but CC-BY-4.0; the gate is access "
             "control, not a licence term. Chosen by AI4Bharat 'allowing "
             "commercial usage'."),
    "globe_v2": Corpus(
        name="GLOBE_V2",
        licence="CC0-1.0",
        train_releasable=True,
        source_url="https://huggingface.co/datasets/MushanW/GLOBE_V2",
        note="CC0 imposes no attribution condition. Note E4: unusable for "
             "speaker-VERIFICATION work (EER 20.0% vs 2.46% on LibriTTS-R), "
             "which is a data-quality finding, not a licence one."),
    "libritts_r": Corpus(
        name="LibriTTS-R",
        licence="CC-BY-4.0",
        train_releasable=True,
        source_url="https://huggingface.co/datasets/blabble-io/libritts_r",
        attribution="LibriTTS-R (Google), CC BY 4.0 — https://www.openslr.org/141/"),

    # ---- present so nobody re-proposes them. RESEARCH/08 section 4. ----
    "expresso": Corpus(
        name="Expresso",
        licence="CC-BY-NC-4.0",
        train_releasable=False,
        source_url="https://huggingface.co/datasets/ylacombe/expresso",
        note="section 4.4. NC. This is the corpus that blocks Indic-Mio, where it "
             "is a DECLARED DIRECT training input in the model's own YAML."),
    "emilia": Corpus(
        name="Emilia",
        licence="CC-BY-NC-4.0",
        train_releasable=False,
        source_url="https://huggingface.co/datasets/amphion/Emilia-Dataset",
        note="section 3.12. NC under its binding terms. Enters the Mio chain "
             "twice transitively, via MioTTS-0.6B and via MioCodec."),
    "indictts_iitm": Corpus(
        name="IITM IndicTTS",
        licence="other:IITM-EULA",
        train_releasable=False,
        source_url="https://www.iitm.ac.in/donlab/indictts/database",
        note="section 3.9 / 4.7. AI4Bharat relabels this CC-BY-4.0; the recovered "
             "EULA section 2.2 forbids onward sublicensing and section 5 mandates a "
             "notice. UNRESOLVED -- this is the open question behind "
             "indic-parler-tts being CONDITIONAL. Flip only on written "
             "confirmation from IITM (NEEDS-FROM-YOU section 2)."),
}


def resolve(name: str) -> Corpus:
    """Look up a corpus, failing closed on anything unregistered."""
    if name not in REGISTRY:
        raise TrainingLicenceError(
            f"{name!r} is not in the training-data registry. An UNKNOWN corpus "
            f"and an UNLICENSED corpus carry the same risk (RESEARCH/08 "
            f"section 8.4 rule 1), so this fails rather than defaulting. Trace its "
            f"chain, then add it. Known: {sorted(REGISTRY)}")
    return REGISTRY[name]


def assert_trainable(mix: Iterable[str]) -> list[Corpus]:
    """
    Call this BEFORE the first optimiser step.

    Raises unless every corpus in the mix may be released in trained weights.
    Returns the resolved mix, which belongs in the checkpoint metadata beside
    `backend_version` -- without it, a routine data change silently alters what
    the released weights may be licensed under.
    """
    mix = list(mix)
    if not mix:
        raise TrainingLicenceError("empty training mix")
    resolved = [resolve(n) for n in mix]
    bad = [c for c in resolved if not c.train_releasable]
    if bad:
        raise TrainingLicenceError(
            "training mix contains corpora whose weights could not be "
            "released:\n" + "\n".join(
                f"  - {c.name} ({c.licence}): {c.note}" for c in bad) +
            "\n\nThis is Q3, and Q3 costs a retrain, not a config change.")
    return resolved


def emit_attribution(mix: Iterable[str], model_name: str = "Alaap",
                     path: Optional[str] = None) -> str:
    """
    Generate the NOTICE text FROM THE MIX, never by hand.

    A hand-maintained attribution file is wrong the moment the mix changes, and
    nothing catches it. Generating it from the same list the trainer asserts on
    means the two cannot disagree.
    """
    resolved = assert_trainable(mix)
    lines = [
        f"# Attribution — {model_name}",
        "",
        "This model's weights were trained on the corpora below. Generated "
        "from the training mix by `alaap.provenance.emit_attribution`; do not "
        "edit by hand — change the mix instead.",
        "",
        "| Corpus | Licence | Source |",
        "|---|---|---|",
    ]
    for c in resolved:
        lines.append(f"| {c.name} | {c.licence} | <{c.source_url}> |")
    notices = [c.attribution for c in resolved if c.attribution]
    if notices:
        lines += ["", "## Required notices", ""]
        lines += [f"- {n}" for n in notices]
        lines += [
            "",
            "Each corpus above marked CC BY 4.0 is used under the Creative "
            "Commons Attribution 4.0 International Licence "
            "(<https://creativecommons.org/licenses/by/4.0/>), which is "
            "provided without warranties of any kind.",
            "",
            "**These materials were modified**: the audio was used to train the "
            "model weights distributed here, and was resampled, trimmed and "
            "acoustically annotated in the process. This statement discharges "
            "CC BY 4.0 section 3(a)(1)(B), which requires indicating modification.",
        ]
    text = "\n".join(lines) + "\n"
    if path:
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
    return text


def checkpoint_metadata(mix: Iterable[str], backend_version: str) -> dict:
    """
    What to record inside the checkpoint, so the licence position travels with
    the weights rather than living in a README somebody forgets to update.
    """
    resolved = assert_trainable(mix)
    return {
        "backend_version": backend_version,
        "training_mix": [asdict(c) for c in resolved],
        "attribution_required": [c.attribution for c in resolved if c.attribution],
        "all_train_releasable": True,
    }


if __name__ == "__main__":  # a quick look at the registry
    print(json.dumps({k: asdict(v) for k, v in REGISTRY.items()}, indent=2))
