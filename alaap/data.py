"""
Corpus streaming with speaker-aware sampling.

Only corpora that passed the licence audit in RESEARCH/08 are wired up here.
Adding a corpus without tracing its upstream chain violates invariant I4.

GLOBE_V2 (CC0) is fine for GEOMETRY work (per-dimension statistics, norms,
effective rank) -- 23,519 speakers, streams without a bulk download, and
carries speaker_id / accent / age / gender so per-demographic slicing
(invariant I10) is free.

>>> GLOBE_V2 IS NOT SUITABLE FOR SPEAKER-VERIFICATION WORK. <<<
E4 measured ECAPA-TDNN at EER 20.0% on GLOBE_V2 but 2.46% on LibriTTS-R
using the identical code path -- an 8x difference that isolates the corpus,
not the pipeline. Its speaker labels and/or its enhancement processing do
not preserve speaker identity reliably enough to calibrate C_same.
Use libritts_r for anything that depends on speaker labels being true.

Note: `datasets` wants torchcodec to decode audio. We sidestep that by
reading raw bytes (`Audio(decode=False)`) and decoding with soundfile,
which avoids a heavyweight dependency.

INDIC CORPORA AND THE LICENCE CHAIN (invariant I4)
--------------------------------------------------
Every Indic corpus with real speaker diversity is GATED on the Hub:
ai4bharat/indicvoices_r, ai4bharat/IndicVoices and ai4bharat/Rasa all refuse
anonymous access. Gating is not a licence problem -- IndicVoices-R is
CC-BY-4.0, which passes the audit -- it is an access-control click.

Until that token exists we stream the SPRINGLab MIRRORS, which are ungated
and carry identical content. Their licence chain is:

    content   IndicVoices-R, AI4Bharat, CC-BY-4.0   (upstream, explicit)
    mirror    SPRINGLab/IndicVoices-R_*             (NO licence declared)

The mirror declares no licence at all -- not a contradiction, an omission.
I4 says trace the chain, and the chain says the content is CC-BY-4.0. That
is good enough to DEVELOP against and not good enough to SHIP against, so
the mirrors are marked dev_only=True and the gate in stream_clips refuses
them unless the caller says so explicitly. Re-point at the gated originals
once HF_TOKEN lands; the schemas match, so nothing else changes.

RESEARCH/08 separately flags SPRINGLab/IndicTTS_* for a licence
CONTRADICTION (YAML cc-by-4.0 vs prose deferring to the original Indic TTS
terms). That is a different and worse problem, and those are not wired up.
"""
from __future__ import annotations

import io
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterator, Optional

import numpy as np

SR = 24000  # Qwen3-TTS extract_speaker_embedding asserts sr == 24000


class LicenceError(RuntimeError):
    """Raised when a corpus is read in a way its licence chain does not support."""


def _label(field_name: str, value):
    """ClassLabel int -> its string, leaving anything else alone."""
    names = CLASS_LABELS.get(field_name)
    if names is None or not isinstance(value, (int, np.integer)) or             isinstance(value, bool):
        return value
    return names[value] if 0 <= int(value) < len(names) else str(value)

# corpus -> (hf_id, config, split, licence, speaker key)
CORPORA = {
    "globe_v2":   ("MushanW/GLOBE_V2", None, "train", "CC0", "speaker_id"),
    "libritts_r": ("blabble-io/libritts_r", "dev", "dev.clean", "CC-BY-4.0", "speaker_id"),
    "libritts_r_train": ("blabble-io/libritts_r", "clean", "train.clean.100", "CC-BY-4.0", "speaker_id"),
    "libritts_r_test": ("blabble-io/libritts_r", "dev", "test.clean", "CC-BY-4.0", "speaker_id"),
    # ~900 speakers -- 4x train.clean.100. S2's top improvement was "scale the
    # fit set"; 206 speakers was the binding constraint on mapper quality.
    "libritts_r_360": ("blabble-io/libritts_r", "clean", "train.clean.360", "CC-BY-4.0", "speaker_id"),

    # --- Indic. See the licence-chain note in the module docstring. ---
    # Mirrors of IndicVoices-R: ungated, but declare no licence themselves.
    # 26,318 utterances / ~1,000 speakers for Hindi. 48 kHz, resampled to SR.
    "indicvoices_r_hi": ("SPRINGLab/IndicVoices-R_Hindi", None, "train",
                         "CC-BY-4.0 upstream; UNDECLARED on mirror", "speaker_id"),
    "indicvoices_r_bn": ("SPRINGLab/IndicVoices-R_Bengali", None, "train",
                         "CC-BY-4.0 upstream; UNDECLARED on mirror", "speaker_id"),
    "indicvoices_r_ta": ("SPRINGLab/IndicVoices-R_Tamil", None, "train",
                         "CC-BY-4.0 upstream; UNDECLARED on mirror", "speaker_id"),
    # The real thing. Gated (gated="auto", so one click) -- needs HF_TOKEN.
    "indicvoices_r": ("ai4bharat/indicvoices_r", None, "train",
                      "CC-BY-4.0", "speaker_id"),
}

# Corpora we may develop against but must not ship from, because their
# licence is inherited rather than declared. stream_clips refuses these
# unless the caller passes dev_only=True, so it can never happen by accident.
DEV_ONLY = {"indicvoices_r_hi", "indicvoices_r_bn", "indicvoices_r_ta"}

# BCP-47 language of each corpus, where it is single-language.
CORPUS_LANG = {"globe_v2": "en", "libritts_r": "en", "libritts_r_train": "en",
               "libritts_r_test": "en", "libritts_r_360": "en",
               "indicvoices_r_hi": "hi", "indicvoices_r_bn": "bn",
               "indicvoices_r_ta": "ta"}

# The mirrors store demographics as ClassLabel INTEGERS. Left raw, Clip.gender
# would be 0 and every downstream slice would read "0" as a category name.
CLASS_LABELS = {
    "gender":    ["Female", "Male", "Other"],
    "age_group": ["18-30", "30-45", "45-60", "60+"],
    "area":      ["Rural", "Urban"],
    "scenario":  ["Extempore", "Read"],
}

# Reference annotations the SPRINGLab mirrors ship alongside the audio. They
# come from Data-Speech, i.e. a DIFFERENT toolchain from alaap.acoustics, so
# they are an independent cross-check on our own measurements -- see S4.
REFERENCE_ANNOTATIONS = ("utterance_pitch_mean", "utterance_pitch_std",
                         "snr", "c50", "speaking_rate", "duration")


@dataclass
class Clip:
    wav: np.ndarray          # float32 mono @ SR
    speaker_id: str
    duration: float
    accent: Optional[str] = None
    age: Optional[str] = None
    gender: Optional[str] = None
    text: Optional[str] = None
    language: Optional[str] = None
    meta: dict = field(default_factory=dict)


def stream_clips(corpus: str = "globe_v2",
                 n: int = 1000,
                 per_speaker: int = 1,
                 min_dur: float = 2.0,
                 max_dur: float = 15.0,
                 min_speakers: int | None = None,
                 skip: int = 0,
                 seed: int = 0,
                 dev_only: bool = False,
                 progress_every: int = 200) -> list[Clip]:
    """
    Collect up to `n` clips, at most `per_speaker` from any one speaker.

    With per_speaker=1 you get maximum speaker diversity (what E3 wanted).
    With per_speaker=20 you get repeated speakers, which is what E4 needs to
    measure C_same (intra-speaker similarity).

    `skip` drops the first N rows of the stream. GLOBE_V2 streams in a stable
    order, so two calls with overlapping ranges return OVERLAPPING SPEAKERS.
    Use skip to build genuinely disjoint fit/test splits -- E4's first run
    had 169/169 speakers leak because both calls started from row 0.

    `dev_only` must be set to read a corpus in DEV_ONLY -- one whose licence
    is inherited from upstream rather than declared on the artefact we are
    actually reading. It is an explicit acknowledgement, not a convenience
    flag, and it exists so that shipping from an unlicensed mirror has to be
    a decision somebody typed rather than a default nobody noticed (I4).
    """
    import soundfile as sf
    import librosa
    from datasets import load_dataset, Audio

    if corpus not in CORPORA:
        raise ValueError(f"unknown corpus {corpus!r}; known: {list(CORPORA)}")
    if corpus in DEV_ONLY and not dev_only:
        raise LicenceError(
            f"{corpus!r} carries no declared licence of its own "
            f"({CORPORA[corpus][3]}). It is fine to develop against and not "
            f"fine to ship from. Pass dev_only=True to acknowledge that, or "
            f"use the gated original once HF_TOKEN is set.")
    hf_id, config, split, licence, spk_key = CORPORA[corpus]
    lang = CORPUS_LANG.get(corpus)

    ds = (load_dataset(hf_id, config, split=split, streaming=True) if config
          else load_dataset(hf_id, split=split, streaming=True))
    ds = ds.cast_column("audio", Audio(decode=False))

    clips: list[Clip] = []
    per_spk: dict[str, int] = defaultdict(int)
    scanned = skipped = 0

    if skip:
        ds = ds.skip(skip)

    for r in ds:
        scanned += 1
        spk = str(r.get(spk_key, "?"))
        if per_spk[spk] >= per_speaker:
            continue
        try:
            wav, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
        except Exception:
            skipped += 1
            continue
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        dur = len(wav) / sr
        if not (min_dur <= dur <= max_dur):
            continue
        if sr != SR:
            wav = librosa.resample(y=wav.astype(np.float32), orig_sr=int(sr), target_sr=SR)
        # guard: some GLOBE clips exceed +/-1.0 and the mel frontend warns
        peak = float(np.abs(wav).max())
        if peak > 1.0:
            wav = wav / peak

        clips.append(Clip(
            wav=wav.astype(np.float32), speaker_id=spk, duration=round(dur, 3),
            accent=r.get("accent"),
            age=_label("age_group", r.get("age_group")) if "age_group" in r
                else r.get("age"),
            gender=_label("gender", r.get("gender")),
            # `normalized` first: on the Indic mirrors it is the field the
            # upstream pipeline actually transcribed against.
            text=(r.get("normalized") or r.get("transcript") or r.get("text")
                  or r.get("text_normalized") or r.get("text_original")),
            language=lang,
            meta={k: r[k] for k in REFERENCE_ANNOTATIONS if k in r} |
                 {k: _label(k, r[k]) for k in ("area", "scenario") if k in r} |
                 {k: r[k] for k in ("district", "state", "occupation",
                                    "task_name") if k in r},
        ))
        per_spk[spk] += 1

        if progress_every and len(clips) % progress_every == 0:
            print(f"      {len(clips)}/{n} clips | {len(per_spk)} speakers "
                  f"| {scanned} scanned", flush=True)

        done_n = len(clips) >= n
        done_spk = min_speakers is not None and len(per_spk) >= min_speakers \
            and all(v >= per_speaker for k, v in per_spk.items() if v > 0)
        if done_n:
            break

    print(f"      collected {len(clips)} clips / {len(per_spk)} speakers "
          f"({scanned} scanned, {skipped} undecodable) [{licence}]", flush=True)
    return clips


def group_by_speaker(clips: list[Clip]) -> dict[str, list[int]]:
    g: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(clips):
        g[c.speaker_id].append(i)
    return dict(g)
