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
"""
from __future__ import annotations

import io
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterator, Optional

import numpy as np

SR = 24000  # Qwen3-TTS extract_speaker_embedding asserts sr == 24000

# corpus -> (hf_id, config, split, licence, speaker key)
CORPORA = {
    "globe_v2":   ("MushanW/GLOBE_V2", None, "train", "CC0", "speaker_id"),
    "libritts_r": ("blabble-io/libritts_r", "dev", "dev.clean", "CC-BY-4.0", "speaker_id"),
    "libritts_r_train": ("blabble-io/libritts_r", "clean", "train.clean.100", "CC-BY-4.0", "speaker_id"),
    "libritts_r_test": ("blabble-io/libritts_r", "dev", "test.clean", "CC-BY-4.0", "speaker_id"),
    # ~900 speakers -- 4x train.clean.100. S2's top improvement was "scale the
    # fit set"; 206 speakers was the binding constraint on mapper quality.
    "libritts_r_360": ("blabble-io/libritts_r", "clean", "train.clean.360", "CC-BY-4.0", "speaker_id"),
}


@dataclass
class Clip:
    wav: np.ndarray          # float32 mono @ SR
    speaker_id: str
    duration: float
    accent: Optional[str] = None
    age: Optional[str] = None
    gender: Optional[str] = None
    text: Optional[str] = None
    meta: dict = field(default_factory=dict)


def stream_clips(corpus: str = "globe_v2",
                 n: int = 1000,
                 per_speaker: int = 1,
                 min_dur: float = 2.0,
                 max_dur: float = 15.0,
                 min_speakers: int | None = None,
                 skip: int = 0,
                 seed: int = 0,
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
    """
    import soundfile as sf
    import librosa
    from datasets import load_dataset, Audio

    if corpus not in CORPORA:
        raise ValueError(f"unknown corpus {corpus!r}; known: {list(CORPORA)}")
    hf_id, config, split, licence, spk_key = CORPORA[corpus]

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
            accent=r.get("accent"), age=r.get("age"), gender=r.get("gender"),
            text=(r.get("transcript") or r.get("text") or r.get("text_normalized")
                  or r.get("text_original")),
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
