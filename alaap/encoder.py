"""
Frozen speaker encoder + the cache-embeddings-once layer.

RESEARCH/GRAND-PLAN and PHASE-00 both insist this be a first-class module
rather than a script: once embeddings are on disk, mapper training reads
vectors and never audio, which turns a GPU-bound problem into a
laptop-friendly one.

Ground truth for the API, read from
  envs/qwen3/Lib/site-packages/qwen_tts/core/models/modeling_qwen3_tts.py:1941

    def extract_speaker_embedding(self, audio, sr):
        assert sr == 24000, "Only support 24kHz audio"
        mels = mel_spectrogram(torch.from_numpy(audio).unsqueeze(0),
                               n_fft=1024, num_mels=128, sampling_rate=24000,
                               hop_size=256, win_size=1024, fmin=0, fmax=12000
                              ).transpose(1, 2)
        return self.speaker_encoder(mels.to(self.device).to(self.dtype))[0]

Takes a MONO FLOAT32 NUMPY ARRAY at exactly 24 kHz.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Iterable, Optional

import numpy as np

DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
SR = 24000

# measured in E3 on an RTX 3060 6GB
KNOWN_MODELS = {
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base": {"enc_dim": 1024, "vram_gb": 2.02},
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base": {"enc_dim": 2048, "vram_gb": None},
}


class SpeakerEncoder:
    """Thin wrapper over the frozen Qwen3-TTS speaker encoder."""

    def __init__(self, model_id: str = DEFAULT_MODEL, device: str = "cuda",
                 dtype: str = "bfloat16"):
        import torch
        from qwen_tts import Qwen3TTSModel

        self.model_id = model_id
        self._torch = torch
        td = {"bfloat16": torch.bfloat16, "float16": torch.float16,
              "float32": torch.float32}[dtype]
        self.wrapper = Qwen3TTSModel.from_pretrained(
            model_id, dtype=td, device_map=device)
        self.model = self.wrapper.model
        self.model.eval()
        assert self.model.speaker_encoder_sample_rate == SR, \
            f"expected 24kHz, got {self.model.speaker_encoder_sample_rate}"

    @property
    def dim(self) -> int:
        return int(self.model.config.speaker_encoder_config.enc_dim)

    def vram_gb(self) -> float:
        if not self._torch.cuda.is_available():
            return 0.0
        return self._torch.cuda.memory_allocated() / 2 ** 30

    def embed(self, wav: np.ndarray) -> np.ndarray:
        """One mono float32 array @24kHz -> (D,) float32."""
        with self._torch.inference_mode():
            e = self.model.extract_speaker_embedding(audio=wav, sr=SR)
        return e.detach().float().cpu().numpy().reshape(-1)

    def embed_many(self, wavs: Iterable[np.ndarray], progress_every: int = 200
                   ) -> np.ndarray:
        out = []
        for i, w in enumerate(wavs):
            out.append(self.embed(w))
            if progress_every and (i + 1) % progress_every == 0:
                print(f"      embedded {i+1}", flush=True)
        return np.stack(out).astype(np.float32)


# ------------------------------------------------------------------ cache
class EmbeddingCache:
    """
    Content-addressed embedding cache.

    Key = sha1(model_id | corpus | n | per_speaker | min_dur | max_dur | seed).
    Anything that changes the embeddings changes the key, so a stale cache
    can never silently poison an experiment.
    """

    def __init__(self, root: str = "cache/embeddings"):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)

    @staticmethod
    def key(**parts) -> str:
        blob = json.dumps(parts, sort_keys=True, default=str)
        return hashlib.sha1(blob.encode()).hexdigest()[:16]

    def path(self, key: str) -> str:
        return os.path.join(self.root, f"{key}.npz")

    def has(self, key: str) -> bool:
        return os.path.exists(self.path(key))

    def put(self, key: str, Z: np.ndarray, meta: list[dict], spec: dict) -> str:
        p = self.path(key)
        np.savez_compressed(p, Z=Z.astype(np.float32),
                            meta=json.dumps(meta), spec=json.dumps(spec))
        return p

    def get(self, key: str) -> tuple[np.ndarray, list[dict], dict]:
        d = np.load(self.path(key), allow_pickle=False)
        return (d["Z"].astype(np.float32),
                json.loads(str(d["meta"])),
                json.loads(str(d["spec"])))

    def list(self) -> list[dict]:
        out = []
        for f in sorted(os.listdir(self.root)):
            if not f.endswith(".npz"):
                continue
            try:
                d = np.load(os.path.join(self.root, f), allow_pickle=False)
                spec = json.loads(str(d["spec"]))
                out.append({"key": f[:-4], "shape": list(d["Z"].shape), **spec})
            except Exception:
                pass
        return out


def build_embeddings(corpus: str = "globe_v2", n: int = 1000, per_speaker: int = 1,
                     model_id: str = DEFAULT_MODEL, min_dur: float = 2.0,
                     max_dur: float = 15.0, seed: int = 0, skip: int = 0,
                     cache_root: str = "cache/embeddings",
                     force: bool = False):
    """
    Stream -> embed -> cache. Returns (Z, meta, spec).
    Re-running with the same spec is instant.
    """
    from . import data as _data

    spec = dict(model_id=model_id, corpus=corpus, n=n, per_speaker=per_speaker,
                min_dur=min_dur, max_dur=max_dur, seed=seed, skip=skip)
    cache = EmbeddingCache(cache_root)
    key = EmbeddingCache.key(**spec)

    if cache.has(key) and not force:
        Z, meta, spec_ = cache.get(key)
        print(f"      cache HIT {key} -> Z{Z.shape}", flush=True)
        return Z, meta, spec_

    print(f"      cache MISS {key}; building ...", flush=True)
    clips = _data.stream_clips(corpus=corpus, n=n, per_speaker=per_speaker,
                               min_dur=min_dur, max_dur=max_dur, seed=seed, skip=skip)
    enc = SpeakerEncoder(model_id)
    print(f"      encoder dim={enc.dim} vram={enc.vram_gb():.2f}GB", flush=True)
    Z = enc.embed_many([c.wav for c in clips])
    meta = [{"speaker_id": c.speaker_id, "duration": c.duration, "accent": c.accent,
             "age": c.age, "gender": c.gender} for c in clips]
    cache.put(key, Z, meta, spec)
    print(f"      cached {key} -> Z{Z.shape}", flush=True)
    return Z, meta, spec


# ------------------------------------------------- independent SV encoder
class IndependentSV:
    """
    An INDEPENDENT speaker-verification encoder, for eval axis 2.

    RESEARCH/06 rule: never score identity consistency with the same encoder
    used for conditioning -- that is marking your own homework. This is a
    different architecture (ECAPA-TDNN, 16 kHz fbank) trained with a different
    objective (discriminative ASV) from Qwen3-TTS's conditioning encoder.

    LICENCE NOTE: speechbrain code is Apache-2.0, but this checkpoint is
    trained on VoxCeleb, which RESEARCH/08 flags as never properly licensed.
    That is acceptable HERE because it is a measurement instrument only --
    never trained on, never served, never shipped. It must not appear in any
    `public_servable` adapter. See NEEDS-FROM-YOU.md.
    """
    MODEL = "speechbrain/spkrec-ecapa-voxceleb"
    SR = 16000

    def __init__(self, device: str = "cuda", savedir: str = "cache/sv"):
        import torch
        from speechbrain.inference.speaker import EncoderClassifier
        self._torch = torch
        self.device = device
        self.clf = EncoderClassifier.from_hparams(
            source=self.MODEL, savedir=savedir,
            run_opts={"device": device})

    def embed(self, wav: np.ndarray, sr: int = SR) -> np.ndarray:
        import librosa
        if sr != self.SR:
            wav = librosa.resample(y=wav.astype(np.float32), orig_sr=sr, target_sr=self.SR)
        t = self._torch.from_numpy(np.asarray(wav, dtype=np.float32)).unsqueeze(0)
        with self._torch.inference_mode():
            e = self.clf.encode_batch(t.to(self.device))
        return e.squeeze().detach().float().cpu().numpy().reshape(-1)

    def embed_many(self, wavs, sr: int = SR, progress_every: int = 200) -> np.ndarray:
        out = []
        for i, w in enumerate(wavs):
            out.append(self.embed(w, sr=sr))
            if progress_every and (i + 1) % progress_every == 0:
                print(f"      [SV] embedded {i+1}", flush=True)
        return np.stack(out).astype(np.float32)
