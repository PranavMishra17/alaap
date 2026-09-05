"""
The renderer adapter — the seam that keeps the CosyVoice failure from
repeating (scope section 2, PHASE-06).

Every backend hides behind one interface. When a better TTS ships, only an
adapter changes.

Two things enforced here that live in code rather than documentation:

  * `public_servable` (invariant I3). Five upstream licence traps were found
    in one research pass; a rule that lives only in a README will eventually
    be violated by a config change. `load_backend` refuses.

  * `direction_support` (RESEARCH/10). A backend that cannot honour a
    Direction field must say so, not silently ignore it. "Approximate"
    without a published bound is REJECT.

The Qwen3 Tier-1 injection path, verified against the shipped source:

    item = VoiceClonePromptItem(ref_code=None,
                                ref_spk_embedding=<torch tensor (D,)>,
                                x_vector_only_mode=True,
                                icl_mode=False, ref_text=None)
    wavs, sr = model.generate_voice_clone(text=..., language=...,
                                          voice_clone_prompt=[item])

`ref_code=None` + `x_vector_only_mode=True` means the vector is the SOLE
identity input -- no reference audio in the loop. That is what makes the
two-tower split expressible on this backend.

Vendor caveat, from the Qwen3-TTS model card, quoted because it matters:
"If you set x_vector_only_mode=True, only the speaker embedding is used so
ref_text is not required, but cloning quality may be reduced."
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

import numpy as np


class Honouring(enum.Enum):
    HONOURED = "honoured"        # backend implements this field properly
    APPROXIMATE = "approximate"  # implemented, but only within a published bound
    REJECT = "reject"            # not supported; requests must fail loudly


@dataclass
class Direction:
    """
    Per-line performance. Timbre lives on the Identity; this never does
    (invariant I5). Mint from neutral, store neutral, apply Direction at render.

    `intensity` defaults to 0.6, not 1.0 -- RESEARCH/10 found three
    independent sources converging on that ceiling before artefacts appear.
    """
    emotion: Optional[dict[str, float]] = None
    intensity: float = 0.6
    style: Optional[str] = None
    rate: Optional[float] = None
    pitch_var: Optional[float] = None
    loudness: Optional[float] = None
    target_seconds: Optional[float] = None
    timing_tolerance: Optional[float] = None
    pauses: Optional[list[tuple[int, float]]] = None
    emphasis: Optional[list[int]] = None
    seed: Optional[int] = None
    strict: bool = False          # fail rather than degrade

    def requested_fields(self) -> list[str]:
        return [f for f in ("emotion", "style", "rate", "pitch_var", "loudness",
                            "target_seconds", "pauses", "emphasis")
                if getattr(self, f) is not None]


@dataclass
class Audio:
    wav: np.ndarray
    sample_rate: int
    backend_id: str
    backend_version: str
    degradations: list[str] = field(default_factory=list)
    watermarked: bool = False


class LicenceGateError(RuntimeError):
    pass


@runtime_checkable
class Renderer(Protocol):
    backend_id: str
    backend_version: str
    identity_tier: int
    languages: list[str]
    public_servable: bool
    direction_support: dict[str, Honouring]

    def render_from_vector(self, vec: np.ndarray, text: str, language: str,
                           direction: Direction | None = None) -> Audio: ...
    def render_from_audio(self, wav: np.ndarray, sr: int, text: str, language: str,
                          ref_text: str | None = None,
                          direction: Direction | None = None) -> Audio: ...


# ---------------------------------------------------------------- Qwen3 Base
class Qwen3BaseRenderer:
    """
    Tier-1 renderer. The only open backend found whose speaker vector is both
    externally addressable AND the sole identity input (RESEARCH/03).

    Direction support is deliberately almost all REJECT: source inspection
    found `generate_voice_clone` has no `instruct` parameter and never passes
    `instruct_ids`. Per-line control on this checkpoint comes from
    training-free emotion direction vectors (experiment E0), not the API.
    """
    backend_id = "qwen3-tts-base"
    identity_tier = 1
    # Verified from the shipped validator, not the model card. NO INDIC
    # LANGUAGES -- confirms RESEARCH/03 from source.
    languages = ["english", "chinese", "french", "german", "italian",
                 "japanese", "korean", "portuguese", "russian", "spanish"]
    # the API wants full names, not ISO codes
    LANG_ALIAS = {"en": "english", "zh": "chinese", "fr": "french",
                  "de": "german", "it": "italian", "ja": "japanese",
                  "ko": "korean", "pt": "portuguese", "ru": "russian",
                  "es": "spanish"}
    public_servable = True          # Apache-2.0 code AND weights, chain traced
    direction_support = {
        "emotion": Honouring.REJECT,       # no instruct channel; E0 supplies this
        "style": Honouring.REJECT,
        "rate": Honouring.REJECT,
        "pitch_var": Honouring.REJECT,
        "loudness": Honouring.REJECT,
        "target_seconds": Honouring.REJECT,
        "pauses": Honouring.REJECT,
        "emphasis": Honouring.REJECT,
    }

    # Generation cap -- cheap insurance, NOT the fix for the stall below.
    #
    # E0 appeared to hang: one render sat for >8 minutes with the GPU busy.
    # The first hypothesis was unbounded generation, so this cap was added.
    # It was the WRONG diagnosis. A controlled test showed RTF was ~70 with
    # the cap, without it, and at 500 -- i.e. the cap changed nothing.
    #
    # The real cause was environmental: five ORPHANED python processes from
    # earlier background runs were holding 5,550 of 6,144 MiB of VRAM, and
    # the laptop GPU was at 87 C with SW Thermal Slowdown ACTIVE (1740 vs
    # 2100 MHz). Memory pressure plus throttling, not the model.
    #
    # The cap stays because an unbounded generation loop is still a real
    # serving hazard, and at 12 Hz 2000 tokens far exceeds any dialogue line.
    # But the operational lesson is the one that matters: REAP BACKGROUND
    # PROCESSES, and check nvidia-smi before believing a performance number.
    DEFAULT_MAX_NEW_TOKENS = 2000

    def __init__(self, model_id: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                 device: str = "cuda", dtype: str = "bfloat16",
                 max_new_tokens: int | None = None):
        import torch
        from qwen_tts import Qwen3TTSModel
        self._torch = torch
        td = {"bfloat16": torch.bfloat16, "float16": torch.float16,
              "float32": torch.float32}[dtype]
        self.wrapper = Qwen3TTSModel.from_pretrained(model_id, dtype=td,
                                                     device_map=device)
        self.model = self.wrapper.model
        self.model.eval()
        self.model_id = model_id
        self.backend_version = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens or self.DEFAULT_MAX_NEW_TOKENS

    # ------------------------------------------------------------- direction
    def _lang(self, language: str) -> str:
        lang = self.LANG_ALIAS.get(language.lower(), language.lower())
        if lang not in self.languages and lang != "auto":
            raise ValueError(
                f"{self.backend_id} does not support {language!r}. "
                f"Supported: {sorted(self.languages)}. "
                f"(No Indic languages -- see RESEARCH/04 and PHASE-05.)")
        return lang

    def _apply_direction(self, d: Direction | None) -> list[str]:
        if d is None:
            return []
        degr = []
        for f in d.requested_fields():
            h = self.direction_support.get(f, Honouring.REJECT)
            if h is Honouring.REJECT:
                msg = f"{f}: not supported by {self.backend_id}"
                if d.strict:
                    raise NotImplementedError(msg)
                degr.append(msg)
        return degr

    # ---------------------------------------------------------------- render
    def _prompt_from_vector(self, vec: np.ndarray):
        from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem
        t = self._torch.as_tensor(np.asarray(vec, dtype=np.float32)).reshape(-1)
        t = t.to(self.model.device).to(self.model.dtype)
        return VoiceClonePromptItem(ref_code=None, ref_spk_embedding=t,
                                    x_vector_only_mode=True, icl_mode=False,
                                    ref_text=None)

    def render_from_vector(self, vec: np.ndarray, text: str, language: str = "en",
                           direction: Direction | None = None, **kw) -> Audio:
        degr = self._apply_direction(direction)
        item = self._prompt_from_vector(vec)
        wavs, sr = self.model_generate(text, self._lang(language), [item], **kw)
        return Audio(wav=wavs[0], sample_rate=sr, backend_id=self.backend_id,
                     backend_version=self.backend_version, degradations=degr)

    def render_from_audio(self, wav: np.ndarray, sr: int, text: str,
                          language: str = "en", ref_text: str | None = None,
                          direction: Direction | None = None, **kw) -> Audio:
        degr = self._apply_direction(direction)
        items = self.wrapper.create_voice_clone_prompt(
            ref_audio=(np.asarray(wav, dtype=np.float32), sr),
            ref_text=ref_text, x_vector_only_mode=(ref_text is None))
        wavs, out_sr = self.model_generate(text, self._lang(language), items, **kw)
        return Audio(wav=wavs[0], sample_rate=out_sr, backend_id=self.backend_id,
                     backend_version=self.backend_version, degradations=degr)

    def model_generate(self, text, language, items, **kw):
        kw.setdefault("max_new_tokens", self.max_new_tokens)
        return self.wrapper.generate_voice_clone(
            text=text, language=language, voice_clone_prompt=items, **kw)

    def extract_vector(self, wav: np.ndarray, sr: int = 24000) -> np.ndarray:
        """Round-trip helper: the encoder side of the same model."""
        import librosa
        if sr != self.model.speaker_encoder_sample_rate:
            wav = librosa.resample(y=np.asarray(wav, dtype=np.float32),
                                   orig_sr=sr,
                                   target_sr=self.model.speaker_encoder_sample_rate)
        with self._torch.inference_mode():
            e = self.model.extract_speaker_embedding(
                audio=np.asarray(wav, dtype=np.float32),
                sr=self.model.speaker_encoder_sample_rate)
        return e.detach().float().cpu().numpy().reshape(-1)


# ------------------------------------------------------------- licence gate
#
# Verdicts from RESEARCH/08. NEVER add an entry without tracing the upstream
# chain -- five models were found declaring a licence their upstream did not
# support (VoiceSculptor/Llasa-3B, IndicF5/F5-TTS, SPRING_F5/F5-TTS,
# Zonos/VoxBlink2, Indic-Mio/Emilia+Expresso).
SERVABLE = {
    "qwen3-tts-base": True, "qwen3-tts-voicedesign": True, "voxcpm2": True,
    "moss-voicegenerator": True, "kokoro": True, "chatterbox": True,
    "parler-tts": True, "indic-parler-tts": True, "cosyvoice2": True,
    "cosyvoice3": True,
    # NOT servable -- upstream chain fails
    "indicf5": False, "spring_f5": False, "indic-mio": False, "dhvaani": False,
    "voicesculptor": False, "llasa-3b": False, "xcodec2": False, "f5-tts": False,
    "xtts-v2": False, "indextts2": False, "vibevoice": False, "zonos-v0.1": False,
    # measurement instruments only -- never served
    "ecapa-voxceleb": False,
}


def load_backend(renderer, is_public_deployment: bool = False):
    """
    Invariant I3, enforced in code.

    A backend whose upstream licence chain does not permit public serving
    cannot be loaded into a public deployment, regardless of what any config
    file says.
    """
    bid = getattr(renderer, "backend_id", None)
    declared = getattr(renderer, "public_servable", False)
    known = SERVABLE.get(bid)
    if known is None:
        raise LicenceGateError(
            f"{bid!r} is not in the licence audit (RESEARCH/08). Trace its "
            f"upstream chain and add it to SERVABLE before use. Invariant I4.")
    if declared != known:
        raise LicenceGateError(
            f"{bid!r} declares public_servable={declared} but the audit says "
            f"{known}. Reconcile against RESEARCH/08 before proceeding.")
    if is_public_deployment and not known:
        raise LicenceGateError(
            f"{bid!r} is NOT publicly servable (RESEARCH/08). Refusing to load "
            f"it into a public deployment.")
    return renderer
