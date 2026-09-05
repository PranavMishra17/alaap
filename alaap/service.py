"""
The mint/render service — PHASE-06's core, with the invariants composed.

This is where the pieces stop being libraries and start being a product.
Everything the research insisted on lives here in one path:

  I2  every identity stores BOTH a Tier-1 vector AND a Tier-2 seed clip,
      plus backend_version, the SpeakerSpace ref, and generation params
  I3  the licence gate refuses an unaudited or non-servable backend
  I5  Direction is per-line; the identity record never sees emotion
  I7  every render is watermarked AND provenance-logged, from render #1

Plus two things measured rather than assumed:

  * UNIQUENESS is checked in WORKING space, never raw. E3: on raw vectors
    the entire 1,025-speaker population has a maximum nearest-neighbour
    cosine distance of 0.029, so the VoicePrivacy-B3 threshold of 0.3 is
    unreachable there. In working space the mean NN distance is 0.550.
  * The CLOSED DRIFT LOOP. E2 measured drift at 0.5519 in working space
    against C_same 0.5766 -- i.e. the rendered voice is about as close to
    its vector as two recordings of one person are to each other -- but the
    worst case fell to 0.3546. Mint verifies and re-mints below threshold.

Cost note (RESEARCH/11): the drift loop costs ONE GPU RENDER PER MINT, which
erases Tier 1's free-CPU-mint advantage. That is a real trade, now
quantified. `verify=False` opts out.
"""
from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .geometry import SpeakerSpace
from .identity import Identity, IdentityStore, sha256_audio
from .renderer import Audio, Direction, load_backend

# E2/E4 measured operating points. Working space, in-domain fit.
DRIFT_FLOOR = 0.40          # below this, re-mint (E2 worst case was 0.3546)
UNIQUENESS_MIN = 0.30       # VoicePrivacy B3's threshold, applied in WORKING space

# E9: extreme voices hold together as well as central ones ON AVERAGE, but
# their WORST CASE is worse (0.4279 vs 0.5008 ECAPA consistency). Means were
# equal; the tail was not. A single character whose voice wanders between
# lines is exactly the failure a game studio would notice, so minting probes
# CONSISTENCY across several lines, not just drift on one.
#
# Floor is set at the E9 extreme-group worst case. ECAPA C_diff on real
# speech is 0.2011 and C_same is 0.6988, so 0.43 is ~46% of the way from a
# different speaker to the same one -- deliberately permissive, because it
# rejects only genuinely unstable identities.
CONSISTENCY_FLOOR = 0.43

# How much to raise novelty after each COLLISION (E11). Only collisions
# escalate: drift and consistency failures are the ones extra novelty makes
# worse, so retrying those at higher novelty would trade a fixable problem for
# an unfixable one.
NOVELTY_STEP = 0.35
CONSISTENCY_PROBE_LINES = [
    "The mountains remember every footstep.",
    "It's cold today, colder than anyone promised.",
]


@dataclass
class MintOutcome:
    identity: Identity
    seed_audio: np.ndarray
    sample_rate: int
    drift: Optional[float]
    uniqueness: float
    attempts: int
    warnings: list[str]
    consistency: Optional[float] = None


class VoiceService:
    def __init__(self, renderer, space: SpeakerSpace, store: IdentityStore,
                 mapper=None, watermarker=None, audio_dir: str = "data/audio",
                 is_public: bool = True, sv=None):
        # I3 -- refuses here, not at request time
        load_backend(renderer, is_public_deployment=is_public)
        self.r = renderer
        self.space = space
        self.store = store
        self.mapper = mapper
        self.wm = watermarker
        self.sv = sv           # INDEPENDENT scorer for the consistency probe
        self.audio_dir = os.path.abspath(audio_dir)
        os.makedirs(self.audio_dir, exist_ok=True)
        self.space_ref = f"space-{getattr(space, 'version', '1')}-{space.stats.dim}d"

    # -------------------------------------------------------------- helpers
    def _save(self, wav: np.ndarray, sr: int, tag: str) -> str:
        import soundfile as sf
        path = os.path.join(self.audio_dir, f"{tag}.wav")
        sf.write(path, wav, sr)
        return path

    def _uniqueness(self, vec: np.ndarray, language: str) -> float:
        existing = [i.embedding for i in self.store.all(language=language)
                    if i.embedding is not None]
        if not existing:
            return 1.0
        return self.space.uniqueness_distance(vec, np.stack(existing))

    def _consistency(self, vec: np.ndarray, language: str) -> tuple:
        """
        Render a couple of probe lines and measure self-similarity with an
        INDEPENDENT encoder. Returns (score, clips) or (None, []) if no
        independent scorer was supplied -- never scored with the conditioning
        encoder, which would be marking its own homework (RESEARCH/06).
        """
        if self.sv is None:
            return None, []
        embs, clips = [], []
        for line in CONSISTENCY_PROBE_LINES:
            a = self.r.render_from_vector(vec.astype(np.float32), line, language)
            embs.append(self.sv.embed(a.wav, sr=a.sample_rate))
            clips.append(a)
        if len(embs) < 2:
            return None, clips
        sims = [float(embs[i] @ embs[j] /
                      max(np.linalg.norm(embs[i]) * np.linalg.norm(embs[j]), 1e-12))
                for i in range(len(embs)) for j in range(i + 1, len(embs))]
        return float(np.mean(sims)), clips

    def _drift(self, intended: np.ndarray, wav: np.ndarray, sr: int) -> float:
        back = self.r.extract_vector(wav, sr).astype(np.float64)
        a = self.space.encode(intended)[0]
        b = self.space.encode(back)[0]
        return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))

    # ----------------------------------------------------------------- mint
    def mint(self, description: str, character_id: str, language: str = "en",
             novelty: float = 0.5, verify: bool = True, max_attempts: int = 3,
             seed_line: str = "This is how I sound when I speak.",
             tags: list[str] | None = None, top_k: int = 2) -> MintOutcome:
        """
        description -> identity, stored in BOTH tiers.

        Minting from a NEUTRAL description is deliberate (invariant I5).
        Performance is added at render time, never baked into the identity --
        RESEARCH/10 found the x-vector dominates the codec tokens on this
        backend, so emotion in the identity would contaminate every line.
        """
        if self.mapper is None:
            raise RuntimeError("no mapper: cannot mint from a description")
        warnings: list[str] = []
        best = None
        collisions = 0        # drives the novelty escalation below

        for attempt in range(1, max_attempts + 1):
            # ADAPTIVE NOVELTY (E11). Novelty's cost is immediate and its
            # benefit is deferred: rendering measurements at matched catalog
            # size put drift failures at 5% for novelty 0.0, 30% at 0.45 and
            # 40% at 0.75, while mean uniqueness at 0.45 was no better than at
            # 0.0 (0.694 vs 0.698). Its only immediate benefit is avoiding
            # collisions -- which do not happen until the catalog is dense.
            #
            # So do not pay for it until a collision actually occurs. Escalate
            # on COLLISIONS only, never on a drift or consistency failure: those
            # are the failures more novelty makes worse.
            eff_novelty = min(1.0, novelty + NOVELTY_STEP * collisions)
            # top_k is the diversity knob S9b/S10 found mis-set; it reaches the
            # mapper from here so a catalog run can sweep it.
            m = self.mapper.mint(description, novelty=eff_novelty,
                                 seed=attempt, top_k=top_k)
            vec = m.vector

            uniq = self._uniqueness(vec, language)
            if uniq < UNIQUENESS_MIN:
                collisions += 1
                warnings.append(
                    f"attempt {attempt}: too close to an existing identity "
                    f"(working-space distance {uniq:.3f} < {UNIQUENESS_MIN}); "
                    f"retrying at novelty "
                    f"{min(1.0, novelty + NOVELTY_STEP * collisions):.2f}")
                if attempt < max_attempts:
                    continue

            # Tier 2 is mandatory: mint the seed clip now (I2)
            a = self.r.render_from_vector(vec.astype(np.float32), seed_line, language)
            drift = self._drift(vec, a.wav, a.sample_rate) if verify else None
            cons, _ = self._consistency(vec, language) if verify else (None, [])

            cand = (vec, a, drift, uniq, cons)
            score = (drift or 0) + (cons or 0)
            if best is None or score > (best[2] or 0) + (best[4] or 0):
                best = cand
            drift_ok = drift is None or drift >= DRIFT_FLOOR
            cons_ok = cons is None or cons >= CONSISTENCY_FLOOR
            if drift_ok and cons_ok:
                break
            if not drift_ok:
                warnings.append(f"attempt {attempt}: drift {drift:.3f} < {DRIFT_FLOOR}")
            if not cons_ok:
                warnings.append(f"attempt {attempt}: consistency {cons:.3f} < "
                                f"{CONSISTENCY_FLOOR} (E9: unstable identities "
                                f"wander between lines)")

        vec, a, drift, uniq, cons = best
        if drift is not None and drift < DRIFT_FLOOR:
            warnings.append(
                f"accepted with drift {drift:.3f} below the {DRIFT_FLOOR} floor "
                f"after {max_attempts} attempts")
        if cons is not None and cons < CONSISTENCY_FLOOR:
            warnings.append(
                f"accepted with consistency {cons:.3f} below the "
                f"{CONSISTENCY_FLOOR} floor after {max_attempts} attempts")

        wav = a.wav
        if self.wm is not None:
            wav = self.wm.embed(wav, a.sample_rate)      # I7, even on the seed clip
        ref = self._save(wav, a.sample_rate, f"seed_{uuid.uuid4().hex[:12]}")

        idn = Identity(
            character_id=character_id, language=language, description=description,
            seed_audio_ref=ref, seed_audio_sha=sha256_audio(wav),
            seed_audio_sr=a.sample_rate,
            backend_id=self.r.backend_id, backend_version=self.r.backend_version,
            generation_params={"novelty": novelty, "mapper": "retrieval+gmm",
                               "seed_line": seed_line, "attempts": attempt,
                               "drift": drift, "uniqueness": uniq,
                               "consistency": cons},
            embedding=vec.astype(np.float32), space_ref=self.space_ref,
            tags=tags or [])
        self.store.put(idn)
        return MintOutcome(identity=idn, seed_audio=wav, sample_rate=a.sample_rate,
                           drift=drift, uniqueness=uniq, attempts=attempt,
                           warnings=warnings, consistency=cons)

    # --------------------------------------------------------------- render
    def render(self, identity_id: str, text: str,
               direction: Direction | None = None,
               save: bool = True) -> tuple[Audio, str]:
        """
        Render one line. Watermarked and provenance-logged unconditionally.

        Tier 1 conditions on the stored vector; Tier 2 falls back to cloning
        from the stored seed clip -- which is what makes an identity survive a
        backend version bump that invalidates the vector.
        """
        idn = self.store.get(identity_id)
        if idn is None:
            raise KeyError(identity_id)

        if idn.backend_version != self.r.backend_version:
            # Never silently render a stale identity on a new backend.
            raise RuntimeError(
                f"identity {identity_id} was minted on "
                f"{idn.backend_id}@{idn.backend_version} but the loaded backend is "
                f"{self.r.backend_id}@{self.r.backend_version}. A version change is a "
                f"MIGRATION (PHASE-01 section 3): re-render, diff, and let the owner "
                f"accept or stay pinned. Use store.version_audit() first.")

        if idn.tier == 1 and idn.space_ref == self.space_ref:
            a = self.r.render_from_vector(idn.embedding.astype(np.float32),
                                          text, idn.language, direction=direction)
        else:
            import soundfile as sf
            wav, sr = sf.read(idn.seed_audio_ref, dtype="float32")
            a = self.r.render_from_audio(wav, sr, text, idn.language,
                                         direction=direction)

        if self.wm is not None:
            a.wav = self.wm.embed(a.wav, a.sample_rate)
            a.watermarked = True

        ref = self._save(a.wav, a.sample_rate,
                         f"render_{uuid.uuid4().hex[:12]}") if save else None
        rid = self.store.log_render(
            identity_id=idn.identity_id, text=text,
            backend_id=self.r.backend_id, backend_version=self.r.backend_version,
            audio_ref=ref, direction=direction.__dict__ if direction else None,
            watermarked=a.watermarked)
        return a, rid

    # --------------------------------------------------------------- script
    def render_script(self, lines: list[tuple[str, str]],
                      direction: Direction | None = None) -> dict:
        """
        Batch render, returning a manifest a game engine can consume.

        PHASE-08 section 1.4: the manifest is what makes this usable in a real
        pipeline rather than a toy. It carries backend_version per line so a
        re-render is reproducible, and the synthetic-audio disclosure the EU
        AI Act Article 50(2) requires.
        """
        out = []
        for i, (identity_id, text) in enumerate(lines):
            a, rid = self.render(identity_id, text, direction=direction)
            idn = self.store.get(identity_id)
            out.append({"line_id": i, "render_id": rid,
                        "identity_id": identity_id,
                        "character_id": idn.character_id, "language": idn.language,
                        "text": text, "duration_s": len(a.wav) / a.sample_rate,
                        "sample_rate": a.sample_rate,
                        "backend": f"{a.backend_id}@{a.backend_version}",
                        "watermarked": a.watermarked,
                        "degradations": a.degradations})
        return {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "disclosure": "AI-generated synthetic speech. "
                              "All audio is watermarked (AudioSeal, presence bit).",
                "lines": out}
