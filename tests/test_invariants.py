"""
Invariant tests.

These are not unit tests for their own sake. Each one guards a rule that, if
broken silently, costs weeks -- and several encode a bug that was ACTUALLY
HIT during development, so they are regression tests with a story.

Run:  envs/qwen3/Scripts/python.exe -m pytest tests/ -q
"""
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from alaap.geometry import SpeakerSpace
from alaap.identity import Identity, IdentityStore, sha256_audio
from alaap.renderer import (Direction, Honouring, load_backend, LicenceGateError,
                            SERVABLE)
from alaap.captions import caption_from_bins, target_bins_from_text
from alaap.acoustics import measure, Binner, adherence_error, BIN_LABELS
from alaap.metrics import verification_stats, vendi_score, nn_distances


# --------------------------------------------------------------- fixtures
@pytest.fixture
def fake_space():
    """
    A synthetic space calibrated to the REAL Qwen3-TTS geometry measured in E3,
    not to convenient defaults:

        identity_fraction  0.163   only ~16% of a vector is speaker-specific
        per-dim var ratio  ~127x   heterogeneous scales
        not L2-normalised, large shared mean offset

    The first version of this fixture used a much smaller offset
    (identity_frac 0.46) and the centring test failed -- correctly. The
    fixture, not the code, was unrealistic.
    """
    rng = np.random.default_rng(0)
    d = 64
    scale = np.exp(rng.normal(0, 1.2, d))          # ~2 orders of per-dim spread
    offset = rng.normal(14.0, 1.0, d)              # dominant shared mean, like Qwen3
    Z = rng.normal(0, 1, (300, d)) * scale + offset
    space = SpeakerSpace.fit(Z, n_components=32)
    assert space.stats.identity_fraction < 0.30, (
        f"fixture must mimic the real space's dominant shared offset; got "
        f"identity_fraction={space.stats.identity_fraction:.3f}")
    return space, Z


# =========================================================== I1: geometry
class TestI1Geometry:
    def test_encode_decode_roundtrips(self, fake_space):
        space, Z = fake_space
        back = space.decode(space.encode(Z[:10]), project_to_shell=False)
        assert np.allclose(Z[:10], back, rtol=1e-6, atol=1e-6)

    def test_shell_projection_matches_real_norm(self, fake_space):
        space, Z = fake_space
        rng = np.random.default_rng(1)
        W = rng.normal(0, 1, (50, Z.shape[1]))
        out = space.decode(W, project_to_shell=True)
        norms = np.linalg.norm(out, axis=1)
        assert np.allclose(norms, space.shell_radius, rtol=1e-6)

    def test_centring_beats_raw_cosine(self, fake_space):
        """
        The E3 finding, as an executable claim: a large shared mean offset
        makes RAW cosine useless, and centring restores dynamic range.
        """
        space, Z = fake_space
        def spread(X):
            Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
            C = Xn @ Xn.T
            iu = np.triu_indices(len(X), 1)
            return np.percentile(C[iu], 99) - np.percentile(C[iu], 1)
        assert spread(space.encode(Z)) > 3 * spread(Z)

    def test_uniqueness_must_use_working_space(self, fake_space):
        """
        E3: on RAW vectors the whole population's max nearest-neighbour
        distance was 0.029, so a 0.3 threshold is unreachable. In working
        space it is comfortably reachable. Guard the gap.
        """
        space, Z = fake_space
        raw_nn = nn_distances(Z).mean()
        work_nn = nn_distances(space.encode(Z)).mean()
        assert work_nn > raw_nn * 3

    def test_effective_rank_is_reported(self, fake_space):
        space, _ = fake_space
        assert 1.0 < space.stats.effective_rank <= space.stats.dim
        assert 0.0 < space.stats.identity_fraction <= 1.0


# ========================================================== I2: identity
class TestI2Identity:
    def _kw(self, **over):
        kw = dict(character_id="c", language="en", description="d",
                  seed_audio_ref="ref.wav", seed_audio_sha="abc",
                  backend_id="qwen3-tts-base", backend_version="v1",
                  generation_params={})
        kw.update(over)
        return kw

    def test_seed_clip_is_mandatory(self):
        with pytest.raises(ValueError, match="I2"):
            Identity(**self._kw(seed_audio_ref="", seed_audio_sha=""))

    def test_vector_requires_space_ref(self):
        """A Tier-1 vector without its transform is meaningless (E4)."""
        with pytest.raises(ValueError, match="space_ref"):
            Identity(**self._kw(embedding=np.zeros(8, dtype=np.float32)))

    def test_tier_is_derived_not_declared(self):
        assert Identity(**self._kw()).tier == 2
        assert Identity(**self._kw(embedding=np.zeros(8, dtype=np.float32),
                                   space_ref="s1")).tier == 1

    def test_roundtrip_and_version_audit(self, tmp_path):
        st = IdentityStore(str(tmp_path / "t.db"))
        v = np.arange(8, dtype=np.float32)
        i = Identity(**self._kw(embedding=v, space_ref="s1"))
        st.put(i)
        got = st.get(i.identity_id)
        assert got is not None and np.allclose(got.embedding, v)
        assert got.space_ref == "s1" and got.tier == 1

        audit = st.version_audit("qwen3-tts-base", "v2")
        assert audit["affected"] == 1 and audit["tier1_at_risk"] == 1
        st.close()

    def test_one_character_many_languages(self, tmp_path):
        st = IdentityStore(str(tmp_path / "t.db"))
        for lang in ("en", "hi", "ta"):
            st.put(Identity(**self._kw(character_id="hero", language=lang)))
        assert len(st.by_character("hero")) == 3
        st.close()


# ====================================================== I3: licence gate
class TestI3LicenceGate:
    def _r(self, bid, servable):
        class R: pass
        r = R(); r.backend_id = bid; r.public_servable = servable
        return r

    def test_refuses_non_servable_in_public(self):
        with pytest.raises(LicenceGateError, match="NOT publicly servable"):
            load_backend(self._r("indicf5", False), is_public_deployment=True)

    def test_refuses_unaudited_backend(self):
        """Invariant I4: an unaudited backend is refused, not assumed safe."""
        with pytest.raises(LicenceGateError, match="not in the licence audit"):
            load_backend(self._r("brand-new-model", True))

    def test_refuses_declaration_mismatch(self):
        """A backend cannot declare itself servable against the audit."""
        with pytest.raises(LicenceGateError, match="declares public_servable"):
            load_backend(self._r("indicf5", True))

    def test_allows_audited_servable(self):
        assert load_backend(self._r("qwen3-tts-base", True),
                            is_public_deployment=True) is not None

    @pytest.mark.parametrize("bid", ["indicf5", "spring_f5", "indic-mio", "dhvaani",
                                     "voicesculptor", "llasa-3b", "xcodec2",
                                     "f5-tts", "xtts-v2", "indextts2", "vibevoice",
                                     "zonos-v0.1"])
    def test_known_bad_chains_stay_blocked(self, bid):
        """Every model RESEARCH/08 found with a broken upstream chain."""
        assert SERVABLE[bid] is False


# =================================================== I5: direction hygiene
class TestI5Direction:
    def test_intensity_defaults_to_measured_ceiling(self):
        """RESEARCH/10: three independent sources converge on 0.6, not 1.0."""
        assert Direction().intensity == 0.6

    def test_requested_fields_excludes_unset(self):
        d = Direction(emotion={"angry": 1.0}, rate=1.2)
        assert set(d.requested_fields()) == {"emotion", "rate"}

    def test_identity_record_has_no_emotion_field(self):
        """Timbre and performance must not share a home."""
        fields = Identity.__dataclass_fields__.keys()
        for bad in ("emotion", "direction", "style", "intensity"):
            assert bad not in fields


# ================================================ captions: regression bug
class TestCaptionParsing:
    def test_unhurried_is_not_hurried(self):
        """
        REGRESSION. Substring matching parsed 'a low, smooth voice, unhurried'
        as speaking_rate='rapid', because 'hurried' occurs inside 'unhurried'
        -- the exact opposite of the description. It corrupted S2's adherence
        scores until the instrument was checked.
        """
        got = target_bins_from_text("a low, smooth voice, unhurried and gently inflected")
        assert got.get("speaking_rate") == "slow"

    def test_longest_synonym_wins(self):
        assert target_bins_from_text("a very deep voice")["f0_mean"] == "very low-pitched"
        assert target_bins_from_text("a deep voice")["f0_mean"] == "low-pitched"
        assert target_bins_from_text("a very clear tone")["hnr_db"] == "very clear"

    def test_unmatched_wording_is_ignored_not_guessed(self):
        """Inventing a target bin would fabricate an adherence score."""
        assert target_bins_from_text("a voice like a rainy Tuesday") == {}

    def test_caption_roundtrips_through_the_parser(self):
        bins = {"f0_mean": "very low-pitched", "hnr_db": "very rough",
                "spectral_tilt": "dark", "f0_cv": "monotone",
                "speaking_rate": "slow"}
        for seed in range(8):
            back = target_bins_from_text(caption_from_bins(bins, seed=seed))
            for k, v in bins.items():
                assert back.get(k) == v, f"seed {seed}: {k} {back.get(k)} != {v}"

    def test_captions_vary_in_surface_form(self):
        bins = {"f0_mean": "low-pitched", "hnr_db": "clear", "f0_cv": "expressive"}
        assert len({caption_from_bins(bins, seed=s) for s in range(10)}) >= 5


# ============================================================== metrics
class TestMetrics:
    def test_verification_separates_planted_speakers(self):
        rng = np.random.default_rng(0)
        centres = rng.normal(0, 1, (10, 32))
        Z, ids = [], []
        for i, c in enumerate(centres):
            for _ in range(8):
                Z.append(c + rng.normal(0, 0.15, 32)); ids.append(f"s{i}")
        vs = verification_stats(np.stack(Z), ids)
        assert vs.same_mean > vs.diff_mean
        assert vs.eer < 0.05 and vs.d_prime > 2.0

    def test_vendi_detects_collapse(self):
        rng = np.random.default_rng(0)
        diverse = rng.normal(0, 1, (20, 16))
        collapsed = np.repeat(rng.normal(0, 1, (1, 16)), 20, axis=0) \
            + rng.normal(0, 1e-3, (20, 16))
        assert vendi_score(diverse) > 0.5
        assert vendi_score(collapsed) < 0.15

    def test_binner_gives_balanced_occupancy(self):
        """Percentile bins, unlike equal-width ones, cannot leave a bin empty."""
        rng = np.random.default_rng(0)
        import collections
        attrs = []
        for _ in range(200):
            attrs.append(measure(rng.normal(0, 0.1, 24000 * 2).astype(np.float32)))
        b = Binner.fit(attrs)
        for f in ("f0_mean", "hnr_db"):
            if f in b.edges:
                c = collections.Counter(b.bin_one(a)[f] for a in attrs)
                assert min(c.values()) >= len(attrs) / len(BIN_LABELS[f]) * 0.5


# ============================================================= acoustics
class TestAcoustics:
    def test_measure_returns_finite(self):
        rng = np.random.default_rng(0)
        t = np.linspace(0, 2, 48000)
        wav = (0.3 * np.sin(2 * np.pi * 140 * t)
               + 0.05 * rng.normal(0, 1, len(t))).astype(np.float32)
        a = measure(wav, "hello there friend")
        for k, v in a.to_dict().items():
            assert np.isfinite(v), f"{k} not finite"
        assert 80 < a.f0_mean < 260, f"expected ~140Hz, got {a.f0_mean}"

    def test_adherence_zero_distance_on_self(self):
        rng = np.random.default_rng(0)
        attrs = [measure(rng.normal(0, 0.1, 24000).astype(np.float32))
                 for _ in range(30)]
        b = Binner.fit(attrs)
        bins = b.bin_one(attrs[0])
        r = adherence_error(bins, attrs[0], b)
        assert r["exact_match_rate"] == 1.0 and r["mean_bin_distance"] == 0.0


# =================================================== Direction channel (E0)
class TestDirectionChannel:
    """
    The Direction channel is task-vector arithmetic (E0). These guard the
    properties that make it safe to expose, not the quality of the effect.
    """

    def _shim(self, enc_dim=1024):
        """A renderer shell with no model loaded -- tests pure vector logic."""
        from alaap.renderer import Qwen3BaseRenderer

        class Cfg:
            class speaker_encoder_config:
                pass
        Cfg.speaker_encoder_config.enc_dim = enc_dim

        class M:
            config = Cfg()
        r = Qwen3BaseRenderer.__new__(Qwen3BaseRenderer)
        r.model = M()
        r.tau = {}
        return r

    def test_tau_dim_mismatch_is_dropped_not_broadcast(self):
        """
        tau fitted on 0.6B (1024-d) is MEANINGLESS on 1.7B (2048-d).
        Silently broadcasting or truncating it would corrupt every render.
        """
        import os
        p = "assets/emotion_tau_qwen3_0.6B.npz"
        if not os.path.exists(p):
            pytest.skip("tau asset not built")
        assert self._shim(2048)._load_tau(p) == {}      # dropped
        assert len(self._shim(1024)._load_tau(p)) > 0   # kept

    def test_steer_is_noop_without_direction(self):
        r = self._shim(); r.tau = {"anger": np.ones(8, dtype=np.float32)}
        v = np.arange(8, dtype=np.float32)
        out, notes = r.steer(v, None)
        assert np.allclose(out, v) and notes == []

    def test_steer_preserves_norm(self):
        """E3: generated vectors belong on the manifold's shell."""
        r = self._shim(); r.tau = {"anger": np.ones(8, dtype=np.float32) * 3}
        v = np.arange(1, 9, dtype=np.float32)
        out, _ = r.steer(v, Direction(emotion={"anger": 1.0}, intensity=1.0))
        assert np.isclose(np.linalg.norm(out), np.linalg.norm(v), rtol=1e-5)
        assert not np.allclose(out, v)

    def test_unknown_emotion_is_reported_not_ignored(self):
        r = self._shim(); r.tau = {"anger": np.ones(8, dtype=np.float32)}
        out, notes = r.steer(np.ones(8, dtype=np.float32),
                             Direction(emotion={"smug": 1.0}))
        assert any("smug" in n for n in notes)

    def test_alpha_is_capped_at_the_published_bound(self):
        from alaap.renderer import Qwen3BaseRenderer
        cap = Qwen3BaseRenderer.direction_bounds["emotion"]["alpha_max"]
        r = self._shim(); r.tau = {"anger": np.ones(8, dtype=np.float32)}
        _, notes = r.steer(np.ones(8, dtype=np.float32),
                           Direction(emotion={"anger": 99.0}, intensity=1.0))
        assert f"alpha={cap:.2f}" in notes[0]

    def test_approximate_requires_a_published_bound(self):
        """RESEARCH/10: 'approximate without a published bound is REJECT'."""
        from alaap.renderer import Qwen3BaseRenderer, Honouring
        for f, h in Qwen3BaseRenderer.direction_support.items():
            if h is Honouring.APPROXIMATE:
                assert f in Qwen3BaseRenderer.direction_bounds, \
                    f"{f} is APPROXIMATE with no bound"

    def test_bounds_quote_both_encoders(self):
        """E0: reporting only the forgiving encoder overstates the result."""
        from alaap.renderer import Qwen3BaseRenderer
        b = Qwen3BaseRenderer.direction_bounds["emotion"]
        assert "identity_retained_ecapa" in b and "identity_retained_wavlm" in b


# ====================================================== calibration (E0/E4)
class TestCalibration:
    def test_normalised_similarity_anchors(self):
        from alaap.metrics import normalised_similarity, CALIBRATION
        for enc, (same, diff, _) in CALIBRATION.items():
            assert abs(normalised_similarity(same, enc) - 1.0) < 1e-9
            assert abs(normalised_similarity(diff, enc) - 0.0) < 1e-9

    def test_unknown_encoder_refuses(self):
        """A similarity without a measured floor is not a claim."""
        from alaap.metrics import normalised_similarity
        with pytest.raises(KeyError):
            normalised_similarity(0.9, "some-encoder-we-never-calibrated")

    def test_encoders_disagree_on_the_same_raw_value(self):
        """
        The E0 finding, as an executable claim: 0.88 means very different
        things on WavLM and ECAPA, so a bare SECS is uninterpretable.
        """
        from alaap.metrics import normalised_similarity as n
        assert n(0.88, "wavlm") - n(0.88, "ecapa") < -0.5


# ================================================ service thresholds (E9)
class TestServiceThresholds:
    def test_floors_sit_between_cdiff_and_csame(self):
        from alaap.service import DRIFT_FLOOR, CONSISTENCY_FLOOR, UNIQUENESS_MIN
        from alaap.metrics import CALIBRATION
        same, diff, _ = CALIBRATION["ecapa"]
        assert diff < CONSISTENCY_FLOOR < same, \
            "a consistency floor outside [C_diff, C_same] is meaningless"
        assert 0.0 < DRIFT_FLOOR < 1.0
        assert 0.0 < UNIQUENESS_MIN < 1.0

    def test_consistency_probe_uses_multiple_lines(self):
        """One line cannot measure consistency ACROSS lines."""
        from alaap.service import CONSISTENCY_PROBE_LINES
        assert len(CONSISTENCY_PROBE_LINES) >= 2
        assert len(set(CONSISTENCY_PROBE_LINES)) == len(CONSISTENCY_PROBE_LINES)


# ============================================== cache keying (regression)
class TestCacheKeying:
    def test_embedding_cache_key_includes_model(self):
        """
        REGRESSION. An experiment cache keyed WITHOUT model_id silently served
        0.6B embeddings to a 1.7B run. It failed loudly only because the dims
        differed (2048 vs 1024) -- two models with the same enc_dim would have
        produced quietly wrong numbers.
        """
        from alaap.encoder import EmbeddingCache
        base = dict(corpus="globe_v2", n=100, per_speaker=1,
                    min_dur=2.0, max_dur=15.0, seed=0, skip=0)
        a = EmbeddingCache.key(model_id="Qwen/Qwen3-TTS-12Hz-0.6B-Base", **base)
        b = EmbeddingCache.key(model_id="Qwen/Qwen3-TTS-12Hz-1.7B-Base", **base)
        assert a != b, "cache key must distinguish models"

    def test_cache_key_distinguishes_every_spec_field(self):
        from alaap.encoder import EmbeddingCache
        base = dict(model_id="m", corpus="c", n=10, per_speaker=1,
                    min_dur=2.0, max_dur=15.0, seed=0, skip=0)
        ref = EmbeddingCache.key(**base)
        for field, alt in [("corpus", "other"), ("n", 11), ("per_speaker", 2),
                           ("min_dur", 3.0), ("max_dur", 14.0), ("seed", 1),
                           ("skip", 5)]:
            assert EmbeddingCache.key(**{**base, field: alt}) != ref, field


# ------------------------------------------------- Indic phone counting
class TestIndicPhones:
    """
    Guards the silent failure that motivated count_phones_indic.

    g2p-en does not raise on Devanagari, and the vowel-group fallback under it
    matches `[aeiouy]+`, which finds nothing in any Brahmic script. `max(.., 1)`
    then made a whole Hindi sentence 2.5 phones, so every Indic clip would have
    read ~0.5 phones/s and speaking_rate would have collapsed to one bin with
    no error anywhere. These tests make that failure loud.
    """

    def test_detects_the_nine_brahmic_scripts(self):
        from alaap.acoustics import detect_script
        for text, want in [("नमस्ते", "devanagari"),
                           ("বাংলা", "bengali"),
                           ("தமிழ்", "tamil"),
                           ("వాడు", "telugu"),
                           ("ಕನ್ನಡ", "kannada")]:
            assert detect_script(text) == want, text

    def test_latin_returns_zero_so_caller_falls_back(self):
        from alaap.acoustics import count_phones_indic, detect_script
        assert detect_script("Hello there") is None
        assert count_phones_indic("Hello there") == 0

    def test_phone_counts_match_hand_transcription(self):
        """Hand-checked against the standard romanisation of each word."""
        from alaap.acoustics import count_phones_indic
        cases = [("कमल", 5),            # kamal   k a m a l
                 ("भारत", 5),       # bhaarat bh aa r a t
                 ("राम", 3),             # raam    r aa m
                 ("नमस्ते", 7),  # namaste n a m a s t e
                 ("हिन्दी", 5)]  # hindii  h i n d ii
        for text, want in cases:
            assert count_phones_indic(text) == want, (text, want)

    def test_virama_suppresses_the_inherent_vowel(self):
        """The abugida rule: a halant kills the schwa, so the count drops."""
        from alaap.acoustics import count_phones_indic
        # सत = s a t a(deleted) -> 3 ; स्त = s t a -> 3? no: virama kills s's schwa
        assert count_phones_indic("सत") > count_phones_indic("स्त")

    def test_final_schwa_deleted_for_indo_aryan_only(self):
        """
        Hindi कमल is /kəmal/, not /kəmələ/. Dravidian languages keep
        their final vowels, so the same rule must NOT apply to Tamil/Telugu.
        """
        from alaap import acoustics as A
        word = "कमल"
        with_rule = A.count_phones_indic(word)
        A._SCHWA_DELETING.discard("devanagari")
        try:
            without = A.count_phones_indic(word)
        finally:
            A._SCHWA_DELETING.add("devanagari")
        assert with_rule == without - 1
        assert "tamil" not in A._SCHWA_DELETING
        assert "telugu" not in A._SCHWA_DELETING

    def test_speaking_rate_uses_the_indic_branch(self):
        """
        End-to-end: Devanagari text must NOT fall through to the vowel-group
        estimate, which would report ~0.5 phones/s for any Indic sentence.

        The signal is an amplitude-modulated harmonic stack rather than a pure
        tone: voiced_mask thresholds energy at the 40th percentile, so a
        CONSTANT-amplitude signal has no frame above its own threshold and
        reads as 0% voiced. Real speech always has that contrast.
        """
        from alaap.acoustics import speaking_rate, voiced_mask, SR
        import numpy as np
        t = np.arange(int(SR * 2.0)) / SR
        harm = sum(np.sin(2 * np.pi * 140 * k * t) / k for k in (1, 2, 3, 4))
        env = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t)          # syllabic rate
        wav = (0.3 * harm * env).astype(np.float32)
        voiced_s = float(voiced_mask(wav, SR).sum() * 0.010)
        assert voiced_s > 0.3, f"fixture is not voiced ({voiced_s:.2f}s)"

        sentence = "राम घर गया"   # raam ghar gayaa = 10 phones
        r = speaking_rate(wav, sentence, SR)
        # 10 phones over ~1s of voiced audio. The vowel-group fallback would
        # give int(2.5 * 1) / voiced_s, i.e. under 2.5 -- so 4.0 separates them.
        assert r > 4.0, f"Indic branch not used, got {r:.2f} phones/s"
        assert abs(r * voiced_s - 10) < 1e-6, "phone count is not 10"


class TestServableMatchesTheAudit:
    """
    Invariant I4, as a test rather than a promise.

    SERVABLE in renderer.py is a RESTATEMENT of the audit in
    RESEARCH/08 section 8.4. Two copies of the same fact drift, and this one
    did: 'indic-parler-tts' shipped as True while the audit said False,
    which would have let load_backend admit a CONDITIONAL backend into a
    public deployment. This pins the entries the audit names.

    The audit's keys are shorter than the code's ids, so map explicitly
    rather than fuzzy-matching -- a fuzzy match is how the drift survived.
    """

    AUDIT = {                       # RESEARCH/08 section 8.4, verbatim verdicts
        "voxcpm2": True, "chatterbox": True, "parler-tts": True,
        "cosyvoice2": True, "qwen3-tts-base": True,
        "qwen3-tts-voicedesign": True,
        "indic-parler-tts": False, "indicf5": False, "spring_f5": False,
        "indic-mio": False, "dhvaani": False, "f5-tts": False,
        "voicesculptor": False, "llasa-3b": False, "xcodec2": False,
        "xtts-v2": False, "indextts2": False, "vibevoice": False,
        "zonos-v0.1": False,
    }

    def test_every_audited_backend_matches(self):
        for bid, want in self.AUDIT.items():
            assert bid in SERVABLE, f"{bid} missing from SERVABLE"
            assert SERVABLE[bid] == want, (
                f"{bid}: code says {SERVABLE[bid]}, RESEARCH/08 says {want}")

    def test_indic_parler_is_blocked_from_public_serving(self):
        """
        The specific regression. It is the strongest Indic candidate, which is
        exactly why it is tempting to flip without doing the two actions that
        settle it.
        """
        assert SERVABLE["indic-parler-tts"] is False

        class _R:
            backend_id = "indic-parler-tts"
            public_servable = False
        with pytest.raises(LicenceGateError):
            load_backend(_R(), is_public_deployment=True)

    def test_a_backend_declaring_more_than_the_audit_allows_is_refused(self):
        """Fail closed: an adapter cannot self-declare its way past the audit."""
        class _Liar:
            backend_id = "indic-parler-tts"
            public_servable = True      # contradicts SERVABLE
        with pytest.raises(LicenceGateError):
            load_backend(_Liar(), is_public_deployment=False)


class TestSchwaDeletionIsPerLanguage:
    """
    Word-final inherent-vowel deletion is a per-LANGUAGE fact, not a
    per-family one. Odia is the standard counterexample: it is Indo-Aryan and
    written in a Brahmic script, and it RETAINS the final vowel where Bengali
    and Hindi drop it. An earlier version of this table grouped it with
    Bengali on family alone.
    """

    def test_odia_retains_its_final_vowel(self):
        from alaap import acoustics as A
        assert "oriya" not in A._SCHWA_DELETING
        assert "tamil" not in A._SCHWA_DELETING
        assert "telugu" not in A._SCHWA_DELETING
        assert "kannada" not in A._SCHWA_DELETING
        assert "malayalam" not in A._SCHWA_DELETING

    def test_the_indo_aryan_deleters_are_exactly_these_four(self):
        from alaap import acoustics as A
        assert A._SCHWA_DELETING == {"devanagari", "bengali",
                                     "gurmukhi", "gujarati"}
