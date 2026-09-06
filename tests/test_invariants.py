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


class TestTrainingDataGate:
    """
    RESEARCH/08 section 8.5: "public_servable guards Q2. NOTHING currently guards
    Q3 -- and Q3 is the one that costs a retrain."

    Q3 is whether weights trained on a corpus may be released. Unlike a
    serving mistake, this one is baked into the parameters. ADR-006 makes it
    live: route C (train our own Indic tower) is the only Indic route no
    third party can veto, and this gate is what stands between it and an
    unreleasable checkpoint.
    """

    def test_a_clean_mix_passes_and_resolves(self):
        from alaap.provenance import assert_trainable
        mix = assert_trainable(["indicvoices_r", "libritts_r", "globe_v2"])
        assert len(mix) == 3
        assert all(c.train_releasable for c in mix)

    def test_a_noncommercial_corpus_blocks_the_mix(self):
        """Expresso is the corpus that blocks Indic-Mio (RESEARCH/08 4.4)."""
        from alaap.provenance import assert_trainable, TrainingLicenceError
        with pytest.raises(TrainingLicenceError, match="Expresso"):
            assert_trainable(["indicvoices_r", "expresso"])

    def test_the_unresolved_iitm_eula_blocks_the_mix(self):
        """
        IITM IndicTTS is the open question behind indic-parler-tts being
        CONDITIONAL. It must stay blocked until IITM confirms in writing.
        """
        from alaap.provenance import assert_trainable, TrainingLicenceError
        with pytest.raises(TrainingLicenceError):
            assert_trainable(["indictts_iitm"])

    def test_an_unknown_corpus_fails_closed(self):
        """An unknown artefact and an unlicensed one carry the same risk."""
        from alaap.provenance import assert_trainable, TrainingLicenceError
        with pytest.raises(TrainingLicenceError, match="not in the training-data"):
            assert_trainable(["some_corpus_nobody_traced"])

    def test_empty_mix_is_refused(self):
        from alaap.provenance import assert_trainable, TrainingLicenceError
        with pytest.raises(TrainingLicenceError):
            assert_trainable([])

    def test_cc_by_corpus_must_carry_an_attribution_line(self):
        """
        A CC-BY corpus with no NOTICE line would ship weights in breach of
        section 3(a)(1), so the dataclass refuses to be constructed that way.
        """
        from alaap.provenance import Corpus
        with pytest.raises(ValueError, match="attribution"):
            Corpus(name="X", licence="CC-BY-4.0", train_releasable=True,
                   source_url="https://example.invalid")

    def test_attribution_is_generated_from_the_mix(self):
        """
        The file must be derived, not hand-written -- a hand-maintained NOTICE
        is wrong the first time the mix changes and nothing catches it.
        """
        from alaap.provenance import emit_attribution
        text = emit_attribution(["indicvoices_r", "libritts_r"])
        assert "IndicVoices-R" in text and "LibriTTS-R" in text
        assert "CC BY 4.0" in text
        # CC-BY 3(a)(1)(B): modification must be indicated
        assert "modified" in text.lower()
        # and it must not silently emit for an unreleasable mix
        import pytest as _p
        from alaap.provenance import TrainingLicenceError
        with _p.raises(TrainingLicenceError):
            emit_attribution(["emilia"])

    def test_checkpoint_metadata_pins_the_mix_to_the_weights(self):
        """
        Without this, a routine data change silently alters what the released
        weights may be licensed under.
        """
        from alaap.provenance import checkpoint_metadata
        m = checkpoint_metadata(["indicvoices_r"], backend_version="v0.1")
        assert m["backend_version"] == "v0.1"
        assert m["training_mix"][0]["name"] == "IndicVoices-R"
        assert m["attribution_required"] and m["all_train_releasable"] is True


class TestMapperGenerativeBranchDoesNotCollapse:
    """
    E12 measured nearest-neighbour distance 0.000 between minted voices at
    novelty >= 0.90 -- i.e. minting 300 voices produced exact duplicates.

    Cause: `GaussianMixture.sample()` calls check_random_state(self.random_state)
    on every call, and random_state was a fixed int, so every call returned the
    IDENTICAL batch of points. The generative branch therefore had at most 64
    distinct outcomes available for the entire life of the mapper, no matter how
    many different descriptions it was asked for.

    That is a capacity ceiling on the whole catalog, and it was invisible at the
    default novelty=0.0 because the generative branch has zero weight there.
    """

    @staticmethod
    def _tiny_mapper(seed=0):
        """A mapper over synthetic anchors -- no corpus or GPU needed."""
        from alaap.geometry import SpeakerSpace
        from alaap.mapper import RetrievalMapper

        class _FakeText:
            model_id = "fake"

            def encode(self, x):
                one = isinstance(x, str)
                xs = [x] if one else list(x)
                out = np.zeros((len(xs), 8), dtype=np.float64)
                for i, s in enumerate(xs):
                    r = np.random.default_rng(abs(hash(s)) % (2 ** 31))
                    v = r.standard_normal(8)
                    out[i] = v / np.linalg.norm(v)
                return out

        rng = np.random.default_rng(seed)
        Z = rng.standard_normal((200, 64)) * 3.0 + 1.5
        space = SpeakerSpace.fit(Z, n_components=16)
        caps = [f"voice number {i} with a distinct manner" for i in range(200)]
        return RetrievalMapper(space, _FakeText(), pca_dims=8).fit(caps, Z)

    def test_repeated_gmm_draws_differ(self):
        """The direct cause: two draws with different rngs must not match."""
        m = self._tiny_mapper()
        a = m._sample_gmm(np.random.default_rng(1), 32)
        b = m._sample_gmm(np.random.default_rng(2), 32)
        assert not np.allclose(a, b), "gmm sampling ignores the caller's rng"

    def test_gmm_sampling_is_reproducible_for_one_seed(self):
        """...while staying deterministic, so a mint is reproducible."""
        m = self._tiny_mapper()
        a = m._sample_gmm(np.random.default_rng(7), 16)
        b = m._sample_gmm(np.random.default_rng(7), 16)
        assert np.allclose(a, b)

    def test_high_novelty_mints_are_distinct(self):
        """The observable symptom: no exact duplicates across many mints."""
        m = self._tiny_mapper()
        V = np.stack([m.mint(f"a voice of kind {i}", novelty=1.0, seed=i).vector
                      for i in range(60)])
        d = np.linalg.norm(V[:, None, :] - V[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        assert d.min() > 1e-6, "minted duplicate voices at novelty=1.0"


class TestIsolationMetricPassesItsOwnControl:
    """
    This metric exists because its predecessor failed silently.

    The first version compared log-likelihoods under a full-covariance GMM
    fitted to the reference speakers. On GLOBE_V2 that scored REAL held-out
    speakers at 0-1% -- the same as Gaussian noise -- because a 24-component
    full-covariance mixture over 50 dimensions fitted to ~1,250 points models
    proximity to its own training set, not plausibility. Every conclusion
    drawn from it was invalid.

    So the control is the test: a usable typicality metric MUST place real
    unseen speakers near the middle and noise at the extreme. If a future
    change breaks that, it breaks here rather than in an experiment.
    """

    @staticmethod
    def _split(seed=0, n=600, dim=32):
        """Two disjoint halves of a correlated, clustered population."""
        rng = np.random.default_rng(seed)
        centres = rng.standard_normal((4, dim)) * 4.0
        A = np.linalg.qr(rng.standard_normal((dim, dim)))[0][:, :8]
        X = np.vstack([centres[rng.integers(4)] + (A @ rng.standard_normal(8)) * 2.0
                       for _ in range(n)])
        return X[: n // 2], X[n // 2:]

    def test_real_unseen_speakers_land_near_the_middle(self):
        from alaap.metrics import isolation_pct
        held_out, reference = self._split()
        p = isolation_pct(held_out, reference)
        assert 25 <= p <= 75, f"real unseen speakers scored {p:.0f}%, expected ~50"

    def test_noise_is_maximally_isolated(self):
        from alaap.metrics import isolation_pct
        held_out, reference = self._split()
        rng = np.random.default_rng(1)
        noise = rng.standard_normal(held_out.shape) * reference.std()
        assert isolation_pct(noise, reference) >= 90

    def test_shuffling_the_dimensions_is_detected(self):
        """
        Destroying the correlation structure while keeping every marginal
        identical. A metric that only looks at per-dimension ranges misses this.
        """
        from alaap.metrics import isolation_pct
        held_out, reference = self._split()
        rng = np.random.default_rng(2)
        shuf = held_out.copy()
        for j in range(shuf.shape[1]):
            shuf[:, j] = shuf[rng.permutation(len(shuf)), j]
        assert isolation_pct(shuf, reference) > isolation_pct(held_out, reference) + 15


class TestFormantsAndVTL:
    """
    E14 found the catalog is limited by the DESCRIPTION, not the mapper: among
    real speakers, bin distance predicts voice distance at only rho = 0.260,
    and the mapper already transports 90% of that. Vocal-tract length was added
    to widen the description, because it is a strong correlate of perceived
    identity and largely independent of F0.

    The first implementation ran LPC at the native 24 kHz with order 26 and
    INVERTED the anatomy -- females measured longer vocal tracts than males,
    and F2/F3 came out lower for females. At 24 kHz most of those poles model
    the 5-12 kHz region, which carries no formant information. The fix is the
    standard recipe: resample to 2 x max_formant, ~2 poles per formant.

    Validated on 200 GLOBE_V2 clips with gender labels:
        VTL      female 15.53 cm  <  male 16.37 cm   (Cohen's d 0.52)
        F1       female 410 Hz    >  male 355 Hz
        F2       female 1513      >  male 1442
        F3       female 2662      >  male 2554
        VTL vs F0  r = -0.415     (correct sign, and not redundant with pitch)

    That needs the corpus, so it is not a unit test. These are the parts that
    can be checked without a network: a longer tube must measure longer.
    """

    @staticmethod
    def _vowel(f_scale=1.0, f0=120.0, sr=24000, dur=1.2):
        """
        Source-filter synthesis: a glottal pulse train through resonators at
        scaled formant frequencies. Scaling the formants by `f_scale` models a
        vocal tract 1/f_scale times as long.
        """
        from scipy.signal import lfilter
        n = int(sr * dur)
        t = np.arange(n) / sr
        src = np.zeros(n)
        src[:: max(int(sr / f0), 1)] = 1.0          # glottal pulses
        src = src - 0.95 * np.roll(src, 1)
        y = src
        for f, bw in ((500 * f_scale, 60), (1500 * f_scale, 90),
                      (2500 * f_scale, 120), (3500 * f_scale, 150)):
            r = np.exp(-np.pi * bw / sr)
            th = 2 * np.pi * f / sr
            y = lfilter([1.0], [1.0, -2 * r * np.cos(th), r * r], y)
        y = y / (np.abs(y).max() + 1e-9) * 0.8
        return (y * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)

    def test_formants_recover_synthesised_resonances(self):
        from alaap.acoustics import formants
        f = formants(self._vowel(1.0), 24000, n=4)
        assert np.isfinite(f[:3]).all(), f
        for got, want in zip(f[:3], (500, 1500, 2500)):
            assert abs(got - want) < 0.25 * want, f"got {f}, wanted ~500/1500/2500"

    def test_a_longer_tube_measures_longer(self):
        """
        The property that made the first implementation detectably wrong.
        Formants scaled UP by 1.25 means a tract 1/1.25 as long, so VTL must
        come out SMALLER.
        """
        from alaap.acoustics import formants, vocal_tract_length
        long_tract = vocal_tract_length(formants(self._vowel(1.0), 24000))
        short_tract = vocal_tract_length(formants(self._vowel(1.25), 24000))
        assert np.isfinite(long_tract) and np.isfinite(short_tract)
        assert short_tract < long_tract, (short_tract, long_tract)

    def test_formant_recovery_on_the_synthetic_is_accurate_for_f1(self):
        """
        Split out of the test above, because they were asserting two different
        things and only one of them is a property of the VTL formula.

        The old version required VTL to scale by 1.1-1.4x when the formants
        were scaled by 1.25x. That is really a claim about how accurately LPC
        recovers formants from this synthetic, and it is not uniformly good:
        F1 comes back at 711 Hz against a 700 Hz target, but F3 at 1884 Hz
        against 2600 Hz -- the estimator locks onto a spurious pole. Feeding
        that into any VTL formula dilutes the ratio through no fault of the
        formula.

        So: DIRECTION is asserted of the VTL estimator (above), and RECOVERY
        ACCURACY is asserted here, of the formant tracker, on the formant it
        actually recovers well.
        """
        from alaap.acoustics import formants
        f_lo = formants(self._vowel(1.0), 24000)
        f_hi = formants(self._vowel(1.25), 24000)
        assert abs(f_lo[0] - 500.0) < 90.0, f"F1 recovery: {f_lo[0]:.0f} vs 500"
        ratio = f_hi[0] / f_lo[0]
        assert 1.12 < ratio < 1.38, f"F1 scaled by {ratio:.3f}, expected ~1.25"

    def test_vtl_is_not_a_restatement_of_pitch(self):
        """Same tract, different F0 -> VTL must barely move."""
        from alaap.acoustics import formants, vocal_tract_length
        a = vocal_tract_length(formants(self._vowel(1.0, f0=110), 24000))
        b = vocal_tract_length(formants(self._vowel(1.0, f0=200), 24000))
        assert abs(a - b) / max(a, 1e-9) < 0.20, (a, b)

    def test_missing_formants_are_nan_not_invented(self):
        from alaap.acoustics import formants, vocal_tract_length
        assert np.isnan(vocal_tract_length(np.array([np.nan, np.nan])))
        assert np.isnan(formants(np.zeros(200, dtype=np.float32), 24000)).all()

    def test_old_caches_still_load_without_the_new_fields(self):
        """Attributes gained fields after several corpora were measured."""
        from alaap.acoustics import Attributes
        a = Attributes.from_dict({
            "f0_mean": 150.0, "f0_std": 20.0, "f0_range": 80.0,
            "speaking_rate": 12.0, "hnr_db": 10.0, "jitter": 0.01,
            "shimmer": 0.05, "spectral_tilt": -6.0, "snr_db": 30.0,
            "duration_s": 4.0, "voiced_frac": 0.6})
        assert np.isnan(a.vtl_cm) and a.f0_cv > 0


class TestCaptionAxesAreLive:
    """
    The synonym table was keyed `f0_std` for the whole life of the v2 binner.

    f0_std is a DEAD axis: S2 run 4 replaced it with f0_cv because f0_std in Hz
    correlates r=+0.72 with f0_mean. BIN_LABELS has no f0_std entry, and
    `adherence_error` skips any target bin whose axis it cannot find -- so every
    natural synonym for expressiveness, including the bare word "monotone",
    was parsed into an axis that was then silently discarded and never scored.

    Nothing failed. The score just quietly stopped covering that axis.
    """

    def test_every_parsed_axis_is_a_real_binned_axis(self):
        """The guard that would have caught it, for any future axis rename."""
        from alaap.captions import target_bins_from_text
        probes = ["monotone", "flat and dull", "highly animated", "expressive",
                  "lively", "a very deep gravelly voice speaking very slowly",
                  "squeaky and shrill", "warm and mellow", "raspy", "brisk"]
        seen = set()
        for p in probes:
            seen |= set(target_bins_from_text(p))
        assert seen, "the probes parsed nothing at all"
        unknown = seen - set(BIN_LABELS)
        assert not unknown, f"parser emits axes that cannot be scored: {unknown}"

    def test_bare_monotone_lands_on_the_live_axis(self):
        from alaap.captions import target_bins_from_text
        b = target_bins_from_text("calm and monotone")
        assert b.get("f0_cv") == "monotone", b
        assert "f0_std" not in b

    def test_every_parsed_label_exists_in_its_axis(self):
        """A label that is not in BIN_LABELS[axis] would raise in adherence."""
        from alaap.captions import target_bins_from_text
        for p in ["monotone", "very deep", "squeaky", "gravelly", "unhurried",
                  "very bright", "crystalline", "racing", "mellow"]:
            for axis, label in target_bins_from_text(p).items():
                assert label in BIN_LABELS[axis], (p, axis, label)


class TestAdaptiveNovelty:
    """
    E11 measured novelty's trade at matched catalog size (first 20 voices):

        novelty   clean   drift below floor   mean uniqueness
           0.00     85%                  5%             0.698
           0.45     50%                 30%             0.694
           0.75     30%                 40%             0.759

    novelty=0.45 bought NO mean-uniqueness gain over 0.0 while costing six
    times the drift failures. The cost lands on the first voice; the benefit --
    avoiding collisions -- does not matter until the catalog is dense (the
    control's collisions became serious between voice 30 and 40).

    So novelty is escalated only when a collision actually happens, and only
    for collisions: drift and consistency failures are exactly the ones more
    novelty makes worse.
    """

    class _Mapper:
        """Records the novelty each attempt was made at."""

        def __init__(self, vectors):
            self.vectors, self.seen = list(vectors), []

        def mint(self, description, novelty=0.5, seed=None):
            self.seen.append(round(float(novelty), 4))
            v = self.vectors[min(len(self.seen) - 1, len(self.vectors) - 1)]

            class _R:
                vector = v
                anchor_score = 0.9
                score_kind = "cosine"
            return _R()

    @staticmethod
    def _service(mapper, uniq_values, drift=0.9, cons=0.9):
        """A VoiceService with the rendering path stubbed out."""
        from alaap.service import VoiceService
        svc = VoiceService.__new__(VoiceService)
        svc.mapper = mapper
        svc._uniq_queue = list(uniq_values)
        svc._uniqueness = lambda vec, lang: (
            svc._uniq_queue.pop(0) if svc._uniq_queue else 1.0)
        svc._drift = lambda *a, **k: drift
        svc._consistency = lambda *a, **k: (cons, [])
        return svc

    def _run(self, uniq_values, n_attempts=3):
        """Drive only the attempt loop, which is what this test is about."""
        from alaap.service import UNIQUENESS_MIN, NOVELTY_STEP
        mapper = self._Mapper([np.ones(8) * (i + 1) for i in range(n_attempts)])
        svc = self._service(mapper, uniq_values)
        collisions, novelty = 0, 0.0
        for attempt in range(1, n_attempts + 1):
            eff = min(1.0, novelty + NOVELTY_STEP * collisions)
            mapper.mint("a voice", novelty=eff, seed=attempt)
            if svc._uniqueness(None, "en") < UNIQUENESS_MIN:
                collisions += 1
                continue
            break
        return mapper.seen

    def test_first_attempt_pays_nothing(self):
        """No collision yet, so no novelty cost."""
        assert self._run([0.9])[0] == 0.0

    def test_novelty_rises_only_after_a_collision(self):
        from alaap.service import NOVELTY_STEP
        seen = self._run([0.05, 0.05, 0.9])
        assert seen[0] == 0.0
        assert seen[1] == round(NOVELTY_STEP, 4)
        assert seen[2] == round(2 * NOVELTY_STEP, 4)

    def test_escalation_is_capped_at_one(self):
        from alaap.service import NOVELTY_STEP
        seen = self._run([0.01] * 6, n_attempts=6)
        assert max(seen) <= 1.0
        assert seen[-1] == 1.0 or NOVELTY_STEP * 5 <= 1.0

    def test_only_collisions_escalate_not_drift(self):
        """
        The rule that matters: drift failures must NOT raise novelty, because
        E11 measured that more novelty is what makes drift worse. Guarded by
        reading the source -- the escalation must be driven by the collision
        counter alone.
        """
        import inspect
        from alaap.service import VoiceService
        src = inspect.getsource(VoiceService.mint)
        i = src.index("eff_novelty")
        assert "collisions" in src[i:i + 120], \
            "novelty escalation must be driven by the collision count"
        # and the counter must only be incremented in the uniqueness branch
        before_drift = src.split("drift_ok")[0]
        assert before_drift.count("collisions += 1") == 1
        assert "collisions += 1" not in src.split("drift_ok", 1)[1]


class TestShippedScriptsLoadTheirArtefacts:
    """
    `demo_script_render.py` is the first thing HANDOFF tells a reader to run,
    and it was broken in two ways at once, both silently accumulated:

      * it loaded the 0.6B cached corpus with `Attributes(**a)`, and those
        attributes predate `f0_cv`, so construction raised TypeError;
      * it loaded a **v1** binner, which `Binner.load` now refuses because v1
        bins raw f0_std and uncorrected hnr_db -- the entangled axes S2 run 3
        got wrong.

    Neither is caught by any experiment, because experiments carry their own
    paths. This walks the artefact paths that shipped scripts actually name.
    """

    @staticmethod
    def _referenced_paths(path):
        import re
        src = open(path, encoding="utf-8").read()
        # string literals that look like artefact paths, including ones split
        # across adjacent literals by the formatter
        joined = re.sub(r'"\s*\n\s*"', "", src)
        return re.findall(r'"(experiments/[^"]+\.(?:npz|json))"', joined)

    def test_every_shipped_script_can_load_what_it_names(self):
        import glob
        import os
        from alaap.acoustics import Attributes, Binner
        import json as _json
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        scripts = sorted(glob.glob(os.path.join(root, "scripts", "*.py")))
        if not scripts:
            pytest.skip("no scripts directory")
        checked = 0
        refs = []
        for script in scripts:
            # preflight globs its own paths at runtime; nothing to pin here
            if os.path.basename(script) == "preflight.py":
                continue
            refs += self._referenced_paths(script)
        for rel in refs:
            full = os.path.join(root, rel)
            if not os.path.exists(full):
                continue                               # not built in this checkout
            if rel.endswith(".json") and "binner" in rel:
                Binner.load(full)                      # raises on a v1 binner
                checked += 1
            elif rel.endswith(".npz"):
                d = np.load(full, allow_pickle=True)
                checked += 1
                if "attrs" in d:
                    for a in _json.loads(str(d["attrs"]))[:5]:
                        Attributes.from_dict(a)        # must tolerate old caches
        assert checked, ("no artefacts were actually loaded -- the guard would "
                         "pass on a repo where every path is broken")


class TestHybridRetrieval:
    """
    E15/E15d: the text path is barely better than chance at picking the right
    anchor (rank 129.7 of 350, chance 175), and weighted bin retrieval improves
    cos-to-true-voice by 55-66% across two corpora and two encoders.

    The blend is proportional to how much of the description parsed, because
    E15b measured that generated captions yield 5 of 6 axes while realistic
    user text yields 1.29 and character prose yields 0.50. These pin the
    properties that make it safe to enable.
    """

    _last_Z = None

    @staticmethod
    def _mapper(retrieval="hybrid", seed=0, n=120):
        from alaap.acoustics import Attributes, Binner
        from alaap.captions import caption_from_bins
        from alaap.geometry import SpeakerSpace
        from alaap.mapper import RetrievalMapper

        class _FakeText:
            model_id = "fake"

            def encode(self, x):
                xs = [x] if isinstance(x, str) else list(x)
                out = np.zeros((len(xs), 8))
                for i, t in enumerate(xs):
                    r = np.random.default_rng(abs(hash(t)) % (2 ** 31))
                    v = r.standard_normal(8)
                    out[i] = v / np.linalg.norm(v)
                return out

        # The vectors must actually DEPEND on the attributes, or no axis
        # predicts voice distance, the measured weights are noise, and these
        # tests assert nothing. An earlier version of this fixture drew Z
        # independently of attrs and duly failed for that reason.
        #
        # Ground truth built in here: f0_mean drives the voice strongly,
        # spectral_tilt weakly, and speaking_rate not at all -- which is the
        # ordering E15/E15d measured on real corpora.
        rng = np.random.default_rng(seed)
        f0 = 80 + 200 * rng.random(n)
        tilt = -9 + 4 * rng.random(n)
        rate = 8 + 12 * rng.random(n)
        basis = rng.standard_normal((3, 64))
        Z = (np.outer((f0 - f0.mean()) / f0.std(), basis[0]) * 3.0
             + np.outer((tilt - tilt.mean()) / tilt.std(), basis[1]) * 0.8
             + rng.standard_normal((n, 64)) * 0.6 + 1.5)
        space = SpeakerSpace.fit(Z, n_components=16)
        attrs = [Attributes(
            f0_mean=float(f0[i]), f0_std=20.0,
            f0_cv=float(0.05 + 0.3 * rng.random()), f0_range=80.0,
            speaking_rate=float(rate[i]),
            hnr_db=float(rng.standard_normal() * 4),
            jitter=0.01, shimmer=0.05,
            spectral_tilt=float(tilt[i]),
            snr_db=30.0, duration_s=4.0, voiced_frac=0.6) for i in range(n)]
        binner = Binner.fit(attrs)
        bins = [binner.bin_one(a) for a in attrs]
        caps = [caption_from_bins(b, seed=i) for i, b in enumerate(bins)]
        m = RetrievalMapper(space, _FakeText(), pca_dims=8,
                            retrieval=retrieval).fit(caps, Z, anchor_bins=bins)
        TestHybridRetrieval._last_Z = Z
        return m, caps, bins

    @classmethod
    def _mapper_with_holdout(cls, retrieval="hybrid", seed=0, n=160, n_held=30):
        """Same construction, but the last n_held speakers are never fitted."""
        from alaap.acoustics import Binner
        from alaap.captions import caption_from_bins
        from alaap.geometry import SpeakerSpace
        from alaap.mapper import RetrievalMapper

        m_full, caps, bins = cls._mapper(retrieval=retrieval, seed=seed, n=n)
        Z = cls._last_Z
        keep = slice(0, n - n_held)
        space = SpeakerSpace.fit(Z[keep], n_components=16)
        m = RetrievalMapper(space, m_full.text, pca_dims=8,
                            retrieval=retrieval).fit(
            caps[keep], Z[keep], anchor_bins=bins[keep])
        held = [(caps[i], Z[i:i + 1]) for i in range(n - n_held, n)]
        return m, held

    def test_hybrid_requires_anchor_bins(self):
        """Failing closed beats silently behaving like the text path."""
        from alaap.mapper import RetrievalMapper
        from alaap.geometry import SpeakerSpace

        class _T:
            model_id = "f"

            def encode(self, x):
                xs = [x] if isinstance(x, str) else list(x)
                return np.ones((len(xs), 4)) / 2.0

        rng = np.random.default_rng(0)
        Z = rng.standard_normal((40, 16)) + 2.0
        sp = SpeakerSpace.fit(Z, n_components=4)
        m = RetrievalMapper(sp, _T(), pca_dims=4, retrieval="hybrid")
        with pytest.raises(ValueError, match="anchor_bins"):
            m.fit([f"caption {i}" for i in range(40)], Z)

    def test_an_unknown_retrieval_mode_is_refused(self):
        from alaap.mapper import RetrievalMapper
        with pytest.raises(ValueError, match="retrieval must be"):
            RetrievalMapper(None, None, retrieval="bins")

    def test_weights_rank_pitch_above_rate(self):
        """
        The measured ordering, reproduced end to end. f0_mean carries identity;
        speaking_rate is behaviour and should not.
        """
        m, _, _ = self._mapper()
        w = dict(zip(m.axes, m.axis_weights))
        assert "f0_mean" in w and "speaking_rate" in w
        assert w["f0_mean"] > w["speaking_rate"]

    def test_description_with_no_acoustic_words_falls_back_exactly(self):
        """
        alpha = 0 must return the text scores UNTOUCHED, so the hybrid can
        never be worse than the text path on prose it cannot parse.
        """
        m, _, _ = self._mapper()
        sims = np.linspace(-1, 1, len(m.captions))
        out = m._hybrid_scores("the protagonist's best friend", sims)
        assert np.allclose(out, sims)

    def test_parsed_description_changes_the_ranking(self):
        m, _, _ = self._mapper()
        sims = np.zeros(len(m.captions))       # text says everything is equal
        out = m._hybrid_scores("a very deep voice, very rough and gravelly", sims)
        assert out.std() > 0, "bins had no effect on an all-ties text score"

    def test_hybrid_lands_nearer_the_true_held_out_voice(self):
        """
        The end-to-end claim, mirroring how E15 measured it: mint from a
        HELD-OUT speaker's caption and see which method lands nearer that
        speaker's real voice.

        Held-out matters. This fixture's fake text encoder hashes the caption
        string, so querying with an ANCHOR's own caption hands the text path an
        exact-match oracle that real MiniLM does not have -- an earlier version
        of this test did exactly that and measured the oracle, not the method.
        E15 avoided it the same way, by testing on speakers the mapper never saw.

        The margin here is much larger than the real one (about +0.69 mean
        cosine, 5/5 seeds, against E15's +55-66% relative on real corpora),
        because this fixture builds the vectors AS a function of the binned
        axes. That makes it a good regression guard and a bad effect-size
        estimate -- read E15/E15d for the size, this for the direction.
        """
        mh, held = self._mapper_with_holdout(retrieval="hybrid")
        mt, _ = self._mapper_with_holdout(retrieval="text")

        def mean_cos_to_true(m):
            out = []
            for cap, z_true in held:
                v = m.mint(cap, novelty=0.0, seed=0).vector
                a = m.space.encode(v)[0]
                b = m.space.encode(np.asarray(z_true, float))[0]
                out.append(float(a @ b /
                                 max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12)))
            return float(np.mean(out))

        assert mean_cos_to_true(mh) > mean_cos_to_true(mt)


class TestMissingTauIsReportedNotSwallowed:
    """
    The demo asked for emotion on three of its five lines, got none, and
    recorded no problem in its manifest.

    `tau` is per-model -- vectors fitted on 0.6B are 1024-d and meaningless on
    1.7B's 2048-d, so `_load_tau` correctly drops them. But `steer()` then
    returned an empty degradation list whenever `self.tau` was empty, so a
    request that could not be honoured simply vanished.

    `_apply_direction` cannot catch this: it reports fields marked REJECT, and
    emotion is APPROXIMATE -- supported in principle. Whether the vectors exist
    is a runtime fact about the loaded model, and the contract is that anything
    asked for and not delivered comes back as a degradation.
    """

    class _R:
        backend_id = "qwen3-tts-base"
        backend_version = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        tau: dict = {}
        direction_bounds = {"emotion": {"alpha_max": 1.0,
                                        "identity_retained_ecapa": 0.62}}
        steer = None  # bound below

    def _renderer(self, tau=None):
        from alaap.renderer import Qwen3BaseRenderer
        r = self._R()
        r.tau = tau or {}
        r.steer = Qwen3BaseRenderer.steer.__get__(r, self._R)
        return r

    def test_emotion_without_tau_returns_a_degradation(self):
        from alaap.renderer import Direction
        r = self._renderer()
        vec, notes = r.steer(np.ones(8, dtype=np.float32),
                             Direction(emotion={"anger": 1.0}, intensity=0.7))
        assert notes, "emotion was requested, not delivered, and not reported"
        assert "no direction vectors" in notes[0]
        assert "NEUTRAL" in notes[0]

    def test_strict_direction_raises_instead(self):
        from alaap.renderer import Direction
        r = self._renderer()
        with pytest.raises(NotImplementedError):
            r.steer(np.ones(8, dtype=np.float32),
                    Direction(emotion={"sad": 1.0}, strict=True))

    def test_no_emotion_requested_is_still_silent(self):
        """Only an unhonoured REQUEST is a degradation."""
        from alaap.renderer import Direction
        r = self._renderer()
        _, notes = r.steer(np.ones(8, dtype=np.float32), None)
        assert notes == []
        _, notes = r.steer(np.ones(8, dtype=np.float32), Direction())
        assert notes == []

    def test_available_tau_is_applied_and_reported(self):
        from alaap.renderer import Direction
        # tau must NOT be parallel to the vector: steer renormalises to the
        # original norm, so adding a parallel direction returns the input
        # unchanged. That is correct behaviour and a degenerate test.
        tau = np.zeros(8, dtype=np.float32); tau[0] = 0.5
        base = np.ones(8, dtype=np.float32)
        r = self._renderer(tau={"anger": tau})
        vec, notes = r.steer(base, Direction(emotion={"anger": 1.0}, intensity=0.5))
        assert notes and "applied at alpha" in notes[0]
        assert not np.allclose(vec, base), "tau was not applied"
        assert abs(np.linalg.norm(vec) - np.linalg.norm(base)) < 1e-4,             "steer must keep the vector on the shell (E3)"


class TestDirectionBoundNamesItsModel:
    """
    `direction_bounds["emotion"]["identity_retained_ecapa"] = 0.591` was
    measured by E0 on **0.6B**. E10 then made 1.7B the default, so a 1.7B
    render was quoting a 0.6B number as if it described itself -- in the
    degradation note, which ends up in the shipped manifest.

    Refitting tau for 1.7B (scripts/fit_emotion_tau.py) does not fix this: the
    vectors and the identity COST of applying them are separate measurements.
    RESEARCH/10's rule is that an APPROXIMATE field needs a published bound; a
    bound from a different model is not one.
    """

    def test_the_bound_records_which_model_it_came_from(self):
        from alaap.renderer import Qwen3BaseRenderer
        b = Qwen3BaseRenderer.direction_bounds["emotion"]
        assert "measured_on" in b, "a bound must say what it was measured on"
        assert "0.6B" in b["measured_on"]

    def test_the_note_flags_a_bound_from_another_model(self):
        from alaap.renderer import Direction, Qwen3BaseRenderer

        class _R:
            backend_id = "qwen3-tts-base"
            backend_version = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
            direction_bounds = Qwen3BaseRenderer.direction_bounds
            tau = {"anger": np.array([0.5, 0, 0, 0], dtype=np.float32)}

        r = _R()
        r.steer = Qwen3BaseRenderer.steer.__get__(r, _R)
        _, notes = r.steer(np.ones(4, dtype=np.float32),
                           Direction(emotion={"anger": 1.0}))
        assert notes and "NOT re-measured" in notes[0], notes

    def test_no_flag_when_the_bound_matches_the_model(self):
        from alaap.renderer import Direction, Qwen3BaseRenderer

        class _R:
            backend_id = "qwen3-tts-base"
            backend_version = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
            direction_bounds = Qwen3BaseRenderer.direction_bounds
            tau = {"anger": np.array([0.5, 0, 0, 0], dtype=np.float32)}

        r = _R()
        r.steer = Qwen3BaseRenderer.steer.__get__(r, _R)
        _, notes = r.steer(np.ones(4, dtype=np.float32),
                           Direction(emotion={"anger": 1.0}))
        assert notes and "NOT re-measured" not in notes[0], notes


class TestCharacterErrorRate:
    """
    CER is how intelligibility is scored for Indic, and it caught a real
    failure: renders decoded through the wrong codec were fluent-sounding
    but said different words. A listener could not detect that without the
    reference text; CER could.
    """

    def test_identical_is_zero(self):
        from alaap.metrics import cer
        assert cer("नमस्ते आप कैसे हैं", "नमस्ते आप कैसे हैं") == 0.0

    def test_punctuation_and_tags_do_not_count(self):
        """An ASR emits neither the danda nor the backend's <happy> tags."""
        from alaap.metrics import cer
        assert cer("मुझे यह फिल्म बहुत पसंद आई! <happy>",
                   "मुझे यह फिल्म बहुत पसंद आई") == 0.0

    def test_wrong_words_score_high(self):
        """The actual failure: fluent Devanagari, entirely different words."""
        from alaap.metrics import cer
        c = cer("नमस्ते आप कैसे हैं आज मौसम बहुत अच्छा है",
                "अज़्ट उद आयार मुशिलो के लब शिबगो जो लिख चिए आया")
        assert c > 0.5, c

    def test_a_spelling_variant_scores_low(self):
        """whisper's नमस्ते->नमस्ती is not a synthesis error."""
        from alaap.metrics import cer
        c = cer("नमस्ते आप कैसे हैं आज मौसम बहुत अच्छा है",
                "नमस्ती आप कैसे है आज मोसम बहुत अच्छा है")
        assert 0 < c < 0.2, c

    def test_empty_reference_is_nan(self):
        from alaap.metrics import cer
        assert np.isnan(cer("", "anything"))


class TestAnchorScoreIsNotAlwaysASimilarity:
    """
    `anchor_similarity` read 2.05 in S6 -- impossible for a cosine, and it went
    into a committed results table that way. The hybrid path returns a blended
    z-score, which is unbounded. The field now says which it is, and asking for
    a similarity when it is not one raises instead of quietly overstating how
    close the retrieved anchor was.
    """

    def _mapper(self, retrieval):
        from alaap.geometry import SpeakerSpace
        from alaap.mapper import RetrievalMapper
        from alaap.acoustics import Attributes, Binner
        from alaap.captions import caption_from_bins
        rng = np.random.default_rng(0)
        n = 60
        f0 = rng.uniform(80, 260, n)
        attrs = [Attributes(f0_mean=f, f0_std=f * 0.2, f0_range=f * 0.5,
                            f0_cv=0.2, speaking_rate=4.0, hnr_db=15.0,
                            spectral_tilt=-10.0, jitter=0.01, shimmer=0.05,
                            snr_db=30.0, duration_s=5.0, voiced_frac=0.6)
                 for f in f0]
        binner = Binner.fit(attrs)
        bins = [binner.bin_one(a) for a in attrs]
        caps = [caption_from_bins(b, seed=i) for i, b in enumerate(bins)]
        # vectors correlated with f0, so retrieval has real structure to find
        Z = np.hstack([f0[:, None] / 100.0, rng.standard_normal((n, 15))])
        space = SpeakerSpace.fit(Z, n_components=8)
        from alaap.mapper import TextEncoder
        m = RetrievalMapper(space, TextEncoder(), pca_dims=8, retrieval=retrieval)
        return m.fit(caps, Z, anchor_bins=bins if retrieval == "hybrid" else None)

    def test_text_retrieval_reports_a_real_cosine(self):
        r = self._mapper("text").mint("a very deep voice", novelty=0.0, seed=0)
        assert r.score_kind == "cosine"
        assert -1.0 <= r.anchor_score <= 1.0, r.anchor_score
        assert r.anchor_similarity == r.anchor_score

    def test_hybrid_refuses_to_call_its_score_a_similarity(self):
        r = self._mapper("hybrid").mint("a very deep voice", novelty=0.0, seed=0)
        assert r.score_kind == "hybrid_z"
        with pytest.raises(AttributeError, match="not a similarity"):
            r.anchor_similarity


class TestBlendingCostsDiversity:
    """
    S9b's finding, locked as a property rather than a number.

    Minting SLERPs the top-k retrieved anchors. Blending k points on a shell
    lands nearer the centroid the larger k is, so every extra anchor costs
    catalog diversity. That relationship was invisible for the whole project
    because `mint` floored top_k at 2, making pure retrieval (k=1)
    inexpressible -- so the S7-vs-S8 gap read as two competing methods when it
    was one method at two settings.

    Measured on 80 Indic mints against a bound of ~38: k=4 gave 13 effective
    voices, k=2 gave 20, k=1 (retrieval) gave 33.
    """

    def _mapper(self, pca_dims):
        from alaap.acoustics import Attributes, Binner
        from alaap.captions import caption_from_bins
        from alaap.geometry import SpeakerSpace
        from alaap.mapper import RetrievalMapper, TextEncoder
        rng = np.random.default_rng(0)
        n = 80
        f0 = rng.uniform(80, 260, n)
        attrs = [Attributes(f0_mean=f, f0_std=f * 0.2, f0_range=f * 0.5, f0_cv=0.2,
                            speaking_rate=4.0, hnr_db=15.0, spectral_tilt=-10.0,
                            jitter=0.01, shimmer=0.05, snr_db=30.0,
                            duration_s=5.0, voiced_frac=0.6) for f in f0]
        binner = Binner.fit(attrs)
        bins = [binner.bin_one(a) for a in attrs]
        caps = [caption_from_bins(b, seed=i) for i, b in enumerate(bins)]
        Z = np.hstack([f0[:, None] / 100.0, rng.standard_normal((n, 15))])
        space = SpeakerSpace.fit(Z, n_components=12)
        m = RetrievalMapper(space, TextEncoder(), pca_dims=pca_dims,
                            retrieval="text").fit(caps, Z)
        return m, space, bins, binner

    def test_top_k_1_is_reachable_and_returns_its_anchor(self):
        """
        k=1 must reduce to retrieval. Not bit-exact: mint decodes through
        `project_to_shell=True`, which rescales per-dimension in RAW space --
        not a rotation, so it does not preserve the working-space basis
        exactly. The invariant is that k=1 lands ON its anchor's direction
        while k=4 does not.
        """
        m, space, _, _ = self._mapper(12)
        r1 = m.mint("a very deep voice", novelty=0.0, seed=0, top_k=1)
        r4 = m.mint("a very deep voice", novelty=0.0, seed=0, top_k=4)

        def cos_to_anchor(res):
            a = space.from_pca(np.pad(m.P[res.anchors[0]][None, :],
                                      ((0, 0), (0, space.components.shape[0] - m.k))))[0]
            w = space.encode(res.vector)[0]
            return float(a @ w / max(np.linalg.norm(a) * np.linalg.norm(w), 1e-12))

        c1, c4 = cos_to_anchor(r1), cos_to_anchor(r4)
        assert c1 > 0.99, f"top_k=1 did not return its anchor: cos {c1:.4f}"
        assert c1 > c4, f"blending 4 anchors was not further from the top anchor: {c1:.4f} vs {c4:.4f}"

    def test_more_anchors_contracts_toward_the_centroid(self):
        from alaap.captions import caption_from_bins
        from alaap.catalog import sample_cells
        m, space, _, _ = self._mapper(12)
        cells = sample_cells(24, 0)
        descs = [caption_from_bins(c, seed=i) for i, c in enumerate(cells)]
        rad = {}
        for tk in (1, 4):
            M = np.vstack([space.encode(m.mint(x, novelty=0.0, seed=i,
                                               top_k=tk).vector)[0]
                           for i, x in enumerate(descs)])
            rad[tk] = float(np.linalg.norm(M - M.mean(0), axis=1).mean())
        assert rad[4] < rad[1], (
            f"blending 4 anchors did not contract against 1: {rad}")

    def test_truncating_the_basis_starves_the_discarded_components(self):
        """
        The other half of the mechanism: mint zero-pads beyond pca_dims, so
        minted voices barely differ there while real speakers differ a lot. On
        the Indic corpus the measured ratio was 0.0069 against 17.23.

        Not exactly zero, for the same reason as above: the shell projection
        leaks a little energy back. The invariant is the RATIO.
        """
        from alaap.captions import caption_from_bins
        from alaap.catalog import sample_cells
        m, space, _, _ = self._mapper(6)
        cells = sample_cells(12, 0)
        M = np.vstack([space.encode(m.mint(caption_from_bins(c, seed=i),
                                           novelty=0.0, seed=i).vector)[0]
                       for i, c in enumerate(cells)])
        real = space.encode(m.Z_fit) if hasattr(m, "Z_fit") else None
        tail_mint = (M @ space.components.T)[:, 6:].var(0).sum()
        # real speakers' tail variance, from the same fitted space
        rng = np.random.default_rng(0)
        f0 = rng.uniform(80, 260, 80)
        Z = np.hstack([f0[:, None] / 100.0, rng.standard_normal((80, 15))])
        tail_real = (space.encode(Z) @ space.components.T)[:, 6:].var(0).sum()
        assert tail_mint < 0.25 * tail_real, (
            "minted voices should be starved of variance beyond pca_dims; "
            f"mint {tail_mint:.4f} vs real {tail_real:.4f}")


class TestRateDirection:
    """
    S14's rate control, and the properties that made it shippable.

    Every one of these is a property stated before the experiment ran, which
    is why they are worth locking: the measured numbers will move if the
    backend changes, but a rate control that shifts pitch, or silently ignores
    its own bound, is broken regardless of the numbers.
    """

    def _speech(self, sr=24000, secs=2.0, f0=140.0):
        """A harmonic buzz -- enough structure for pitch to be measurable."""
        t = np.arange(int(sr * secs)) / sr
        w = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 12))
        env = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t)     # syllable-ish envelope
        return (w * env / np.abs(w * env).max()).astype(np.float32)

    def test_duration_scales_with_the_requested_rate(self):
        from alaap.timing import retime
        w = self._speech()
        for r in (0.7, 1.0, 1.43):
            out = retime(w, r)
            assert abs(len(out) / (len(w) / r) - 1.0) < 0.02, (
                f"rate {r}: got {len(out)} samples, wanted ~{len(w)/r:.0f}")

    def test_pitch_does_not_move(self):
        """
        The property ADR-012 requires: direction must not touch f0_mean, the
        dominant identity axis. A phase vocoder preserves it; naive resampling
        does not, and S14 used exactly that contrast as its negative control.
        """
        from alaap.acoustics import f0_track
        from alaap.timing import retime
        sr = 24000
        w = self._speech(sr=sr)
        base = float(np.nanmean(f0_track(w, sr)))
        for r in (0.7, 1.43):
            got = float(np.nanmean(f0_track(retime(w, r), sr)))
            assert abs(got - base) < 0.06 * base, (
                f"rate {r} moved f0 from {base:.1f} to {got:.1f} Hz")

    def test_naive_resampling_would_fail_that_test(self):
        """The negative control, as a test: proves the check above has teeth."""
        import librosa
        from alaap.acoustics import f0_track
        sr = 24000
        w = self._speech(sr=sr)
        base = float(np.nanmean(f0_track(w, sr)))
        naive = librosa.resample(w, orig_sr=int(sr * 1.43), target_sr=sr)
        got = float(np.nanmean(f0_track(naive, sr)))
        assert abs(got - base) > 0.2 * base, (
            "resampling did not shift pitch, so the pitch check cannot "
            f"distinguish the two methods: {base:.1f} -> {got:.1f}")

    def test_out_of_bound_rates_clamp_and_report(self):
        from alaap.timing import RATE_MAX, RATE_MIN, clamp_rate
        r, notes = clamp_rate(3.0)
        assert r == RATE_MAX and notes and "bound" in notes[0]
        r, notes = clamp_rate(0.1)
        assert r == RATE_MIN and notes
        r, notes = clamp_rate(1.0)
        assert r == 1.0 and notes == []

    def test_a_bound_measured_elsewhere_is_reported_not_hidden(self):
        """
        ADR-011's rule, as a test. The rate bound was measured on the Indic
        path; a Qwen3 render quoting it must say so, exactly as the emotion
        bound already does.
        """
        from alaap.renderer import Qwen3BaseRenderer
        b = Qwen3BaseRenderer.direction_bounds["rate"]
        assert b["measured_on"] and "indic" in b["measured_on"].lower()

    def test_rate_is_approximate_and_therefore_needs_its_bound(self):
        """RESEARCH/10: 'approximate without a published bound is REJECT'."""
        from alaap.renderer import Honouring, Qwen3BaseRenderer
        assert Qwen3BaseRenderer.direction_support["rate"] is Honouring.APPROXIMATE
        assert "rate" in Qwen3BaseRenderer.direction_bounds


class TestVocalTractLength:
    """
    The estimator was at chance until S17, and the reason is worth locking:
    formant dispersion averages the gaps THROUGH F2, the vowel-dependent
    formant, cancelling what F1 and F3 know. Measured gender effect sizes
    (males positive) across four corpora: dispersion gave +0.11, -0.17, +0.06,
    +0.11; F1+F3 gives +0.70, +1.03, +0.44, +0.68.
    """

    def test_males_read_longer_than_females(self):
        """Typical adult F1/F3; a male tract is longer, so formants are lower."""
        from alaap.acoustics import vocal_tract_length as vtl
        female = vtl(np.array([500.0, 1800.0, 2900.0, 3800.0]))
        male = vtl(np.array([420.0, 1500.0, 2500.0, 3400.0]))
        assert male > female, f"male {male:.1f} should exceed female {female:.1f}"

    def test_values_are_in_a_plausible_band(self):
        """
        The uniform-tube model OVERESTIMATES absolute length, because real
        formants are not a neutral schwa's. A 500/2900 Hz pair reads 16.3 cm
        where a phonetician would say ~14. That is accepted deliberately: this
        axis is percentile-binned, so only the ORDERING is used, and no
        corpus-specific calibration is applied that would have to be re-fitted
        per corpus.

        The band is therefore wide, and the number should not be quoted as an
        anatomical measurement.
        """
        from alaap.acoustics import vocal_tract_length as vtl
        for f1, f3 in ((500.0, 2900.0), (420.0, 2500.0), (350.0, 2300.0)):
            v = vtl(np.array([f1, 1500.0, f3, 3400.0]))
            assert 10.0 < v < 25.0, f"F1={f1} F3={f3} gave {v:.1f} cm"

    def test_f2_does_not_affect_the_estimate(self):
        """
        The whole point of the change. F2 swings 800-2200 Hz across vowels; if
        it moved the answer, the estimate would track the vowel rather than
        the speaker.
        """
        from alaap.acoustics import vocal_tract_length as vtl
        a = vtl(np.array([450.0, 900.0, 2600.0, 3500.0]))
        b = vtl(np.array([450.0, 2100.0, 2600.0, 3500.0]))
        assert abs(a - b) < 1e-9, f"F2 changed the estimate: {a} vs {b}"

    def test_missing_formants_give_nan_not_a_number(self):
        from alaap.acoustics import vocal_tract_length as vtl
        assert np.isnan(vtl(np.array([450.0, 1500.0, np.nan, 3500.0])))
        assert np.isnan(vtl(np.array([450.0, 1500.0])))


class TestNaturalnessGate:
    """
    S20's gate, locked as properties. The numbers will move with the corpus;
    what must not move is that it stays a naturalness measure rather than
    becoming the pitch detector RESEARCH/06 warns about.
    """

    def test_typical_and_atypical_separate(self):
        from alaap.metrics import naturalness_isolation
        rng = np.random.default_rng(0)
        ref = rng.standard_normal((80, 32))
        typical = rng.standard_normal((1, 32))
        far = rng.standard_normal((1, 32)) + 8.0
        assert naturalness_isolation(typical, ref) < naturalness_isolation(far, ref)

    def test_held_out_reference_scores_mid_range(self):
        """The validation isolation_pct itself requires: real scores ~50, not ~0 or ~100."""
        from alaap.metrics import naturalness_isolation
        rng = np.random.default_rng(1)
        pool = rng.standard_normal((160, 32))
        ref, held = pool[:80], pool[80:]
        s = np.mean([naturalness_isolation(h[None, :], ref) for h in held])
        assert 25.0 < s < 75.0, f"held-out reference scored {s:.1f}"

    def test_the_documented_floor_is_above_real_speech(self):
        from alaap.metrics import NATURALNESS_FLOOR, NATURALNESS_REAL_TYPICAL
        assert NATURALNESS_FLOOR > NATURALNESS_REAL_TYPICAL + 20
