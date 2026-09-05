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
                "spectral_tilt": "dark", "f0_std": "monotone",
                "speaking_rate": "slow"}
        for seed in range(8):
            back = target_bins_from_text(caption_from_bins(bins, seed=seed))
            for k, v in bins.items():
                assert back.get(k) == v, f"seed {seed}: {k} {back.get(k)} != {v}"

    def test_captions_vary_in_surface_form(self):
        bins = {"f0_mean": "low-pitched", "hnr_db": "clear", "f0_std": "expressive"}
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
