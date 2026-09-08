"""
S23 -- does the naturalness gate's layer choice transfer to English?

`S20b` swept all 12 WavLM base+ layers on Hindi and moved the gate to LAYER 5.
`S20`'s own "Not established" section says the reference is 70 Hindi clips from
one corpus and that scoring English against it would measure LANGUAGE rather
than naturalness. So the layer choice is a Hindi result until somebody runs it
on English. This runs it.

WHAT IS DELIBERATELY NOT COPIED FROM S20b

  The CODEC ROUNDTRIP ARM IS ABSENT, for two reasons and only one of them is
  the machine.

  1. It would be the WRONG CODEC. S20b round-tripped through MioCodec because
     that is the Indic path. The English path is Qwen3-TTS. A MioCodec
     roundtrip of GLOBE_V2 audio measures a codec English never passes through.
  2. It is the only arm that needs a model resident beside WavLM, and this box
     has ~2 GB free against the ~3 GB floor in HANDOFF 8b.

  So this answers "which layer, for English" and NOT "does the English gate see
  its codec". The second question is real and untouched; see RESULTS.md.

  THE SYNTHETIC ARM MUST BE ENGLISH. run_layer_sweep.py hardcodes
  S6-indic-mint and S7-indic-catalog -- Hindi renders. Pointing an English
  reference at those would separate them on language and report a triumphant
  gate. English renders come from E11-catalog and S15-english-listening.

THE FOUR CHECKS, from S20, asserted before any number is read. A layer must
pass ALL of them; the biggest separation is not the winner.

  C1  held-out REAL scores near 50%. isolation_pct is a percentile within the
      reference's own radii, so real-vs-real MUST be typical. A layer that
      scores real speech at 90% is not measuring naturalness.
  C2  synthetic scores HIGHER than real, and |t| >= 2. S20b's first pass
      declared three layers winners on a bare `gap >= 5` with a standard error
      of ~6. An effect size is not a test.
  C3  |r| with mean F0 stays low. RESEARCH/06: humans -0.059, DNSMOS -0.788.
      A layer that separates by tracking pitch is the documented trap, and is
      REJECTED rather than celebrated -- layer 0 was, at -0.578.
  C4  real-vs-real is a tie. A layer that tells two halves of the same held-out
      real set apart is reading channel or speaker, not naturalness.

    envs/qwen3/Scripts/python.exe experiments/S23-english-gate/run_english_sweep.py --phase audio
    envs/qwen3/Scripts/python.exe experiments/S23-english-gate/run_english_sweep.py --phase feats
    envs/qwen3/Scripts/python.exe experiments/S23-english-gate/run_english_sweep.py --phase sweep
"""
import argparse
import glob
import io
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", write_through=True)
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from alaap.acoustics import f0_track
from alaap.metrics import isolation_pct

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--phase", default="sweep", choices=["audio", "feats", "sweep"])
ap.add_argument("--corpus", default="globe_v2")
ap.add_argument("--n", type=int, default=150)
# Point the IDENTICAL code path at S20b's cached Hindi features, so the
# corrected statistics (within-real r, control averaged over 200 splits) are
# applied to the run that chose layer 5 in the first place. One definition of
# the statistics, two corpora -- if the corrections change the Hindi verdict,
# that is a correction to a committed result, not a new experiment.
ap.add_argument("--feats", default=None, help="override the features file")
ap.add_argument("--label", default="English / GLOBE_V2")
args = ap.parse_args()

SR_CORPUS, SR_W = 24000, 16000
HINDI_LAYER = 5                     # what S20b chose, and what this may not confirm
AUDIO = os.path.join(OUT, f"audio_{args.corpus}_{args.n}.npz")
FEATS = args.feats or os.path.join(OUT, f"feats_{args.corpus}_{args.n}.npz")

# English renders only. Hindi ones would be separated on LANGUAGE.
SYNTH_PATTERNS = [
    "experiments/E11-catalog/out/audio/*.wav",          # English catalog, Qwen3-TTS
    "experiments/S15-english-listening/out/pair*.wav",  # English listening set
]

import librosa

# ------------------------------------------------------------- phase audio
if args.phase == "audio":
    from alaap.data import CORPUS_LANG, stream_clips
    assert CORPUS_LANG.get(args.corpus) == "en", (
        f"{args.corpus} is not an English corpus; this experiment exists because "
        f"the gate's reference must match the language it scores")
    print(f"[audio] {args.n} real {args.corpus} clips (no codec, no LM)")
    payload, f0s, k = {}, [], 0
    for c in stream_clips(args.corpus, n=args.n, per_speaker=1,
                          min_dur=2.0, max_dur=8.0, progress_every=25):
        w = np.asarray(c.wav, np.float32)
        payload[f"R{k}"] = librosa.resample(w, orig_sr=SR_CORPUS, target_sr=SR_W)
        f0s.append(float(np.nanmean(f0_track(w, SR_CORPUS))))
        k += 1
    payload["n"] = k
    payload["f0"] = np.array(f0s, float)
    np.savez_compressed(AUDIO, **payload)
    print(f"        {k} real clips -> {AUDIO}")
    sys.exit(0)

# ------------------------------------------------------------- phase feats
if args.phase == "feats":
    import soundfile as sf
    import torch
    import torchaudio
    if not os.path.exists(AUDIO):
        sys.exit(f"run --phase audio first (missing {AUDIO})")
    z = np.load(AUDIO, allow_pickle=True)
    m = torchaudio.pipelines.WAVLM_BASE_PLUS.get_model().eval()
    NL = 12

    def all_layers(x):
        """One forward pass gives every layer -- sweeping 12 costs one."""
        if len(x) < SR_W // 2:
            return None
        with torch.no_grad():
            hs, _ = m.extract_features(
                torch.from_numpy(np.asarray(x, np.float32)).unsqueeze(0), num_layers=NL)
        return np.stack([h.squeeze(0).mean(0).numpy() for h in hs])

    R = np.stack([v for v in (all_layers(z[f"R{i}"]) for i in range(int(z["n"])))
                  if v is not None])
    print(f"[feats] real {R.shape}")

    S, sf0, srcs = [], [], []
    for pat in SYNTH_PATTERNS:
        got = sorted(glob.glob(pat))
        print(f"        {len(got):>3} from {pat}")
        for f in got:
            w, sr = sf.read(f, dtype="float32")
            if w.ndim > 1:
                w = w.mean(1)
            v = all_layers(librosa.resample(w, orig_sr=sr, target_sr=SR_W))
            if v is None:
                continue
            S.append(v)
            sf0.append(float(np.nanmean(f0_track(
                librosa.resample(w, orig_sr=sr, target_sr=SR_CORPUS), SR_CORPUS))))
            srcs.append(os.path.basename(os.path.dirname(os.path.dirname(f))))
    S = np.stack(S)
    print(f"        synthetic {S.shape}")
    np.savez_compressed(FEATS, R=R, S=S, f0=z["f0"], sf0=np.array(sf0, float),
                        srcs=json.dumps(srcs))
    print(f"        -> {FEATS}")
    sys.exit(0)

# ------------------------------------------------------------- phase sweep
if not os.path.exists(FEATS):
    sys.exit(f"run --phase audio then --phase feats (missing {FEATS})")
z = np.load(FEATS, allow_pickle=True)
R, S, f0, sf0 = z["R"], z["S"], z["f0"], z["sf0"]
NL = R.shape[1]
rng = np.random.default_rng(0)
idx = rng.permutation(len(R))
half = len(R) // 2
ref_i, held_i = idx[:half], idx[half:]
print(f"[sweep] {args.label}")
print(f"        reference {len(ref_i)} real | held-out {len(held_i)} | "
      f"synthetic {len(S)} | {NL} layers")
print(f"        real F0 {np.nanmin(f0):.0f}-{np.nanmax(f0):.0f} Hz, "
      f"synthetic {np.nanmin(sf0):.0f}-{np.nanmax(sf0):.0f} Hz")

C1_LO, C1_HI = 35.0, 65.0
C4_MAX, T_MIN = 0.35, 2.0

# C3's bar is SIGNIFICANCE, not an effect-size cut. A fixed 0.35 was inherited
# from S20b's pooled r and is toothless against the within-real correlation --
# it passed all 12 layers, which is a check that has stopped checking. The
# question a pitch-trap screen asks is "is this correlation distinguishable from
# zero at this n", which is the same lesson S20b earned when it called three
# layers winners on `gap >= 5` with a standard error of 6.
_n_real_held = len(R) - len(R) // 2
R_CRIT = float(2.0 / np.sqrt(max(_n_real_held - 3, 1)))   # ~|r| for |z| = 2

print()
print(f"  {'layer':>5} {'real':>7} {'synth':>7} {'gap':>7} {'t':>6} "
      f"{'rPOOL':>7} {'rREAL':>7} {'rSYN':>7} {'ctrlD':>6}  verdict")
print(f"        (C3 rejects |rREAL| >= {R_CRIT:.3f}, the value distinguishable "
      f"from zero at n={_n_real_held})")
print("  " + "-" * 92)
rows = []
for L in range(NL):
    ref = R[ref_i, L, :]
    s_held = np.array([isolation_pct(R[i, L, :][None, :], ref) for i in held_i])
    s_syn = np.array([isolation_pct(s[None, :], ref) for s in S[:, L, :]])
    gap = float(s_syn.mean() - s_held.mean())
    se = float(np.sqrt(s_syn.var(ddof=1) / len(s_syn) +
                       s_held.var(ddof=1) / len(s_held)))
    t = gap / se if se > 0 else 0.0
    # r(f0) POOLED over real+synthetic is confounded by group. Synthetic clips
    # score ~50 points higher AND have a different F0 distribution, so a pooled
    # correlation reads that group difference as a pitch effect. S20b reported
    # the pooled number; on Hindi it happened to land at -0.068 and the flaw
    # never showed. On English it lands at +0.34, next to the reject threshold,
    # which is what exposed it. The question "is this a pitch detector" is a
    # WITHIN-group question, so both are reported and the within one decides.
    def _r(x, y):
        m = np.isfinite(x) & np.isfinite(y)
        return float(np.corrcoef(x[m], y[m])[0, 1]) if m.sum() > 10 else np.nan

    rp = _r(np.concatenate([f0[held_i], sf0]), np.concatenate([s_held, s_syn]))
    r_real, r_syn = _r(f0[held_i], s_held), _r(sf0, s_syn)
    rw = max(abs(r_real), abs(r_syn))

    # The real-vs-real control over ONE arbitrary split is a single draw, and
    # this sweep had layers passing at 0.08 and failing at 0.46 on splits that
    # differ only by which half a clip fell in. Averaged over many.
    ds = []
    crng = np.random.default_rng(7)
    for _ in range(200):
        q = crng.permutation(len(s_held))
        h1, h2 = s_held[q[: len(q) // 2]], s_held[q[len(q) // 2:]]
        ds.append(abs(h1.mean() - h2.mean()) /
                  max(np.sqrt((h1.var(ddof=1) + h2.var(ddof=1)) / 2), 1e-9))
    d = float(np.mean(ds))
    d_p95 = float(np.percentile(ds, 95))

    c1, c2 = C1_LO <= s_held.mean() <= C1_HI, (t >= T_MIN and gap > 0)
    # WHICH pitch correlation decides. The pre-committed check copied S20b's
    # POOLED r, which is confounded: synthetic clips score ~50 points higher AND
    # have a different F0 distribution, so the pooled number partly measures the
    # group split. The three disagree here, so one has to be chosen, and the
    # argument is a priori rather than fitted:
    #
    #   WITHIN REAL is the confound-free probe. Naturalness is ~constant across
    #   real clips, so any correlation with F0 there is the metric responding to
    #   pitch and nothing else. It is also the comparison RESEARCH/06 made --
    #   humans at r = -0.059 were rating real speech.
    #   WITHIN SYNTHETIC is ambiguous: naturalness genuinely varies there, and
    #   E11's catalog was sampled ACROSS the pitch axis by construction, so if
    #   the TTS renders pitch extremes worse a correlation is real rather than a
    #   defect of the gate.
    #
    # Stated plainly because the decisive statistic was changed after seeing the
    # numbers. All three are reported so the choice can be disagreed with.
    c3, c4 = abs(r_real) < R_CRIT, d < C4_MAX
    if c1 and c2 and c3 and c4:
        v = "USABLE"
    elif c2 and not c3:
        v = f"pitch trap (r_real={r_real:+.2f})"
    elif c2 and not c4:
        v = f"fails real-vs-real (d={d:.2f})"
    elif not c1:
        v = f"real not typical ({s_held.mean():.0f}%)"
    else:
        v = "no significant separation"
    rows.append({"layer": L, "real": float(s_held.mean()), "synth": float(s_syn.mean()),
                 "gap": gap, "se": se, "t": float(t), "r_f0_pooled": rp,
                 "r_f0_real": r_real, "r_f0_synth": r_syn, "r_f0_within_max": rw,
                 "ctrl_d": d, "ctrl_d_p95": d_p95,
                 "C1": bool(c1), "C2": bool(c2), "C3": bool(c3), "C4": bool(c4),
                 "verdict": v})
    print(f"  {L:>5} {s_held.mean():>6.1f}% {s_syn.mean():>6.1f}% {gap:>+7.1f} "
          f"{t:>6.2f} {rp:>+7.3f} {r_real:>+7.3f} {r_syn:>+7.3f} {d:>6.2f}  {v}")

json.dump(rows, io.open(os.path.join(OUT, f"layer_sweep_{_tag}.json"), "w",
                        encoding="utf-8"), indent=2)

_tag = "hi" if args.feats else "en"
good = [r for r in rows if r["verdict"] == "USABLE"]
h = rows[HINDI_LAYER]
print()
print("=" * 78)
print(f"  LAYER {HINDI_LAYER} (S20b's choice) on {args.label}:")
print(f"    real {h['real']:.1f}%  synth {h['synth']:.1f}%  gap {h['gap']:+.1f} "
      f"(t = {h['t']:.2f})  ctrl d {h['ctrl_d']:.2f}")
print(f"    r(f0): pooled {h['r_f0_pooled']:+.3f} | WITHIN REAL {h['r_f0_real']:+.3f}"
      f" (|r_crit| = {R_CRIT:.3f}) | within synth {h['r_f0_synth']:+.3f}")
print(f"    -> {h['verdict']}")
print()
if good:
    best = min(good, key=lambda r: abs(r["r_f0_real"]))
    print(f"  {len(good)} of {NL} layers pass all four checks: "
          f"{', '.join(str(r['layer']) for r in good)}")
    print(f"  Cleanest on pitch: layer {best['layer']} "
          f"(r_real = {best['r_f0_real']:+.3f}, t = {best['t']:.2f})")
    if h["verdict"] == "USABLE":
        print(f"  Layer {HINDI_LAYER} IS usable here.")
    else:
        print(f"  Layer {HINDI_LAYER} is NOT usable here -- it fails C3 on this")
        print(f"  corpus, so the layer choice does not carry over unexamined.")
else:
    print(f"  NO layer passes all four checks on English. The gate does not")
    print(f"  transfer as built. Before concluding it cannot, note what this")
    print(f"  design could resolve: n = {len(S)} synthetic vs {len(held_i)} real.")
    b = max(rows, key=lambda r: abs(r["t"]))
    need = int(np.ceil(len(S) * (T_MIN / max(abs(b["t"]), 1e-6)) ** 2))
    print(f"  Best |t| = {abs(b['t']):.2f} at layer {b['layer']}; reaching |t| = 2")
    print(f"  there would need ~{need} synthetic clips.")
print("=" * 78)
print(f"\n  -> {os.path.join(OUT, 'layer_sweep_en.json')}")
