"""
E10 — 0.6B vs 1.7B: does the bigger speaker space actually help?

Three open questions converge on this, so one run answers all three:

  E4-O2  Does 1.7B (enc_dim 2048) discriminate speakers better than 0.6B
         (1024)? 0.6B reached EER 4.35% with an in-domain transform.
  E0     The paper used 1.7B; we used 0.6B and saw more identity loss under
         steering than it reported. A better-separated space is the single
         most likely explanation.
  PHASE-00  Does 1.7B fit a 6GB card at all? Weights are 3.59GB vs 0.6B's
         1.70GB, and 0.6B measured 2.02GB resident.

Everything is measured on IDENTICAL audio, with an IN-DOMAIN SpeakerSpace fit
on a disjoint speaker half (E4's finding: a cross-domain transform buys almost
nothing, an in-domain one halves EER).

Scored with ECAPA -- independent of both conditioning encoders, and the
stricter of the two available (EER 2.58% vs WavLM's 5.34% on real speech).

Outputs -> experiments/E10/out/
"""
import argparse, gc, json, os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from alaap.data import stream_clips, SR
from alaap.encoder import SpeakerEncoder, IndependentSV
from alaap.geometry import SpeakerSpace
from alaap.metrics import verification_stats

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="*", default=["Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                                                "Qwen/Qwen3-TTS-12Hz-1.7B-Base"])
ap.add_argument("--n", type=int, default=800)
ap.add_argument("--per-speaker", type=int, default=10)
ap.add_argument("--corpus", default="libritts_r_train")
ap.add_argument("--n-drift", type=int, default=6, help="identities to render")
ap.add_argument("--line", default="The mountains remember every footstep.")
args = ap.parse_args()

# ------------------------------------------------------------ shared audio
print(f"[1/3] streaming {args.n} clips from {args.corpus}")
clips = stream_clips(corpus=args.corpus, n=args.n, per_speaker=args.per_speaker,
                     min_dur=3.0, max_dur=12.0)
ids = [c.speaker_id for c in clips]
wavs = [c.wav for c in clips]
print(f"      {len(clips)} clips / {len(set(ids))} speakers")

results = {}
for model_id in args.models:
    tag = model_id.split("/")[-1]
    print(f"\n{'='*70}\n{tag}\n{'='*70}")
    import torch
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    gc.collect()

    # ---------------------------------------------------------- fit / VRAM
    t0 = time.time()
    try:
        enc = SpeakerEncoder(model_id)
    except Exception as e:
        print(f"      FAILED TO LOAD: {type(e).__name__}: {str(e)[:120]}")
        results[tag] = {"loaded": False, "error": str(e)[:200]}
        continue
    load_s = time.time() - t0
    vram_load = torch.cuda.memory_allocated() / 2**30
    print(f"      loaded in {load_s:.0f}s | dim {enc.dim} | VRAM {vram_load:.2f} GB")

    t0 = time.time()
    Z = enc.embed_many(wavs, progress_every=400).astype(np.float64)
    embed_s = time.time() - t0
    vram_peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"      embedded {len(Z)} in {embed_s:.0f}s "
          f"({len(Z)/embed_s:.1f}/s) | peak VRAM {vram_peak:.2f} GB")

    # --------------------------------------- in-domain space, disjoint half
    uniq = sorted(set(ids)); idsa = np.asarray(ids)
    rng = np.random.default_rng(0)
    half = {uniq[i] for i in rng.permutation(len(uniq))[:len(uniq)//2]}
    fit_mask = np.array([s in half for s in ids])
    space = SpeakerSpace.fit(Z[fit_mask], n_components=min(256, int(fit_mask.sum())-1))

    raw = verification_stats(Z, ids)
    work = verification_stats(space.encode(Z), ids)
    print(f"      EER raw {raw.eer*100:.2f}%  ->  in-domain space "
          f"{work.eer*100:.2f}%  (d' {work.d_prime:.2f})")
    print(f"      {space}")

    # --------------------------------------------------------------- drift
    del enc; torch.cuda.empty_cache(); gc.collect()
    from alaap.renderer import Qwen3BaseRenderer, load_backend
    r = Qwen3BaseRenderer(model_id); load_backend(r, is_public_deployment=True)
    sv = IndependentSV()

    # one centroid per speaker, take the first n_drift
    cents = []
    for s in uniq[:args.n_drift]:
        cents.append((s, Z[idsa == s].mean(0)))
    drifts, rtfs, ecs = [], [], []
    for s, v in cents:
        t0 = time.time()
        try:
            au = r.render_from_vector(v.astype(np.float32), args.line, "en")
        except Exception as e:
            print(f"      !! render {s}: {e}"); continue
        dt = time.time() - t0
        dur = len(au.wav) / au.sample_rate
        back = r.extract_vector(au.wav, au.sample_rate).astype(np.float64)
        a, b = space.encode(v)[0], space.encode(back)[0]
        drifts.append(float(a @ b / max(np.linalg.norm(a)*np.linalg.norm(b), 1e-12)))
        rtfs.append(dt / max(dur, 1e-6))
        ecs.append(sv.embed(au.wav, sr=au.sample_rate))
    render_peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"      drift (working space) {np.mean(drifts):.4f} "
          f"+/- {np.std(drifts):.4f} | RTF {np.mean(rtfs):.2f} "
          f"| peak VRAM {render_peak:.2f} GB")

    results[tag] = {
        "loaded": True, "dim": enc_dim if (enc_dim := int(Z.shape[1])) else None,
        "load_s": load_s, "vram_load_gb": vram_load, "vram_peak_gb": render_peak,
        "embed_per_s": len(Z)/embed_s,
        "eer_raw": raw.eer, "eer_indomain": work.eer,
        "d_prime_indomain": work.d_prime,
        "c_same": work.same_mean, "c_diff": work.diff_mean,
        "effective_rank": space.stats.effective_rank,
        "identity_fraction": space.stats.identity_fraction,
        "shell_radius": space.stats.shell_radius,
        "drift_mean": float(np.mean(drifts)) if drifts else None,
        "drift_std": float(np.std(drifts)) if drifts else None,
        "rtf": float(np.mean(rtfs)) if rtfs else None,
    }
    del r, sv; torch.cuda.empty_cache(); gc.collect()

json.dump(results, open(os.path.join(OUT, "results.json"), "w"), indent=2)

print()
print("=" * 86)
print("E10 RESULT — 0.6B vs 1.7B")
print("=" * 86)
ok = [k for k in results if results[k].get("loaded")]
if not ok:
    print("  nothing loaded")
else:
    rows = [("dim", "dim", "{:.0f}"), ("VRAM peak GB", "vram_peak_gb", "{:.2f}"),
            ("embed clips/s", "embed_per_s", "{:.1f}"), ("RTF", "rtf", "{:.2f}"),
            ("EER raw %", "eer_raw", "{:.2%}"),
            ("EER in-domain %", "eer_indomain", "{:.2%}"),
            ("d-prime", "d_prime_indomain", "{:.2f}"),
            ("effective rank", "effective_rank", "{:.1f}"),
            ("identity fraction", "identity_fraction", "{:.3f}"),
            ("shell radius", "shell_radius", "{:.2f}"),
            ("drift (working)", "drift_mean", "{:.4f}")]
    print(f"  {'':<20}" + "".join(f"{k:>26}" for k in ok))
    print("  " + "-"*20 + "".join(" " + "-"*25 for _ in ok))
    for lbl, key, fmt in rows:
        cells = []
        for k in ok:
            v = results[k].get(key)
            cells.append(fmt.format(v) if v is not None else "-")
        print(f"  {lbl:<20}" + "".join(f"{c:>26}" for c in cells))
print()
print(f"  reference: 0.6B measured EER 4.35% in-domain (E4), drift 0.5519 (E2)")
print("=" * 86)
