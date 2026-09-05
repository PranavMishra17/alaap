"""
End-to-end demo — the "minimum shippable thing" (GRAND-PLAN section 5).

    description -> minted identity (both tiers, stored)
                -> multi-line script render with per-line Direction
                -> watermarked audio + a game-engine manifest

This is S0 + S1 + a slice of S6/S8 running as one pipeline, and it exercises
every invariant at once:

    I2  each identity stores a vector AND a seed clip AND backend_version
    I3  the licence gate refuses an unaudited backend at construction
    I5  Direction is per line; identities are minted neutral
    I7  every render watermarked and provenance-logged

Runs on 1.7B with the v2 (decorrelated) binner -- the only corpus/binner
pair whose numbers the project still believes.

    envs/qwen3/Scripts/python.exe scripts/demo_script_render.py
"""
import json, os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from alaap.acoustics import Attributes, Binner
from alaap.captions import caption_from_bins
from alaap.geometry import SpeakerSpace
from alaap.encoder import IndependentSV
from alaap.mapper import TextEncoder, RetrievalMapper
from alaap.identity import IdentityStore
from alaap.renderer import Qwen3BaseRenderer, Direction
from alaap.watermark import Watermarker
from alaap.service import VoiceService

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
OUT = os.path.abspath("demo_out")
os.makedirs(OUT, exist_ok=True)

CAST = [
    ("mentor",   "a very deep voice, very rough and gravelly, speaking very slowly"),
    ("scout",    "a high voice, crisp-toned, speaking quickly and highly animated"),
    ("innkeeper","a mid-range voice, warm-toned, at a steady pace"),
]

SCRIPT = [
    ("mentor",   "The mountains remember every footstep, even the ones you regret.",
     None),
    ("scout",    "There's smoke on the ridge. Two fires, maybe three.",
     Direction(emotion={"fear": 1.0}, intensity=0.6)),
    ("innkeeper","Sit down, both of you. Nobody rides out on an empty stomach.",
     None),
    ("mentor",   "Then we leave at first light, and we do not look back.",
     Direction(emotion={"sad": 1.0}, intensity=0.5)),
    ("scout",    "I told you exactly what would happen!",
     Direction(emotion={"anger": 1.0}, intensity=0.7)),
]

# ------------------------------------------------------------ 1. the mapper
print("[1/5] loading the mapper (cached GLOBE_V2 corpus)")
# 1.7B corpus + the v2 (decorrelated) binner. The 0.6B artefacts next to
# these are stale in two ways that both surfaced as hard errors: their
# cached attributes predate f0_cv, so Attributes(**a) raises, and their
# binner is v1, which Binner.load refuses because it bins raw f0_std and
# uncorrected hnr_db -- the entangled axes S2 run 3 got wrong.
d = np.load("experiments/S2/out/"
            "corpus_globe_v2_2500_1_Qwen3-TTS-12Hz-17B-Base.npz", allow_pickle=True)
Z = d["Z"].astype(np.float64)
# from_dict, not Attributes(**a): it derives fields added after a corpus was
# measured instead of raising on them.
attrs = [Attributes.from_dict(a) for a in json.loads(str(d["attrs"]))]
binner = Binner.load("experiments/S2/out/"
                     "binner_globe_v2_Qwen3-TTS-12Hz-17B-Base.json")
caps = [caption_from_bins(binner.bin_one(a), seed=i) for i, a in enumerate(attrs)]
space = SpeakerSpace.fit(Z, n_components=150)
mapper = RetrievalMapper(space, TextEncoder(), pca_dims=50).fit(caps, Z)
print(f"      {len(caps)} pairs | {space}")

# ------------------------------------------------------------ 2. the service
print("[2/5] building the service (licence gate + watermark + store)")
store = IdentityStore(os.path.join(OUT, "identities.db"))
svc = VoiceService(renderer=Qwen3BaseRenderer(MODEL), space=space, store=store,
                   mapper=mapper, watermarker=Watermarker(),
                   sv=IndependentSV(), audio_dir=os.path.join(OUT, "audio"),
                   is_public=True)
print(f"      backend {svc.r.backend_id} | space_ref {svc.space_ref}")
print(f"      tau available: {sorted(svc.r.tau)}")

# --------------------------------------------------------------- 3. mint
print(f"[3/5] minting {len(CAST)} characters "
      f"(novelty 0.0; service.mint escalates it only on a collision -- E11)")
ids = {}
for name, desc in CAST:
    t0 = time.time()
    out = svc.mint(desc, character_id=name, language="en", novelty=0.0,
                   verify=True, tags=["demo"])
    ids[name] = out.identity.identity_id
    print(f"      {name:<10} tier={out.identity.tier} "
          f"drift={out.drift:.3f} " if out.drift else f"      {name:<10} ",
          end="")
    print(f"consistency={out.consistency:.3f} " if out.consistency else "",
          end="")
    print(f"uniq={out.uniqueness:.3f} attempts={out.attempts} "
          f"({time.time()-t0:.0f}s)")
    for w in out.warnings:
        print(f"        ! {w}")

# ------------------------------------------------------------- 4. render
print(f"[4/5] rendering a {len(SCRIPT)}-line script")
lines, t0 = [], time.time()
for i, (who, text, direction) in enumerate(SCRIPT):
    a, rid = svc.render(ids[who], text, direction=direction)
    lines.append({"line_id": i, "render_id": rid, "character": who,
                  "text": text, "duration_s": round(len(a.wav)/a.sample_rate, 2),
                  "watermarked": a.watermarked,
                  "direction": ({k: v for k, v in direction.__dict__.items()
                                 if v is not None and k != "strict"}
                                if direction else None),
                  "degradations": a.degradations})
    print(f"      [{i}] {who:<10} {len(a.wav)/a.sample_rate:>5.2f}s  "
          f"wm={a.watermarked}  {text[:44]}")
    for g in a.degradations:
        print(f"          ~ {g}")
print(f"      rendered in {time.time()-t0:.0f}s")

# ---------------------------------------------------- 5. manifest + verify
print("[5/5] manifest + watermark verification")
manifest = {
    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "disclosure": "AI-generated synthetic speech. All audio carries an "
                  "AudioSeal watermark (presence bit). Required under EU AI "
                  "Act Article 50(2), in force since 2026-08-02.",
    "backend": f"{svc.r.backend_id}@{svc.r.backend_version}",
    "space_ref": svc.space_ref,
    "characters": [{"character_id": n, "identity_id": ids[n],
                    "description": d, "tier": store.get(ids[n]).tier}
                   for n, d in CAST],
    "lines": lines,
}
json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=2)

# verify the watermark actually survives on disk
import soundfile as sf, glob
wm = svc.wm
checked = []
for f in sorted(glob.glob(os.path.join(OUT, "audio", "render_*.wav")))[:5]:
    w, sr = sf.read(f, dtype="float32")
    checked.append(wm.detect(w, sr).probability)
print(f"      watermark detection on {len(checked)} rendered files: "
      f"{[round(x,3) for x in checked]}")

st = store.stats()
print()
print("=" * 78)
print("DEMO COMPLETE")
print("=" * 78)
print(f"  identities   {st['identities']} ({st['tier1']} tier-1, {st['tier2']} tier-2)")
print(f"  characters   {st['characters']} | languages {st['languages']}")
print(f"  renders      {st['renders']} ({st['renders_watermarked']} watermarked)")
print(f"  audio        {OUT}\\audio")
print(f"  manifest     {OUT}\\manifest.json")
print(f"  provenance   {OUT}\\identities.db  (render_log table)")
print("=" * 78)
