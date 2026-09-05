import io, json, os, sys, warnings
warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", write_through=True)
import numpy as np
sys.path.insert(0, os.path.abspath("."))
from alaap.acoustics import Attributes, Binner, BIN_LABELS
from alaap.geometry import SpeakerSpace

FIVE = ["f0_mean","spectral_tilt","hnr_db","f0_cv","speaking_rate"]

def spearman(x,y):
    x,y=np.asarray(x,float),np.asarray(y,float)
    rx=np.argsort(np.argsort(x)).astype(float); ry=np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx,ry)[0,1])

def unit(A):
    A=np.asarray(A,float); return A/(np.linalg.norm(A,axis=1,keepdims=True)+1e-12)

def run(name, path, dedup):
    d=np.load(path, allow_pickle=True)
    Z=d["Z"].astype(np.float64)
    attrs=[Attributes.from_dict(a) for a in json.loads(str(d["attrs"]))]
    if dedup and "ids" in d.files:
        ids=[str(x) for x in d["ids"]]
        seen,keep=set(),[]
        for i,s in enumerate(ids):
            if s not in seen: seen.add(s); keep.append(i)
        Z=Z[keep]; attrs=[attrs[i] for i in keep]
        note=f"deduplicated {len(ids)} clips -> {len(keep)} speakers"
    else:
        note="one clip per speaker already"
    n=len(Z)
    space=SpeakerSpace.fit(Z, n_components=min(150,n-1))
    E=space.encode(Z)
    binner=Binner.fit(attrs)
    BIDX={a:{l:k for k,l in enumerate(BIN_LABELS[a])} for a in FIVE}
    bins=[binner.bin_one(a) for a in attrs]
    B=np.array([[BIDX[a][b[a]] for a in FIVE] for b in bins],float)
    rng=np.random.default_rng(0)
    iu,ju=np.triu_indices(n,k=1)
    if len(iu)>80000:
        s=rng.choice(len(iu),80000,replace=False); iu,ju=iu[s],ju[s]
    dv=(1.0-unit(E)@unit(E).T)[iu,ju]
    rho5=spearman(np.abs(B[iu]-B[ju]).sum(1), dv)
    rho_f0=spearman(np.abs(B[iu,0]-B[ju,0]), dv)
    print(f"  {name}")
    print(f"    {note}; {n} speakers, {Z.shape[1]}-dim embeddings, {len(iu):,} pairs")
    print(f"    5-axis bin distance vs voice distance   rho = {rho5:.3f}")
    print(f"    f0 alone                                rho = {rho_f0:.3f}")
    return rho5, rho_f0

print("  Does E14's finding survive a different corpus AND a different encoder?")
print()
a=run("GLOBE_V2 / 1.7B (E14's corpus)",
      "experiments/S2/out/corpus_globe_v2_2500_1_Qwen3-TTS-12Hz-17B-Base.npz", False)
print()
b=run("LibriTTS-R train.clean.360 / 0.6B",
      "experiments/S2/out/corpus_libritts_r_360_2000_2.npz", True)
print()
print(f"  5-axis rho: {a[0]:.3f} vs {b[0]:.3f}")
print(f"  f0 alone  : {a[1]:.3f} vs {b[1]:.3f}")
print(f"  f0 beats the 5-axis sum on both corpora: "
      f"{a[1]>a[0]} / {b[1]>b[0]}")
