"""
Preflight — one command that answers "what is blocked right now?"

Written because the answer was spread across NEEDS-FROM-YOU.md, DECISIONS.md,
two research documents and a stale binner file, and because two of the things
that bit hardest this week were both *silent*: a licence gate that disagreed
with its own audit, and a v1 binner that would have re-entangled the acoustic
axes. Neither raised anything until something downstream produced a wrong
number.

This checks the things that fail quietly. It is not a test suite -- the tests
cover invariants in code. This covers the ENVIRONMENT: credentials, corpus
reachability, artefact versions, and whether the code still agrees with the
audit.

Exit code is the number of BLOCKED checks, so it can gate a script.

    envs/qwen3/Scripts/python.exe scripts/preflight.py
    envs/qwen3/Scripts/python.exe scripts/preflight.py --network   # slower
"""
import argparse
import io
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", write_through=True)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

ap = argparse.ArgumentParser()
ap.add_argument("--network", action="store_true",
                help="also probe the Hub (slow, needs connectivity)")
args = ap.parse_args()

OK, WARN, BLOCK = "ok", "warn", "BLOCKED"
results: list[tuple[str, str, str]] = []


def check(name):
    def deco(fn):
        try:
            status, detail = fn()
        except Exception as e:
            status, detail = BLOCK, f"{type(e).__name__}: {str(e)[:110]}"
        results.append((name, status, detail))
        return fn
    return deco


# --------------------------------------------------------------- credentials
@check("HF token")
def _hf():
    tok = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
           or "")
    if not tok and os.path.exists(".hf_token"):
        tok = io.open(".hf_token", encoding="utf-8").read().strip()
    if not tok:
        p = os.path.expanduser("~/.cache/huggingface/token")
        if os.path.exists(p):
            tok = io.open(p, encoding="utf-8").read().strip()
    if not tok:
        return BLOCK, ("absent. Gated corpora (ai4bharat/indicvoices_r, Rasa, "
                       "IndicVoices) and indic-parler-tts are unreachable. "
                       "See NEEDS-FROM-YOU section 1")
    return OK, f"present ({tok[:6]}...{len(tok)} chars)"


@check("Parler gate terms")
def _gate():
    for p in ("GATE-TERMS-indic-parler.txt", "docs/GATE-TERMS-indic-parler.txt"):
        if os.path.exists(p) and os.path.getsize(p) > 40:
            return OK, f"captured in {p}"
    return BLOCK, ("not captured. indic-parler-tts stays public_servable=False "
                   "until somebody reads what the gate actually says "
                   "(RESEARCH/08 section 4.7)")


# --------------------------------------------------------- code vs the audit
@check("SERVABLE matches RESEARCH/08")
def _servable():
    from alaap.renderer import SERVABLE
    audit = {"indic-parler-tts": False, "indicf5": False, "indic-mio": False,
             "dhvaani": False, "f5-tts": False, "xtts-v2": False,
             "llasa-3b": False, "zonos-v0.1": False, "vibevoice": False,
             "parler-tts": True, "cosyvoice2": True, "chatterbox": True,
             "voxcpm2": True}
    bad = [f"{k}: code={SERVABLE.get(k)} audit={v}"
           for k, v in audit.items() if SERVABLE.get(k) != v]
    if bad:
        return BLOCK, "; ".join(bad)
    return OK, f"{len(audit)} audited backends agree"


@check("no Indic language on the working backend")
def _lang():
    from alaap.renderer import Qwen3BaseRenderer
    langs = set(Qwen3BaseRenderer.LANG_ALIAS)
    indic = {"hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "or"}
    if langs & indic:
        return OK, f"backend now speaks {sorted(langs & indic)}"
    return WARN, ("expected. Qwen3-TTS covers "
                  f"{len(langs)} languages, none Indian -- ADR-006. "
                  "Indic rendering needs a different tower")


# --------------------------------------------------------------- artefacts
@check("binner versions")
def _binners():
    import glob
    v1, v2 = [], []
    for p in glob.glob("experiments/**/binner*.json", recursive=True):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        (v2 if d.get("version", 1) >= 2 else v1).append(os.path.basename(p))
    if not v2:
        return BLOCK, ("no v2 binner exists. v1 bins raw f0_std and "
                       "uncorrected hnr_db, both entangled with pitch")
    if v1:
        return WARN, (f"{len(v2)} v2, {len(v1)} v1 present. Binner.load "
                      f"refuses v1, so a stale default is a hard stop, not a "
                      f"silent error. v1: {', '.join(sorted(v1)[:3])}")
    return OK, f"{len(v2)} v2 binners, no v1"


@check("emotion tau for the DEFAULT model")
def _tau():
    """
    Checking a hardcoded 0.6B path reported 'ok' while the DEFAULT model had no
    usable tau at all -- which is exactly how emotion came to do nothing on
    1.7B without anyone noticing. tau is per-model, so check the one the
    default backend would actually load.
    """
    import glob
    import numpy as np
    from alaap.encoder import DEFAULT_MODEL
    tag = "1.7B" if "1.7B" in DEFAULT_MODEL else "0.6B"
    want = 2048 if tag == "1.7B" else 1024
    p = f"assets/emotion_tau_qwen3_{tag}.npz"
    others = [os.path.basename(x) for x in glob.glob("assets/emotion_tau_*.npz")
              if os.path.basename(x) != os.path.basename(p)]
    if not os.path.exists(p):
        return BLOCK, (f"{p} absent, so Direction(emotion=...) renders NEUTRAL "
                       f"on {DEFAULT_MODEL.split('/')[-1]}. tau is per-model; "
                       f"rebuild with scripts/fit_emotion_tau.py"
                       + (f" (present but unusable here: {', '.join(others)})"
                          if others else ""))
    d = np.load(p)
    # the archive also carries provenance keys (model, n_speakers, source,
    # dim) -- counting those as emotions overstates what Direction can do
    emo = sorted(k[4:] for k in d.files if k.startswith("tau_"))
    got = int(d["dim"]) if "dim" in d.files else None
    if got is not None and got != want:
        return BLOCK, (f"{p} is {got}-d but {tag} needs {want}-d -- it will be "
                       f"dropped at load and emotion will render NEUTRAL")
    spk = f"{int(d['n_speakers'])} speakers" if "n_speakers" in d.files else ""
    return OK, f"{len(emo)} emotions ({', '.join(emo)}) {spk} @ {want}-d".strip()


@check("dev-only corpora are gated in code")
def _devonly():
    from alaap.data import DEV_ONLY, stream_clips, LicenceError
    if not DEV_ONLY:
        return WARN, "DEV_ONLY is empty"
    try:
        stream_clips(sorted(DEV_ONLY)[0], n=1)
        return BLOCK, "an unlicensed mirror streamed without dev_only=True"
    except LicenceError:
        return OK, f"{len(DEV_ONLY)} mirrors refuse to stream unacknowledged"


# ----------------------------------------------------------------- network
if args.network:
    @check("gated Indic corpora reachable")
    def _gated():
        from datasets import get_dataset_config_names
        try:
            get_dataset_config_names("ai4bharat/indicvoices_r")
            return OK, "ai4bharat/indicvoices_r is readable"
        except Exception as e:
            return BLOCK, f"still gated: {str(e)[:90]}"

    @check("ungated mirror reachable")
    def _mirror():
        from datasets import load_dataset
        ds = load_dataset("SPRINGLab/IndicVoices-R_Hindi", split="train",
                          streaming=True)
        r = next(iter(ds.take(1)))
        return OK, f"streams; fields include {sorted(r)[:4]}"


# ------------------------------------------------------------------ report
W = max(len(n) for n, _, _ in results)
print()
print("=" * 78)
print("ALAAP PREFLIGHT")
print("=" * 78)
mark = {OK: "  ok   ", WARN: " warn  ", BLOCK: "BLOCKED"}
for name, status, detail in results:
    print(f"  [{mark[status]}] {name:<{W}}  {detail}")
blocked = sum(1 for _, s, _ in results if s == BLOCK)
warned = sum(1 for _, s, _ in results if s == WARN)
print("-" * 78)
print(f"  {len(results)} checks | {blocked} blocked | {warned} warnings")
if blocked:
    print("  -> read NEEDS-FROM-YOU.md; the blocked items need a human")
if not args.network:
    print("  (--network also probes the Hub)")
print("=" * 78)
sys.exit(blocked)
