"""
Convert microsoft/wavlm-base-plus-sv to safetensors.

Why this exists: the upstream repo ships pytorch_model.bin only, and
transformers 4.57 refuses torch.load on torch < 2.6 (CVE-2025-32434).
Upgrading torch would break qwen-tts, which pins transformers==4.57.3.
Converting once is the option that neither disables a security check nor
destabilises the environment.

    envs/qwen3/Scripts/python.exe scripts/convert_wavlm.py
"""
import os, shutil, torch, warnings
warnings.filterwarnings("ignore")
from huggingface_hub import snapshot_download
from safetensors.torch import save_file

src = snapshot_download("microsoft/wavlm-base-plus-sv")
dst = os.path.abspath("cache/wavlm-base-plus-sv")
os.makedirs(dst, exist_ok=True)
for f in os.listdir(src):
    if f.startswith(".") or f == "pytorch_model.bin":
        continue
    p = os.path.join(src, f)
    if os.path.isfile(p):
        shutil.copy2(p, dst)
sd = torch.load(os.path.join(src, "pytorch_model.bin"), map_location="cpu",
                weights_only=True)
sd = {k: v.clone() for k, v in sd.items() if hasattr(v, "shape")}
save_file(sd, os.path.join(dst, "model.safetensors"), metadata={"format": "pt"})
print("converted ->", dst)
